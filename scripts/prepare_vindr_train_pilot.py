#!/usr/bin/env python3
"""
Select a small, diverse VinDr-CXR training pilot and create the three inputs
required by GL-LCM:

    CXR/<image>.png
    Mask/<image>.png
    Masked_CXR/<image>.png

The official GL-LCM repository uses lungs-segmentation for mask preparation.
This script calls that package's published inference API and then cleans the
union of the left/right lung masks.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as lung_inference


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a 5-10-image VinDr train pilot for GL-LCM."
    )
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotations_csv", type=Path, default=None)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument(
        "--preferred_labels",
        type=str,
        default=(
            "Cardiomegaly,Pleural effusion,Lung Opacity,"
            "Aortic enlargement,No finding"
        ),
        help="Comma-separated labels used to make the sample diverse.",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="resnet34",
        choices=["resnet34", "densenet121"],
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.20,
        help="Probability threshold passed to lungs_segmentation.inference.",
    )
    parser.add_argument(
        "--min_component_area_frac",
        type=float,
        default=0.005,
        help="Remove connected components below this image-area fraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate pilot files even when they already exist.",
    )
    return parser.parse_args()


def discover_images(images_root: Path) -> dict[str, Path]:
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    image_map: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_map.setdefault(path.stem, path)

    if not image_map:
        raise RuntimeError(f"No supported images found under: {images_root}")

    return image_map


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def select_image_ids(
    image_map: dict[str, Path],
    annotations_csv: Path | None,
    num_images: int,
    preferred_labels: list[str],
    seed: int,
) -> tuple[list[str], dict[str, list[str]]]:
    rng = random.Random(seed)
    available_ids = sorted(image_map)

    if num_images < 1:
        raise ValueError("--num_images must be at least 1")
    num_images = min(num_images, len(available_ids))

    labels_by_image: dict[str, list[str]] = {image_id: [] for image_id in available_ids}

    if annotations_csv is None or not annotations_csv.is_file():
        chosen = rng.sample(available_ids, num_images)
        return chosen, labels_by_image

    ann = pd.read_csv(annotations_csv)
    id_col = find_column(
        ann.columns,
        ["image_id", "dicom_id", "image", "id"],
    )
    class_col = find_column(
        ann.columns,
        ["class_name", "label", "finding", "finding_name"],
    )

    if id_col is None:
        id_col = str(ann.columns[0])

    ann[id_col] = ann[id_col].astype(str).map(lambda x: Path(x).stem)
    ann = ann[ann[id_col].isin(image_map)].copy()

    if class_col is not None:
        ann[class_col] = ann[class_col].astype(str)
        grouped = ann.groupby(id_col)[class_col].apply(
            lambda values: sorted(set(map(str, values)))
        )
        for image_id, labels in grouped.items():
            labels_by_image[str(image_id)] = list(labels)

    chosen: list[str] = []
    used: set[str] = set()

    if class_col is not None:
        for preferred_label in preferred_labels:
            candidates = ann.loc[
                ann[class_col].str.casefold() == preferred_label.casefold(),
                id_col,
            ].drop_duplicates().astype(str).tolist()
            candidates = [x for x in candidates if x not in used]
            if candidates:
                selected = rng.choice(sorted(candidates))
                chosen.append(selected)
                used.add(selected)
                if len(chosen) >= num_images:
                    break

    remaining = [image_id for image_id in available_ids if image_id not in used]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, num_images - len(chosen))])

    return chosen, labels_by_image


def to_uint8_grayscale(path: Path, size: int) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read: {path}")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image_float = image.astype(np.float32)
        finite = np.isfinite(image_float)
        if not finite.any():
            raise RuntimeError(f"Image contains no finite values: {path}")
        lo, hi = np.percentile(image_float[finite], [0.5, 99.5])
        if hi <= lo:
            lo = float(image_float[finite].min())
            hi = float(image_float[finite].max())
        if hi <= lo:
            image = np.zeros_like(image_float, dtype=np.uint8)
        else:
            image = np.clip((image_float - lo) / (hi - lo), 0.0, 1.0)
            image = np.rint(image * 255.0).astype(np.uint8)

    interpolation = cv.INTER_AREA if max(image.shape[:2]) >= size else cv.INTER_CUBIC
    image = cv.resize(image, (size, size), interpolation=interpolation)
    return image


def union_model_masks(mask_object: object) -> np.ndarray:
    """
    Convert the package's left/right mask return value to one 2D binary mask.
    Handles list, tuple, [2,H,W], [1,2,H,W], [H,W,2], or a single [H,W] mask.
    """
    if isinstance(mask_object, (list, tuple)):
        arrays = [np.asarray(x) for x in mask_object]
        arrays = [np.squeeze(x) for x in arrays if np.asarray(x).size > 0]
        if not arrays:
            raise RuntimeError("Lung segmentation returned an empty mask list")
        target_h = max(arr.shape[-2] for arr in arrays)
        target_w = max(arr.shape[-1] for arr in arrays)
        resized = [
            cv.resize(
                arr.astype(np.float32),
                (target_w, target_h),
                interpolation=cv.INTER_NEAREST,
            )
            for arr in arrays
        ]
        return np.any(np.stack(resized, axis=0) > 0, axis=0)

    arr = np.asarray(mask_object)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr > 0

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected lung-mask shape: {arr.shape}")

    # The class/channel axis is normally the smallest axis (usually size 2).
    channel_axis = int(np.argmin(arr.shape))
    arr = np.moveaxis(arr, channel_axis, 0)
    return np.any(arr > 0, axis=0)


def clean_mask(
    mask: np.ndarray,
    output_size: int,
    min_component_area_frac: float,
) -> np.ndarray:
    mask = cv.resize(
        mask.astype(np.uint8),
        (output_size, output_size),
        interpolation=cv.INTER_NEAREST,
    )
    mask = mask > 0

    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, close_kernel)
    mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, open_kernel)
    mask = ndimage.binary_fill_holes(mask_u8 > 0)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    min_area = max(
        1,
        int(round(min_component_area_frac * output_size * output_size)),
    )

    components: list[tuple[int, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label_id))

    # A valid frontal CXR should usually contain two lung components. Keep the
    # two largest valid components, while allowing a connected bilateral mask.
    components.sort(reverse=True)
    keep_ids = [label_id for _, label_id in components[:2]]

    if not keep_ids:
        raise RuntimeError(
            "No valid lung-mask component survived post-processing. "
            "Inspect this image manually."
        )

    cleaned = np.isin(labels, keep_ids)
    cleaned = ndimage.binary_fill_holes(cleaned)

    coverage = float(cleaned.mean())
    if not 0.10 <= coverage <= 0.80:
        raise RuntimeError(
            f"Implausible lung-mask coverage {coverage:.3f}; expected 0.10-0.80"
        )

    return cleaned.astype(np.uint8) * 255


def count_components(mask_u8: np.ndarray) -> int:
    n, _, _, _ = cv.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8),
        connectivity=8,
    )
    return max(0, n - 1)


def main() -> None:
    args = parse_args()

    image_map = discover_images(args.images_root)
    preferred_labels = [
        x.strip() for x in args.preferred_labels.split(",") if x.strip()
    ]

    selected_ids, labels_by_image = select_image_ids(
        image_map=image_map,
        annotations_csv=args.annotations_csv,
        num_images=args.num_images,
        preferred_labels=preferred_labels,
        seed=args.seed,
    )

    cxr_dir = args.output_dir / "CXR"
    mask_dir = args.output_dir / "Mask"
    masked_dir = args.output_dir / "Masked_CXR"
    for directory in (cxr_dir, mask_dir, masked_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Found {len(image_map)} VinDr train images")
    print(f"[Data] Selected {len(selected_ids)} images: {selected_ids}")
    print(f"[LungSeg] Loading model: {args.segmentation_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lung_model = create_model(args.segmentation_model)
    lung_model = lung_model.to(device).eval()
    print(f"[LungSeg] Device: {device}")

    manifest_rows: list[dict[str, object]] = []

    for image_id in tqdm(selected_ids, desc="Preparing VinDr pilot"):
        source_path = image_map[image_id]
        output_name = f"{image_id}.png"
        cxr_path = cxr_dir / output_name
        mask_path = mask_dir / output_name
        masked_path = masked_dir / output_name

        if (
            not args.overwrite
            and cxr_path.is_file()
            and mask_path.is_file()
            and masked_path.is_file()
        ):
            cxr = cv.imread(str(cxr_path), cv.IMREAD_GRAYSCALE)
            mask_u8 = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            if cxr is None or mask_u8 is None:
                raise RuntimeError(f"Could not reopen existing pilot files for {image_id}")
        else:
            cxr = to_uint8_grayscale(source_path, args.image_size)
            if not cv.imwrite(str(cxr_path), cxr):
                raise RuntimeError(f"Failed to save: {cxr_path}")

            with torch.inference_mode():
                _, raw_masks = lung_inference.inference(
                    lung_model,
                    str(cxr_path),
                    args.mask_threshold,
                )

            union_mask = union_model_masks(raw_masks)
            mask_u8 = clean_mask(
                union_mask,
                output_size=args.image_size,
                min_component_area_frac=args.min_component_area_frac,
            )

            masked_cxr = cv.bitwise_and(cxr, cxr, mask=mask_u8)

            if not cv.imwrite(str(mask_path), mask_u8):
                raise RuntimeError(f"Failed to save: {mask_path}")
            if not cv.imwrite(str(masked_path), masked_cxr):
                raise RuntimeError(f"Failed to save: {masked_path}")

        manifest_rows.append(
            {
                "image_id": image_id,
                "filename": output_name,
                "source_path": str(source_path),
                "labels": "|".join(labels_by_image.get(image_id, [])),
                "image_size": args.image_size,
                "mask_coverage": float((mask_u8 > 0).mean()),
                "mask_components": count_components(mask_u8),
                "cxr_path": str(cxr_path),
                "mask_path": str(mask_path),
                "masked_cxr_path": str(masked_path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "pilot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config_path = args.output_dir / "pilot_prepare_config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "images_root": str(args.images_root),
                "annotations_csv": (
                    str(args.annotations_csv) if args.annotations_csv else None
                ),
                "num_images": len(selected_ids),
                "seed": args.seed,
                "image_size": args.image_size,
                "preferred_labels": preferred_labels,
                "segmentation_model": args.segmentation_model,
                "mask_threshold": args.mask_threshold,
                "min_component_area_frac": args.min_component_area_frac,
                "device": str(device),
            },
            f,
            indent=2,
        )

    file_list_path = args.output_dir / "pilot_files.txt"
    file_list_path.write_text(
        "".join(f"{image_id}.png\n" for image_id in selected_ids)
    )

    print(f"[Output] Manifest: {manifest_path}")
    print(f"[Output] File list: {file_list_path}")
    print(manifest[["image_id", "labels", "mask_coverage", "mask_components"]])


if __name__ == "__main__":
    main()
