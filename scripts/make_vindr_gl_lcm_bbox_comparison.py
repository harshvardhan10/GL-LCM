#!/usr/bin/env python3
"""
Create pathology-box overlays and zoomed pathology crops for a VinDr-CXR
GL-LCM pilot.

Expected pilot layout:
    PILOT_DIR/
        CXR/
        Mask/
        outputs_szch/Fusion_BS/
        outputs_jsrt/Fusion_BS/
        pilot_manifest.csv

The VinDr box coordinates are defined on the original VinDr image. The pilot
preparation script resized each image directly to a square (normally 1024x1024),
so this script scales x and y coordinates independently into pilot coordinates.

Outputs:
    OUTPUT_DIR/
        vindr_train_gl_lcm_bbox_comparison.png
        full/<image_id>_boxed.png
        crops/<image_id>__<class>__<index>.png
        bbox_review_index.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_CANDIDATES = ["image_id", "dicom_id", "image", "id"]
CLASS_CANDIDATES = ["class_name", "label", "finding", "finding_name"]
RAD_CANDIDATES = ["rad_id", "radiologist_id", "reader_id"]
XMIN_CANDIDATES = ["x_min", "xmin", "x1", "left"]
YMIN_CANDIDATES = ["y_min", "ymin", "y1", "top"]
XMAX_CANDIDATES = ["x_max", "xmax", "x2", "right"]
YMAX_CANDIDATES = ["y_max", "ymax", "y2", "bottom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay VinDr pathology boxes on Original/SZCH/JSRT pilot images "
            "and create enlarged pathology crops."
        )
    )
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--annotations_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--szch_dir", type=Path, default=None)
    parser.add_argument("--jsrt_dir", type=Path, default=None)
    parser.add_argument(
        "--include_labels",
        type=str,
        default="",
        help="Optional comma-separated pathology labels. Empty means all boxed labels.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.50,
        help=(
            "Suppress strongly overlapping same-class boxes, usually duplicate "
            "annotations from different radiologists. Set to 1.0 to keep all boxes."
        ),
    )
    parser.add_argument(
        "--crop_context",
        type=float,
        default=0.30,
        help="Fractional context added around each pathology box.",
    )
    parser.add_argument("--display_size", type=int, default=384)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 means all pilot images.",
    )
    return parser.parse_args()


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    exact = {str(c).strip().casefold(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.casefold() in exact:
            return exact[candidate.casefold()]
    return None


def load_gray(path: Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_original_shape(source_path: str, fallback_shape: tuple[int, int]) -> tuple[int, int]:
    path = Path(source_path)
    if path.is_file():
        image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if image is not None:
            return int(image.shape[0]), int(image.shape[1])
    return fallback_shape


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "finding"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_same_class(boxes: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    """Keep representative boxes while removing strongly overlapping duplicates."""
    if boxes.empty or iou_threshold >= 1.0:
        return boxes.copy()

    kept_frames: list[pd.DataFrame] = []
    for _, group in boxes.groupby("class_name", sort=False):
        group = group.copy()
        group["_area"] = (
            (group["x2"] - group["x1"]).clip(lower=0)
            * (group["y2"] - group["y1"]).clip(lower=0)
        )
        group = group.sort_values("_area", ascending=False)

        selected_indices: list[int] = []
        selected_boxes: list[np.ndarray] = []
        for index, row in group.iterrows():
            box = row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if all(iou_xyxy(box, existing) < iou_threshold for existing in selected_boxes):
                selected_indices.append(index)
                selected_boxes.append(box)

        kept_frames.append(group.loc[selected_indices].drop(columns="_area"))

    if not kept_frames:
        return boxes.iloc[0:0].copy()
    return pd.concat(kept_frames, ignore_index=True)


def scale_boxes(
    rows: pd.DataFrame,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> pd.DataFrame:
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    result = rows.copy()

    coords = result[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if coords.size == 0:
        return result

    # Support both absolute-pixel and normalized [0,1] annotations.
    if np.nanmax(coords) <= 1.5:
        result["x1"] *= source_w
        result["x2"] *= source_w
        result["y1"] *= source_h
        result["y2"] *= source_h

    result["x1"] *= target_w / source_w
    result["x2"] *= target_w / source_w
    result["y1"] *= target_h / source_h
    result["y2"] *= target_h / source_h

    x1 = np.minimum(result["x1"].to_numpy(), result["x2"].to_numpy())
    x2 = np.maximum(result["x1"].to_numpy(), result["x2"].to_numpy())
    y1 = np.minimum(result["y1"].to_numpy(), result["y2"].to_numpy())
    y2 = np.maximum(result["y1"].to_numpy(), result["y2"].to_numpy())

    result["x1"] = np.clip(x1, 0, target_w - 1)
    result["x2"] = np.clip(x2, 0, target_w - 1)
    result["y1"] = np.clip(y1, 0, target_h - 1)
    result["y2"] = np.clip(y2, 0, target_h - 1)

    result = result[(result["x2"] - result["x1"] >= 2) & (result["y2"] - result["y1"] >= 2)]
    return result.reset_index(drop=True)


def draw_boxes(gray: np.ndarray, boxes: pd.DataFrame) -> np.ndarray:
    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    height, width = gray.shape[:2]
    thickness = max(2, round(max(height, width) / 350))
    font_scale = max(0.5, max(height, width) / 1300)

    # RGB values because the canvas is RGB.
    box_color = (255, 45, 45)
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)

    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = [
            int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")
        ]
        label = str(row["class_name"])
        cv.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

        (text_w, text_h), baseline = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = min(height - 1, label_y1 + text_h + baseline + 6)
        label_x2 = min(width - 1, x1 + text_w + 8)
        cv.rectangle(
            canvas,
            (x1, label_y1),
            (label_x2, label_y2),
            background_color,
            thickness=-1,
        )
        cv.putText(
            canvas,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv.LINE_AA,
        )
    return canvas


def crop_with_context(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    context: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    cx1 = max(0, int(math.floor(x1 - context * box_w)))
    cy1 = max(0, int(math.floor(y1 - context * box_h)))
    cx2 = min(width, int(math.ceil(x2 + context * box_w)))
    cy2 = min(height, int(math.ceil(y2 + context * box_h)))

    crop = image[cy1:cy2, cx1:cx2]
    relative_box = (
        int(round(x1 - cx1)),
        int(round(y1 - cy1)),
        int(round(x2 - cx1)),
        int(round(y2 - cy1)),
    )
    crop_bounds = (cx1, cy1, cx2, cy2)
    return crop, relative_box, crop_bounds


def draw_single_box(gray: np.ndarray, box: tuple[int, int, int, int], label: str) -> np.ndarray:
    frame = pd.DataFrame(
        [{"class_name": label, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}]
    )
    return draw_boxes(gray, frame)


def save_triptych(
    original: np.ndarray,
    szch: np.ndarray,
    jsrt: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    context: float,
    output_path: Path,
    dpi: int,
) -> tuple[int, int, int, int]:
    original_crop, relative_box, crop_bounds = crop_with_context(original, box, context)
    szch_crop, _, _ = crop_with_context(szch, box, context)
    jsrt_crop, _, _ = crop_with_context(jsrt, box, context)

    images = [
        draw_single_box(original_crop, relative_box, label),
        draw_single_box(szch_crop, relative_box, label),
        draw_single_box(jsrt_crop, relative_box, label),
    ]
    titles = ["Original crop", "SZCH crop", "JSRT crop"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for axis, image, title in zip(axes[0], images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(label, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return crop_bounds


def resolve_annotation_schema(annotations: pd.DataFrame) -> dict[str, str]:
    mapping = {
        "image_id": find_column(annotations.columns, ID_CANDIDATES),
        "class_name": find_column(annotations.columns, CLASS_CANDIDATES),
        "x1": find_column(annotations.columns, XMIN_CANDIDATES),
        "y1": find_column(annotations.columns, YMIN_CANDIDATES),
        "x2": find_column(annotations.columns, XMAX_CANDIDATES),
        "y2": find_column(annotations.columns, YMAX_CANDIDATES),
    }
    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise ValueError(
            "Could not detect required annotation columns "
            f"{missing}. Available columns: {list(annotations.columns)}"
        )
    return {key: str(value) for key, value in mapping.items()}


def main() -> None:
    args = parse_args()
    pilot_dir = args.pilot_dir.resolve()
    output_dir = (args.output_dir or (pilot_dir / "bbox_review")).resolve()
    szch_dir = (args.szch_dir or (pilot_dir / "outputs_szch" / "Fusion_BS")).resolve()
    jsrt_dir = (args.jsrt_dir or (pilot_dir / "outputs_jsrt" / "Fusion_BS")).resolve()

    manifest_path = pilot_dir / "pilot_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {manifest_path}")
    if not args.annotations_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {args.annotations_csv}")

    manifest = pd.read_csv(manifest_path)
    if args.max_images > 0:
        manifest = manifest.head(args.max_images).copy()
    if manifest.empty:
        raise RuntimeError("Pilot manifest is empty")

    annotations_raw = pd.read_csv(args.annotations_csv)
    schema = resolve_annotation_schema(annotations_raw)

    annotations = annotations_raw.rename(
        columns={
            schema["image_id"]: "image_id",
            schema["class_name"]: "class_name",
            schema["x1"]: "x1",
            schema["y1"]: "y1",
            schema["x2"]: "x2",
            schema["y2"]: "y2",
        }
    ).copy()
    annotations["image_id"] = annotations["image_id"].astype(str).map(lambda x: Path(x).stem)
    annotations["class_name"] = annotations["class_name"].astype(str).str.strip()

    for column in ("x1", "y1", "x2", "y2"):
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    annotations = annotations.dropna(subset=["x1", "y1", "x2", "y2"])

    include_labels = {
        value.strip().casefold()
        for value in args.include_labels.split(",")
        if value.strip()
    }
    if include_labels:
        annotations = annotations[
            annotations["class_name"].str.casefold().isin(include_labels)
        ].copy()

    pilot_ids = set(manifest["image_id"].astype(str))
    annotations = annotations[annotations["image_id"].isin(pilot_ids)].copy()

    if annotations.empty:
        raise RuntimeError(
            "No boxed annotations matched the pilot image IDs. Check that the "
            "correct VinDr training annotation CSV was provided."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    full_dir = output_dir / "full"
    crop_dir = output_dir / "crops"
    full_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    montage_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    total_boxes = 0

    for _, manifest_row in manifest.iterrows():
        image_id = str(manifest_row["image_id"])
        filename = str(manifest_row["filename"])

        original_path = pilot_dir / "CXR" / filename
        mask_path = pilot_dir / "Mask" / filename
        szch_path = szch_dir / filename
        jsrt_path = jsrt_dir / filename

        for required in (original_path, mask_path, szch_path, jsrt_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required pilot image missing: {required}")

        original = load_gray(original_path)
        mask = load_gray(mask_path)
        szch = load_gray(szch_path)
        jsrt = load_gray(jsrt_path)

        target_shape = original.shape[:2]
        for name, image in (("mask", mask), ("SZCH", szch), ("JSRT", jsrt)):
            if image.shape[:2] != target_shape:
                raise RuntimeError(
                    f"Shape mismatch for {image_id}: Original={target_shape}, "
                    f"{name}={image.shape[:2]}"
                )

        source_shape = load_original_shape(
            str(manifest_row.get("source_path", "")),
            fallback_shape=target_shape,
        )

        image_boxes = annotations[annotations["image_id"] == image_id][
            ["class_name", "x1", "y1", "x2", "y2"]
        ].copy()
        image_boxes = scale_boxes(image_boxes, source_shape, target_shape)
        image_boxes = nms_same_class(image_boxes, args.nms_iou)
        total_boxes += len(image_boxes)

        annotated = {
            "Original + GT": draw_boxes(original, image_boxes),
            "Lung mask + GT": draw_boxes(mask, image_boxes),
            "SZCH + GT": draw_boxes(szch, image_boxes),
            "JSRT + GT": draw_boxes(jsrt, image_boxes),
        }

        # Per-image full-resolution comparison.
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
        for axis, (title, image) in zip(axes[0], annotated.items()):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        labels = ", ".join(image_boxes["class_name"].astype(str).tolist())
        fig.suptitle(f"{image_id}: {labels or 'No valid bounding box'}", fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        full_path = full_dir / f"{image_id}_boxed.png"
        fig.savefig(full_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # One enlarged triptych per retained pathology box.
        per_class_counter: dict[str, int] = {}
        for _, box_row in image_boxes.iterrows():
            class_name = str(box_row["class_name"])
            per_class_counter[class_name] = per_class_counter.get(class_name, 0) + 1
            box_number = per_class_counter[class_name]
            box = tuple(float(box_row[key]) for key in ("x1", "y1", "x2", "y2"))
            crop_name = (
                f"{image_id}__{safe_filename(class_name)}__{box_number:02d}.png"
            )
            crop_path = crop_dir / crop_name
            crop_bounds = save_triptych(
                original=original,
                szch=szch,
                jsrt=jsrt,
                box=box,
                label=class_name,
                context=args.crop_context,
                output_path=crop_path,
                dpi=args.dpi,
            )
            index_rows.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "box_index": box_number,
                    "x1_pilot": box[0],
                    "y1_pilot": box[1],
                    "x2_pilot": box[2],
                    "y2_pilot": box[3],
                    "crop_x1": crop_bounds[0],
                    "crop_y1": crop_bounds[1],
                    "crop_x2": crop_bounds[2],
                    "crop_y2": crop_bounds[3],
                    "crop_path": str(crop_path),
                    "full_comparison_path": str(full_path),
                }
            )

        montage_rows.append(
            {
                "image_id": image_id,
                "labels": labels,
                "images": annotated,
            }
        )

    if total_boxes == 0:
        raise RuntimeError("No valid boxes remained after scaling/filtering")

    # Combined annotated contact sheet.
    columns = ["Original + GT", "Lung mask + GT", "SZCH + GT", "JSRT + GT"]
    rows = len(montage_rows)
    fig, axes = plt.subplots(
        rows,
        len(columns),
        figsize=(4.0 * len(columns), 3.8 * rows),
        squeeze=False,
    )
    for row_index, entry in enumerate(montage_rows):
        for col_index, title in enumerate(columns):
            image = entry["images"][title]
            resized = cv.resize(
                image,
                (args.display_size, args.display_size),
                interpolation=cv.INTER_AREA,
            )
            axis = axes[row_index, col_index]
            axis.imshow(resized)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title, fontsize=12)
            if col_index == 0:
                label_text = str(entry["image_id"])
                if entry["labels"]:
                    label_text += "\n" + str(entry["labels"])
                axis.set_ylabel(label_text, fontsize=8)

    fig.suptitle(
        "VinDr-CXR GL-LCM pilot with ground-truth pathology boxes",
        fontsize=16,
        y=0.997,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    montage_path = output_dir / "vindr_train_gl_lcm_bbox_comparison.png"
    fig.savefig(montage_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    index = pd.DataFrame(index_rows)
    index_path = output_dir / "bbox_review_index.csv"
    index.to_csv(index_path, index=False)

    print("=" * 72)
    print(f"Pilot images processed: {len(montage_rows)}")
    print(f"Retained pathology boxes: {total_boxes}")
    print(f"Combined montage: {montage_path}")
    print(f"Per-image comparisons: {full_dir}")
    print(f"Pathology crops: {crop_dir}")
    print(f"Review index: {index_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
