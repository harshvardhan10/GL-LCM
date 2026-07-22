#!/usr/bin/env python3
"""
Generate final SZCH GL-LCM bone-suppressed MIMIC-CXR images.

Production properties
---------------------
- Reads the verified lung-mask manifest directly.
- Loads original CXR and native-resolution binary lung mask.
- Creates every intermediate in memory only.
- Runs the same SZCH global/local GL-LCM logic used by the validated pilot.
- Saves only the final fused 256x256 grayscale PNG.
- Uses deterministic per-image seeds based on the global manifest index.
- Supports Slurm array sharding and safe resume.
- Writes a per-shard status CSV continuously.

Expected manifest entry
-----------------------
{
    "image": "/absolute/path/to/source.jpg",
    "caption": ["No Finding"],
    "mask_relpath": "02/<uuid>.png",
    "chexmask_view": "lung"
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch
from diffusers import LCMScheduler
from PIL import Image
from tqdm import tqdm


DOMAIN_CONFIG = {
    "szch": {
        "clip_rate": 0.025,
        "initial_clip_sample_range_g": 2.0,
        "initial_clip_sample_range_l": 3.5,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final-only SZCH GL-LCM MIMIC-CXR images."
    )
    parser.add_argument("--repo_codes_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--status_dir", type=Path, required=True)
    parser.add_argument("--vqgan_checkpoint", type=Path, required=True)
    parser.add_argument("--unet_checkpoint", type=Path, required=True)

    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--items_per_shard", type=int, default=500)

    parser.add_argument("--gl_lcm_size", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--min_mask_component_area", type=int, default=100)
    parser.add_argument("--png_compression", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even when a valid 256x256 PNG already exists.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on the first failed image.",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=25,
        help="Abort after this many errors in one shard; ignored with --fail_fast.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit within the selected shard.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    required_dirs = [
        (args.repo_codes_dir, "GL-LCM codes directory"),
        (args.mask_root, "lung-mask root"),
    ]
    required_files = [
        (args.manifest, "lung-mask manifest"),
        (args.vqgan_checkpoint, "VQGAN checkpoint"),
        (args.unet_checkpoint, "UNet checkpoint"),
    ]

    for path, name in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"{name} not found: {path}")

    for path, name in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    if args.shard_index < 0:
        raise ValueError("--shard_index must be >= 0")
    if args.items_per_shard <= 0:
        raise ValueError("--items_per_shard must be > 0")
    if args.gl_lcm_size <= 0 or args.output_size <= 0:
        raise ValueError("Image sizes must be > 0")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png_compression must be in [0, 9]")
    if args.max_errors <= 0:
        raise ValueError("--max_errors must be > 0")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Manifest must be a non-empty JSON list: {path}")

    required_keys = {"image", "mask_relpath"}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Manifest entry {index} is not an object")
        missing = required_keys - item.keys()
        if missing:
            raise KeyError(
                f"Manifest entry {index} is missing required keys: {sorted(missing)}"
            )
        if item.get("chexmask_view") not in (None, "lung"):
            raise ValueError(
                f"Manifest entry {index} has chexmask_view="
                f"{item.get('chexmask_view')!r}, expected 'lung'"
            )

    return data


def safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"mask_relpath must be relative, got: {value}")
    if ".." in relative.parts:
        raise ValueError(f"mask_relpath contains '..': {value}")
    return relative


def read_native_gray(path: Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    if image.ndim != 2:
        raise RuntimeError(f"Expected grayscale image at {path}, got {image.shape}")
    return image


def resize_cxr(image: np.ndarray, size: int) -> np.ndarray:
    interpolation = (
        cv.INTER_AREA
        if image.shape[0] >= size and image.shape[1] >= size
        else cv.INTER_CUBIC
    )
    return cv.resize(image, (size, size), interpolation=interpolation)


def resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    resized = cv.resize(mask_u8, (size, size), interpolation=cv.INTER_NEAREST)
    return np.where(resized >= 128, 255, 0).astype(np.uint8)


def clean_fusion_mask(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    binary = np.where(mask_u8 >= 128, 1, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask_u8, dtype=np.uint8)
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == component_id] = 255

    if not np.any(cleaned):
        raise RuntimeError("The lung mask is empty after component filtering")

    return cleaned


def image_to_tensor(image_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image_u8.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_unit_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    array = np.squeeze(array)
    array = np.clip(array * 0.5 + 0.5, 0.0, 1.0)
    return array.astype(np.float32)


def load_serialized_module(path: Path, device: torch.device) -> torch.nn.Module:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(
            f"Expected a serialized torch.nn.Module in {path}, got {type(obj)}"
        )
    return obj.to(device).eval()


def make_scheduler(
    *,
    num_train_timesteps: int,
    clip_sample_range: float,
    num_infer_steps: int,
) -> LCMScheduler:
    scheduler = LCMScheduler(
        num_train_timesteps=num_train_timesteps,
        clip_sample=True,
        clip_sample_range=clip_sample_range,
    )
    scheduler.set_timesteps(num_infer_steps)
    return scheduler


def run_gl_lcm(
    *,
    source_path: Path,
    mask_path: Path,
    model: torch.nn.Module,
    vqgan: torch.nn.Module,
    gl_lcm_size: int,
    output_size: int,
    num_train_timesteps: int,
    num_infer_steps: int,
    alpha: float,
    seed: int,
    min_mask_component_area: int,
    device: torch.device,
) -> np.ndarray:
    """
    Return only the final fused output as output_size x output_size uint8.
    No intermediate file is written.
    """
    source_native = read_native_gray(source_path)
    mask_native = read_native_gray(mask_path)

    if source_native.shape != mask_native.shape:
        raise ValueError(
            "Image/mask geometry mismatch: "
            f"image={source_native.shape}, mask={mask_native.shape}, "
            f"image_path={source_path}, mask_path={mask_path}"
        )

    cxr_u8 = resize_cxr(source_native, gl_lcm_size)
    mask_u8 = resize_binary_mask(mask_native, gl_lcm_size)
    mask_u8 = clean_fusion_mask(mask_u8, min_mask_component_area)

    # Construct the local-view conditioning image entirely in memory.
    masked_cxr_u8 = cv.bitwise_and(cxr_u8, cxr_u8, mask=mask_u8)

    cxr_tensor = image_to_tensor(cxr_u8, device)
    masked_cxr_tensor = image_to_tensor(masked_cxr_u8, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    cfg = DOMAIN_CONFIG["szch"]

    with torch.inference_mode():
        cxr_latent = vqgan.encode_stage_2_inputs(cxr_tensor)
        masked_cxr_latent = vqgan.encode_stage_2_inputs(masked_cxr_tensor)

        noise = torch.randn(
            cxr_latent.shape,
            generator=generator,
            device=device,
            dtype=cxr_latent.dtype,
        )

        sample = torch.cat((noise, cxr_latent), dim=1)
        masked_sample = torch.cat((noise, masked_cxr_latent), dim=1)

        base_scheduler = make_scheduler(
            num_train_timesteps=num_train_timesteps,
            clip_sample_range=cfg["initial_clip_sample_range_g"],
            num_infer_steps=num_infer_steps,
        )

        for step_index, timestep in enumerate(base_scheduler.timesteps):
            timestep_on_device = timestep.to(device)
            timestep_batch = timestep_on_device.reshape(1).long()

            residual = model(sample, timestep_batch)
            local_residual_raw = model(masked_sample, timestep_batch)

            # Fast Local-Enhanced Guidance approximation used by the pilot.
            masked_residual = (
                (1.0 - alpha) * residual
                + alpha * local_residual_raw
            )

            global_scheduler = make_scheduler(
                num_train_timesteps=num_train_timesteps,
                clip_sample_range=(
                    cfg["initial_clip_sample_range_g"]
                    + cfg["clip_rate"] * step_index
                ),
                num_infer_steps=num_infer_steps,
            )
            local_scheduler = make_scheduler(
                num_train_timesteps=num_train_timesteps,
                clip_sample_range=(
                    cfg["initial_clip_sample_range_l"]
                    + cfg["clip_rate"] * step_index
                ),
                num_infer_steps=num_infer_steps,
            )

            sample = global_scheduler.step(
                residual,
                timestep_on_device,
                sample,
            ).prev_sample

            masked_sample = local_scheduler.step(
                masked_residual,
                timestep_on_device,
                masked_sample,
            ).prev_sample

            # Re-append the conditioning latent after every scheduler step.
            sample = torch.cat((sample[:, :4], cxr_latent), dim=1)
            masked_sample = torch.cat(
                (masked_sample[:, :4], masked_cxr_latent),
                dim=1,
            )

        global_bs = tensor_to_unit_image(vqgan.decode(sample[:, :4]))
        local_bs = tensor_to_unit_image(vqgan.decode(masked_sample[:, :4]))

    # Preserve original black padding/background.
    background = cxr_u8 == 0
    global_bs[background] = 0.0
    local_bs[background] = 0.0

    lung_pixels = mask_u8 > 0
    if np.any(lung_pixels):
        local_bs[lung_pixels] += (
            float(global_bs[lung_pixels].mean())
            - float(local_bs[lung_pixels].mean())
        )

    global_u8 = np.rint(np.clip(global_bs, 0.0, 1.0) * 255.0).astype(np.uint8)
    local_u8 = np.rint(np.clip(local_bs, 0.0, 1.0) * 255.0).astype(np.uint8)

    x, y, width, height = cv.boundingRect(mask_u8)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Empty mask bounding box: {mask_path}")

    center = (x + width // 2, y + height // 2)

    global_bgr = cv.cvtColor(global_u8, cv.COLOR_GRAY2BGR)
    local_bgr = cv.cvtColor(local_u8, cv.COLOR_GRAY2BGR)

    fusion_bgr = cv.seamlessClone(
        local_bgr,
        global_bgr,
        mask_u8,
        center,
        cv.MONOCHROME_TRANSFER,
    )
    fusion_u8 = cv.cvtColor(fusion_bgr, cv.COLOR_BGR2GRAY)
    fusion_u8[background] = 0

    final_u8 = cv.resize(
        fusion_u8,
        (output_size, output_size),
        interpolation=cv.INTER_AREA,
    )

    if final_u8.shape != (output_size, output_size):
        raise RuntimeError(
            f"Unexpected final output shape: {final_u8.shape}"
        )

    return final_u8


def valid_existing_output(path: Path, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False

    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    return (
        image is not None
        and image.shape == (expected_size, expected_size)
        and image.dtype == np.uint8
    )


def atomic_save_png(
    image_u8: np.ndarray,
    output_path: Path,
    compression: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )

    try:
        Image.fromarray(image_u8, mode="L").save(
            temp_path,
            format="PNG",
            compress_level=compression,
            optimize=False,
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_run_config(
    *,
    args: argparse.Namespace,
    total_entries: int,
    start_index: int,
    end_index: int,
    path: Path,
) -> None:
    payload = {
        "manifest": str(args.manifest),
        "mask_root": str(args.mask_root),
        "output_root": str(args.output_root),
        "repo_codes_dir": str(args.repo_codes_dir),
        "vqgan_checkpoint": str(args.vqgan_checkpoint),
        "unet_checkpoint": str(args.unet_checkpoint),
        "shard_index": args.shard_index,
        "items_per_shard": args.items_per_shard,
        "start_index_inclusive": start_index,
        "end_index_exclusive": end_index,
        "total_manifest_entries": total_entries,
        "gl_lcm_size": args.gl_lcm_size,
        "output_size": args.output_size,
        "num_infer_steps": args.num_infer_steps,
        "num_train_timesteps": args.num_train_timesteps,
        "alpha": args.alpha,
        "base_seed": args.base_seed,
        "min_mask_component_area": args.min_mask_component_area,
        "png_compression": args.png_compression,
        "device": args.device,
        "overwrite": args.overwrite,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Required so torch.load can import GL-LCM repository classes.
    repo_codes = str(args.repo_codes_dir.resolve())
    if repo_codes not in sys.path:
        sys.path.insert(0, repo_codes)
    __import__("model")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device requests CUDA, but torch.cuda.is_available() is False"
        )
    device = torch.device(args.device)

    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)

    manifest = load_manifest(args.manifest)

    start_index = args.shard_index * args.items_per_shard
    end_index = min(start_index + args.items_per_shard, len(manifest))

    if start_index >= len(manifest):
        print(
            f"[Shard] Nothing to process: start={start_index}, "
            f"manifest_size={len(manifest)}"
        )
        return

    selected = manifest[start_index:end_index]
    if args.limit is not None:
        selected = selected[: args.limit]
        end_index = start_index + len(selected)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)

    status_path = args.status_dir / f"shard_{args.shard_index:05d}.csv"
    config_path = args.status_dir / f"shard_{args.shard_index:05d}_config.json"

    write_run_config(
        args=args,
        total_entries=len(manifest),
        start_index=start_index,
        end_index=end_index,
        path=config_path,
    )

    print("=" * 88)
    print("[MIMIC GL-LCM] Final-only SZCH generation")
    print(f"[Device]          {device}")
    print(f"[Manifest]        {args.manifest}")
    print(f"[Manifest size]   {len(manifest)}")
    print(f"[Shard]           {args.shard_index}")
    print(f"[Range]           [{start_index}, {end_index})")
    print(f"[Images]          {len(selected)}")
    print(f"[Mask root]       {args.mask_root}")
    print(f"[Output root]     {args.output_root}")
    print(f"[Inference steps] {args.num_infer_steps}")
    print(f"[Output size]     {args.output_size}")
    print(f"[Status CSV]      {status_path}")
    print("=" * 88)

    print("[Model] Loading serialized SZCH UNet...")
    model = load_serialized_module(args.unet_checkpoint, device)
    print("[Model] Loading serialized SZCH VQGAN...")
    vqgan = load_serialized_module(args.vqgan_checkpoint, device)

    fieldnames = [
        "global_index",
        "image_id",
        "source_path",
        "mask_path",
        "output_path",
        "status",
        "seed",
        "seconds",
        "error",
    ]

    saved = 0
    skipped = 0
    failed = 0
    successful_seconds: list[float] = []
    shard_started = time.perf_counter()

    # Line buffering plus explicit flush preserves progress during long jobs.
    with status_path.open("w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        iterator = tqdm(
            enumerate(selected, start=start_index),
            total=len(selected),
            desc=f"SZCH shard {args.shard_index}",
        )

        for global_index, item in iterator:
            started = time.perf_counter()
            source_path = Path(item["image"])
            relative_path = safe_relative_path(str(item["mask_relpath"]))
            mask_path = args.mask_root / relative_path
            output_path = args.output_root / relative_path
            image_id = source_path.stem
            seed = args.base_seed + global_index

            row = {
                "global_index": global_index,
                "image_id": image_id,
                "source_path": str(source_path),
                "mask_path": str(mask_path),
                "output_path": str(output_path),
                "status": "",
                "seed": seed,
                "seconds": "",
                "error": "",
            }

            try:
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source image not found: {source_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Lung mask not found: {mask_path}")

                if (
                    not args.overwrite
                    and valid_existing_output(output_path, args.output_size)
                ):
                    row["status"] = "exists_skipped"
                    skipped += 1
                else:
                    final_u8 = run_gl_lcm(
                        source_path=source_path,
                        mask_path=mask_path,
                        model=model,
                        vqgan=vqgan,
                        gl_lcm_size=args.gl_lcm_size,
                        output_size=args.output_size,
                        num_train_timesteps=args.num_train_timesteps,
                        num_infer_steps=args.num_infer_steps,
                        alpha=args.alpha,
                        seed=seed,
                        min_mask_component_area=args.min_mask_component_area,
                        device=device,
                    )

                    atomic_save_png(
                        final_u8,
                        output_path,
                        args.png_compression,
                    )

                    if not valid_existing_output(
                        output_path,
                        args.output_size,
                    ):
                        raise RuntimeError(
                            f"Saved output failed validation: {output_path}"
                        )

                    row["status"] = "saved"
                    saved += 1

                elapsed = time.perf_counter() - started
                row["seconds"] = f"{elapsed:.6f}"

                if row["status"] == "saved":
                    successful_seconds.append(elapsed)

            except Exception as exc:
                failed += 1
                elapsed = time.perf_counter() - started
                row["status"] = "failed"
                row["seconds"] = f"{elapsed:.6f}"
                row["error"] = (
                    f"{type(exc).__name__}: {exc}"
                ).replace("\n", " ")

                print(
                    f"\n[ERROR] global_index={global_index}, "
                    f"image={source_path}\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

            writer.writerow(row)
            handle.flush()

            mean_seconds = (
                sum(successful_seconds) / len(successful_seconds)
                if successful_seconds
                else 0.0
            )
            iterator.set_postfix(
                saved=saved,
                skipped=skipped,
                failed=failed,
                mean_s=f"{mean_seconds:.2f}",
            )

            if row["status"] == "failed":
                if args.fail_fast:
                    raise RuntimeError(
                        "Stopping because --fail_fast is enabled"
                    )
                if failed >= args.max_errors:
                    raise RuntimeError(
                        f"Stopping after {failed} errors "
                        f"(--max_errors={args.max_errors})"
                    )

    shard_elapsed = time.perf_counter() - shard_started
    mean_saved_seconds = (
        sum(successful_seconds) / len(successful_seconds)
        if successful_seconds
        else None
    )

    summary = {
        "shard_index": args.shard_index,
        "start_index_inclusive": start_index,
        "end_index_exclusive": end_index,
        "selected": len(selected),
        "saved": saved,
        "exists_skipped": skipped,
        "failed": failed,
        "total_seconds": shard_elapsed,
        "mean_seconds_per_saved_image": mean_saved_seconds,
        "status_csv": str(status_path),
    }

    summary_path = (
        args.status_dir / f"shard_{args.shard_index:05d}_summary.json"
    )
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 88)
    print("[Done]")
    print(json.dumps(summary, indent=2))
    print("=" * 88)

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
