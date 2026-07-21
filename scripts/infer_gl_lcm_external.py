#!/usr/bin/env python3
"""
Run the official GL-LCM global/local inference logic on an external CXR folder.

This is a path-parameterized adaptation of codes/batch_lcm_eval.py. It preserves:
  - 1024x1024 grayscale preprocessing
  - 50-step LCM sampling by default
  - shared initial noise for global and local paths
  - fast Local-Enhanced Guidance approximation
  - Poisson fusion using OpenCV seamlessClone

Expected input:
  input_dir/<filename>
  mask_dir/<filename>          binary mask, values 0/255
  masked_input_dir/<filename>  input image multiplied by the mask
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from diffusers import LCMScheduler
from tqdm import tqdm


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DOMAIN_CONFIG = {
    "szch": {
        "clip_rate": 0.025,
        "initial_clip_sample_range_g": 2.0,
        "initial_clip_sample_range_l": 3.5,
    },
    "jsrt": {
        "clip_rate": 0.025,
        "initial_clip_sample_range_g": 1.7,
        "initial_clip_sample_range_l": 3.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External-data GL-LCM inference with global/local fusion."
    )
    parser.add_argument("--repo_codes_dir", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--masked_input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vqgan_checkpoint", type=Path, required=True)
    parser.add_argument("--unet_checkpoint", type=Path, required=True)
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIG), required=True)
    parser.add_argument("--file_list", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_mask_component_area", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for path, description in [
        (args.repo_codes_dir, "GL-LCM codes directory"),
        (args.input_dir, "input directory"),
        (args.mask_dir, "mask directory"),
        (args.masked_input_dir, "masked-input directory"),
    ]:
        if not path.is_dir():
            raise FileNotFoundError(f"{description} not found: {path}")

    for path, description in [
        (args.vqgan_checkpoint, "VQGAN checkpoint"),
        (args.unet_checkpoint, "UNet checkpoint"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")


def list_filenames(input_dir: Path, file_list: Path | None) -> list[str]:
    if file_list is not None:
        if not file_list.is_file():
            raise FileNotFoundError(f"File list not found: {file_list}")
        filenames = [
            line.strip()
            for line in file_list.read_text().splitlines()
            if line.strip()
        ]
    else:
        filenames = sorted(
            path.name
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    if not filenames:
        raise RuntimeError("No input files were selected")

    missing = [name for name in filenames if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} listed input files are missing. Examples: {missing[:5]}"
        )
    return filenames


def read_gray_1024(path: Path, image_size: int, nearest: bool = False) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    interpolation = cv.INTER_NEAREST if nearest else (
        cv.INTER_AREA if max(image.shape) >= image_size else cv.INTER_CUBIC
    )
    image = cv.resize(image, (image_size, image_size), interpolation=interpolation)
    return image


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
    # GL-LCM checkpoints are serialized full torch modules, not state_dict files.
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(
            f"Expected a serialized torch.nn.Module in {path}, got {type(obj)}"
        )
    return obj.to(device).eval()


def clean_fusion_mask(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    mask = np.where(mask_u8 >= 128, 255, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == component_id] = 255

    if not np.any(keep):
        raise RuntimeError("The fusion mask is empty after component filtering")

    return keep


def make_scheduler(
    *,
    num_train_timesteps: int,
    clip_sample: bool,
    clip_sample_range: float,
    num_infer_steps: int,
) -> LCMScheduler:
    scheduler = LCMScheduler(
        num_train_timesteps=num_train_timesteps,
        clip_sample=clip_sample,
        clip_sample_range=clip_sample_range,
    )
    scheduler.set_timesteps(num_infer_steps)
    return scheduler


def run_one(
    *,
    filename: str,
    input_dir: Path,
    mask_dir: Path,
    masked_input_dir: Path,
    image_size: int,
    model: torch.nn.Module,
    vqgan: torch.nn.Module,
    domain_cfg: dict[str, float],
    num_train_timesteps: int,
    num_infer_steps: int,
    alpha: float,
    min_mask_component_area: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()

    cxr_u8 = read_gray_1024(input_dir / filename, image_size)
    masked_cxr_u8 = read_gray_1024(masked_input_dir / filename, image_size)
    mask_u8 = read_gray_1024(mask_dir / filename, image_size, nearest=True)
    mask_u8 = clean_fusion_mask(mask_u8, min_mask_component_area)

    cxr_tensor = image_to_tensor(cxr_u8, device)
    masked_cxr_tensor = image_to_tensor(masked_cxr_u8, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

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
            clip_sample=True,
            clip_sample_range=domain_cfg["initial_clip_sample_range_g"],
            num_infer_steps=num_infer_steps,
        )

        for step_index, timestep in enumerate(base_scheduler.timesteps):
            timestep_on_device = timestep.to(device)
            timestep_batch = timestep_on_device.reshape(1).long()

            residual = model(sample, timestep_batch)
            local_residual_raw = model(masked_sample, timestep_batch)

            # Fast LEG approximation from the official evaluator.
            masked_residual = (
                (1.0 - alpha) * residual
                + alpha * local_residual_raw
            )

            global_scheduler = make_scheduler(
                num_train_timesteps=num_train_timesteps,
                clip_sample=True,
                clip_sample_range=(
                    domain_cfg["initial_clip_sample_range_g"]
                    + domain_cfg["clip_rate"] * step_index
                ),
                num_infer_steps=num_infer_steps,
            )
            local_scheduler = make_scheduler(
                num_train_timesteps=num_train_timesteps,
                clip_sample=True,
                clip_sample_range=(
                    domain_cfg["initial_clip_sample_range_l"]
                    + domain_cfg["clip_rate"] * step_index
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

            # Re-append the conditioning CXR latent after every scheduler step.
            sample = torch.cat((sample[:, :4], cxr_latent), dim=1)
            masked_sample = torch.cat(
                (masked_sample[:, :4], masked_cxr_latent),
                dim=1,
            )

        global_bs = tensor_to_unit_image(vqgan.decode(sample[:, :4]))
        local_bs = tensor_to_unit_image(vqgan.decode(masked_sample[:, :4]))

    # Preserve black padding/background.
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
        raise RuntimeError(f"Empty mask bounding box for {filename}")
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

    elapsed = time.perf_counter() - started
    return global_u8, local_u8, fusion_u8, elapsed


def main() -> None:
    args = parse_args()
    validate_paths(args)

    # Required so torch.load can import modules.unet and model definitions used
    # when the repository serialized full model objects.
    repo_codes = str(args.repo_codes_dir.resolve())
    if repo_codes not in sys.path:
        sys.path.insert(0, repo_codes)

    # Import model.py before torch.load to make repository classes importable.
    __import__("model")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    filenames = list_filenames(args.input_dir, args.file_list)

    output_global = args.output_dir / "Global_BS"
    output_local = args.output_dir / "Local_BS"
    output_fusion = args.output_dir / "Fusion_BS"
    for directory in (output_global, output_local, output_fusion):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[GL-LCM] External inference")
    print(f"[GL-LCM] Device:           {device}")
    print(f"[GL-LCM] Domain:           {args.domain}")
    print(f"[GL-LCM] Files:            {len(filenames)}")
    print(f"[GL-LCM] Inference steps:  {args.num_infer_steps}")
    print(f"[GL-LCM] VQGAN:            {args.vqgan_checkpoint}")
    print(f"[GL-LCM] UNet:             {args.unet_checkpoint}")
    print(f"[GL-LCM] Output:           {args.output_dir}")
    print("=" * 80)

    print("[Model] Loading serialized UNet")
    model = load_serialized_module(args.unet_checkpoint, device)
    print("[Model] Loading serialized VQGAN")
    vqgan = load_serialized_module(args.vqgan_checkpoint, device)

    domain_cfg = DOMAIN_CONFIG[args.domain]
    timing_rows: list[dict[str, object]] = []

    total_started = time.perf_counter()

    for index, filename in enumerate(tqdm(filenames, desc=f"GL-LCM {args.domain}")):
        global_path = output_global / filename
        local_path = output_local / filename
        fusion_path = output_fusion / filename

        if (
            not args.overwrite
            and global_path.is_file()
            and local_path.is_file()
            and fusion_path.is_file()
        ):
            timing_rows.append(
                {
                    "filename": filename,
                    "status": "exists_skipped",
                    "seed": args.seed + index,
                    "seconds": None,
                }
            )
            continue

        global_u8, local_u8, fusion_u8, elapsed = run_one(
            filename=filename,
            input_dir=args.input_dir,
            mask_dir=args.mask_dir,
            masked_input_dir=args.masked_input_dir,
            image_size=args.image_size,
            model=model,
            vqgan=vqgan,
            domain_cfg=domain_cfg,
            num_train_timesteps=args.num_train_timesteps,
            num_infer_steps=args.num_infer_steps,
            alpha=args.alpha,
            min_mask_component_area=args.min_mask_component_area,
            device=device,
            seed=args.seed + index,
        )

        for path, image in [
            (global_path, global_u8),
            (local_path, local_u8),
            (fusion_path, fusion_u8),
        ]:
            if not cv.imwrite(str(path), image):
                raise RuntimeError(f"Failed to save output: {path}")

        timing_rows.append(
            {
                "filename": filename,
                "status": "saved",
                "seed": args.seed + index,
                "seconds": elapsed,
            }
        )

    total_elapsed = time.perf_counter() - total_started

    timing_path = args.output_dir / "timings.csv"
    with timing_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "status", "seed", "seconds"],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    config_path = args.output_dir / "inference_config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "repo_codes_dir": str(args.repo_codes_dir),
                "input_dir": str(args.input_dir),
                "mask_dir": str(args.mask_dir),
                "masked_input_dir": str(args.masked_input_dir),
                "output_dir": str(args.output_dir),
                "vqgan_checkpoint": str(args.vqgan_checkpoint),
                "unet_checkpoint": str(args.unet_checkpoint),
                "domain": args.domain,
                "domain_config": domain_cfg,
                "image_size": args.image_size,
                "num_infer_steps": args.num_infer_steps,
                "num_train_timesteps": args.num_train_timesteps,
                "alpha": args.alpha,
                "seed": args.seed,
                "device": str(device),
                "num_files": len(filenames),
                "total_seconds": total_elapsed,
            },
            f,
            indent=2,
        )

    print(f"[Output] Global:  {output_global}")
    print(f"[Output] Local:   {output_local}")
    print(f"[Output] Fusion:  {output_fusion}")
    print(f"[Output] Timings: {timing_path}")
    print(f"[Done] Total seconds: {total_elapsed:.2f}")


if __name__ == "__main__":
    main()
