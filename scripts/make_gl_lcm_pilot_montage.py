#!/usr/bin/env python3
"""Create a contact sheet comparing VinDr originals, masks, SZCH, and JSRT."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--szch_dir", type=Path, required=True)
    parser.add_argument("--jsrt_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display_size", type=int, default=384)
    return parser.parse_args()


def load_gray(path: Path, display_size: int) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return cv.resize(
        image,
        (display_size, display_size),
        interpolation=cv.INTER_AREA,
    )


def main() -> None:
    args = parse_args()
    manifest_path = args.pilot_dir / "pilot_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError("Pilot manifest is empty")

    columns = [
        ("Original", args.pilot_dir / "CXR"),
        ("Lung mask", args.pilot_dir / "Mask"),
        ("SZCH fusion", args.szch_dir / "Fusion_BS"),
        ("JSRT fusion", args.jsrt_dir / "Fusion_BS"),
    ]

    rows = len(manifest)
    fig, axes = plt.subplots(
        rows,
        len(columns),
        figsize=(4.0 * len(columns), 3.8 * rows),
        squeeze=False,
    )

    for row_index, row in manifest.iterrows():
        filename = str(row["filename"])
        labels = str(row.get("labels", ""))
        if labels == "nan":
            labels = ""

        for col_index, (title, directory) in enumerate(columns):
            image = load_gray(directory / filename, args.display_size)
            ax = axes[row_index, col_index]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            if row_index == 0:
                ax.set_title(title, fontsize=13)

            if col_index == 0:
                label_text = str(row["image_id"])
                if labels:
                    label_text += f"\n{labels}"
                ax.set_ylabel(label_text, fontsize=9)

    fig.suptitle(
        "VinDr-CXR train GL-LCM pilot: original vs SZCH/JSRT checkpoints",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.99))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Output] Saved comparison montage: {args.output}")


if __name__ == "__main__":
    main()
