#!/usr/bin/env python3
"""
Finalize the generated MIMIC SZCH bone-suppressed dataset manifest.

The output manifest contains only the fields required by the existing ALBEF
pretraining dataset:
    {"image": "/absolute/output/path.png", "caption": ...}

By default, finalization fails unless every source-manifest entry has a valid
output image with the expected dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2 as cv
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify GL-LCM outputs and build the ALBEF training manifest."
    )
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--missing_csv", type=Path, required=True)
    parser.add_argument("--expected_size", type=int, default=256)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Write a partial manifest instead of failing when outputs are missing.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")

    return data


def safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe mask_relpath: {value}")
    return relative


def validate_png(path: Path, expected_size: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"

    if path.stat().st_size <= 0:
        return False, "empty_file"

    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        return False, "unreadable"

    if image.shape != (expected_size, expected_size):
        return False, f"wrong_shape:{image.shape}"

    return True, ""


def main() -> None:
    args = parse_args()

    source = load_manifest(args.source_manifest)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.missing_csv.parent.mkdir(parents=True, exist_ok=True)

    finalized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, item in enumerate(
        tqdm(source, desc="Verifying bone-suppressed outputs")
    ):
        if "image" not in item or "mask_relpath" not in item:
            raise KeyError(
                f"Source entry {index} requires image and mask_relpath"
            )

        relative_path = safe_relative_path(str(item["mask_relpath"]))
        output_path = args.output_root / relative_path

        valid, reason = validate_png(output_path, args.expected_size)

        if not valid:
            failures.append(
                {
                    "global_index": index,
                    "source_image": item["image"],
                    "output_path": str(output_path),
                    "reason": reason,
                }
            )
            continue

        finalized.append(
            {
                "image": str(output_path.resolve()),
                "caption": item.get("caption", []),
            }
        )

    with args.output_manifest.open("w") as handle:
        json.dump(finalized, handle, indent=2)

    with args.missing_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "global_index",
                "source_image",
                "output_path",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(failures)

    print("=" * 80)
    print(f"Source entries:     {len(source)}")
    print(f"Valid outputs:      {len(finalized)}")
    print(f"Missing/invalid:    {len(failures)}")
    print(f"Output manifest:    {args.output_manifest}")
    print(f"Failure report:     {args.missing_csv}")
    print("=" * 80)

    if failures and not args.allow_missing:
        raise SystemExit(
            "Finalization failed because outputs are missing or invalid. "
            "Resubmit the affected array shards, then run finalization again."
        )


if __name__ == "__main__":
    main()
