#!/usr/bin/env python3
"""
Check whether mask filenames in masks/ exactly match config.yaml segments.mask_path
for each video directory under OurBench.

Usage:
    conda run -n sam3 python tools/check_mask_config_consistency.py \
        --dataset_root data/OurBench

Exit code:
    0: all matched
    1: at least one mismatch or missing file/folder
"""

import argparse
import sys
from pathlib import Path

import yaml


def collect_config_mask_filenames(config_path: Path):
    """Collect mask filenames from config.yaml segments[*].mask_path."""
    if not config_path.exists():
        return None, "config.yaml missing"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return None, f"config parse error: {exc}"

    filenames = set()
    for seg in data.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        mask_path = seg.get("mask_path")
        if isinstance(mask_path, str) and mask_path.strip():
            filenames.add(Path(mask_path).name)

    return filenames, None


def check_video_dir(video_dir: Path):
    """Check one video dir and return mismatch details."""
    masks_dir = video_dir / "masks"
    config_path = video_dir / "config.yaml"

    errors = []

    if not masks_dir.exists():
        return {
            "video": video_dir.name,
            "errors": ["masks/ missing"],
            "extra_in_masks": [],
            "missing_in_masks": [],
            "masks_count": 0,
            "config_count": 0,
        }

    masks_files = {p.name for p in masks_dir.glob("*.mp4")}

    config_files, err = collect_config_mask_filenames(config_path)
    if err:
        errors.append(err)
        config_files = set()

    extra_in_masks = sorted(masks_files - config_files)
    missing_in_masks = sorted(config_files - masks_files)

    return {
        "video": video_dir.name,
        "errors": errors,
        "extra_in_masks": extra_in_masks,
        "missing_in_masks": missing_in_masks,
        "masks_count": len(masks_files),
        "config_count": len(config_files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check masks/ filenames and config.yaml mask_path consistency"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/OurBench",
        help="Root containing video dirs",
    )
    parser.add_argument(
        "--all_dirs",
        action="store_true",
        help="Check all subdirectories (not only those with source.mp4)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ dataset root not found: {dataset_root}")
        return 1

    if args.all_dirs:
        video_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    else:
        video_dirs = sorted(
            [d for d in dataset_root.iterdir() if d.is_dir() and (d / "source.mp4").exists()]
        )

    print(f"Scanning {len(video_dirs)} video directories under {dataset_root} ...")

    mismatches = []
    for video_dir in video_dirs:
        result = check_video_dir(video_dir)
        has_issue = bool(result["errors"] or result["extra_in_masks"] or result["missing_in_masks"])
        if has_issue:
            mismatches.append(result)

    print("\n" + "=" * 72)
    print(f"TOTAL: {len(video_dirs)}")
    print(f"MISMATCHED: {len(mismatches)}")

    for item in mismatches:
        print("-" * 72)
        print(f"[{item['video']}] masks={item['masks_count']} config={item['config_count']}")
        for err in item["errors"]:
            print(f"  ERROR: {err}")
        if item["extra_in_masks"]:
            print("  EXTRA_IN_MASKS: " + ", ".join(item["extra_in_masks"]))
        if item["missing_in_masks"]:
            print("  MISSING_IN_MASKS: " + ", ".join(item["missing_in_masks"]))

    print("=" * 72)
    if not mismatches:
        print("✅ All video dirs matched: masks/ filenames == config.yaml segments.mask_path")
        return 0

    print("⚠️  Found mismatches. Please fix listed videos.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
