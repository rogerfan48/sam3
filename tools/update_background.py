#!/usr/bin/env python3
"""
Regenerate background mask(s) from existing mask videos in OurBench.

The background is computed as the inverse of the union of all non-background
masks found in each video's masks/ directory.

Usage:
    # Single video
    conda run -n sam3 python tools/update_background.py \
        --video_dir data/OurBench/boxer-punching-towards-camera

    # All videos under dataset root
    conda run -n sam3 python tools/update_background.py \
        --dataset_root data/OurBench
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "generate_masks", Path(__file__).parent / "generate_masks.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
save_mask_video = _mod.save_mask_video


def read_mask_video(mask_path):
    """Read a mask video file and return list of boolean masks (H, W)."""
    cap = cv2.VideoCapture(str(mask_path))
    masks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        masks.append(gray > 127)
    cap.release()
    return masks


def update_background(video_dir):
    """Regenerate background.mp4 for a single video directory.

    Reads all non-background .mp4 files from masks/, computes their union,
    then inverts to get the background.

    Args:
        video_dir: Path to video directory (must contain source.mp4 and masks/)

    Returns:
        True on success, False on failure
    """
    video_dir = Path(video_dir)
    source_video = video_dir / "source.mp4"
    masks_dir = video_dir / "masks"
    config_path = video_dir / "config.yaml"

    if not source_video.exists():
        print(f"⚠️  Skipping {video_dir.name}: source.mp4 not found")
        return False

    if not masks_dir.exists():
        print(f"⚠️  Skipping {video_dir.name}: masks/ directory not found")
        return False

    # Collect all mask files except background.mp4
    mask_files = sorted([
        p for p in masks_dir.glob("*.mp4")
        if p.stem != "background"
    ])

    if not mask_files:
        print(f"⚠️  Skipping {video_dir.name}: no foreground masks found in masks/")
        return False

    print(f"\n🎬 {video_dir.name}")
    print(f"  Foreground masks: {[p.name for p in mask_files]}")

    # Get video properties from source
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    cap.release()

    # Read all foreground masks
    print(f"  Reading {len(mask_files)} mask video(s)...")
    all_fg_masks = [read_mask_video(p) for p in mask_files]

    # Compute background frame by frame
    bg_masks = []
    for frame_idx in tqdm(range(total_frames), desc="  Computing background"):
        bg = np.ones((h, w), dtype=bool)
        for fg in all_fg_masks:
            if frame_idx < len(fg):
                bg &= ~fg[frame_idx]
        bg_masks.append(bg)

    # Save background
    output_path = masks_dir / "background.mp4"
    success = save_mask_video(bg_masks, output_path, fps=fps)
    if not success:
        return False

    # Update config.yaml: ensure background segment is listed
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        segments = config.get('segments', [])
        # Remove stale background entry if present
        segments = [s for s in segments if s.get('name') != 'background']
        segments.append({
            'name': 'background',
            'desc': '',
            'prompt': '',
            'mask_path': 'masks/background.mp4',
            'sam3_prompt': 'inverse_of_all_foreground',
        })
        config['segments'] = segments

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)
        print(f"  ✓ config.yaml updated")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate background masks from existing foreground masks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_dir", type=str,
                       help="Single video directory to update")
    group.add_argument("--dataset_root", type=str,
                       help="Process all video directories under this root")

    args = parser.parse_args()

    if args.video_dir:
        video_dirs = [Path(args.video_dir)]
    else:
        dataset_root = Path(args.dataset_root)
        video_dirs = sorted([
            d for d in dataset_root.iterdir()
            if d.is_dir() and (d / "source.mp4").exists()
        ])
        print(f"Found {len(video_dirs)} video directories under {dataset_root}")

    success_count = 0
    failed = []

    for video_dir in video_dirs:
        ok = update_background(video_dir)
        if ok:
            success_count += 1
        else:
            failed.append(video_dir.name)

    print("\n" + "=" * 70)
    print(f"✅ Done: {success_count}/{len(video_dirs)} updated")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  • {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
