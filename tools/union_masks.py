#!/usr/bin/env python3
"""
Union two existing mask videos into one mask video (OurBench structure).

Output encoding is exactly the same as other mask outputs by reusing
`save_mask_video` from tools/generate_masks.py (H.264/yuv420p when available).

Usage:
    conda run -n sam3 python tools/union_masks.py \
        --video_dir data/OurBench/bears-fighting-by-road \
        --mask1 bear_left \
        --mask2 bear_right \
        --output union_bears
"""

import argparse
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import yaml

_spec = importlib.util.spec_from_file_location(
    "generate_masks", Path(__file__).parent / "generate_masks.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
save_mask_video = _mod.save_mask_video


def read_mask_video(mask_path):
    """Read mask video and return list of boolean masks (H, W)."""
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


def resolve_mask_path(masks_dir: Path, mask_arg: str) -> Path:
    """Resolve a mask arg as path or masks/<name>.mp4."""
    candidate = Path(mask_arg)
    if candidate.exists():
        return candidate

    name = mask_arg if mask_arg.endswith(".mp4") else f"{mask_arg}.mp4"
    candidate = masks_dir / name
    return candidate


def update_config(config_path: Path, output_name: str, mask1: str, mask2: str):
    """Add or replace union segment entry in config.yaml."""
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    segments = config.get("segments", [])
    segments = [s for s in segments if s.get("name") != output_name]
    segments.append({
        "name": output_name,
        "description": "",
        "mask_path": f"masks/{output_name}.mp4",
        "sam3_prompt": f"union_of:{mask1},{mask2}",
    })
    config["segments"] = segments

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Union two mask videos into one output mask")
    parser.add_argument("--video_dir", required=True, type=str, help="OurBench video directory")
    parser.add_argument("--mask1", required=True, type=str, help="First mask name or path")
    parser.add_argument("--mask2", required=True, type=str, help="Second mask name or path")
    parser.add_argument("--output", required=True, type=str, help="Output mask name (without .mp4)")
    parser.add_argument("--no_update_config", action="store_true", help="Do not modify config.yaml")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    source_video = video_dir / "source.mp4"
    masks_dir = video_dir / "masks"
    config_path = video_dir / "config.yaml"

    if not source_video.exists():
        raise FileNotFoundError(f"source.mp4 not found: {source_video}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    mask1_path = resolve_mask_path(masks_dir, args.mask1)
    mask2_path = resolve_mask_path(masks_dir, args.mask2)

    if not mask1_path.exists():
        raise FileNotFoundError(f"mask1 not found: {mask1_path}")
    if not mask2_path.exists():
        raise FileNotFoundError(f"mask2 not found: {mask2_path}")

    print(f"📖 Reading mask1: {mask1_path.name}")
    m1 = read_mask_video(mask1_path)
    print(f"📖 Reading mask2: {mask2_path.name}")
    m2 = read_mask_video(mask2_path)

    if len(m1) == 0 or len(m2) == 0:
        raise ValueError("One of the mask videos has no frames")

    # Use source fps for consistent output playback.
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Union frame-by-frame up to max length. Missing frames are treated as all-black.
    h, w = m1[0].shape
    total_frames = max(len(m1), len(m2))
    union_masks = []

    for idx in range(total_frames):
        a = m1[idx] if idx < len(m1) else np.zeros((h, w), dtype=bool)
        b = m2[idx] if idx < len(m2) else np.zeros((h, w), dtype=bool)

        if a.shape != (h, w):
            a = cv2.resize(a.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if b.shape != (h, w):
            b = cv2.resize(b.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        union_masks.append(a | b)

    output_name = args.output[:-4] if args.output.endswith(".mp4") else args.output
    output_path = masks_dir / f"{output_name}.mp4"

    print(f"💾 Saving union mask: {output_path.name}")
    success = save_mask_video(union_masks, output_path, fps=fps)
    if not success:
        raise RuntimeError("Failed to save union mask")

    if not args.no_update_config:
        update_config(config_path, output_name, mask1_path.stem, mask2_path.stem)
        print("✓ config.yaml updated")

    print("✅ Done")


if __name__ == "__main__":
    main()
