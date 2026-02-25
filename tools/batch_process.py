#!/usr/bin/env python3
"""
Batch process multiple videos in OurBench dataset.

Loads SAM3 model once and processes all videos in sequence,
avoiding the overhead of reloading the model for each video.

Usage:
    conda run -n sam3 python tools/batch_process.py --config tools/batch_config.yaml

batch_config.yaml format:
    videos:
      - name: "bears-fighting-by-road"
        sam3_prompts: ["bear", "bear"]
        mask_names: ["bear_left", "bear_right"]
        add_background: true

      - name: "dogs-jump"
        sam3_prompts: ["dog"]
        auto_name: true
        add_background: true
"""

import argparse
from pathlib import Path

import importlib.util
import sys
from pathlib import Path

import yaml

# Load generate_masks from the same directory (tools/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "generate_masks", Path(__file__).parent / "generate_masks.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
process_video = _mod.process_video

from sam3.model_builder import build_sam3_video_predictor


def main():
    parser = argparse.ArgumentParser(description="Batch process OurBench videos")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to batch config YAML file")
    parser.add_argument("--dataset_root", type=str, default="data/OurBench",
                        help="Root directory of OurBench dataset (default: data/OurBench)")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'videos' not in config:
        print("❌ Config file must contain 'videos' list")
        return

    dataset_root = Path(args.dataset_root)
    videos = config['videos']

    print(f"📋 Batch processing {len(videos)} videos")
    print(f"Dataset root: {dataset_root}")

    # Load model once for all videos
    print("\n🔧 Loading SAM3 model...")
    video_predictor = build_sam3_video_predictor()

    success_count = 0
    failed_videos = []

    for i, video_config in enumerate(videos):
        video_name = video_config['name']
        video_dir = dataset_root / video_name

        print(f"\n[{i+1}/{len(videos)}] {video_name}")

        if not video_dir.exists():
            print(f"⚠️  Skipping: directory not found ({video_dir})")
            failed_videos.append(video_name)
            continue

        success = process_video(
            video_predictor=video_predictor,
            video_dir=video_dir,
            sam3_prompts=video_config['sam3_prompts'],
            mask_names=video_config.get('mask_names'),
            auto_name=video_config.get('auto_name', True),
            frame_index=video_config.get('frame_index', 0),
            add_background=video_config.get('add_background', True),
        )

        if success:
            success_count += 1
        else:
            failed_videos.append(video_name)

    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total: {len(videos)}  |  Success: {success_count}  |  Failed: {len(failed_videos)}")
    if failed_videos:
        print("\nFailed videos:")
        for name in failed_videos:
            print(f"  • {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
