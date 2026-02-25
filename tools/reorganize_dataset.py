#!/usr/bin/env python3
"""
Reorganize OurBench dataset to the new structure.

Before:
    data/OurBench/
    ├── source/*.mp4
    └── masks/ (empty)

After:
    data/OurBench/
    ├── video1/
    │   ├── source.mp4
    │   ├── masks/
    │   └── config.yaml
    ├── video2/
    │   ├── source.mp4
    │   ├── masks/
    │   └── config.yaml
    ...
"""

import argparse
import shutil
from pathlib import Path
import yaml


def create_config_yaml(video_name):
    """Create initial config.yaml structure.

    Args:
        video_name: Name of the video (without extension)

    Returns:
        dict: Config structure
    """
    config = {
        'video_name': video_name,
        'prompt': '',  # Full prompt combining all segments (to be filled later)
        'sam3_prompts': [],  # List of prompts used for SAM3 segmentation
        'segments': []  # Will be populated when masks are generated
    }
    return config


def reorganize(source_dir, output_dir, keep_original=True):
    """Reorganize videos into new structure.

    Args:
        source_dir: Directory containing source videos
        output_dir: Root directory for reorganized structure
        keep_original: Keep original files if True, move them if False
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Get all video files
    video_files = sorted(source_dir.glob("*.mp4"))

    if len(video_files) == 0:
        print(f"No video files found in {source_dir}")
        return

    print(f"Found {len(video_files)} videos to reorganize")
    print(f"Output directory: {output_dir}")
    print()

    for video_file in video_files:
        # Get video name without extension
        video_name = video_file.stem

        # Create video directory
        video_dir = output_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)

        # Create masks subdirectory
        masks_dir = video_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Copy or move source video
        dest_video = video_dir / "source.mp4"
        if dest_video.exists():
            print(f"⚠️  Skipping {video_name} (already exists)")
            continue

        if keep_original:
            shutil.copy2(video_file, dest_video)
            action = "Copied"
        else:
            shutil.move(str(video_file), str(dest_video))
            action = "Moved"

        # Create config.yaml
        config = create_config_yaml(video_name)
        config_path = video_dir / "config.yaml"

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        print(f"✓ {action} {video_name}")
        print(f"  └─ Created: {video_dir.relative_to(output_dir.parent)}/")

    print()
    print(f"✓ Reorganized {len(video_files)} videos")
    print(f"✓ Structure created in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize OurBench dataset into new structure"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/OurBench/source",
        help="Directory containing source videos (default: data/OurBench/source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/OurBench",
        help="Root directory for reorganized structure (default: data/OurBench)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)"
    )

    args = parser.parse_args()

    reorganize(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        keep_original=not args.move
    )


if __name__ == "__main__":
    main()
