#!/usr/bin/env python3
"""
Generate object masks from videos using SAM3 (OurBench structure version).

Usage:
    python tools/generate_masks.py --video_dir data/OurBench/video_name \
                                    --mask_names "person_1" "person_2" "ball" \
                                    --sam3_prompts "person" "person" "ball"

Or process single video with automatic detection:
    python tools/generate_masks.py --video_dir data/OurBench/video_name \
                                    --sam3_prompts "person" "ball" \
                                    --auto_name

This script will:
1. Load the video from video_dir/source.mp4
2. For each SAM3 prompt, detect and segment all instances
3. Save masks to video_dir/masks/ with specified names
4. Update video_dir/config.yaml with mask information
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def save_mask_video(masks, output_path, fps=30, size=None):
    """Save binary masks as a video file using imageio-ffmpeg (H.264/yuv420p).

    Encodes masks as grayscale content in an H.264/yuv420p MP4, which is
    compatible with virtually all players (VSCode, browsers, VLC, etc.).
    Requires imageio-ffmpeg: pip install imageio-ffmpeg

    Args:
        masks: List of binary masks (H, W) as numpy arrays
        output_path: Path to save the video
        fps: Frames per second
        size: Optional (width, height) to resize masks
    """
    if len(masks) == 0:
        print(f"⚠️  Warning: No masks to save for {output_path}")
        return False

    try:
        import imageio
    except ImportError:
        print(f"⚠️  imageio not installed, falling back to opencv")
        return save_mask_video_opencv(masks, output_path, fps, size)

    # Get dimensions
    h, w = masks[0].shape
    if size:
        w, h = size

    try:
        # Use imageio-ffmpeg to write H.264/yuv420p — compatible with all players.
        # yuv420p requires RGB input (3-channel), so we stack grayscale 3x.
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=1,  # disable auto-padding to preserve exact source dimensions
        )
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            if size and (mask.shape[1] != w or mask.shape[0] != h):
                mask_uint8 = cv2.resize(mask_uint8, size)
            # Stack to RGB so ffmpeg can encode as yuv420p without color shifts
            frame_rgb = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
            writer.append_data(frame_rgb)
        writer.close()
        print(f"  ✓ Saved: {output_path.name}")
        return True
    except Exception as e:
        print(f"⚠️  imageio-ffmpeg failed: {e}, trying opencv fallback")
        return save_mask_video_opencv(masks, output_path, fps, size)


def save_mask_video_opencv(masks, output_path, fps=30, size=None):
    """Fallback: Save binary masks using OpenCV with mp4v codec.

    Args:
        masks: List of binary masks (H, W) as numpy arrays
        output_path: Path to save the video
        fps: Frames per second
        size: Optional (width, height) to resize masks
    """
    if len(masks) == 0:
        return False

    # Get dimensions
    h, w = masks[0].shape
    if size:
        w, h = size

    # Use mp4v codec (more widely supported in opencv)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=False)

    if not out.isOpened():
        print(f"⚠️  Error: Could not open video writer for {output_path}")
        return False

    for mask in masks:
        # Convert boolean to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Resize if needed
        if size and (mask.shape[1] != w or mask.shape[0] != h):
            mask_uint8 = cv2.resize(mask_uint8, size)

        out.write(mask_uint8)

    out.release()
    print(f"  ✓ Saved: {output_path.name} (opencv mp4v)")
    return True


def extract_masks_for_prompt(video_predictor, session_id, prompt, frame_index=0):
    """Extract all object masks for a given text prompt.

    Args:
        video_predictor: SAM3 video predictor instance
        session_id: Video session ID
        prompt: Text prompt (e.g., "dog", "cat")
        frame_index: Frame to add the prompt on

    Returns:
        dict: {obj_id: [list of masks per frame]}
    """
    print(f"\n📝 Processing SAM3 prompt: '{prompt}'")

    # Add text prompt
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_index,
            text=prompt
        )
    )

    # Propagate through video (following official example notebook)
    outputs_generator = video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    )

    # Collect masks for each object
    object_masks = {}
    frame_count = 0

    for response in tqdm(outputs_generator, desc=f"  Tracking '{prompt}'"):
        frame_count += 1

        # Extract frame_index and outputs from response
        if not isinstance(response, dict):
            continue

        frame_idx = response.get("frame_index")
        outputs = response.get("outputs")

        if outputs is None or not isinstance(outputs, dict):
            continue

        # Get masks for all detected objects
        masks = outputs.get("out_binary_masks", [])  # Shape: (N_objects, H, W)
        obj_ids = outputs.get("out_obj_ids", [])

        for obj_id, mask in zip(obj_ids, masks):
            if obj_id not in object_masks:
                object_masks[obj_id] = []
            object_masks[obj_id].append(mask)

    print(f"  ✓ Found {len(object_masks)} object(s) across {frame_count} frames")
    return object_masks


def generate_background_mask(video_path, all_foreground_masks):
    """Generate background mask by inverting all foreground masks.

    Args:
        video_path: Path to original video
        all_foreground_masks: List of foreground mask videos (each is list of frames)

    Returns:
        List of background masks (one per frame)
    """
    # Read video to get dimensions
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    h, w = frame.shape[:2]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    background_masks = []

    for frame_idx in tqdm(range(total_frames), desc="  Generating background"):
        # Start with all white (everything is background)
        bg_mask = np.ones((h, w), dtype=bool)

        # Subtract all foreground objects
        for fg_obj_masks in all_foreground_masks:
            if frame_idx < len(fg_obj_masks):
                bg_mask = bg_mask & ~fg_obj_masks[frame_idx]

        background_masks.append(bg_mask)

    return background_masks


def update_config_yaml(config_path, sam3_prompts, mask_info, add_background=False):
    """Update config.yaml with mask information.

    Args:
        config_path: Path to config.yaml
        sam3_prompts: List of SAM3 prompts used
        mask_info: List of dicts with keys: name, mask_path, sam3_prompt
        add_background: Whether background was generated
    """
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Update SAM3 prompts
    config['sam3_prompts'] = sam3_prompts

    # Update segments
    config['segments'] = []
    for info in mask_info:
        config['segments'].append({
            'name': info['name'],
            'desc': '',  # To be filled later
            'prompt': '',
            'mask_path': info['mask_path'],
            'sam3_prompt': info['sam3_prompt'],
        })

    # Add background if generated
    if add_background:
        config['segments'].append({
            'name': 'background',
            'desc': '',
            'prompt': '',
            'mask_path': 'masks/background.mp4',
            'sam3_prompt': 'inverse_of_all_foreground',
        })

    # Keep desc/prompt fields (to be filled later)
    if 'desc' not in config:
        config['desc'] = ''
    if 'prompt' not in config:
        config['prompt'] = ''

    # Save config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n✓ Updated config: {config_path}")


def process_video(video_predictor, video_dir, sam3_prompts, mask_names=None,
                  auto_name=False, frame_index=0, add_background=False):
    """Process a single video with a pre-loaded SAM3 predictor.

    Args:
        video_predictor: Pre-loaded SAM3 video predictor instance
        video_dir: Path to video directory (must contain source.mp4)
        sam3_prompts: List of SAM3 text prompts
        mask_names: Optional list of custom mask names
        auto_name: If True, auto-generate names like "prompt_1", "prompt_2"
        frame_index: Frame index to add prompts on
        add_background: If True, generate background mask

    Returns:
        True on success, False on failure
    """
    video_dir = Path(video_dir)
    source_video = video_dir / "source.mp4"
    if not source_video.exists():
        print(f"⚠️  Source video not found: {source_video}")
        return False

    masks_dir = video_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    config_path = video_dir / "config.yaml"

    # Get video properties
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 70)
    print(f"🎬 Processing: {video_dir.name}")
    print("=" * 70)
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print(f"SAM3 Prompts: {sam3_prompts}")

    # Start video session
    print("🎥 Starting video session...")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(source_video)
        )
    )
    session_id = response["session_id"]

    # Process each prompt
    all_foreground_masks = []
    all_detected_objects = []  # [(prompt, obj_idx, masks), ...]
    mask_info = []

    for prompt in sam3_prompts:
        object_masks = extract_masks_for_prompt(
            video_predictor, session_id, prompt, frame_index
        )

        for obj_idx, (obj_id, masks) in enumerate(object_masks.items()):
            if len(masks) > 0:
                all_detected_objects.append((prompt, obj_idx, masks))
                all_foreground_masks.append(masks)

        video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

    # Close session to free GPU memory before saving
    video_predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )

    # Determine mask names
    if mask_names:
        if len(mask_names) != len(all_detected_objects):
            print(f"\n⚠️  ERROR: Number of mask names ({len(mask_names)}) "
                  f"doesn't match detected objects ({len(all_detected_objects)})")
            print("Please provide correct number of mask_names")
            return False
    elif auto_name:
        mask_names = []
        prompt_counts = {}
        for prompt, obj_idx, _ in all_detected_objects:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            mask_names.append(f"{prompt}_{prompt_counts[prompt]}")
    else:
        print(f"\n⚠️  ERROR: Please provide mask_names or set auto_name=True")
        print(f"Detected {len(all_detected_objects)} objects:")
        for i, (prompt, obj_idx, _) in enumerate(all_detected_objects):
            print(f"  {i+1}. From prompt '{prompt}' (object {obj_idx+1})")
        return False

    # Save masks
    print(f"\n💾 Saving {len(all_detected_objects)} mask videos...")
    for (prompt, obj_idx, masks), mask_name in zip(all_detected_objects, mask_names):
        output_path = masks_dir / f"{mask_name}.mp4"
        success = save_mask_video(masks, output_path, fps=fps)
        if success:
            mask_info.append({
                'name': mask_name,
                'mask_path': f"masks/{mask_name}.mp4",
                'sam3_prompt': prompt
            })

    # Generate background mask if requested
    if add_background:
        print("\n🌄 Generating background mask...")
        bg_masks = generate_background_mask(source_video, all_foreground_masks)
        if len(bg_masks) > 0:
            save_mask_video(bg_masks, masks_dir / "background.mp4", fps=fps)

    # Update config.yaml
    update_config_yaml(config_path, sam3_prompts, mask_info, add_background)

    print("\n" + "=" * 70)
    print("✅ SUMMARY")
    print("=" * 70)
    print(f"Video: {video_dir.name}")
    print(f"Masks generated: {len(mask_info)}")
    for info in mask_info:
        print(f"  • {info['name']} (prompt: '{info['sam3_prompt']}')")
    if add_background:
        print(f"  • background")
    print(f"Masks saved to: {masks_dir}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate object masks using SAM3 (OurBench structure)")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to video directory (e.g., data/OurBench/video_name)")
    parser.add_argument("--sam3_prompts", nargs="+", required=True,
                        help="SAM3 prompts for segmentation (e.g., 'person' 'ball')")
    parser.add_argument("--mask_names", nargs="*", default=None,
                        help="Custom names for masks (must match number of detected objects)")
    parser.add_argument("--auto_name", action="store_true",
                        help="Auto-generate mask names (prompt_1, prompt_2, etc.)")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index to add prompts on (default: 0)")
    parser.add_argument("--add_background", action="store_true",
                        help="Generate background mask (inverse of all objects)")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    print("\n🔧 Loading SAM3 model...")
    video_predictor = build_sam3_video_predictor()

    process_video(
        video_predictor=video_predictor,
        video_dir=video_dir,
        sam3_prompts=args.sam3_prompts,
        mask_names=args.mask_names,
        auto_name=args.auto_name,
        frame_index=args.frame_index,
        add_background=args.add_background,
    )


if __name__ == "__main__":
    main()
