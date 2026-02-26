# SAM3 Multi-Object Mask Generation Project

## 🎯 Project Goal
Create multi-object video dataset for CV research by:
1. Selecting videos from MSVBench dataset
2. Generating individual object masks using SAM3
3. Creating prompts for each object (future work)

## 🚀 Quick Start

### Current Status
- ✅ 55 videos selected and reorganized in `data/OurBench/`
- ✅ Each video has: `source.mp4`, `masks/`, `config.yaml`
- ✅ Mask video codec fixed (H.264/yuv420p, plays everywhere)
- ✅ Batch processing runs model only once (no per-video reload)
- 🔜 Generate masks for all videos (batch config in progress)

### Process Single Video
```bash
conda run -n sam3 python tools/generate_masks.py \
    --video_dir data/OurBench/bears-fighting-by-road \
    --sam3_prompts "bear" "bear" \
    --mask_names "bear_left" "bear_right" \
    --add_background
```

### Batch Process Multiple Videos
```bash
# Edit tools/batch.yaml with your videos, then:
conda run -n sam3 python tools/batch_process.py --config tools/batch.yaml
```

### Regenerate Background After Manual Edits
If you delete unwanted masks from `masks/`, recompute background:
```bash
# Single video
conda run -n sam3 python tools/update_background.py \
    --video_dir data/OurBench/boxer-punching-towards-camera

# All videos at once
conda run -n sam3 python tools/update_background.py \
    --dataset_root data/OurBench
```

### Union Two Existing Masks
Create one new mask from the union (OR) of two existing masks in the same video:
```bash
conda run -n sam3 python tools/union_masks.py \
  --video_dir data/OurBench/bears-fighting-by-road \
  --mask1 bear_left \
  --mask2 bear_right \
  --output bear_union
```

Notes:
- `--mask1` and `--mask2` can be either mask names (without `.mp4`) or full paths
- Output is saved as `masks/<output>.mp4` using the same encoding path as other masks (H.264/yuv420p when available)
- `config.yaml` is updated automatically unless `--no_update_config` is set

### Audit masks/ vs config.yaml Consistency
Check that `masks/*.mp4` filenames exactly match `segments[].mask_path` entries in `config.yaml`:
```bash
conda run -n sam3 python tools/check_mask_config_consistency.py \
  --dataset_root data/OurBench
```

This tool reports, per video directory:
- `EXTRA_IN_MASKS`: files in `masks/` but not listed in `config.yaml`
- `MISSING_IN_MASKS`: files listed in `config.yaml` but not present in `masks/`
- Missing folders/files or config parse errors

### Tips
- **Unsure how many objects?** Use `--auto_name` first, then rename/delete as needed
- **Distinguish similar objects?** Use specific prompts: `"person in red"`, `"person in blue"`
- **Object not detected?** Try a different `--frame_index` or more specific prompt (e.g. `"grizzly bear"`)
- **Person mask includes held objects (gloves etc.)** — this is correct SAM3 behavior; overlapping masks are fine for compositing
- **Deleted unwanted masks?** Run `update_background.py` to fix the background

### Note on HuggingFace Access
SAM3 model requires HuggingFace access:
1. Visit https://huggingface.co/facebook/sam3 and request access
2. Create token at https://huggingface.co/settings/tokens
3. Login: `conda run -n sam3 huggingface-cli login`

## Environment Setup

### Conda Environment
```bash
conda activate sam3
# Environment already created with Python 3.12
# All dependencies installed via: pip install -e ".[notebooks]"
```

### Key Dependencies
- SAM3 (editable install)
- PyTorch 2.10.0 with CUDA 12.8
- imageio + imageio-ffmpeg (mask video encoding — **required**, installed separately)
- opencv-python-headless, decord, pycocotools
- timm, numpy, tqdm, ftfy, huggingface_hub
- einops, psutil, scikit-image, hydra-core, omegaconf

> `imageio-ffmpeg` is listed in `pyproject.toml` under `[notebooks]` extras.
> If setting up a fresh environment: `pip install -e ".[notebooks]"` installs it automatically.

## Dataset Structure

### MSVBench Location
```
data/MSVBench/MSVBench/
├── mask/        # Contains mask videos
├── source/      # Contains source videos
└── info.xlsx    # Metadata
```

### OurBench Structure
```
data/OurBench/
├── video_name/
│   ├── source.mp4           # Original video
│   ├── masks/
│   │   ├── object_a.mp4     # Foreground mask (white=object, black=bg)
│   │   ├── object_b.mp4
│   │   └── background.mp4   # Inverse of union of all foreground masks
│   └── config.yaml
...
```

### config.yaml Format
```yaml
video_name: "video_name"
prompt: ""  # Full prompt (to be filled later)
sam3_prompts:
  - "person"
  - "ball"
segments:
  - name: "person_1"
    description: ""
    mask_path: "masks/person_1.mp4"
    sam3_prompt: "person"
  - name: "ball"
    description: ""
    mask_path: "masks/ball.mp4"
    sam3_prompt: "ball"
  - name: "background"
    description: ""
    mask_path: "masks/background.mp4"
    sam3_prompt: "inverse_of_all_foreground"
```

## Mask Generation Workflow

### Step 1: Generate Masks (Single Video)

```bash
conda run -n sam3 python tools/generate_masks.py \
    --video_dir data/OurBench/video_name \
    --sam3_prompts "person" "ball" \
    --mask_names "player_1" "soccer_ball" \
    --add_background
```

Auto-naming (useful when unsure of object count):
```bash
conda run -n sam3 python tools/generate_masks.py \
    --video_dir data/OurBench/video_name \
    --sam3_prompts "person" "person" "ball" \
    --auto_name \
    --add_background
# Output: person_1.mp4, person_2.mp4, ball_1.mp4, background.mp4
```

Parameters:
- `--video_dir`: Path to video directory (must contain `source.mp4`)
- `--sam3_prompts`: SAM3 text prompts
- `--mask_names`: Custom names (must match number of detected objects)
- `--auto_name`: Auto-generate names as `prompt_1`, `prompt_2`, etc.
- `--add_background`: Generate background mask
- `--frame_index`: Frame to add prompt on (default: 0; try other frames if detection fails)

### Step 2: Batch Processing

`batch_process.py` loads the SAM3 model **once** and processes all videos in sequence.

```bash
conda run -n sam3 python tools/batch_process.py --config tools/batch.yaml
```

`batch.yaml` format:
```yaml
videos:
  # auto_name: true and add_background: true are defaults, no need to specify
  - name: "dogs-jump"
    sam3_prompts: ["dog"]

  # Override with custom mask_names if you know exact object count
  - name: "bears-fighting-by-road"
    sam3_prompts: ["bear", "bear"]
    mask_names: ["bear_left", "bear_right"]

  # Use frame_index if detection fails on frame 0
  - name: "some-video"
    sam3_prompts: ["person"]
    frame_index: 30
```

### Step 3: Manual Cleanup + Background Update

After batch runs, you may want to delete unwanted masks (e.g. SAM3 detected 5 people but you only want 3):
```bash
rm data/OurBench/video_name/masks/person_4.mp4
rm data/OurBench/video_name/masks/person_5.mp4

# Then recompute background from remaining masks
conda run -n sam3 python tools/update_background.py \
    --video_dir data/OurBench/video_name
```

### How It Works
1. Loads SAM3 video predictor (once per session)
2. For each video: starts a session with `source.mp4`
3. For each SAM3 prompt:
   - Adds text prompt to specified frame
   - Propagates through entire video
   - Saves all detected instances
   - Resets session for next prompt
4. Closes session (frees GPU memory)
5. Optionally generates background = inverse of union of all foreground masks
6. Updates `config.yaml`

## SAM3 Key Concepts

### Model Capabilities
- Open-vocabulary segmentation (any text concept)
- Video tracking with temporal consistency
- Detects ALL instances of a concept automatically
- Supports text, point, box prompts

### Session Management
```python
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor()

response = predictor.handle_request({"type": "start_session", "resource_path": "video.mp4"})
session_id = response["session_id"]

predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "dog"})

for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    frame_idx = response["frame_index"]
    masks = response["outputs"]["out_binary_masks"]  # (N_objects, H, W) boolean
    obj_ids = response["outputs"]["out_obj_ids"]

predictor.handle_request({"type": "reset_session", "session_id": session_id})   # clear for next prompt
predictor.handle_request({"type": "close_session", "session_id": session_id})   # free GPU memory
```

### Mask Video Encoding
Masks are saved as **H.264/yuv420p MP4** using `imageio-ffmpeg`.
This format plays in VSCode, browsers, VLC, and all modern players.
- Grayscale mask → stacked to 3-channel RGB → encoded as yuv420p
- Falls back to OpenCV `mp4v` if imageio-ffmpeg is unavailable (less compatible)

## Current Status & TODO

✅ Environment setup complete
✅ MSVBench dataset cloned (200 videos)
✅ OurBench dataset created (~52 selected videos)
✅ All scripts working (generate, batch, update_background)
✅ Mask video codec fixed (H.264/yuv420p)
🔜 Complete `tools/batch.yaml` for all videos
🔜 Run full batch mask generation
🔜 Prompt generation (descriptions for each segment)
🔜 Dataset assembly (final format)

## Script Reference

| Script | Purpose |
|--------|---------|
| `tools/generate_masks.py` | Generate masks for a single video (also exposes `process_video()` for batch use) |
| `tools/batch_process.py` | Batch process videos from a YAML config (model loaded once) |
| `tools/update_background.py` | Recompute background from existing foreground masks (no model needed) |
| `tools/union_masks.py` | Union two existing mask videos into one new mask and optionally update `config.yaml` |
| `tools/check_mask_config_consistency.py` | Verify each video's `masks/*.mp4` exactly matches `config.yaml` `segments[].mask_path` entries |
| `tools/reorganize_dataset.py` | Convert flat structure to per-video directory structure |

## Resources
- SAM3 Repo: `/work/rogerfan48/sam3`
- Example Notebook: `examples/sam3_video_predictor_example.ipynb`
- MSVBench: `data/MSVBench/MSVBench/`
- OurBench: `data/OurBench/`
- Tools: `tools/`
