#!/bin/bash
# Convert all mask videos to H.264 for better compatibility

if [ -z "$1" ]; then
    echo "Usage: $0 <video_directory>"
    echo "Example: $0 data/OurBench/dogs-jump"
    exit 1
fi

VIDEO_DIR="$1"
MASKS_DIR="$VIDEO_DIR/masks"

if [ ! -d "$MASKS_DIR" ]; then
    echo "Error: Masks directory not found: $MASKS_DIR"
    exit 1
fi

echo "Converting masks in: $MASKS_DIR"
echo ""

for mask in "$MASKS_DIR"/*.mp4; do
    if [ ! -f "$mask" ]; then
        continue
    fi

    filename=$(basename "$mask")
    temp="${mask}.temp.mp4"

    echo "Converting: $filename"
    ffmpeg -i "$mask" -c:v libx264 -pix_fmt yuv420p -y "$temp" 2>&1 | grep -v "frame=" | grep -v "size="

    if [ $? -eq 0 ]; then
        mv "$temp" "$mask"
        echo "  ✓ Done"
    else
        echo "  ✗ Failed"
        rm -f "$temp"
    fi
    echo ""
done

echo "All masks converted!"
