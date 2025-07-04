#!/bin/bash

# Usage: ./code/scripts/download_images.sh <video_id> [--data-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Downloading RGB frames: $VIDEO_ID"
echo "Using data root: $DATA_ROOT"

# Paths
TEMP_DIR="$DATA_ROOT/temp_downloads"
IMAGES_DIR="$DATA_ROOT/images/$VIDEO_ID"
TARBALL_PATH="$TEMP_DIR/${VIDEO_ID}_frames.tar"
DOWNLOAD_DONE_TOKEN="$IMAGES_DIR/download.done"

# Skip if already downloaded
if [ -f "$DOWNLOAD_DONE_TOKEN" ]; then
    echo "Frames for $VIDEO_ID already downloaded."
    exit 0
fi

mkdir -p "$TEMP_DIR"
mkdir -p "$IMAGES_DIR"

# Determine tarball download URL
participant=$(echo "$VIDEO_ID" | cut -d'_' -f1)
video_num=$(echo "$VIDEO_ID" | cut -d'_' -f2)

echo "Downloading tarball to $TARBALL_PATH"
if [ "$video_num" -lt 100 ]; then
    URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/$participant/$VIDEO_ID.tar"
    wget -c "$URL" -O "$TARBALL_PATH" || {
        echo "Primary URL (train) failed, trying alternative (test) for $VIDEO_ID"
        ALT_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/$participant/$VIDEO_ID.tar"
        wget -c "$ALT_URL" -O "$TARBALL_PATH"
    }
else
    URL="https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/$participant/rgb_frames/$VIDEO_ID.tar"
    wget -c "$URL" -O "$TARBALL_PATH"
fi

echo "Extracting $TARBALL_PATH to $IMAGES_DIR"
tar -xf "$TARBALL_PATH" -C "$IMAGES_DIR"

echo "Cleaning up temporary tarball"
rm "$TARBALL_PATH"

# Mark as downloaded
touch "$DOWNLOAD_DONE_TOKEN"

echo "Images for $VIDEO_ID downloaded successfully!"