#!/bin/bash

# Usage: ./code/scripts/download_masks.sh <video_id> [--data-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Downloading 2D masks: $VIDEO_ID"
echo "Using data root: $DATA_ROOT"

# Paths
TEMP_DIR="$DATA_ROOT/temp_downloads"
MASKS_PATH="$DATA_ROOT/aggregated/$VIDEO_ID/mask_annotations.json"
MASKS_DIR="$(dirname "$MASKS_PATH")"
ZIP_PATH="$TEMP_DIR/${VIDEO_ID}_masks.zip"

# Skip if already downloaded
if [ -f "$MASKS_PATH" ]; then
    echo "Masks for $VIDEO_ID already downloaded."
    exit 0
fi

mkdir -p "$TEMP_DIR"
mkdir -p "$MASKS_DIR"

echo "Downloading zip to $ZIP_PATH"
URL="https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/train/${VIDEO_ID}_interpolations.zip"
wget -c "$URL" -O "$ZIP_PATH" || {
    echo "Primary URL (train) failed, trying alternative (val) for $VIDEO_ID"
    ALT_URL="https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/val/${VIDEO_ID}_interpolations.zip"
    wget -c "$ALT_URL" -O "$ZIP_PATH"
}

echo "Unzipping $ZIP_PATH to $MASKS_DIR"
unzip -q "$ZIP_PATH" -d "$MASKS_DIR"
mv "$MASKS_DIR/${VIDEO_ID}_interpolations.json" "$MASKS_PATH"

echo "Cleaning up temporary files"
rm "$ZIP_PATH"

echo "2D masks for $VIDEO_ID downloaded successfully!"
