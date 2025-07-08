#!/bin/bash

# Usage: ./code/scripts/extract_feat_2D.sh <video_id> [--storage-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Extracting 2D feature: $VIDEO_ID"
echo "Using storage root: $STORAGE_ROOT"

# Paths
OUTPUT_PATH="$STORAGE_ROOT/results/$VIDEO_ID/feat/2D_feat.pkl"

if [ -f "$OUTPUT_PATH" ]; then
    echo "2D features already exist for $VIDEO_ID at $OUTPUT_PATH"
    exit 0
fi

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

python code/tracking_code/extract_feat/save_feat_batch_2D.py \
    --output_path "$OUTPUT_PATH" \
    --data_path "$STORAGE_ROOT/data/aggregated/$VIDEO_ID" \
    --frames_path "$STORAGE_ROOT/data/images/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"
