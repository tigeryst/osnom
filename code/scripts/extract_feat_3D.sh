#!/bin/bash

# Usage: ./code/scripts/extract_feat_3D.sh <video_id> [--data-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Extracting 3D feature: $VIDEO_ID"
echo "Using data root: $DATA_ROOT"

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

python code/tracking_code/extract_feat/save_feat_batch_3D.py \
    --output_path "results/$VIDEO_ID/feat/3D_feat.pkl" \
    --data_path "$DATA_ROOT/aggregated/$VIDEO_ID" \
    --frames_path "$DATA_ROOT/images/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"
