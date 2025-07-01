#! /bin/bash

# Usage: ./code/tracking_code/scripts/extract_feat_3D.sh <video_id>

# Exit on error or unset variable
set -eu

VIDEO_ID=$1
echo $VIDEO_ID

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

# Paths
OUTPUT_PATH = "results"
DATA_PATH = "data/aggregated"
FRAMES_PATH = "data/images"

python code/tracking_code/extract_feat/save_feat_batch_3D.py \
    --output_dir "$OUTPUT_PATH" \
    --data_path "$DATA_PATH/$VIDEO_ID" \
    --mesh_path "$DATA_PATH/$VIDEO_ID" \
    --frames_path "$FRAMES_PATH/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"
