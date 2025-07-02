#! /bin/bash

# Usage: ./code/tracking_code/scripts/extract_feat_3D.sh <video_id>

# Exit on error or unset variable
set -eu

VIDEO_ID=$1
echo $VIDEO_ID

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

python code/tracking_code/extract_feat/save_feat_batch_3D.py \
    --output_dir "results/$VIDEO_ID/feat/3D_feat.pkl" \
    --data_path "data/aggregated/$VIDEO_ID" \
    --frames_path "data/images/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"
