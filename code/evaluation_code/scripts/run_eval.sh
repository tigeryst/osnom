#! /bin/bash

# Usage: ./code/evaluation_code/scripts/run_eval.sh <video_id>

# Exit on error or unset variable
set -eu

VIDEO_ID=$1
echo $VIDEO_ID

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

python code/evaluation_code/evaluate.py \
    --output_path "results/$VIDEO_ID/eval.pkl" \
    --results_path "results/$VIDEO_ID/track/results.pkl" \
    --video_info_path "data/EPIC_100_video_info.csv" \
    --frames_path "data/images/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"