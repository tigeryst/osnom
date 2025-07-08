#!/bin/bash

# Usage: ./code/scripts/run_eval.sh <video_id> [--storage-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Evaluating: $VIDEO_ID"
echo "Using storage root: $STORAGE_ROOT"

# Paths
OUTPUT_PATH="$STORAGE_ROOT/results/$VIDEO_ID/eval.pkl"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Evaluation already exist for $VIDEO_ID at $OUTPUT_PATH"
    exit 0
fi

# Initialize conda in this shell session
eval "$(conda shell.bash hook)"

conda activate OSNOM

python code/evaluation_code/evaluate.py \
    --output_path "$OUTPUT_PATH" \
    --results_path "$STORAGE_ROOT/results/$VIDEO_ID/track/results.pkl" \
    --frames_path "$STORAGE_ROOT/data/images/$VIDEO_ID" \
    --kitchen "$VIDEO_ID"