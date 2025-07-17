#!/bin/bash

# Usage: ./code/scripts/batch_download.sh [--storage-root <path>]

set -eu

source ./code/scripts/common/log.sh

source ./code/scripts/common/parse_args.sh "$@"
parse_batch_args "$@"

echo "Using storage root: $STORAGE_ROOT"

VIDEO_IDS_PATH="code/scripts/videos.txt"

while IFS= read -r VIDEO_ID; do
    # Skip empty lines
    [ -z "$VIDEO_ID" ] && continue

    log INFO "Downloading assets: $VIDEO_ID"

    # Download RGB frames to $STORAGE_ROOT/data/images/$VIDEO_ID
    ./code/scripts/download_images.sh $VIDEO_ID --storage-root "$STORAGE_ROOT"
    log INFO "Images downloaded for $VIDEO_ID"

    # $STORAGE_ROOT/data/colmap_models/sparse/$VIDEO_ID
    ./code/scripts/download_sparse.sh $VIDEO_ID --storage-root "$STORAGE_ROOT"
    log INFO "Sparse model downloaded for $VIDEO_ID"

    # $STORAGE_ROOT/data/aggregated/$VIDEO_ID/mask_annotations.json
    ./code/scripts/download_masks.sh $VIDEO_ID --storage-root "$STORAGE_ROOT"
    log INFO "2D masks downloaded for $VIDEO_ID"
    # $STORAGE_ROOT/data/aggregated/$VIDEO_ID/poses.json
    ./code/scripts/download_poses.sh $VIDEO_ID --storage-root "$STORAGE_ROOT"
    log INFO "3D poses downloaded for $VIDEO_ID"

done <"$VIDEO_IDS_PATH"