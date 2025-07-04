#!/bin/bash

# Usage: ./code/scripts/batch.sh [--data-root <path>] [--visualize]

set -eu

source ./code/scripts/common/log.sh

source ./code/scripts/common/parse_args.sh "$@"
parse_data_root_args "$@"

# echo "Processing video: $VIDEO_ID"
echo "Using data root: $DATA_ROOT"

VIDEO_IDS_PATH="code/scripts/videos.txt"

while IFS= read -r VIDEO_ID; do
    # Skip empty lines
    [ -z "$VIDEO_ID" ] && continue

    log INFO "Processing video: $VIDEO_ID"

    # Assumes $DATA_ROOT/frame_mapping.json has been downloaded
    # Assumes Depth-Anything repo has been cloned to ../Depth-Anything
    # Assumes Dropbox API token is set in .dropbox_token

    # Download RGB frames to $DATA_ROOT/images/$VIDEO_ID
    ./code/scripts/download_images.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "Images downloaded for $VIDEO_ID"

    # $DATA_ROOT/colmap_models/sparse/$VIDEO_ID
    ./code/scripts/download_sparse.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "Sparse model downloaded for $VIDEO_ID"

    # $DATA_ROOT/aggregated/$VIDEO_ID/mask_annotations.json
    ./code/scripts/download_masks.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "2D masks downloaded for $VIDEO_ID"
    # $DATA_ROOT/aggregated/$VIDEO_ID/poses.json
    ./code/scripts/download_poses.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "3D poses downloaded for $VIDEO_ID"

    # $DATA_ROOT/colmap_models/dense3D/$VIDEO_ID
    ./code/mesh_generation_code/reconstruct_mesh.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "Dense mesh reconstructed for $VIDEO_ID"

    # Copy mesh
    MESH_NAME="fused-minpix15-meshed-delaunay-qreg5.ply"
    SOURCE_MESH_PATH="$DATA_ROOT/colmap_models/dense3D/$VIDEO_ID/$MESH_NAME"
    TARGET_MESH_PATH="$DATA_ROOT/aggregated/$VIDEO_ID/$MESH_NAME"
    if [ ! -f "$TARGET_MESH_PATH" ]; then
        cp "$SOURCE_MESH_PATH" "$TARGET_MESH_PATH"
        log INFO "Mesh copied to $TARGET_MESH_PATH"
    else
        log INFO "Mesh already exists at $TARGET_MESH_PATH, skipping copy"
    fi

    # Extract features
    ./code/scripts/extract_feat_2D.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "2D features extracted for $VIDEO_ID"
    ./code/scripts/extract_feat_3D.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "3D features extracted for $VIDEO_ID"

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate OSNOM

    # Track objects
    if $VISUALIZE; then
        ./code/scripts/run_lmk.py $VIDEO_ID --data-root $DATA_ROOT --visualize
    else
        ./code/scripts/run_lmk.py $VIDEO_ID --data-root $DATA_ROOT
    fi
    log INFO "LMK tracking completed for $VIDEO_ID"

    # Evaluate
    ./code/scripts/run_eval.sh $VIDEO_ID --data-root $DATA_ROOT
    log INFO "Evaluation completed for $VIDEO_ID"

    log INFO "Batch processing completed for $VIDEO_ID"

done <"$VIDEO_IDS_PATH"
