#!/bin/bash

# Usage: ./code/scripts/reconstruct_mesh.sh <video_id> [--storage-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

echo "Reconstructing mesh: $VIDEO_ID"
echo "Using storage root: $STORAGE_ROOT"

# Paths
IMAGE_PATH="$STORAGE_ROOT/data/images/$VIDEO_ID"
CAMERAS_PATH="$STORAGE_ROOT/data/colmap_models/sparse/$VIDEO_ID/sparse/0"
DENSE3D_PATH="$STORAGE_ROOT/data/colmap_models/dense3D/$VIDEO_ID"

mkdir -p $DENSE3D_PATH

# =========
# Hyperparameters
minpix=15
# =========

# Skip whole shebang if final mesh already exists
FINAL_MESH_PATH="$DENSE3D_PATH/fused-minpix$minpix-meshed-delaunay-qreg5.ply"
if [ -f "$FINAL_MESH_PATH" ]; then
    echo "Final mesh already exists for $VIDEO_ID at $FINAL_MESH_PATH"
    exit 0
fi

# Prompt for COLMAP_COMMAND if not set in the environment
if [ -z "${COLMAP_COMMAND:-}" ]; then
    read -p "Enter COLMAP_COMMAND [default: colmap]: " user_input
    COLMAP_COMMAND="${user_input:-colmap}"
fi

echo "Using COLMAP_COMMAND: $COLMAP_COMMAND"

echo "Running: colmap image_undistorter"
$COLMAP_COMMAND image_undistorter \
    --image_path $IMAGE_PATH \
    --input_path $CAMERAS_PATH \
    --output_path $DENSE3D_PATH \
    --output_type COLMAP \
    --copy_policy soft-link \
    --max_image_size 2000

echo "Running: colmap patch_match_stereo"
$COLMAP_COMMAND patch_match_stereo \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0

echo "Running: colmap stereo_fusion (min pix$minpix)"
$COLMAP_COMMAND stereo_fusion \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.min_num_pixels $minpix \
    --output_path $DENSE3D_PATH/fused-minpix$minpix.ply

echo "Running: colmap poisson_mesher"
$COLMAP_COMMAND poisson_mesher \
    --input_path $DENSE3D_PATH/fused-minpix$minpix.ply \
    --output_path $DENSE3D_PATH/fused-minpix$minpix-meshed-poisson-d10-t5.ply \
    --PoissonMeshing.depth 10 \
    --PoissonMeshing.num_threads 5 \
    --PoissonMeshing.trim 5

echo "Copying fused-minpix$minpix.ply to fused.ply"
cp $DENSE3D_PATH/fused-minpix$minpix.ply $DENSE3D_PATH/fused.ply
cp $DENSE3D_PATH/fused-minpix$minpix.ply.vis $DENSE3D_PATH/fused.ply.vis
echo "Running: colmap delaunay_mesher"
$COLMAP_COMMAND delaunay_mesher \
    --input_path $DENSE3D_PATH \
    --output_path $FINAL_MESH_PATH \
    --DelaunayMeshing.quality_regularization 5
