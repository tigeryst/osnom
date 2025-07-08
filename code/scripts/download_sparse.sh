#!/bin/bash

# Usage: ./code/scripts/download_sparse.sh <video_id> [--storage-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

# Ask user for Dropbox API token if not set
source ./code/scripts/common/auth.sh
get_dropbox_auth_token

echo "Downloading sparse reconstruction: $VIDEO_ID"
echo "Using storage root: $STORAGE_ROOT"

# Paths
SPARSE_DIR="$STORAGE_ROOT/data/colmap_models/sparse/$VIDEO_ID"
TEMP_DIR="$STORAGE_ROOT/data/temp_downloads"
ZIP_PATH="$TEMP_DIR/${VIDEO_ID}_sparse.zip"
DOWNLOAD_DONE_TOKEN="$SPARSE_DIR/download.done"

# Skip if already downloaded
if [ -f "$DOWNLOAD_DONE_TOKEN" ]; then
    echo "Sparse reconstruction for $VIDEO_ID already downloaded."
    exit 0
fi

mkdir -p "$TEMP_DIR"
mkdir -p "$SPARSE_DIR"

echo "Downloading zip to $ZIP_PATH"

curl --location --request POST 'https://content.dropboxapi.com/2/sharing/get_shared_link_file' \
    --header "Authorization: Bearer ${DROPBOX_AUTH_TOKEN}" \
    --header "Dropbox-API-Arg: {\"url\": \"https://www.dropbox.com/scl/fo/0wtphqqyp4fu6bd7dhbfs/h?dl=0&rlkey=ju21graeixi6vpecrf7rqurpt\", \"path\": \"/${VIDEO_ID}.zip\"}" \
    --output "$ZIP_PATH"

echo "Unzipping $ZIP_PATH to $SPARSE_DIR"
unzip -q "$ZIP_PATH" -d "$SPARSE_DIR"

echo "Cleaning up temporary files"
rm "$ZIP_PATH"

# Mark as downloaded
touch "$DOWNLOAD_DONE_TOKEN"

echo "Sparse reconstruction for $VIDEO_ID downloaded successfully!"
