#!/bin/bash

# Usage: ./code/scripts/download_poses.sh <video_id> [--data-root <path>]

set -eu

# Parse arguments
source ./code/scripts/common/parse_args.sh "$@"
parse_video_args "$@"

# Ask user for Dropbox API token if not set
source ./code/scripts/common/auth.sh
get_dropbox_auth_token

echo "Downloading camera poses: $VIDEO_ID"
echo "Using data root: $DATA_ROOT"

# Paths
POSES_PATH="$DATA_ROOT/aggregated/$VIDEO_ID/poses.json"

# Skip if already downloaded
if [ -f "$POSES_PATH" ]; then
    echo "Poses for $VIDEO_ID already downloaded."
    exit 0
fi

mkdir -p "$(dirname "$POSES_PATH")"

echo "Downloading poses to $POSES_PATH"

curl --location --request POST 'https://content.dropboxapi.com/2/sharing/get_shared_link_file' \
    --header "Authorization: Bearer ${DROPBOX_AUTH_TOKEN}" \
    --header "Dropbox-API-Arg: {\"url\": \"https://www.dropbox.com/scl/fo/onqyany4ze39pknck49ir/AKTS2LUt3WxFd02z7GdLYqM?rlkey=fc8gb6dz1pi6r89b30ma43x3m&e=2&dl=0\", \"path\": \"/${VIDEO_ID}.json\"}" \
    --output "$POSES_PATH"

echo "Poses for $VIDEO_ID downloaded successfully!"
