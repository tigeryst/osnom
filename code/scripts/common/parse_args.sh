#!/bin/bash

# Usage:
#   source ./code/scripts/common/parse_video_args.sh "$@"

function _parse_data_root() {
    # Expect arguments starting with optional --data-root

    # Sets DATA_ROOT, defaults to 'data'
    DATA_ROOT="data"

    while [[ $# -gt 0 ]]; do
        case "$1" in
        --data-root)
            if [ $# -lt 2 ]; then
                echo "Error: --data-root requires a value."
                exit 1
            fi
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "$2"
            echo "$USAGE_MSG"
            exit 1
            ;;
        esac
    done
    export DATA_ROOT
}

function parse_video_args() {
    USAGE_MSG="Usage: $0 <video_id> [--data-root <path>]"

    if [ $# -lt 1 ]; then
        echo "$USAGE_MSG"
        exit 1
    fi

    VIDEO_ID=$1
    shift

    _parse_data_root "$@"

    export VIDEO_ID
}

function parse_data_root_args() {
    USAGE_MSG="Usage: $0 [--data-root <path>]"

    _parse_data_root "$@"
}
