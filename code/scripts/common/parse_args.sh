#!/bin/bash

# Usage:
#   source ./code/scripts/common/parse_video_args.sh "$@"

function parse_video_args() {
    local usage_msg="Usage: $0 <video_id> [--storage-root <path>]"

    if [ $# -lt 1 ]; then
        echo "$usage_msg"
        exit 1
    fi

    VIDEO_ID=$1
    shift

    # Expect arguments starting with optional --storage-root

    # Sets STORAGE_ROOT, defaults to '.'
    STORAGE_ROOT="."

    while [[ $# -gt 0 ]]; do
        case "$1" in
        --storage-root)
            if [ $# -lt 2 ]; then
                echo "Error: --storage-root requires a value."
                exit 1
            fi
            STORAGE_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "$2"
            echo "$usage_msg"
            exit 1
            ;;
        esac
    done

    export VIDEO_ID
    export STORAGE_ROOT
}

function parse_batch_args() {
    local usage_msg="Usage: $0 [--storage-root <path>] [--visualize]"

    STORAGE_ROOT="."
    VISUALIZE="false"  # default: visualization disabled

    while [[ $# -gt 0 ]]; do
        case "$1" in
        --storage-root)
            if [ $# -lt 2 ]; then
                echo "Error: --storage-root requires a value."
                exit 1
            fi
            STORAGE_ROOT="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE="true"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--storage-root <path>] [--visualize]"
            exit 1
            ;;
        esac
    done

    export STORAGE_ROOT
    export VISUALIZE
}
