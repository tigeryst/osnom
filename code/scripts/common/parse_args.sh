#!/bin/bash

# Usage:
#   source ./code/scripts/common/parse_video_args.sh "$@"

function parse_video_args() {
    local usage_msg="Usage: $0 <video_id> [--data-root <path>]"

    if [ $# -lt 1 ]; then
        echo "$usage_msg"
        exit 1
    fi

    VIDEO_ID=$1
    shift

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
            echo "$usage_msg"
            exit 1
            ;;
        esac
    done

    export VIDEO_ID
    export DATA_ROOT
}

function parse_batch_args() {
    local usage_msg="Usage: $0 [--data-root <path>] [--visualize]"

    DATA_ROOT="data"
    VISUALIZE="false"  # default: visualization disabled

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
        --visualize)
            VISUALIZE="true"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data-root <path>] [--visualize]"
            exit 1
            ;;
        esac
    done

    export DATA_ROOT
    export VISUALIZE
}
