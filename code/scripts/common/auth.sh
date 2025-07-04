#!/bin/bash

# Usage:
#   source ./code/scripts/common/auth.sh "$@"

function get_dropbox_auth_token() {
    if [ -n "${DROPBOX_AUTH_TOKEN:-}" ]; then
        # Already set in environment — nothing to do
        return
    fi

    local token_path=".dropbox_token"

    if [ -f "$token_path" ]; then
        # Read token from file if it exists
        echo "Reading Dropbox API token from $token_path"
        DROPBOX_AUTH_TOKEN=$(<"$token_path")
        if [ -z "$DROPBOX_AUTH_TOKEN" ]; then
            echo "Error: $token_path exists but is empty."
            exit 1
        fi
        export DROPBOX_AUTH_TOKEN
        return
    fi

    # Interactively prompt for token
    echo "DROPBOX_AUTH_TOKEN is not set."
    echo "If you don't have a token, you can generate one by following the instructions at:"
    echo "  https://dropbox.tech/developers/generate-an-access-token-for-your-own-account"
    echo
    echo -n "Enter your Dropbox API token: "
    # Disable echo to hide input (password prompt)
    stty -echo
    trap 'stty echo' EXIT
    read -r DROPBOX_AUTH_TOKEN
    stty echo
    trap - EXIT
    echo
    if [ -z "$DROPBOX_AUTH_TOKEN" ]; then
        echo "Error: No token provided."
        exit 1
    fi

    export DROPBOX_AUTH_TOKEN
}
