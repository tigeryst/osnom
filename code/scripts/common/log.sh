log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Colors: INFO=cyan, SUCCESS=green, ERROR=red, WARN=yellow
    case "$level" in
        INFO)
            color="\033[1;36m" ;;   # bright cyan
        SUCCESS)
            color="\033[1;32m" ;;   # bright green
        WARN)
            color="\033[1;33m" ;;   # bright yellow
        ERROR)
            color="\033[1;31m" ;;   # bright red
        *)
            color="\033[0m" ;;      # reset
    esac

    reset="\033[0m"
    # Print with timestamp, level, and color
    echo -e "${timestamp} [${color}${level}${reset}] $message"
}
