import os
import subprocess

# Constants
BASE_PATH = "."
OUTPUT_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, "data", "aggregated")
FRAMES_PATH = os.path.join(BASE_PATH, "data", "images")

# Command Template
BASE_COMMAND = [
    "python",
    os.path.join(
        BASE_PATH, "code", "tracking_code", "extract_feat", "save_feat_batch_2D.py"
    ),
    "--output_dir",
    OUTPUT_PATH,
]


def get_full_command(video):
    """Construct the full command for the given video."""
    return BASE_COMMAND + [
        "--data_path",
        os.path.join(BASE_PATH, "data", "aggregated", video),
        "--frames_path",
        os.path.join(FRAMES_PATH, video),
        "--kitchen",
        video,
    ]


# List videos with complete data to process
videos = []
for video in os.listdir(DATA_PATH):
    dir = os.path.join(DATA_PATH, video)
    if os.path.isdir(dir):
        has_masks = os.path.exists(os.path.join(dir, "mask_annotations.json"))
        has_poses = os.path.exists(os.path.join(dir, "poses.json"))
        has_frames = os.path.exists(os.path.join(FRAMES_PATH, video))

        if has_masks and has_poses and has_frames:
            videos.append(video)

# Main Execution
for video in videos:
    print(f"Processing video: {video}")
    full_command = get_full_command(video)
    print("Executing command:", full_command)
    subprocess.run(full_command)
