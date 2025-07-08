# Usage: python code/tracking_code/scripts/run_lmk.py <video_id> [--storage-root <path>]

import sys
import os
sys.path.append(os.path.join('code', 'tracking_code'))

import wandb
from tracking.tracker import PHALP
import argparse

# Parameters

# Weight assigned to appearance features in the model.
BETA_0 = 2  
# Weight assigned to location features in the model.
BETA_1 = 13  
# Number of past appearance features to consider for aggregation.
PAST_LOOKBACK = 100  
# Threshold used in the Hungarian algorithm for assignment problems.
# Determines when associations are made or rejected based on costs.
HUNGARIAN_TH = 10  
# Type of distance metric to use:
# - 'A0': Considers only appearance features.
# - 'L0': Considers only location features.
# - 'AL': Combines appearance and location features.
DISTANCE_TYPE = 'AL'

# Helper Functions
def get_sweep_config(video_id, storage_root, visualize):
    return {
        "method": "grid",
        "parameters": {
            "use_velocity": {"value": False},
            "activation": {"values": ["sigmoid"]},
            "T": {"value": 1},
            "beta_0": {"values": [BETA_0]},
            "beta_1": {"values": [BETA_1]},
            "beta_2": {"values": [1.0]},
            "beta_3": {"values": [1.0]},
            "output_path": {"values": [os.path.join("storage_root", "results", video_id, "track")]},
            "data_path": {"values": [os.path.join(storage_root, "data", "aggregated", video_id)]},
            "feat_path_2d": {"values": [os.path.join("storage_root", "results", video_id, "feat", "2D_feat.pkl")]},
            "feat_path_3d": {"values": [os.path.join("storage_root", "results", video_id, "feat", "3D_feat.pkl")]},
            "frames_path": {"values": [os.path.join(storage_root, "data", "images", video_id)]},
            "kitchen": {"values": [video_id]},
            "use_unproj": {"values": [False]},
            "visualize": {"values": [visualize]},
            "save_res": {"values": [True]},
            "random": {"values": [False]},
            "n_init": {"values": [5]},
            "predict": {"values": ["A"]},
            "distance_type": {"values": [DISTANCE_TYPE]},
            "alpha": {"values": [0.1]},
            "hungarian_th": {"values": [HUNGARIAN_TH]},
            "track_history": {"values": [7]},
            "max_age_track": {"values": [10000000]},
            "encode_type": {"values": ["4c"]},
            "past_lookback": {"values": [PAST_LOOKBACK]},
            "detection_type": {"values": ["mask"]},
            "shot": {"values": [0]},
            "aggregation": {"values": ["mean"]},
            "model": {"values": ["dino"]},
            "use_pred": {"value": False},
            "experiment_name": {"value": "demo"},
            "track_dataset": {"value": "demo"},
            "device": {"value": "cuda"},
            "train": {"value": False},
            "debug": {"value": False},
            "use_gt": {"value": False},
            "batch_id": {"value": -1},
            "verbose": {"value": False},
            "detect_shots": {"value": False},
            "video_seq": {"value": None},
        }
    }

def main():
    parser = argparse.ArgumentParser(description="LMK tracking")
    parser.add_argument("video_id", help="Video ID")
    parser.add_argument(
        "--storage-root",
        default=".",
        help="Data root directory path (default: current directory)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization during tracking",
    )

    args = parser.parse_args()

    print(f"Tracking: {args.video_id}")
    print(f"Using storage root: {args.storage_root}")

    wandb.login()
    sweep_config = get_sweep_config(args.video_id, args.storage_root, args.visualize)
    sweep_id = wandb.sweep(sweep_config, project="tuning")

    phalp = PHALP()
    wandb.agent(sweep_id, phalp.track)

if __name__ == "__main__":
    main()
