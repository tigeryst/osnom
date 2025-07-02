import sys
import os
sys.path.append(os.path.join("code", "tracking_code"))
import torch.nn as nn
from util import (
    get_colors,
    read_data_all,
    get_object_bbs_new,
    extract_dino_features_batch
)
import torchvision.transforms as T
import pickle
import torch
from tqdm import tqdm
import argparse
import json
import cv2
from PIL import Image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class PHALP(nn.Module):
    """
    PHALP: A class for object tracking and appearance feature extraction using DINOv2.
    """
    def __init__(self, output_path, data_path, frames_path, kitchen):
        """
        Initializes the PHALP class.
        
        Args:
            output_path (str): File path to save extracted features.
            data_path (str): Path to the dataset containing poses and masks.
            frames_path (str): Path to the folder containing video frames.
            kitchen (str): Identifier for the specific kitchen or dataset instance.
        """
        super(PHALP, self).__init__()

        # Initialize class attributes
        self.RGB_tuples = get_colors()  # Predefined RGB tuples for visualization
        self.kitchen = kitchen
        self.path_to_save = output_path

        self.data_path = data_path
        self.frames_path = frames_path

        # Load poses from JSON file
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)

        # Read data including masks, camera poses, and frames
        self.masks, _, self.camera_poses, self.frames, _ = read_data_all(
            self.data_path, kitchen, True
        )

        # Extract object bounding boxes from annotations
        self.bbs_dict = get_object_bbs_new(self.masks['video_annotations'])

        # Select the appropriate device (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device: ', self.device)

        # Load the DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        self.model = self.model.to(self.device)

        # Define image transformation pipeline
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Resize to fixed size
            T.ToTensor(),  # Convert to tensor
            T.Normalize(  # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Move the entire model to the selected device
        self.to(self.device)

        # Set the model to evaluation mode
        self.eval()

    def track(self):
        """
        Track objects across frames and extract 2D appearance features.

        Saves features and corresponding metadata into a pickle file.
        """
        save_dict_2D = {}

        # Set model to evaluation mode
        self.eval()

        # Initialize temporary storage for batch processing
        batched_bbs = []  # Batched bounding boxes
        frames_bbs = []  # Corresponding image frames
        frame_names = []  # Frame names
        batched_objs = []  # Associated object identifiers

        # Iterate over sorted frame names
        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            # Get bounding boxes and object IDs for the frame
            bbs, objs = self.bbs_dict[f"{self.kitchen}_{frame_name}"]

            # Skip frames with no detected objects
            if len(bbs) == 0:
                continue

            # Read and preprocess the image frame
            image_frame = cv2.imread(os.path.join(self.frames_path, f"{frame_name}.jpg"))
            image_frame = cv2.resize(image_frame, (854, 480))
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
            if not isinstance(image_frame, Image.Image):
                image_frame = Image.fromarray(image_frame)

            # Accumulate data for batch processing
            if len(bbs) > 0:
                batched_bbs.extend(bbs)
                batched_objs.extend(objs)
                frames_bbs.extend([image_frame] * len(bbs))
                frame_names.extend([frame_name] * len(bbs))

            # Process batches of bounding boxes
            if len(batched_bbs) >= 100:
                appe_features = extract_dino_features_batch(
                    frames_bbs, batched_bbs, self.model, self.device
                )
                j = 0
                unique_elements_ordered = []

                # Retain unique frame names in order of appearance
                for item in frame_names:
                    if item not in unique_elements_ordered:
                        unique_elements_ordered.append(item)
                
                # Map features to frames
                for i, s in enumerate(unique_elements_ordered):
                    if frame_names.count(s) == len(self.bbs_dict[f"{self.kitchen}_{s}"][0]):
                        save_dict_2D[s] = appe_features[j:j + frame_names.count(s)].cpu().numpy()
                        j += frame_names.count(s)

                # Clear the temporary storage
                batched_bbs = []
                batched_objs = []
                frames_bbs = []
                frame_names = []

        # Handle any remaining bounding boxes in the last batch
        if len(batched_bbs) > 0:
            appe_features = extract_dino_features_batch(
                frames_bbs, batched_bbs, self.model, self.device
            )
            j = 0
            unique_elements_ordered = []

            # Retain unique frame names in order of appearance
            for item in frame_names:
                if item not in unique_elements_ordered:
                    unique_elements_ordered.append(item)

            # Map features to frames
            for i, s in enumerate(unique_elements_ordered):
                if frame_names.count(s) == len(self.bbs_dict[f"{self.kitchen}_{s}"][0]):
                    save_dict_2D[s] = appe_features[j:j + frame_names.count(s)].cpu().numpy()
                    j += frame_names.count(s)

        # Create the output directory if it does not exist
        os.makedirs(os.path.dirname(self.path_to_save), exist_ok=True)

        # Save the extracted 2D features to disk
        with open(self.path_to_save, 'wb') as f:
            pickle.dump(save_dict_2D, f)


def main():
    parser = argparse.ArgumentParser(description="Extract 2D features")
    parser.add_argument("--output_path", required=True, help="Output file path")
    parser.add_argument("--data_path", required=True, help="Data directory path containing poses and masks")
    parser.add_argument("--frames_path", required=True, help="RGB frames directory path")
    parser.add_argument("--kitchen", required=True, help="Video ID")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_path, args.data_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
