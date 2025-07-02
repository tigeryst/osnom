import sys
import os
sys.path.append(os.path.join("code", "tracking_code"))
import torch.nn as nn
from util import (
    get_colors,
    read_data_1,
    get_object_bbs_seg,
    get_camera_pose_1,
    extract_3d_features,
    get_depth_shared
)
import pickle
import pyrender
import trimesh
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2
import json
import requests

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torchvision.transforms as transforms
from torchvision.transforms import Compose
import torch.nn.functional as F

with open(os.path.join("data", "scaling_scores_dict.json"), "r") as f:
    rescale_scores = json.load(f)

def get_depth_anything(image, depth_anything_model, transform, device='cuda'):
    image = image / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_anything_model(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    return depth

def get_depth_anything_metric(image, model, device='cuda'):
    image_pil = transforms.ToPILImage()(image)
    image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor, dataset='nyu')

    if isinstance(pred, dict):
        pred = pred.get('metric_depth', pred.get('out'))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    depth = pred[0]

    h, w = image.shape[:2]
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    return depth


def draw_random_semantic_mask(ann):
    ann = np.asarray(ann, dtype=np.int32)  # Ensure the annotation is an integer numpy array

    # Create an empty image with the same dimensions as the annotation mask
    mask = np.zeros((ann.shape[0], ann.shape[1], 3), dtype=np.uint8)

    # Generate a random color for each unique value in the annotation mask, excluding the background (assumed to be 0)
    unique_classes = np.unique(ann)
    colors = {cls: np.random.randint(0, 256, size=3, dtype=np.uint8) for cls in unique_classes if cls != 0}

    # Fill the mask with random colors according to the annotation mask
    for cls, color in colors.items():
        mask[ann == cls] = color

    return mask


def align_depth1_to_depth2(depth1, depth2, mask, subsample_size=None):
    # depth1, depth
    depth1_masked = depth1[mask].cpu()
    depth2_masked = depth2[mask]
    if subsample_size is not None:
        subsample = np.random.randint(0, mask.sum(), subsample_size)
        depth1_masked = depth1_masked[subsample]
        depth2_masked = depth2_masked[subsample]

    # Solve for depth_masked * alpha + beta = rend_depth_masked
    A = np.stack([depth1_masked, np.ones_like(depth1_masked)], axis=-1)
    # A = torch.stack([depth1_masked, torch.ones_like(depth1_masked)], dim=-1)

    alpha, beta = np.linalg.lstsq(A, depth2_masked, rcond=None)[0]

    depth1_aligned = depth1 * alpha + beta
    print(f'alpha: {alpha}, beta: {beta}')

    return depth1_aligned.cpu().numpy()
def rename_keys(kitchen, original_dict, mapping_dict):
    # Create a new dictionary with the keys renamed
    renamed_dict = {}
    for old_key, sub_dict in original_dict.items():
        try:
            new_key = mapping_dict[kitchen][
                kitchen + '_' + old_key + '.jpg']
            renamed_dict[new_key.split('.')[0]] = sub_dict
        except:
            renamed_dict[old_key] = sub_dict

    return renamed_dict

class PHALP(nn.Module):
    """
    PHALP class is responsible for processing 3D object feature extraction using depth estimation.
    It handles loading and running models for depth prediction and feature extraction.
    """
    def __init__(self, output_path, data_path, frames_path, kitchen):
        """
        Initializes the PHALP class, loads models for depth estimation, and prepares paths for data processing.

        Args:
            output_path (str): File path to save extracted features.
            data_path (str): Path to dataset containing poses, masks, and 3D mesh.
            frames_path (str): Directory for video frames.
            kitchen (str): Identifier for the specific kitchen or dataset instance.
        """
        super(PHALP, self).__init__()

        # Add Depth-Anything path to system path
        path_to_depth_anything = os.path.join("..", 'Depth-Anything')
        sys.path.append(path_to_depth_anything)

        # Import Depth-Anything model and utilities
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        ENCODER = 'vitl'

        # Load Depth-Anything model
        currdir = os.getcwd()
        os.chdir(path_to_depth_anything)
        depth_anything = DepthAnything.from_pretrained(
            f'LiheYoung/depth_anything_{ENCODER}14'
        ).to(DEVICE).eval()
        os.chdir(currdir)

        # Define image transformation for Depth-Anything
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Lambda function for depth estimation
        self.image_to_depth = lambda image: get_depth_anything(
            image, depth_anything, transform, device=DEVICE
        )

        # Import Zoedepth model for metric depth estimation
        sys.path.append(os.path.join(path_to_depth_anything, 'metric_depth'))
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        # Function to download checkpoint files if they don't exist
        def download_to_path(url, path):
            if os.path.exists(path):
                print(f'File {path} already exists')
                return
            response = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for data in tqdm(response.iter_content(1024 * 1024)):
                    f.write(data)

        # Change directory for checkpoint downloads
        os.chdir(os.path.join(path_to_depth_anything, 'metric_depth'))
        Path('checkpoints').mkdir(exist_ok=True)

        # Download pre-trained models
        download_to_path(
            'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth',
            'checkpoints/depth_anything_vitl14.pth'
        )
        download_to_path(
            'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt',
            'checkpoints/depth_anything_metric_depth_indoor.pt'
        )

        # Load Zoedepth model for metric depth
        config = get_config('zoedepth', "eval", 'nyu')
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        model_metric = build_model(config).to(DEVICE).eval()
        os.chdir(currdir)

        # Lambda function for metric depth estimation
        self.image_to_depth_metric = lambda image: get_depth_anything_metric(
            image, model_metric, device=DEVICE
        )

        # Initialize paths and directories
        self.RGB_tuples = get_colors()
        self.kitchen = kitchen
        self.path_to_save = output_path

        self.data_path = data_path
        self.frames_path = frames_path

        # Load pose data
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)

        # Read other necessary data
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, 
                                                                    kitchen, True)

        # Load frame mapping data
        with open(os.path.join('data', 'frame_mapping.json')) as f:
            self.mapping_dense = json.load(f)

        # Process bounding boxes and segmentations
        self.bbs_dict = get_object_bbs_seg(self.masks['video_annotations'])
        self.bbs_dict = rename_keys(self.kitchen, self.bbs_dict, self.mapping_dense)

        # Set device for computation
        self.device = DEVICE
        print('Device: ', self.device)

        # Set model to evaluation mode
        self.eval()

    def track(self):
        """
        Performs the 3D feature extraction and depth estimation for each frame in the video.

        The method extracts bounding boxes and 3D features for objects in the frames,
        aligns the depth estimates, and saves the resulting 3D features in a pickle file.

        Returns:
            None
        """
        save_dict = {}  # Dictionary to store extracted 3D features

        # Load the 3D mesh for the kitchen
        mesh = trimesh.load(os.path.join(self.data_path, 'fused-minpix15-meshed-delaunay-qreg5.ply'), force='mesh')
        mesh.apply_scale(rescale_scores[self.kitchen])

        # Camera parameters: size and focal length
        image_size = [self.poses['camera']['width'], self.poses['camera']['height']]
        focal_length = self.poses['camera']['params'][:4]
        camera_intrinsics = [image_size[0], image_size[1]] + focal_length

        # Create Pyrender camera with the appropriate field of view
        camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(image_size[1] / (2 * focal_length[1])))
        SCENE = pyrender.Scene()
        RENDERER = pyrender.OffscreenRenderer(image_size[0], image_size[1])

        # Add mesh and camera to the scene
        pmesh = pyrender.Mesh.from_trimesh(mesh)
        SCENE.add(pmesh)
        CAMERA_NODE = SCENE.add(camera, pose=np.eye(4))

        batched_bbs = []  # List to store bounding boxes from all frames
        frame_names = []  # List to store names of frames
        res = (854, 480)  # Resolution for depth maps

        # Iterate over frames and process each
        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            segments, bbs, objs = self.bbs_dict[f"{self.kitchen}_{frame_name}"]
            if len(bbs) == 0:
                continue  # Skip frames with no objects detected

            # Get camera pose for the current frame
            camera_pose = get_camera_pose_1(self.camera_poses, frame_name)
            depth = get_depth_shared(camera_pose, SCENE, CAMERA_NODE, RENDERER)
            frame_path = os.path.join(self.frames_path, f"{frame_name}.jpg")

            # Estimate depth for the frame
            try:
                image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            except:
                continue  # Skip frames that cannot be loaded

            try:
                rend_depth = depth
            except FileNotFoundError:
                continue  # Skip frames where depth is not found

            # Create segmentation masks for detected objects
            height, width = depth.shape[1], depth.shape[0]
            mask_frames = []
            for s in segments:
                mask = np.zeros((480, 854), dtype=np.uint8)
                cv2.fillPoly(mask, np.int32([s]), color=1)
                resized_mask = cv2.resize(mask, (456, 256), interpolation=cv2.INTER_NEAREST)
                out_image = np.stack((resized_mask * 255,) * 3, axis=-1)
                mask_frames.append(out_image)

            if mask_frames:
                segs_all = np.stack(mask_frames, axis=0)
                segs_final = (segs_all > 0).any(axis=-1).any(axis=0)

            # Estimate depth metric and align with rendered depth
            depth_metric = self.image_to_depth_metric(image)
            if mask_frames:
                mask = (~segs_final) & (rend_depth > 1e-6)
            else:
                mask = rend_depth > 1e-6

            depth_metric_aligned = align_depth1_to_depth2(depth_metric, rend_depth, mask, None)

            if bbs:
                batched_bbs.extend(bbs)
                frame_names.extend([frame_name] * len(bbs))
                loca_features, loc_3d_objs, r3d_objs = extract_3d_features(
                    bbs, objs, camera_pose, depth_metric_aligned, camera_intrinsics,
                    self.data_path, frame_name, res
                )
                loca_features = loca_features.squeeze(1)
                features_3d = loca_features[:, :3]
                save_dict[frame_name] = (features_3d.cpu().numpy(), r3d_objs.cpu().numpy(), objs)

        # Create the output directory if it does not exist
        os.makedirs(os.path.dirname(self.path_to_save), exist_ok=True)

        # Save the extracted 3D features to disk
        with open(self.path_to_save, 'wb') as f:
            pickle.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Extract 3D features")
    parser.add_argument("--output_path", required=True, help="Output file path")
    parser.add_argument("--data_path", required=True, help="Data directory path containing poses, masks, and mesh")
    parser.add_argument("--frames_path", required=True, help="RGB frames directory path")
    parser.add_argument("--kitchen", required=True, help="Video ID")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_path, args.data_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
