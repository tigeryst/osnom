import sys
import os
sys.path.append(os.path.join("code", "tracking_code"))
import torch.nn as nn
from util import *
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
rescale_scores = {
    'P03_04': 0.9744356799,
    'P05_08': 0.6808696067,
    'P06_05': 0.5479392007,
    'P24_08': 0.573,
    'P14_05': 0.639,
    'P07_10': 0.7000619555,
    'P07_101': 0.8526057764,
    'P09_14': 0.807294715,
    'P11_106': 0.7854287263,
    'P12_04': 0.747035204,
    'P12_101': 0.4651162791,
    'P19_04': 0.5028916269,
    'P22_114': 0.8423322495,
    'P23_04': 0.6576783953,
    'P23_112': 0.6657656638,
    'P24_05': 0.5391126206,
    'P25_05': 0.7407407407,
    'P26_124': 0.9245177484,
    'P28_103': 1.143961883,
    'P30_107': 0.8297412452,
    'P01_14': 1.265822785,
    'P02_09': 1.025499033,
    'P04_05': 0.6309944472,
    'P06_101': 0.7367947953,
    'P08_05': 0.6336934825,
    'P10_04': 0.7426248074,
    'P11_05': 0.904159132,
    'P13_08': 0.5752085131,
    'P14_08': 0.7861017216,
    'P15_08': 0.350262697,
    'P17_04': 0.4575528245,
    'P18_05': 0.4780114723,
    'P30_05': 0.6166178511,
    'P32_06': 0.3060069158,
    'P11_101': 0.8367080558,
    'P35_105': 0.826446281,
    'P02_121': 0.8695652174,
    'P15_02': 0.5714285714,
    'P02_107': 0.9950248756,
    'P02_132': 1.342281879,
    'P32_01': 0.6097560976,
    'P22_01': 0.8240626288,
    'P06_11': 1.069518717,
    'P02_102': 1.388888889,
    'P35_109': 0.956937799,
    'P02_124': 0.8220304151,
    'P08_16': 0.5917159763,
    'P02_122': 1.27388535,
    'P37_101': 0.9550642281,
    'P28_110': 0.66,
    'P20_03': 1.321003963,
    'P02_09': 1.020408163,
    'P04_121': 0.4246284501,
    'P37_103': 0.5221932115,
    'P02_101': 0.9615384615,
    'P01_03': 1.739130435,
    'P27_101': 0.9756097561,
    'P28_113': 1.538461538,
    'P01_103': 1.136363636,
    'P30_111': 0.9049773756,
    'P03_13': 0.78125,
    'P02_132': 1.360544218,
    'P23_02': 0.7462686567,
    'P28_14': 1.321615014,
    'P27_101': 0.9708737864,
    'P28_113': 1.324503311,
    'P01_103': 1.19047619,
    'P30_111': 1.209189843,
    'P03_13': 0.4842615012,
    'P02_132': 1.307189542,
    'P23_02': 0.7490636704,
    'P28_14': 0.9900990099,
    'P01_104': 0.826446281,
    'P01_01': 1.234567901,
    'P28_13': 0.3853564547,
    'P03_17': 0.4975124378,
    'P03_13': 0.7434944238,
    'P02_01': 0.9478672986,
    'P18_07': 0.7662835249,
    'P04_21': 0.6872852234,
    'P04_25': 0.6309148265,
    'P03_113': 0.5899705015,
    'P03_24': 0.7633587786,
    'P03_11': 0.6666666667,
    'P23_05': 0.8097165992,
    'P01_09': 1.19760479,
    'P10_04': 0.7604562738,
    'P04_26': 0.422832981,
    'P08_21': 0.701754386,
    'P11_16': 0.9049773756,
    'P13_08': 0.8230452675,
    'P03_23': 0.4938271605,
    'P18_07': 0.6993006993,
    'P02_124': 0.8403361345,
    'P11_104': 0.3710575139,
    'P03_03': 0.625,
    'P18_06': 0.4140786749,
    'P04_33': 0.6349206349,
    'P02_128': 1.081081081,
    'P07_08': 0.826446281,
    'P12_02': 1.069518717,
    'P18_03': 0.8163265306,
    'P03_101': 0.8333333333,
    'P37_101': 0.9174311927,
    'P28_06': 0.9259259259,
    'P25_107': 0.4545454545,
    'P03_112': 0.9433962264,
    'P28_14': 1.369863014,
    'P04_33': 0.7042253521,
    'P28_101': 1.156069364,
    'P26_110': 0.5617977528,
    'P22_117': 0.9302325581,
    'P28_112': 1.526717557,
    'P32_01': 0.5698005698,
    'P37_103': 0.4807692308,
    'P04_02': 0.9661835749,
    'P04_109': 0.5025125628,
    'P04_101': 0.6756756757,
    'P03_11': 0.6666666667,
    'P23_05': 0.8968609865,
    'P30_101': 0.5221932115,
    'P02_130': 1.03626943,
    'P08_16': 0.4424778761,
    'P06_110': 0.4210526316,
    'P06_13': 0.583090379,
    'P04_11': 0.5494505495,
    'P02_122': 1.27388535,
    'P11_101': 0.5319148936,
    'P04_110': 0.7547169811,
    'P08_09': 0.6622516556,
    'P06_01': 0.7299270073,
    'P06_07': 0.487804878,
    'P12_03': 0.7246376812,
    'P05_01': 0.6006006006,
    'P02_07': 1.324503311,
    'P06_107': 0.7874015748,
    'P06_102': 0.625,
    'P06_14': 0.8583690987,
    'P28_109': 1.360544218,
    'P02_109': 1.265822785,
    'P04_114': 0.4608294931,
    'P04_03': 0.9302325581,
    'P11_102': 1.149425287,
    'P03_05': 0.4618937644,
    'P11_105': 1.030927835,
    'P22_07': 0.826446281,
    'P03_123': 0.4535147392,
    'P06_103': 0.9900990099,
    'P30_112': 0.4184100418,
    'P01_05': 1.19760479,
    'P02_03': 1.219512195,
    'P06_09': 0.6269592476,
    'P06_12': 1.183431953,
    'P17_01': 1.098901099,
    'P02_135': 1.104972376,
    'P11_103': 0.7407407407,
    'P04_04': 0.9852216749
}

import torchvision.transforms as transforms
from torchvision.transforms import Compose
import torch.nn.functional as F


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

    Attributes:
        output_dir (str): Directory to save extracted features.
        data_path (str): Path to dataset containing poses, masks, and annotations.
        mesh_path (str): Path to 3D mesh data.
        frames_path (str): Directory for video frames.
        kitchen (str): Identifier for the specific kitchen or dataset instance.
    """
    def __init__(self, output_dir, data_path, mesh_path, frames_path, kitchen):
        """
        Initializes the PHALP class, loads models for depth estimation, and prepares paths for data processing.

        Args:
            output_dir (str): Directory to save extracted features.
            data_path (str): Path to dataset containing poses, masks, and annotations.
            mesh_path (str): Path to 3D mesh data.
            frames_path (str): Directory for video frames.
            kitchen (str): Identifier for the specific kitchen or dataset instance.
        """
        super(PHALP, self).__init__()

        # Add Depth-Anything path to system path
        path_to_depth_anything = './Depth-Anything'
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
        self.path_to_save = output_dir
        self.kitchen = kitchen
        self.mesh_path = mesh_path
        self.output_dir_name = f"saved_feat_3D/{self.kitchen}"
        self.path_to_save = os.path.join(self.path_to_save, self.output_dir_name)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

        self.data_path = data_path
        self.frames_path = frames_path

        # Load pose data
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)

        # Read other necessary data
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, 
                                                                    '', kitchen, True)

        # Load frame mapping data
        with open('./data/dense_frame_mapping_corrected.json') as f:
        with open(os.path.join('data', 'dense_frame_mapping_corrected.json')) as f:
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
        mesh_path = os.path.join(self.mesh_path, 'fused-minpix15-meshed-delaunay-qreg5.ply')
        mesh = trimesh.load(mesh_path, force='mesh')
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

        # Save the extracted 3D features to disk
        with open(os.path.join(self.path_to_save, f'3D_feat_{self.kitchen}.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--mesh_path", required=True, help="Data path")
    parser.add_argument("--frames_path", required=True, help="Frames path")
    parser.add_argument("--kitchen", required=True, help="Frames path")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_dir, args.data_path, args.mesh_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
