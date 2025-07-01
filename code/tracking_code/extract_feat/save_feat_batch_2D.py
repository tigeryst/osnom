import sys
import os
sys.path.append(os.path.join("code", "tracking_code"))
import torch.nn as nn
from util import *
import pickle
import torch
from tqdm import tqdm
import argparse
import json
import cv2
from PIL import Image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
rescale_scores = {
    'P03_04': 0.9744356799,
    'P05_08': 0.6808696067,
    'P06_05': 0.5479392007,
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
    'P24_08': 3.95,
    'P14_05': 2.13684,
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


class PHALP(nn.Module):
    """
    PHALP: A class for object tracking and appearance feature extraction using DINOv2.

    Attributes:
        output_dir (str): Path to save extracted features.
        data_path (str): Path to the dataset containing poses and masks.
        frames_path (str): Path to the folder containing video frames.
        kitchen (str): Identifier for the specific kitchen or dataset instance.
    """
    def __init__(self, output_dir, data_path, frames_path, kitchen):
        super(PHALP, self).__init__()

        # Initialize class attributes
        self.RGB_tuples = get_colors()  # Predefined RGB tuples for visualization
        self.path_to_save = output_dir
        self.kitchen = kitchen
        self.output_dir_name = f"saved_feat_2D/{self.kitchen}"
        self.path_to_save = os.path.join(self.path_to_save, self.output_dir_name)

        # Create the output directory if it does not exist
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

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

        # Save the extracted features and metadata to a file
        with open(os.path.join(self.path_to_save, f"2D_feat_{self.kitchen}.pkl"), 'wb') as f:
            pickle.dump(save_dict_2D, f)


def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--frames_path", required=True, help="Frames path")
    parser.add_argument("--kitchen", required=True, help="Frames path")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_dir, args.data_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
