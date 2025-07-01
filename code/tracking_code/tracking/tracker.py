import torch.nn as nn
import numpy as np
import torch
import pickle
import json
import os
import cv2
from tqdm import tqdm
from PIL import Image
import wandb
from external.deep_sort_ import nn_matching
from external.deep_sort_.detection import Detection
from external.deep_sort_.tracker import Tracker
from util import get_colors, read_data_1, get_object_bbs_seg, visualize_mask

with open(os.path.join("data", "items_dict.json"), "r") as f:
    items_dict = json.load(f)

class PHALP(nn.Module):

    def __init__(self):
        super(PHALP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device:', self.device)
        self.to(self.device)
        self.eval()

    def setup_deepsort(self):
        print("Setting up DeepSort...")
        metric = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.hungarian_th, self.cfg.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.max_age_track, n_init=self.cfg.n_init,
                               phalp_tracker=self, dims=[4096, 4096, 99])

    def track(self, config=None, debug=False):
        wb = wandb.init(config=config)
        wb.name = f"output_{wandb.config.distance_type}_a{wandb.config.alpha}_h{wandb.config.hungarian_th}_pl{wandb.config.past_lookback}_agg{wandb.config.aggregation}_{wandb.config.model}_{wandb.config.kitchen}_beta0{wandb.config.beta_0}_beta1{wandb.config.beta_1}"
        self.cfg = wb.config
        self.save_res = self.cfg.save_res
        self.path_to_save = os.path.join(self.cfg.output_dir, self.cfg.dir_name, "tune_output")
        os.makedirs(self.path_to_save, exist_ok=True)
        self.RGB_tuples = get_colors()
        self.kitchen = self.cfg.kitchen
        self.base_path = self.cfg.base_path
        participant = self.kitchen.split('_')[0]

        self.data_path = f"{self.base_path}/data/aggregated/{self.cfg.kitchen}/"
        self.frames_path = f"{self.base_path}/data/images/{self.cfg.kitchen}/"
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)
        self.masks, _, self.camera_poses, self.frames, _ = read_data_1(self.data_path, '', self.cfg.kitchen, True)

        with open(f"{self.base_path}/saved_feat_3D/{self.cfg.kitchen}/3D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
            self.all_loca = pickle.load(file)
        with open(f"{self.base_path}/saved_feat_2D/{self.cfg.kitchen}/2D_feat_{self.cfg.kitchen}.pkl", 'rb') as file:
            self.all_feat = pickle.load(file)

        self.bbs_dict = get_object_bbs_seg(self.masks['video_annotations'])
        visual_store_ = ['tracked_ids', 'tracked_bbox', 'tracked_gt', 'tid', 'bbox', 'tracked_time', 'features', 'loca', 'radius', 'size', 'img_path', 'img_name', 'conf']
        final_visuals_dic = {}
        tracked_frames = []

        self.setup_deepsort()

        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            _, bbs, objs = self.bbs_dict.get(f"{self.cfg.kitchen}_{frame_name}", (None, [], []))
            if t_ == len(self.frames) - 1 and self.save_res:
                with open(os.path.join(self.path_to_save, 'results.pkl'), 'wb') as f:
                    pickle.dump(final_visuals_dic, f)

            detections = []
            removed_indices = []

            if bbs:
                try:
                    appe_features = self.all_feat[frame_name]
                    features_3d = self.all_loca[frame_name]
                except KeyError:
                    print('Frame features not found!')
                    with open(os.path.join(self.path_to_save, 'results.pkl'), 'wb') as f:
                        pickle.dump(final_visuals_dic, f)
                    continue

                if self.cfg.kitchen in items_dict:
                    duplicates = items_dict[self.cfg.kitchen]
                    gt_copy = objs.copy()
                    for item in duplicates:
                        if item in gt_copy:
                            index = gt_copy.index(item)
                            objs.remove(item)
                            removed_indices.append(index)
                    bbs = [b for i, b in enumerate(bbs) if i not in removed_indices]

                feat_3D = np.delete(features_3d[0], removed_indices, axis=0)
                radius = np.delete(features_3d[1], removed_indices, axis=0)
                appe = np.delete(appe_features, removed_indices, axis=0)

                for i in range(len(objs)):
                    detection_data = {
                        "bbox": np.array([bbs[i][0], bbs[i][1], (bbs[i][2] - bbs[i][0]), (bbs[i][3] - bbs[i][1])]),
                        "conf": 1.0,
                        "appe": appe[i],
                        "loca": feat_3D[i],
                        "radius": radius[i],
                        "size": [480, 854],
                        "img_path": frame_name[0] + "/" + frame_name[1],
                        "img_name": frame_name[1],
                        "ground_truth": objs[i],
                        "time": t_,
                    }
                    detections.append(Detection(detection_data))

            self.tracker.predict()
            _, statistics = self.tracker.update(detections, t_, frame_name)

            final_visuals_dic.setdefault(frame_name, {'time': t_, 'frame': frame_name})
            for key_ in visual_store_:
                final_visuals_dic[frame_name].setdefault(key_, [])
            for track in self.tracker.tracks:
                if frame_name not in tracked_frames:
                    tracked_frames.append(frame_name)
                track_id = track.track_id
                track_data_hist = track.track_data['history'][-1]
                if track.time_since_update == 0:
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    final_visuals_dic[frame_name]['tracked_gt'].append(track_data_hist['ground_truth'])
                    final_visuals_dic[frame_name]['loca'].append(track_data_hist['loca'])
                    final_visuals_dic[frame_name]['radius'].append(track_data_hist['radius'])

            if self.cfg.visualize and final_visuals_dic[frame_name]['tracked_ids']:
                cv_image = cv2.imread(os.path.join(self.frames_path, f"{frame_name}.jpg"))
                cv_image = cv2.resize(cv_image, (854, 480))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                for bbox, tr_id in zip(final_visuals_dic[frame_name]['tracked_bbox'], final_visuals_dic[frame_name]['tracked_ids']):
                    cv_image = visualize_mask(cv_image, None, bbox, color=np.array(self.RGB_tuples[tr_id]), text=f"track id: {tr_id}")
                img = Image.fromarray(cv_image)
                img.save(os.path.join(self.path_to_save, f"{frame_name}.jpg"))

            else:
                print('No detections')