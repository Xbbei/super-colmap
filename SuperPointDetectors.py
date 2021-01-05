from .superpoint import SuperPoint
import cv2
import numpy as np
import torch
from .utils import *
import json
import argparse
from tqdm import tqdm
import os

class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperPoint detector config: ")
        print(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        print("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # print("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            # "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict

def get_super_points_from_scenes(image_path, result_dir):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    spd = SuperPointDetector()
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        ret_dict = spd(cv2.imread(image_name))
        with open(os.path.join(result_dir, name + ".json"), 'w') as f:
            json.dump(ret_dict, f)

def get_super_points_from_scenes_return(image_path):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    spd = SuperPointDetector()
    sps = {}
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        ret_dict = spd(cv2.imread(image_name))
        sps[name] = ret_dict
    return sps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points detector')
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=False, default="../superpoints", help="real result_file = args.image_path + args.result_dir")
    args = parser.parse_args()
    result_dir = os.path.join(args.image_path, args.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    get_super_points_from_scenes(args.image_path, result_dir)