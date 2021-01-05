import cv2
import numpy as np
import torch

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)