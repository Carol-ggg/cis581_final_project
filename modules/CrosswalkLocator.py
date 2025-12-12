"""Detect crosswalk/traffic-light classes using YOLOv5 PyTorch (.pt) weights."""

import os
import cv2
import torch
import numpy as np

from config import crosswalk_class_names, crosswalk_yolov5_pt
from modules.LensOpticCalculator import RatioProportionCalculator, LimitVal, SafetyLevel

_crosswalk_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _crosswalk_model
    if _crosswalk_model is not None:
        return _crosswalk_model

    repo_dir = os.path.join(os.path.dirname(__file__), "..", "yolov5")
    repo_dir = os.path.abspath(repo_dir)
    _crosswalk_model = torch.hub.load(repo_dir, "custom", path=crosswalk_yolov5_pt, source="local", force_reload=False)
    _crosswalk_model.to(_device)
    _crosswalk_model.eval()
    return _crosswalk_model


def FindCrosswalkObjects(img, target_object_depth_val, monocular_depth_val):
    return []
