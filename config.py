# Contains the initialization of necessary variables, models, and etc comming from the config.yml file

import yaml
import cv2
import torch
from ultralytics import YOLO

# Load the YAML config file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

coco_names = config["model_path"]["coco_names"]
vehicle_names = config["model_path"]["vehicle_names"]
licenseplate_names = config["model_path"]["licenseplate_names"]
coco_yolo_cfg = config["model_path"]["coco_yolo_cfg"]
coco_yolo_weights = config["model_path"]["coco_yolo_weights"]
vehicle_yolo_cfg = config["model_path"]["vehicle_yolo_cfg"]
vehicle_yolo_weights = config["model_path"]["vehicle_yolo_weights"]
licenseplate_yolo_cfg = config["model_path"]["licenseplate_yolo_cfg"]
licenseplate_yolo_weights = config["model_path"]["licenseplate_yolo_weights"]
MDE_model_path = config["model_path"]["MDE_model_path"]

# Crosswalk / traffic-light model paths
crosswalk_names = config["model_path"]["crosswalk_names"]
crosswalk_yolov5_pt = config["model_path"]["crosswalk_yolov5_pt"]
crosswalk_yolov5_onnx = config["model_path"]["crosswalk_yolov5_onnx"]

sensor_height_mm = config["camera_information"]["sensor_height_mm"]
sensor_height_px = config["camera_information"]["sensor_height_px"]
focal_length = config["camera_information"]["focal_length"]

real_object_height = config["target_object"]["real_object_height"]
target = config["target_object"]["target"]

crosswalk_class_names = []
with open(crosswalk_names, "rt") as f:
    crosswalk_class_names = f.read().rstrip("\n").split("\n")

class_names = []
with open(coco_names, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

target_names = []
with open(coco_names, "rt") as f:
    target_names = f.read().rstrip("\n").split("\n")

# Initialize a single YOLOv11 model (COCO weights) for all detections.
yolo_device = 0 if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(coco_yolo_weights)

# Initializes the MDE Model to be used throughout the script.
mde_model = cv2.dnn.readNet(MDE_model_path)
mde_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
mde_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Checkpoint of sucessfully initializing the necessary models.
print("Yolo Initialization Successful")
print("Depth Estimation Model Initialization Successful")
