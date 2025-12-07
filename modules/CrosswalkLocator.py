"""Detect traffic-light related classes using the YOLOv11 COCO model."""

import cv2
import numpy as np

from config import crosswalk_class_names, yolo_model, yolo_device
from modules.LensOpticCalculator import RatioProportionCalculator, LimitVal, SafetyLevel

_CROSSWALK_LABELS = {"traffic light", "stop sign"}
_CROSSWALK_CLASS_IDS = {idx for idx, name in enumerate(crosswalk_class_names) if name in _CROSSWALK_LABELS}


def FindCrosswalkObjects(img, target_object_depth_val, monocular_depth_val):
    """
    Detects traffic-light related objects via COCO YOLOv3 and overlays distance estimation based on the
    existing target object reference depth.
    Returns a list of detection dicts including safety state.
    """
    if not bool(target_object_depth_val):
        # Need target object to establish reference distance.
        return []

    reference_distance = target_object_depth_val[0]
    reference_point = target_object_depth_val[1]

    conf_threshold = 0.35
    results = yolo_model(img, imgsz=640, conf=conf_threshold, device=yolo_device, verbose=False)
    boxes = results[0].boxes if results else []

    hT, wT, _ = img.shape
    detections_summary = []

    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < conf_threshold or cls_id not in _CROSSWALK_CLASS_IDS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        xcoord = (x + (x + w)) / 2
        ycoord = (y + (y + h)) / 2

        xcoord = LimitVal(xcoord, wT)
        ycoord = LimitVal(ycoord, hT)

        object_depthmap_val = monocular_depth_val[int(ycoord), int(xcoord)]
        computed_distance = RatioProportionCalculator(object_depthmap_val, reference_distance, reference_point)
        safety = SafetyLevel(computed_distance)

        color = safety[1]
        label = f"{crosswalk_class_names[cls_id].upper()}- {computed_distance/1000:.1f}m"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        cv2.circle(img, (int(xcoord), int(ycoord)), 4, color, -1)

        detections_summary.append({
            "label": crosswalk_class_names[cls_id],
            "distance_m": computed_distance/1000.0,
            "midas_depth": float(object_depthmap_val),
            "bbox": (x, y, w, h),
            "confidence": conf,
            "safety": safety[0]
        })

    return detections_summary
