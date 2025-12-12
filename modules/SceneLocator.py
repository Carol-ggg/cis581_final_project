# Contains the function that locates the Surrounding Objects from the original feed.

import cv2
import numpy as np
from config import yolo_model, class_names, yolo_device, target, target_names
from modules.LensOpticCalculator import RatioProportionCalculator, LimitVal, SafetyLevel
from modules.TrafficLightColor import classify_light_color

_TARGET_CLASS_ID = {label: idx for idx, label in enumerate(target_names)}.get(target)


def FindObjects(img, target_object_depth_val, monocular_depth_val):
    detections_summary = []
    if bool(target_object_depth_val):
        img_copy = img.copy()
        reference_distance = target_object_depth_val[0]
        reference_point = target_object_depth_val[1]

        conf_threshold = 0.35
        results = yolo_model(img_copy, imgsz=640, conf=conf_threshold, device=yolo_device, verbose=False)
        boxes = results[0].boxes if results else []

        hT, wT, _ = img_copy.shape
        for box in boxes:
            if float(box.conf) < conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls)
            if _TARGET_CLASS_ID is not None and cls_id == _TARGET_CLASS_ID:
                continue  # avoid duplicating target detections here
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            xcoord = (x + (x + w)) // 2
            ycoord = (y + (y + h)) // 2
            xcoord = LimitVal(xcoord, wT)
            ycoord = LimitVal(ycoord, hT)

            object_depthmap_val = monocular_depth_val[int(ycoord), int(xcoord)]
            cv2.circle(monocular_depth_val, (int(xcoord), int(ycoord)), 3, (255, 0, 255), -1)
            cv2.putText(img, str(round(object_depthmap_val, 5)), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_face = RatioProportionCalculator(object_depthmap_val, reference_distance, reference_point)
            safety = SafetyLevel(output_face)

            color_info = None
            label_text = f'{class_names[cls_id].upper()}- {output_face/1000:.1f}m'
            if class_names[cls_id].lower() == "traffic light":
                color_info = classify_light_color(img_copy, (x, y, w, h))
                if color_info and color_info.get("dominant") not in (None, "unknown"):
                    label_text += f' [{color_info["dominant"].upper()}]'

            cv2.rectangle(img, (x, y), (x + w, y + h), (safety[1]), 4)
            cv2.putText(img, label_text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (safety[1]), 3)
            cv2.circle(img, (int(xcoord), int(ycoord)), 4, (safety[1]), -1)

            det = {
                "label": class_names[cls_id],
                "distance_m": output_face / 1000.0,
                "midas_depth": float(object_depthmap_val),
                "bbox": (x, y, w, h),
                "safety": safety[0]
            }
            if color_info:
                det["color"] = color_info.get("dominant")
                det["color_breakdown"] = color_info.get("percentages")
            detections_summary.append(det)
    else:
        cv2.putText(img, "Place Target Object within the ROI", (300, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return detections_summary
