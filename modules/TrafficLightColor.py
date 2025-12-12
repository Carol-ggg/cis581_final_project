"""Utility to estimate traffic-light color composition inside a detection bbox."""

import cv2
import numpy as np


def _safe_crop(img, bbox):
    """Crop bbox from RGB image with bounds checking."""
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    if x1 >= x2 or y1 >= y2:
        return None
    return img[y1:y2, x1:x2]


def classify_light_color(img_rgb, bbox):
    """
    Estimate color distribution inside a traffic-light bbox.

    Returns dict with:
      - dominant: 'red'|'green'|'yellow'|'white'|'unknown'
      - percentages: {'red': x, 'green': y, 'yellow': z, 'white': w}
    """
    roi = _safe_crop(img_rgb, bbox)
    if roi is None:
        return {"dominant": "unknown", "percentages": {}}

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return {"dominant": "unknown", "percentages": {}}

    # Color masks in HSV
    red_mask1 = cv2.inRange(hsv, (0, 60, 40), (20, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 60, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv, (35, 40, 50), (90, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 40, 40), (100, 255, 255))
    # Widen white detection to include brighter backgrounds
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "white": int(cv2.countNonZero(white_mask)),
    }

    percentages = {k: (v / total_pixels) * 100 for k, v in counts.items()}
    dominant = max(percentages.items(), key=lambda kv: kv[1])[0] if percentages else "unknown"

    # Confidence threshold: if dominant color is weak (<5%), mark as unknown.
    if percentages.get(dominant, 0) < 5.0:
        dominant = "unknown"

    return {"dominant": dominant, "percentages": percentages}
