# modules/DepthSampler.py

import numpy as np
import cv2

def get_stable_depth(monocular_depth_val, bbox, sampling_strategy="grid"):
    """
    Multi-point sampling to get stable depth value
    
    Args:
        monocular_depth_val: MiDaS depth map (numpy array)
        bbox: (x, y, w, h) detection box
        sampling_strategy: Sampling strategy
            - "center": Single point center sampling (original method)
            - "grid": 3x3 grid sampling, remove outliers
            - "edges": Edge sampling (suitable for traffic lights)
            - "adaptive": Adaptive based on object size
    
    Returns:
        depth_value: Stable depth value
    """
    x, y, w, h = bbox
    
    # Ensure coordinates are within image bounds
    img_h, img_w = monocular_depth_val.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    if sampling_strategy == "center":
        # Original method: single point center
        xcoord = x + w // 2
        ycoord = y + h // 2
        return monocular_depth_val[ycoord, xcoord]
    
    elif sampling_strategy == "grid":
        # 3x3 grid sampling
        sample_points = []
        for i in [0.25, 0.5, 0.75]:
            for j in [0.25, 0.5, 0.75]:
                px = int(x + w * i)
                py = int(y + h * j)
                px = max(0, min(px, img_w - 1))
                py = max(0, min(py, img_h - 1))
                sample_points.append((px, py))
        
        depths = [monocular_depth_val[p[1], p[0]] for p in sample_points]
        
        # Remove max and min outliers, take average
        depths_sorted = sorted(depths)
        # Remove 2 values from each end (if enough points)
        if len(depths_sorted) >= 5:
            trimmed = depths_sorted[2:-2]
        else:
            trimmed = depths_sorted[1:-1] if len(depths_sorted) >= 3 else depths_sorted
        
        return np.mean(trimmed)
    
    elif sampling_strategy == "edges":
        # Edge sampling (suitable for objects with clear boundaries like traffic lights)
        sample_points = [
            (x + w // 2, y),           # Top center
            (x + w // 2, y + h),       # Bottom center
            (x, y + h // 2),           # Left center
            (x + w, y + h // 2),       # Right center
            (x + w // 2, y + h // 2),  # Center
        ]
        
        depths = []
        for px, py in sample_points:
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))
            depths.append(monocular_depth_val[py, px])
        
        # Use median (more robust to outliers)
        return np.median(depths)
    
    elif sampling_strategy == "adaptive":
        # Choose strategy based on bbox size
        area = w * h
        if area < 1000:  # Small object, single point sufficient
            return get_stable_depth(monocular_depth_val, bbox, "center")
        elif area < 10000:  # Medium object, edge sampling
            return get_stable_depth(monocular_depth_val, bbox, "edges")
        else:  # Large object, grid sampling
            return get_stable_depth(monocular_depth_val, bbox, "grid")
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


def visualize_sampling(img, bbox, monocular_depth_val, sampling_strategy="grid"):
    """
    Visualize sampling points (for debugging)
    
    Args:
        img: Original image
        bbox: (x, y, w, h)
        monocular_depth_val: depth map
        sampling_strategy: Sampling strategy
    
    Returns:
        vis_img: Image with annotated sampling points
    """
    vis_img = img.copy()
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]
    
    # Draw bbox
    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Annotate sampling points based on strategy
    if sampling_strategy == "center":
        px, py = x + w // 2, y + h // 2
        cv2.circle(vis_img, (px, py), 5, (255, 0, 0), -1)
    
    elif sampling_strategy == "grid":
        for i in [0.25, 0.5, 0.75]:
            for j in [0.25, 0.5, 0.75]:
                px = int(x + w * i)
                py = int(y + h * j)
                cv2.circle(vis_img, (px, py), 3, (255, 0, 0), -1)
    
    elif sampling_strategy == "edges":
        points = [
            (x + w // 2, y),
            (x + w // 2, y + h),
            (x, y + h // 2),
            (x + w, y + h // 2),
            (x + w // 2, y + h // 2),
        ]
        for px, py in points:
            cv2.circle(vis_img, (px, py), 4, (0, 0, 255), -1)
    
    return vis_img