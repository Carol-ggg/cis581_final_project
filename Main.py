# Initialization of Libraries and Dependencies

import cv2
import os
import numpy as np
import argparse
from datetime import datetime

from config import target
from modules.SceneLocator import FindObjects
from modules.ROIConfigurator import ROIConfigurator
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject
from modules.CrosswalkLocator import FindCrosswalkObjects
from modules.PriorityAnalyzer import PriorityAnalyzer

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
parser.add_argument('--source', type=str, default='0', 
                    help='Video source: 0 for webcam, or path to video file')
parser.add_argument('--debug', action='store_true', 
                    help='Enable debug mode for PriorityAnalyzer')
args = parser.parse_args()

# Initialization of camera 
if args.source == '0':
    cam = cv2.VideoCapture(0)  # Use webcam
else:
    cam = cv2.VideoCapture(args.source)  # Use video file
    
# Check if video source opened successfully
if not cam.isOpened():
    print(f"ERROR: Cannot open video source {args.source}")
    exit()

ROIConfigurator(cam)

# Prepare video writer to save processed output
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = "data/processed_output.mp4"
writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Initialize PriorityAnalyzer with debug mode
priority_analyzer = PriorityAnalyzer(debug=args.debug)

# Log files
unsafe_log_file = open("data/unsafe_detections.txt", "w")
unsafe_log_file.write("Frame,Origin,Label,Distance(m),Safety\n")

priority_log_file = open("data/priority_alerts.txt", "w")
priority_log_file.write("Frame,Priority,Type,Message,Distance(m),MiDaS_Depth(mm)\n")

# Statistics log
stats_log_file = open("data/detection_stats.txt", "w")
stats_log_file.write(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
stats_log_file.write(f"Video source: {args.source}\n")
stats_log_file.write(f"Debug mode: {args.debug}\n")
stats_log_file.write("-" * 80 + "\n\n")

frame_idx = 0
total_alerts = {'priority_1': 0, 'priority_2': 0, 'priority_3': 0}

print("\n" + "=" * 80)
print("Starting video processing...")
print("=" * 80 + "\n")

while True:
    success, img = cam.read()
    
    # Check if video ended
    if not success:
        print("Video playback completed or cannot read frame")
        break
        
    img_height, img_width, channels = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                                              

    monocular_depth_val = MonocularEstimator(img)                                                                                                                                                                    
    target_object_depth_val = FindTargetObject(img, target, monocular_depth_val)                                              
    surrounding = FindObjects(img, target_object_depth_val, monocular_depth_val)                                                            
    # crosswalks = FindCrosswalkObjects(img, target_object_depth_val, monocular_depth_val)                                                   

    # Merge all detection results for priority analysis
    all_detections = []
    if surrounding:
        all_detections.extend(surrounding)
    # if crosswalks:
    #     all_detections.extend(crosswalks)
    
    # Priority analysis (pass image width for position detection)
    alerts = priority_analyzer.analyze(all_detections, img_width=img_width, frame_rgb=img)
    
    # Print and log priority alerts
    if alerts:
        print(f"\n--- Frame {frame_idx} ---")
        for alert in alerts:
            # Console output
            print(f"  [Priority {alert['priority']}] {alert['message']}")
            
            # Write to log file
            priority_log_file.write(
                f"{frame_idx},{alert['priority']},{alert['type']},"
                f"\"{alert['message']}\",{alert['distance_m']:.1f},"
                f"{alert.get('midas_depth_mm', 'N/A')}\n"
            )
            
            # Update statistics
            total_alerts[f"priority_{alert['priority']}"] += 1

    # Record unsafe detections
    unsafe = []
    if surrounding:
        unsafe.extend([("surrounding", d) for d in surrounding if d.get("safety") != "Safe"])
    # if crosswalks:
    #     unsafe.extend([("crosswalk", d) for d in crosswalks if d.get("safety") != "Safe"])
    
    for origin, det in unsafe:
        unsafe_log_file.write(
            f"{frame_idx},{origin},{det['label']},{det['distance_m']:.1f},"
            f"{det.get('safety', 'Unknown')}\n"
        )
    
    if unsafe:
        unsafe_text = "; ".join([f"{o}:{d['label']}={d['distance_m']:.1f}m" for o, d in unsafe])
        print(f"  [UNSAFE] {unsafe_text}")

    frame_idx += 1

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    writer.write(img)
    
    cv2.imshow('Depth Map', monocular_depth_val)                                                                            
    cv2.imshow('Monocular Depth Estimation', img)                                                                            
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        print("\nUser interrupted processing...")
        break

# Write statistics
stats_log_file.write(f"\nProcessing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
stats_log_file.write(f"Total frames processed: {frame_idx}\n")
stats_log_file.write(f"\nPriority Alerts Summary:\n")
stats_log_file.write(f"  Priority 1 (Traffic Lights): {total_alerts['priority_1']}\n")
stats_log_file.write(f"  Priority 2 (Vehicles < 3m): {total_alerts['priority_2']}\n")
stats_log_file.write(f"  Priority 3 (Persons < 3m): {total_alerts['priority_3']}\n")
stats_log_file.write(f"  Total Alerts: {sum(total_alerts.values())}\n")

# Close all files and resources
writer.release()
unsafe_log_file.close()
priority_log_file.close()
stats_log_file.close()
cam.release()
cv2.destroyAllWindows()

# Final statistics printout
print("\n" + "=" * 80)
print("Processing Complete! Statistics:")
print("=" * 80)
print(f"Total frames processed: {frame_idx}")
print(f"\nPriority Alert Statistics:")
print(f"  Priority 1 (Traffic Lights): {total_alerts['priority_1']}")
print(f"  Priority 2 (Vehicles < 3m): {total_alerts['priority_2']}")
print(f"  Priority 3 (Persons < 3m): {total_alerts['priority_3']}")
print(f"  Total Alerts: {sum(total_alerts.values())}")
print(f"\nOutput files:")
print(f"  - Processed video: {out_path}")
print(f"  - Unsafe detections log: data/unsafe_detections.txt")
print(f"  - Priority alerts log: data/priority_alerts.txt")
print(f"  - Statistics log: data/detection_stats.txt")
print("=" * 80 + "\n")
