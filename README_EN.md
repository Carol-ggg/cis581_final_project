# Real-Time Monocular Scene Distance Estimation (Quick Guide)

This project combines MiDaS monocular depth estimation with YOLO detection to estimate distances and emit safety alerts in road scenes. Core loop: read camera/video → pick a ROI → MiDaS depth map → compute true distance for a target object → infer distances for surrounding objects → emit priority alerts and save annotated video/logs.

## Environment & Dependencies
- Python 3.8+
- OpenCV (`opencv-python`; build with CUDA if available), NumPy, PyYAML
- PyTorch + `ultralytics` (for YOLOv11 weights)
- Pretrained weights: MiDaS model, COCO/vehicle/license-plate/crosswalk/traffic-light YOLO/YOLOv5 weights and their `.names`

## Quick Start
1. Install deps: `pip install -r requirements.txt` (or install the libs above manually).
2. Prepare weight and class files, then fill paths in `config.yml`:  
   - `model_path`: paths to COCO/vehicle/licenseplate/crosswalk `.names/.cfg/.weights/.pt/.onnx`; `MDE_model_path` for the MiDaS ONNX file.  
   - `camera_information`: `sensor_height_mm`, `sensor_height_px`, `focal_length`.  
   - `target_object`: `target` (class name in the names file) and `real_object_height` (mm).
3. Run: `python Main.py --source 0` for webcam or `--source path/to/video.mp4` for a file. Add `--debug` to see priority analyzer debug prints.
4. Outputs: processed video at `data/processed_output.mp4`; logs at `data/unsafe_detections.txt`, `data/priority_alerts.txt`, `data/detection_stats.txt`.

## Key Modules
- `modules/MonocularEstimator.py`: runs MiDaS, normalizes depth.
- `modules/ROIConfigurator.py`: trackbars to pick ROI for the target object.
- `modules/TargetObjectLocator.py`: YOLO within ROI, reads pixel height & MiDaS depth, computes true distance via thin-lens equation.
- `modules/LensOpticCalculator.py`: thin-lens distance, ratio-proportion inference, safety thresholding.
- `modules/SceneLocator.py`: detect surrounding objects, infer their distances from the target reference; marks safety.
- `modules/DistanceCompensator.py`: adjusts distances for elevated objects (traffic lights) using pitch/height compensation.
- `modules/DepthSampler.py`: multi-point sampling/median to stabilize depth values.
- `modules/TrafficLightColor.py`: crop traffic-light region and estimate dominant color.
- `modules/PriorityAnalyzer.py`: priority alerts (P1 traffic lights, P2 close vehicles, P3 close pedestrians) with simple cross-frame de-duplication.
- `modules/CrosswalkLocator.py` (optional): wrappers for crosswalk/signal models.

## Runtime Flow
1. `MonocularEstimator` produces depth.  
2. `TargetObjectLocator` finds the target in ROI, yields its true distance and MiDaS reference.  
3. `SceneLocator` infers distances for other detections; `DistanceCompensator` corrects elevated targets.  
4. `PriorityAnalyzer` aggregates alerts; the main loop writes video and logs.

## Tips
- Camera parameters in `config.yml` directly affect accuracy—use real sensor size/focal length.  
- ROI too small may miss the target; too large reduces stability—adjust via trackbars.  
- If performance is slow, lower input resolution or use smaller MiDaS/YOLO models.  
- Changing the target class? Update `target_object` and its real height (mm), and ensure the class name exists in the `.names` file.
