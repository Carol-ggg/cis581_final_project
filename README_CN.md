# 实时单目深度距离估计（简要说明）

本项目基于 MiDaS 单目深度估计与 YOLO 检测，实现道路场景的距离估计和安全提示。核心流程：读取视频/相机 → ROI 选定目标物 → MiDaS 深度图 → 目标物真实距离标定 → 推算周围物体距离 → 按优先级输出告警并保存日志/视频。

## 环境与依赖
- Python 3.8+
- OpenCV (`opencv-python`，若有 GPU 可自行编译 CUDA 版)、NumPy、PyYAML
- PyTorch + `ultralytics` (YOLOv11 权重加载)
- 预训练权重：MiDaS 模型、COCO/车辆/车牌/Crosswalk/Traffic-light 等 YOLO/YOLOv5 权重

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`（若无此文件，可按上方列表手动安装）。
2. 准备模型权重与类别文件，并在 `config.yml` 中填好路径：  
   - `model_path`: COCO/vehicle/licenseplate/crosswalk 的 `.names/.cfg/.weights/.pt/.onnx` 路径；`MDE_model_path` 为 MiDaS ONNX。
   - `camera_information`: `sensor_height_mm / sensor_height_px / focal_length`。  
   - `target_object`: `target`（目标类别名，需存在于 names 文件中）与 `real_object_height`（mm）。
3. 运行：`python Main.py --source 0` 使用摄像头，或 `--source path/to/video.mp4` 播放视频。`--debug` 可打印优先级调试信息。
4. 结果：处理后视频保存到 `data/processed_output.mp4`，日志写入 `data/unsafe_detections.txt`、`data/priority_alerts.txt`、`data/detection_stats.txt`。

## 核心模块
- `modules/MonocularEstimator.py`：调用 MiDaS 生成并归一化深度图。
- `modules/ROIConfigurator.py`：用 Trackbar 选择 ROI，便于锁定目标物。
- `modules/TargetObjectLocator.py`：在 ROI 内用 YOLO 找目标物，获取像素高度与 MiDaS 值，用薄透镜公式计算真实距离。
- `modules/LensOpticCalculator.py`：薄透镜距离计算、比例推算、安全阈值判定。
- `modules/SceneLocator.py`：检测周围物体，利用目标物距离与深度值反推其距离，并标记安全/危险。
- `modules/DistanceCompensator.py`：对高位物体（如红绿灯）做俯仰角/高度补偿，估算水平距离。
- `modules/DepthSampler.py`：多点取样/中值等策略，获取更稳的深度值。
- `modules/TrafficLightColor.py`：裁剪红绿灯区域，分析主色用于信号判定。
- `modules/PriorityAnalyzer.py`：基于距离/位置/颜色输出优先级告警（1 级红绿灯，2 级近距离车辆，3 级近距离行人），并做简单的跨帧追踪去重。
- `modules/CrosswalkLocator.py`（可选）：针对人行横道/信号灯模型的检测封装。

## 运行时流程简图
1. `MonocularEstimator` 生成深度图。  
2. `TargetObjectLocator` 在 ROI 找目标物，算出目标实际距离与对应的 MiDaS 深度基准。  
3. `SceneLocator` 用比例法推算其他检测框的距离，`DistanceCompensator` 用于高位目标矫正。  
4. `PriorityAnalyzer` 汇总并输出告警；主循环写入视频与日志。

## 调试与小贴士
- `config.yml` 中的相机参数直接决定物距精度，务必用真实传感器尺寸/焦距。  
- ROI 太小可能漏检目标；太大则基准不稳。可运行时调整 Trackbar。  
- 处理慢时可降低输入分辨率，或换用更小的 MiDaS/YOLO 模型。  
- 如果目标类别换成其他物体，记得更新 `target_object` 与实际高度（mm），确保 names 文件中存在对应类别。
