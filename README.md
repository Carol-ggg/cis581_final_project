# StreetGuide: Vision-based Assistant for Visually Impaired Individuals

StreetGuide is a real-time, camera-only assistant that fuses MiDaS monocular depth estimation and YOLO detection to estimate distances, highlight nearby obstacles, and interpret traffic signals for safer navigation without extra range sensors.

# Monocular Depth Estimation Model 
<img align="left" width = 820 src ="https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/af288b93-84ca-445a-be73-d5bd5e1725c5">
                                       
Throughout this repository **MiDaS - MDE models trained by Ranftl et al. were used.** Compared to various MDE models available, MiDaS' approach in training a model by mixing datasets to cover diverse environments provided an effective and robust MDE model for multiple applications.

Read More @ https://github.com/isl-org/MiDaS.

# Actualizing Method and Physical Concepts Used
### Thin Lens Diagram
<p align="center">
  <img width="700" src="https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/dee5a119-14d4-49c6-a831-8455648921c6">
</p>
In order to actualize depth map values into measurement quantities such as the SI units, physical relationships were inferred using the thin lens diagram together with ground
truths from the camera's information and a known object's pixel height. Given this formulaic constraint, the actualizing approach is limited to scenarios where a target object with known physical height measurement is present within a video feed.

### Inferred Equations

<p align="center">
  <img width= '700' src="https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/7bf8bf9a-d150-4455-8fc6-1d42bc46bfe3">
</p>

From the thin lens model, the following equations are drawn. The proportion between the sensor height (measured in millimeters) and the object's pixel height, as observed in the image feed, is utilized to calculate the object's height on the sensor in millimeters. By dividing this proportion by the overall image resolution height (measured in pixels), the object's height in millimeters on the physical camera sensor is essentially inferred.

In summary, this equation allows us to linearize the observable data to estimate the physical size of an object on the camera sensor and correspondingly use it to accommodate the thin lens diagram. These variables that comprise the thin lens diagram are algebraically re-arranged to compute an object's distance to the camera.  

### Limitations
Given the formulaic constraint, the actualizing method is limited to the following- Accuracy of the camera's information, Scenarios where a **Target Object**, an Object with known physical height measurement, is present within the camera's feed, and that the Target Object is always perpendicular to the camera.

### YOLO Object Detection 
<p align="center">
 <img width= '700' src= "https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/59e98a7f-88b6-4612-be2d-c53b34eeecfb">
</p>

To continuously compute the object's actualized distance, **YOLO Object detection** was used. YOLO's bounding box outputs were extensively used to determine the Target Object's pixel height (h) and accompany the actualization of values through the lens optic equation.

Moreover, YOLO allowed the classification of the Target Object and other Surrounding Objects, and from its bounding box centroids, the Target and Surrounding Object's distances based on the MDE-generated depth map are adequately referenced.
<p align="center">
  <img width= '700' src= https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/9c769079-901e-4be7-bd59-4f6cd1807707 >
  <img width= '700' src= https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/6943aef8-922f-4f3f-bfa4-1899d574f7fe >
</p>
### MiDaS - YOLO
Since the target object's actualized distance, position in feed, and corresponding depth map value from the MiDaS model are calculated. These values are then related to the rest of the depth map via inverse proportion to calculate the surrounding objects' actualized distances. 

Upon testing, the relationship of the generated depth map was observed to be inversely proportional. Whereby within the normalized scale of 0 - 1, closer values to 0 represent that an object is far and vice versa and whose relationship is represented by the equation: 

<p align="center">
 <img width= '700' src= https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/c1ab69ba-3f40-49f7-829d-2fdeaab5c959 >
</p>

The Reference Distance is the Target Object's calculated distance using the equation drawn from the thin lens model, Reference MiDaS Value is the Target Object centroid's MiDaS value, and the Surrounding Object MiDaS Value is/are the Surrounding Object/s centroid's MiDaS value.

### Process Flow
**The novelty of this approach is** the ingenuous use of the bounding box output of the YOLO object detection, precisely the bounding box height, the computable detection centroid, and the lens optic formula, for continuous computation of the target object's distance. Since the generated depth map of MiDaS is presented in normalized values, the output of this equation, via lens optic equation, and its location via centroid computation, is then used as a ground reference value for the remainder of the depth map that is adequately related through inverse proportion.

<p align="center">
  <img width="700" src="https://github.com/LanzDeGuzman/Scene-Distance-Estimation-Using-Monocular-Depth-Estimation/assets/97860488/9a00bd93-4143-4dcc-baf2-dabcd755da5c">
</p>


# Using The Source Code

Dependencies:
  1. Python
  2. OpenCV (CUDA build recommended if available)
  3. NumPy
  4. yaml

How to run:
  1. Place model weights and .names files, then set their paths in `config.yml` (`model_path` section). Set camera parameters (`camera_information`) and choose `target_object` with its real height (mm).
  2. Install packages: `pip install -r requirements.txt` (or install Python, OpenCV, numpy, pyyaml, torch, ultralytics as needed).
  3. Launch: `python Main.py --source 0` for webcam, or `python Main.py --source path/to/video.mp4` for a file. Add `--debug` to print PriorityAnalyzer details.
  4. During startup use the ROI trackbars to center the target object; the target’s true distance seeds the scale for surrounding objects.
  5. Outputs are written to `data/processed_output.mp4`, `data/unsafe_detections.txt`, `data/priority_alerts.txt`, and `data/detection_stats.txt`.

Switching YOLO models:
  - Add new entries in `config.yml` under `model_path` with the relevant `.names/.cfg/.weights/.pt/.onnx`.
  - Mirror those entries in `config.py` when loading names/weights so class labels align with the selected model.
