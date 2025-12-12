# Contains the function that locates the Target Object within the ROI.

import cv2
import numpy as np
from config import yolo_model, target_names, yolo_device
from modules.LensOpticCalculator import LensOpticCalculator, LimitVal

def FindTargetObject(img,target,mde_Model):
    img_copy = img.copy()                                                                                                      # Creates an image copy to have clear feed free from drawn variables.
    img_height, img_width, channels = img_copy.shape
 
    top_bottom_crop = cv2.getTrackbarPos("Top-Bottom Crop", "ROI Size")                                                        # Retrieves the current ROI slider top-bottom crop values.
    left_right_crop = cv2.getTrackbarPos("Left-Right Crop", "ROI Size")                                                        # Retrieves the current ROI slider left-right crop values.

    # Calculates the coordinates of the ROI based on the current trackbar positions.
    x1 = int(img_width / 2 - left_right_crop / 2)
    x2 = int(img_width / 2 + left_right_crop / 2)
    y1 = int(img_height / 2 - top_bottom_crop / 2)
    y2 = int(img_height / 2 + top_bottom_crop / 2)

    # Declares the ROI and draws a reference box on the non-cropped video feed.
    img_roi = img_copy[y1:y2, x1:x2]
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)

    # Converts YOLO's names file indexing to numerical values for convenient referencing.
    yolo_labels_indexing = {label:index for index,label in enumerate(target_names)}

    # Declares offset variables based on the ROI, used in referencing detected targets object within the ROI from the original feed.
    xoff = x1
    yoff = y1
    
    conf_threshold = 0.7                                                                                                       # Detection confidence treshold - 70% Treshold.
    nms_treshold = 0.4                                                                                                         # Detection boxing sensitivity - the Lower the value the more agressive and less boxes.    
    
    results = yolo_model(img_roi, imgsz=640, conf=conf_threshold, device=yolo_device, verbose=False)
    boxes = results[0].boxes if results else []

    best_box = None
    target_index = yolo_labels_indexing.get(target)

    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < conf_threshold or cls_id != target_index:
            continue
        if best_box is None or conf > float(best_box.conf):
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        xcoord = (x+(x+w))/2
        ycoord = (y+(y+h))/2

        xoffset_coord = xcoord+xoff
        yoffset_coord = ycoord+yoff 

        # Inputs the detected centroid to the LimitVal Function to prevent indexing error.
        xoffset_coord = LimitVal(xoffset_coord, img_width)     
        yoffset_coord = LimitVal(yoffset_coord, img_height)

        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR)
        
        # Draws a bounding box and a circle on the target object and its centroid.
        cv2.rectangle(img_roi,(x,y),(x+w,y+h),(255,255,255),3)
        cv2.circle(img,(int(xoffset_coord),int(yoffset_coord)), 3, (0,0,0),2)

        # Also draw on the main image (not just ROI) so it shows up in the main window.
        cv2.rectangle(img,(int(x+xoff),int(y+yoff)),(int(x+xoff+w),int(y+yoff+h)),(255,255,255),3)
        cv2.putText(img, target_names[target_index].upper(), (int(x+xoff), int(y+yoff-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        target_midas_val = mde_Model[int(yoffset_coord),int(xoffset_coord)]                                                    # Retrieves the depth value of the Target Object from the generated MDE feed.
        target_computed_depthmap_val = LensOpticCalculator(h)                                                                  # Computes the Target Object's actualized distance using the Lens Optic Equation, and detected Target Object's pixel height.

        # Overlay distance on ROI and main image
        dist_text = f"{target_computed_depthmap_val/1000:.1f} m"
        cv2.putText(img_roi, dist_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(img, dist_text, (int(x+xoff), int(y+yoff+h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow('ROI',img_roi)
        return (target_computed_depthmap_val,target_midas_val)                                                                 # Returns the Target Object's computed actualized distance and midas value.
