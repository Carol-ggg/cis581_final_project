# Initialization of Libraries and Dependencies

import cv2
import os
import numpy as np
import argparse

from config import target
from modules.SceneLocator import FindObjects
from modules.ROIConfigurator import ROIConfigurator
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject
from modules.CrosswalkLocator import FindCrosswalkObjects

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
parser.add_argument('--source', type=str, default='0', 
                    help='视频源: 0表示摄像头，或者输入视频文件路径')
args = parser.parse_args()

# Initialization of camera 
if args.source == '0':
    cam = cv2.VideoCapture(0)  # 使用摄像头
else:
    cam = cv2.VideoCapture(args.source)  # 使用视频文件
    
# 检查视频源是否成功打开
if not cam.isOpened():
    print(f"错误：无法打开视频源 {args.source}")
    exit()

ROIConfigurator(cam)

while True:
  success, img = cam.read()
  
  # 检查视频是否结束
  if not success:
    print("视频已播放完毕或无法读取帧")
    break
    
  #img = cv2.resize(img, None, fx=0.6, fy=0.6)                                                                            
  
  img_height, img_width, channels = img.shape
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                                              

  monocular_depth_val = MonocularEstimator(img)                                                                                                                                                                    
  target_object_depth_val = FindTargetObject(img,target,monocular_depth_val)                                              
  FindObjects(img,target_object_depth_val,monocular_depth_val)                                                            
  FindCrosswalkObjects(img,target_object_depth_val,monocular_depth_val)                                                   

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                                                                              
  
  cv2.imshow('Depth Map', monocular_depth_val)                                                                            
  cv2.imshow('Monocular Depth Estimation',img)                                                                            
  key = cv2.waitKey(1)
  
  if key == ord("q"):
    break

cam.release()
cv2.destroyAllWindows()
