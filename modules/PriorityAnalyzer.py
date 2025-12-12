# PriorityAnalyzer.py
# Priority-based alert system for assisting visually impaired individuals

import math
from modules.TrafficLightColor import classify_light_color

class PriorityAnalyzer:
    def __init__(self, debug=False):
        # 用于追踪上一帧的状态，避免重复报告
        self.last_traffic_light_state = None
        self.last_traffic_light_distance = None
        self.last_reported_cars = {}  # 改为字典：car_id -> (bbox, distance, frame_count)
        self.last_reported_persons = {}  # 改为字典：person_id -> (bbox, distance, frame_count)
        self.debug = debug
        self.frame_counter = 0
        
        # 配置参数
        self.traffic_light_distance_threshold = 3.0  # 红绿灯距离阈值
        self.vehicle_distance_threshold = 3.0  # 车辆距离阈值
        self.person_distance_threshold = 3.0  # 行人距离阈值
        self.bbox_iou_threshold = 0.3  # IOU阈值，用于判断是否为同一对象
        self.object_timeout_frames = 720  # 对象消失多少帧后从追踪列表移除
        
    def analyze(self, detections_summary, img_width=640, frame_rgb=None):
        """
        分析检测结果并按优先级返回需要报告的信息
        
        Parameters:
        - detections_summary: list of dicts from SceneLocator.FindObjects()
          每个dict包含: label, distance_m, midas_depth (mm), bbox, safety
        - img_width: 图像宽度，用于判断物体位置（左/中/右）
        - frame_rgb: 当前帧的RGB图像，用于检测交通灯颜色
        
        Returns:
        - alerts: list of dicts with priority and message
        """
        alerts = []
        self.frame_counter += 1
        
        # Debug: 打印所有检测到的标签
        if self.debug and detections_summary:
            labels = [d['label'] for d in detections_summary]
            print(f"[DEBUG] Frame {self.frame_counter} - Detected labels: {labels}")
        
        # Priority 1: 交通信号灯检测
        traffic_light_alert = self._check_traffic_lights(detections_summary, frame_rgb)
        if traffic_light_alert:
            alerts.append(traffic_light_alert)
        
        # Priority 2: 车辆距离检测 (< 3m)
        car_alerts = self._check_cars(detections_summary, img_width)
        alerts.extend(car_alerts)
        
        # Priority 3: 行人距离检测 (< 3m)
        person_alerts = self._check_persons(detections_summary, img_width)
        alerts.extend(person_alerts)
        
        # 清理过期对象
        self._cleanup_expired_objects()
        
        return alerts
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个bbox的IOU（交并比）"""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _get_position_text(self, bbox, img_width):
        """根据bbox位置返回方向文本（左/中/右）"""
        x, y, w, h = bbox
        center_x = x + w / 2
        
        # 将图像分为三个区域
        left_boundary = img_width / 3
        right_boundary = 2 * img_width / 3
        
        if center_x < left_boundary:
            return "on the LEFT"
        elif center_x > right_boundary:
            return "on the RIGHT"
        else:
            return "in the CENTER"
    
    def _check_traffic_lights(self, detections, frame_rgb=None):
        """
        检测红绿灯状态变化或接近警告
        只在以下情况报告：
        1. 状态变化（红→绿或绿→红）
        2. 距离 < 3m（接近警告）
        """
        # 扩展支持的交通灯标签列表 - 特别支持 R_Signal 和 G_Signal
        traffic_light_keywords = ['r_signal', 'g_signal', 'traffic light', 'r_light', 'g_light', 
                                 'traffic_light', 'trafficlight', 'stop sign', 'traffic signal']
        
        traffic_lights = []
        for d in detections:
            label_lower = d['label'].lower()
            # 检查是否包含任何关键词
            if any(keyword in label_lower for keyword in traffic_light_keywords):
                # 附带颜色分析（如果有frame）
                color_info = d.get('color_breakdown')
                dominant_color = d.get('color')
                if not color_info and frame_rgb is not None:
                    color_info = classify_light_color(frame_rgb, d['bbox'])
                    dominant_color = color_info.get("dominant") if color_info else None
                d = dict(d)  # copy
                if color_info:
                    d['color'] = dominant_color
                    d['color_breakdown'] = color_info.get('percentages', {})
                traffic_lights.append(d)
                if self.debug:
                    color_dbg = f", color={d.get('color')}" if d.get('color') else ""
                    print(f"[DEBUG] Found traffic light: {d['label']} at {d['distance_m']:.1f}m{color_dbg}")
        
        if not traffic_lights:
            if self.last_traffic_light_state is not None:
                if self.debug:
                    print("[DEBUG] No traffic light detected, resetting state")
            self.last_traffic_light_state = None
            self.last_traffic_light_distance = None
            return None
        
        # 取最近的交通灯
        nearest_light = min(traffic_lights, key=lambda x: x['distance_m'])
        label = nearest_light['label'].lower()
        distance = nearest_light['distance_m']
        
        # 判断红绿灯状态 - 匹配 R_Signal 和 G_Signal
        color_hint = nearest_light.get('color')

        if 'r_signal' in label or 'r_light' in label or 'red' in label or color_hint == 'red':
            current_state = 'red'
            state_text = "RED LIGHT"
        elif 'g_signal' in label or 'g_light' in label or 'green' in label or color_hint == 'green':
            current_state = 'green'
            state_text = "GREEN LIGHT"
        elif 'stop sign' in label:
            current_state = 'stop'
            state_text = "STOP SIGN"
        elif color_hint == 'yellow':
            current_state = 'unknown'
            state_text = "YELLOW LIGHT"
        else:
            current_state = 'unknown'
            state_text = "Traffic Light"
        
        if self.debug:
            print(f"[DEBUG] Traffic light - State: {current_state}, Distance: {distance:.1f}m, Last state: {self.last_traffic_light_state}")
        
        # 决定是否报告
        should_report = False
        report_reason = ""
        
        # 情况1：状态变化
        if current_state != self.last_traffic_light_state:
            should_report = True
            report_reason = "state change"
            if self.debug:
                print(f"[DEBUG] Traffic light state changed: {self.last_traffic_light_state} -> {current_state}")
        
        # 情况2：距离 < 3m（接近警告）
        elif distance < self.traffic_light_distance_threshold:
            # 检查距离是否有显著变化（> 0.5m）
            if self.last_traffic_light_distance is None or abs(distance - self.last_traffic_light_distance) > 0.5:
                should_report = True
                report_reason = "close proximity"
                if self.debug:
                    print(f"[DEBUG] Traffic light close: {distance:.1f}m")
        
        # 更新状态
        self.last_traffic_light_state = current_state
        self.last_traffic_light_distance = distance
        
        if should_report:
            if distance < self.traffic_light_distance_threshold:
                message = f"{state_text} AHEAD! Distance {distance:.1f}m"
            else:
                message = f"{state_text}, Distance {distance:.1f}m"
            
            if self.debug:
                print(f"[DEBUG] Reporting traffic light ({report_reason}): {message}")
            
            return {
                'priority': 1,
                'type': 'traffic_light',
                'state': current_state,
                'message': message,
                'distance_m': distance,
                'midas_depth_mm': nearest_light['midas_depth']
            }
        
        return None
    
    def _check_cars(self, detections, img_width):
        """
        检测距离小于阈值的车辆
        使用IOU追踪，避免重复报告同一车辆
        """
        vehicle_labels = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'vehicle']
        
        # distance_m 已经是米为单位
        close_cars = [d for d in detections 
                     if d['label'].lower() in vehicle_labels 
                     and d['distance_m'] < self.vehicle_distance_threshold]
        
        if self.debug and close_cars:
            print(f"[DEBUG] Found {len(close_cars)} vehicles < {self.vehicle_distance_threshold}m")
        
        alerts = []
        current_cars = {}
        
        for car in close_cars:
            bbox = car['bbox']
            distance = car['distance_m']
            label = car['label']
            
            # 检查是否与已追踪的车辆匹配
            is_new = True
            matched_id = None
            
            for tracked_id, (tracked_bbox, tracked_distance, tracked_frame) in self.last_reported_cars.items():
                iou = self._calculate_iou(bbox, tracked_bbox)
                
                if iou > self.bbox_iou_threshold:
                    # 找到匹配的车辆
                    is_new = False
                    matched_id = tracked_id
                    
                    # 检查距离是否有显著变化（> 0.5m）
                    distance_change = abs(distance - tracked_distance)
                    
                    if distance_change > 0.5:
                        # 距离变化显著，更新报告
                        position = self._get_position_text(bbox, img_width)
                        message = f"WARNING! {label.upper()} {position} at {distance:.1f}m"
                        
                        if self.debug:
                            print(f"[DEBUG] Vehicle distance changed: {tracked_distance:.1f}m -> {distance:.1f}m")
                        
                        alerts.append({
                            'priority': 2,
                            'type': 'vehicle',
                            'message': message,
                            'distance_m': distance,
                            'midas_depth_mm': car['midas_depth'],
                            'label': label,
                            'position': position
                        })
                    
                    break
            
            # 新车辆，报告
            if is_new:
                car_id = f"{label}_{bbox[0]}_{bbox[1]}_{self.frame_counter}"
                position = self._get_position_text(bbox, img_width)
                message = f"WARNING! {label.upper()} {position} at {distance:.1f}m"
                
                if self.debug:
                    print(f"[DEBUG] New vehicle detected: {message}")
                
                alerts.append({
                    'priority': 2,
                    'type': 'vehicle',
                    'message': message,
                    'distance_m': distance,
                    'midas_depth_mm': car['midas_depth'],
                    'label': label,
                    'position': position
                })
                
                current_cars[car_id] = (bbox, distance, self.frame_counter)
            else:
                # 更新已有车辆
                current_cars[matched_id] = (bbox, distance, self.frame_counter)
        
        self.last_reported_cars = current_cars
        return alerts
    
    def _check_persons(self, detections, img_width):
        """
        检测距离小于阈值的行人
        使用IOU追踪，避免重复报告同一行人
        """
        # distance_m 已经是米为单位
        close_persons = [d for d in detections 
                        if d['label'].lower() == 'person' 
                        and d['distance_m'] < self.person_distance_threshold]
        
        if self.debug and close_persons:
            print(f"[DEBUG] Found {len(close_persons)} persons < {self.person_distance_threshold}m")
        
        alerts = []
        current_persons = {}
        
        for person in close_persons:
            bbox = person['bbox']
            distance = person['distance_m']
            
            # 检查是否与已追踪的行人匹配
            is_new = True
            matched_id = None
            
            for tracked_id, (tracked_bbox, tracked_distance, tracked_frame) in self.last_reported_persons.items():
                iou = self._calculate_iou(bbox, tracked_bbox)
                
                if iou > self.bbox_iou_threshold:
                    # 找到匹配的行人
                    is_new = False
                    matched_id = tracked_id
                    
                    # 检查距离是否有显著变化（> 0.5m）
                    distance_change = abs(distance - tracked_distance)
                    
                    if distance_change > 0.5:
                        # 距离变化显著，更新报告
                        position = self._get_position_text(bbox, img_width)
                        message = f"PERSON {position} at {distance:.1f}m"
                        
                        if self.debug:
                            print(f"[DEBUG] Person distance changed: {tracked_distance:.1f}m -> {distance:.1f}m")
                        
                        alerts.append({
                            'priority': 3,
                            'type': 'person',
                            'message': message,
                            'distance_m': distance,
                            'midas_depth_mm': person['midas_depth'],
                            'position': position
                        })
                    
                    break
            
            # 新行人，报告
            if is_new:
                person_id = f"person_{bbox[0]}_{bbox[1]}_{self.frame_counter}"
                position = self._get_position_text(bbox, img_width)
                message = f"PERSON {position} at {distance:.1f}m"
                
                if self.debug:
                    print(f"[DEBUG] New person detected: {message}")
                
                alerts.append({
                    'priority': 3,
                    'type': 'person',
                    'message': message,
                    'distance_m': distance,
                    'midas_depth_mm': person['midas_depth'],
                    'position': position
                })
                
                current_persons[person_id] = (bbox, distance, self.frame_counter)
            else:
                # 更新已有行人
                current_persons[matched_id] = (bbox, distance, self.frame_counter)
        
        self.last_reported_persons = current_persons
        return alerts
    
    def _cleanup_expired_objects(self):
        """清理已经消失超过timeout_frames帧的对象"""
        # 清理车辆
        expired_cars = [
            car_id for car_id, (_, _, frame) in self.last_reported_cars.items()
            if self.frame_counter - frame > self.object_timeout_frames
        ]
        for car_id in expired_cars:
            if self.debug:
                print(f"[DEBUG] Removing expired vehicle: {car_id}")
            del self.last_reported_cars[car_id]
        
        # 清理行人
        expired_persons = [
            person_id for person_id, (_, _, frame) in self.last_reported_persons.items()
            if self.frame_counter - frame > self.object_timeout_frames
        ]
        for person_id in expired_persons:
            if self.debug:
                print(f"[DEBUG] Removing expired person: {person_id}")
            del self.last_reported_persons[person_id]
    
    def reset(self):
        """重置状态追踪（例如在视频流重启时）"""
        self.last_traffic_light_state = None
        self.last_traffic_light_distance = None
        self.last_reported_cars = {}
        self.last_reported_persons = {}
        self.frame_counter = 0
