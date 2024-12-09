'''
-------------------------------------------------
Project Name: 火焰检测
File Name: yolo_flame_detection.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''
from its.read_thermal import detect_flame, load_model
from its.compare_flame_temp import read_temperature_data
import numpy as np

def is_flame_detected(flame_bboxes, temp_data, temp_threshold=40):
    """判断火焰是否存在：基于YOLOv5的边界框和红外温度数据"""
    for bbox in flame_bboxes:
        x1, y1, x2, y2 = bbox
        region_temp = np.mean(temp_data[int(y1):int(y2), int(x1):int(x2)])
        if region_temp > temp_threshold:
            return True
    return False

def fire_detection(image_path, model):
    """火焰检测主逻辑"""
    temp_data = read_temperature_data()
    if temp_data is None:
        return "无法读取温度数据"
    
    flame_bboxes = detect_flame(image_path, model)
    if is_flame_detected(flame_bboxes, temp_data):
        return "检测到火焰"
    else:
        return "未检测到火焰"

