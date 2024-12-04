'''
-------------------------------------------------
Project Name: YOLOv5火焰检测
File Name: detector.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''


import torch
from PIL import Image

def load_model(model_path='models/yolov5_model.pt'):
    """加载YOLOv5模型"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def detect_flame(image_path, model):
    """使用YOLOv5模型进行火焰检测"""
    img = Image.open(image_path)
    results = model(img)
    return results.xywh[0]  # 返回边界框（x1, y1, x2, y2）
