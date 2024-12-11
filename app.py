# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Web火灾检测
File Name: app.py
Author: xiang
Create Date: 2024/12/6
Description：Web端，可上传图片和视频文件进行火灾检测
-------------------------------------------------
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import cv2
from pathlib import Path
import numpy as np
from temperature import TemperatureData

# 根据您的实际项目结构和模型加载代码进行调整
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

app = Flask(__name__, 
            template_folder="web/backend/templates", 
            static_folder="web/backend/static")
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DETECTED_FOLDER = BASE_DIR / 'detected'
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
DETECTED_FOLDER.mkdir(exist_ok=True, parents=True)

device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(str(BASE_DIR / 'weights' / 'best.pt'), device=device)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size([640, 640], s=stride)

# 初始化温度数据类 (根据实际需求修改参数)
temp_data = TemperatureData(width=640, height=640, min_temp=100, max_temp=2000)

@app.route('/')
def index():
    # 渲染 templates/index.html
    return render_template('index.html')

@app.route('/api/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    conf_thres = float(request.form.get('conf_thres', 0.25))
    temp_thres = float(request.form.get('temp_thres', 800))

    image_path = UPLOAD_FOLDER / file.filename
    file.save(str(image_path))

    img0 = cv2.imread(str(image_path))
    if img0 is None:
        return jsonify({'error': 'Cannot read image file'}), 400

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if len(img.shape) == 3:
        img = img[None]

    # 推理
    pred = model(img, augment=False)
    pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False, max_det=1000)

    annotator = Annotator(img0, line_width=3, example=str(names))
    temp_list = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                # 计算中心点获取温度
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                temp = temp_data.get_temperature(x_center, y_center)
                temp_list.append(temp)
                # 如果不是烟雾(smoke)类别且温度超过阈值，显示温度
                if names[c] != "smoke" and temp >= temp_thres:
                    temp_label = f'Temperature: {temp:.2f}°C'
                    cv2.putText(img0, temp_label,
                                (x_center, y_center - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

    # 保存检测后的图像
    detected_image_path = DETECTED_FOLDER / f'detected_{file.filename}'
    cv2.imwrite(str(detected_image_path), annotator.result())

    detected_image_url = f'/detected/{detected_image_path.name}'
    return jsonify({'detected_image_url': detected_image_url})

@app.route('/api/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    conf_thres = float(request.form.get('conf_thres', 0.25))
    temp_thres = float(request.form.get('temp_thres', 800))

    video_path = UPLOAD_FOLDER / file.filename
    file.save(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video file'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detected_video_path = DETECTED_FOLDER / f'detected_{file.filename}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(detected_video_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]

        # 推理
        pred = model(img, augment=False)
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False, max_det=1000)

        annotator = Annotator(frame, line_width=3, example=str(names))
        temp_list = []
        if len(pred):
            det = pred[0]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # 计算中心点获取温度
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)
                    temp = temp_data.get_temperature(x_center, y_center)
                    temp_list.append(temp)
                    if names[c] != "smoke" and temp >= temp_thres:
                        temp_label = f'Temperature: {temp:.2f}°C'
                        cv2.putText(frame, temp_label,
                                    (x_center, y_center - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

        out.write(annotator.result())

    cap.release()
    out.release()

    detected_video_url = f'/detected/{detected_video_path.name}'
    return jsonify({'detected_video_url': detected_video_url})

@app.route('/detected/<filename>')
def send_detected_file(filename):
    return send_from_directory(DETECTED_FOLDER, filename)

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    # 可根据需要实现停止检测逻辑
    return jsonify({'message': '停止检测成功'}), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
