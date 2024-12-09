'''
-------------------------------------------------
Project Name: 主程序入口
File Name: main.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''
import torch
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend

def load_model(weights, device='cpu'):
    """
    加载YOLOv5模型
    """
    model = DetectMultiBackend(weights, device=device)  # 加载YOLOv5模型
    return model

def preprocess_image(img, img_size=(640, 640)):
    """
    预处理图像：调整大小，填充，转换为tensor
    """
    img_resized = letterbox(img, new_shape=img_size)  # 调整图像大小并保持长宽比
    img_resized = img_resized[..., ::-1]  # BGR转RGB
    img_resized = np.ascontiguousarray(img_resized)  # 转换为连续数组
    img_tensor = torch.from_numpy(img_resized).float()  # 转换为Tensor
    img_tensor /= 255.0  # 归一化
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度
    return img_tensor

def detect_fire_with_yolov5(model, img, device='cpu'):
    """
    使用YOLOv5模型进行火焰检测
    """
    img_tensor = preprocess_image(img)
    img_tensor = img_tensor.to(device)
    
    # 模型推理
    with torch.no_grad():
        pred = model(img_tensor)

    # 非极大值抑制（NMS）
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    return pred

def extract_temperature_from_thermal(thermal_img, bbox):
    """
    从热成像图像中提取指定区域的温度
    bbox: [x1, y1, x2, y2] 边界框坐标
    """
    x1, y1, x2, y2 = bbox
    region = thermal_img[y1:y2, x1:x2]
    average_temp = np.mean(region)  # 计算该区域的平均温度
    return average_temp

def plot_detected_fire(img, pred, thermal_img, temp_threshold=50, device='cpu'):
    """
    在图像上绘制火焰检测框，并标注温度信息
    """
    for det in pred[0]:  # 处理每个检测结果
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img.shape).round()  # 缩放坐标
            for *xyxy, conf, cls in det:
                if cls == 0:  # 0代表火焰类别
                    # 获取火焰检测框坐标
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"Fire: {conf:.2f}"
                    
                    # 从热成像图中提取该区域的温度
                    avg_temp = extract_temperature_from_thermal(thermal_img, [x1, y1, x2, y2])
                    
                    # 检查温度是否大于阈值
                    if avg_temp > temp_threshold:
                        label += f" (Temp: {avg_temp:.2f}C)"
                        color = (0, 255, 0)  # 检测到火焰且温度高，标记为绿色
                    else:
                        color = (0, 0, 255)  # 检测到火焰但温度低，标记为红色

                    # 绘制框和标签
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# 主程序
def main(video_path, model_weights, thermal_img_path, temp_threshold=50):
    # 加载模型
    model = load_model(model_weights, device='cuda')
    
    # 读取视频和热成像图
    video_capture = cv2.VideoCapture(video_path)
    thermal_img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)  # 读取热成像图像，假设为灰度图
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 进行火焰检测
        pred = detect_fire_with_yolov5(model, frame)
        
        # 使用热成像数据验证火焰
        frame_with_fire = plot_detected_fire(frame, pred, thermal_img, temp_threshold)
        
        # 显示图像
        cv2.imshow('Fire Detection', frame_with_fire)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'test_video.mp4'  # 输入视频路径
    model_weights = 'yolov5s.pt'  # YOLOv5模型权重
    thermal_img_path = 'thermal_image.png'  # 热成像图像路径
    main(video_path, model_weights, thermal_img_path)

