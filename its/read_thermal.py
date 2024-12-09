'''
-------------------------------------------------
Project Name: 读取热成像仪图像数据的代码(使用FlirOne SDK)
File Name: read_thermal.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''

import numpy as np
import cv2
from flirpy import FlirCamera  # 使用 FlirCamera 而不是 FlirImage，假设 FlirImage 不存在

def read_thermal_image():
    # 初始化 FLIR 摄像头（使用FlirCamera作为示例）
    camera = FlirCamera()
    
    # 获取热成像图像数据
    thermal_image = camera.get_image()
    
    # 通过 camera 获取温度图像（可能是一个2D矩阵）
    thermal_data = np.array(thermal_image, dtype=np.float32)
    
    # 可选择：进行温度数据转换为可视化图像
    # 假设我们想将温度数据转换为温度的可视化图像
    thermal_image_visual = np.uint8(thermal_data * 255 / np.max(thermal_data))  # 归一化为8位图像
    
    # 归一化图像
    normalized_image = cv2.normalize(thermal_image_visual, None, 0, 255, cv2.NORM_MINMAX)
    
    # 显示图像
    cv2.imshow("Normalized Thermal Image", normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    read_thermal_image()



