'''
-------------------------------------------------
Project Name: 比对火焰位置与热成像数据
File Name: compare_flame_temp.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''

import numpy as np

def check_if_flame_in_thermal_image(thermal_img, flame_positions, threshold=50):
    flame_detected = False

    for x1, y1, x2, y2 in flame_positions:
        flame_region = thermal_img[y1:y2, x1:x2]  # 提取火焰区域
        avg_temperature = np.mean(flame_region)  # 计算该区域的平均温度

        if avg_temperature > threshold:
            flame_detected = True
            print(f"Flame detected in region ({x1}, {y1}, {x2}, {y2}) with avg temperature: {avg_temperature}")

    return flame_detected
