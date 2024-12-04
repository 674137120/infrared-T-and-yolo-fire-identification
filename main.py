'''
-------------------------------------------------
Project Name: 主程序入口
File Name: main.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''
from its.detector import load_model
from its.fire_detection import fire_detection

def main(image_path):
    model = load_model('weights/best.pt')  # 加载训练好的YOLOv5模型
    result = fire_detection(image_path, model)    # 检测火焰
    print(result)

if __name__ == "__main__":
    main('data/images/test_image.jpg')
