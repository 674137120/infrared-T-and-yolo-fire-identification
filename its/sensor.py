'''
-------------------------------------------------
Project Name: 温度传感器读取
File Name: sensor.py
Author: XIANG SHI CHAO
Create Date: 2024/12/3
Description：
-------------------------------------------------
'''

import smbus2
import numpy as np

I2C_ADDRESS = 0x33  # MLX90640的I2C地址
bus = smbus2.SMBus(1)  # I2C总线

def read_temperature_data():
    """读取温度数据并返回32x24的矩阵"""
    try:
        data = bus.read_i2c_block_data(I2C_ADDRESS, 0x00, 832)  # 获取传感器数据
        temp_data = np.array(data).reshape((24, 32))  # 假设数据为32x24矩阵
        return temp_data
    except Exception as e:
        print(f"Error reading temperature data: {e}")
        return None
    