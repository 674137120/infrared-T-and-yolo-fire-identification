# temperature.py
import numpy as np

class TemperatureData:
    def __init__(self, width=640, height=640, min_temp=100, max_temp=2000):
        """
        初始化温度数据。

        参数:
            width (int): 图像的宽度（像素）。
            height (int): 图像的高度（像素）。
            min_temp (float): 温度数据的最小值（°C）。
            max_temp (float): 温度数据的最大值（°C）。
        """
        self.width = width
        self.height = height
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temperature_matrix = np.random.uniform(min_temp, max_temp, (height, width))

    def get_temperature(self, x, y):
        """
        获取指定坐标的温度值。

        参数:
            x (int): x 坐标（列）。
            y (int): y 坐标（行）。

        返回:
            float: 指定坐标的温度值，如果坐标超出范围，返回 0。
        """
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.temperature_matrix[y, x]
        else:
            return 0.0  # 默认温度值
