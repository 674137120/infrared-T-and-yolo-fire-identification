# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: ui界面
File Name: window.py
Author: xiang
Create Date: 2024/12/6
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""

import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from tqdm import tqdm
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.conf_thres = 0.25 # 初始化照片界面置信度的值
        self.vid_conf_thres = 0.25 # 初始化视频界面置信度的值
        self.temperature_threshold = 800 # 初始化温度置信度的值
        self.setWindowTitle('火灾检测系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/xf.jpg"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cuda:0'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model = self.model_load(weights="weights/best.pt",
                                     device=self.device)  # todo 指明模型加载的位置的设备
        self.initUI()
        self.reset_vid()
    '''
    ***模型初始化***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        # 选择设备
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # 初始化模型
        model = None
        print("开始加载模型...")

        # 使用 tqdm 显示加载进度
        with tqdm(total=3, desc="模型加载进度", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]") as pbar:
            # 加载权重
            pbar.set_postfix(task="加载权重", refresh=True)  # 更新进度条任务
            model = DetectMultiBackend(weights, device=device, dnn=dnn)
            pbar.update(1)  # 更新进度

            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            pbar.set_postfix(task="配置模型", refresh=True)  # 更新任务状态
            pbar.update(1)  # 更新进度

            # 处理 half precision
            half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
            if pt:
                model.model.half() if half else model.model.float()
            pbar.set_postfix(task="完成模型加载", refresh=True)  # 最后一个任务更新
            pbar.update(1)  # 更新进度

        print("模型加载完成!")
        return model

    def save_detection_result(self):
        """
        保存识别结果的方法
        这里可以选择保存图片或者其他识别框信息。
        """
        # 保存识别结果图像
        if self.right_img.pixmap() is not None:
            img = self.right_img.pixmap().toImage()
            save_path, _ = QFileDialog.getSaveFileName(self, "保存识别结果", "", "Images (*.png *.xpm *.jpg)")
            
            if save_path:
                img.save(save_path)
                QMessageBox.information(self, "保存成功", "识别结果已保存！")
            else:
                QMessageBox.warning(self, "保存失败", "未选择保存路径！")
        else:
            QMessageBox.warning(self, "保存失败", "没有检测到图像！")

    '''
    ***界面初始化***
    '''
    def initUI(self):
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        # 图片识别界面，两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)

        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)

        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        save_result_button = QPushButton("保存识别结果")  # 新增保存结果按钮
        up_img_button.clicked.connect(self.upload_img)  # 上传图片连接
        det_img_button.clicked.connect(self.detect_img)
        save_result_button.clicked.connect(self.save_detection_result)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        save_result_button.setFont(font_main)  # 设置字体

        button_style = """
            QPushButton{color:white}
            QPushButton:hover{background-color: rgb(2,110,180); }
            QPushButton{background-color:rgb(48,124,208)}
            QPushButton{border:2px}
            QPushButton{border-radius:5px}
            QPushButton{padding:5px 5px}
            QPushButton{margin:5px 5px}
        """
        up_img_button.setStyleSheet(button_style)
        det_img_button.setStyleSheet(button_style)
        save_result_button.setStyleSheet(button_style)  # 设置样式

        # 图片滑动条和置信度显示
        self.confidence_label = QLabel(f"当前置信度: {0.25:.2f}")
        self.confidence_label.setFont(font_main)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)  # 滑动条范围1%-100%
        self.conf_slider.setValue(25)  # 默认值25%
        self.conf_slider.setTickInterval(10)  # 每10%一个刻度
        self.conf_slider.valueChanged.connect(self.update_confidence)

        # 布局设置
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.confidence_label, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.conf_slider, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(save_result_button)
        img_detection_widget.setLayout(img_detection_layout)

        '''
        视频界面
        '''

        # 视频检测界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)

        # 视频界面中的按钮样式
        self.webcam_detection_btn.setStyleSheet(button_style)
        self.mp4_detection_btn.setStyleSheet(button_style)
        self.vid_stop_btn.setStyleSheet(button_style)

        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)

        # 视频置信度控件
        self.vid_confidence_label = QLabel(f"当前置信度: {0.25:.2f}")
        self.vid_confidence_label.setFont(font_main)
        self.vid_conf_slider = QSlider(Qt.Horizontal)
        self.vid_conf_slider.setRange(1, 100)  # 滑动条范围1%-100%
        self.vid_conf_slider.setValue(25)  # 默认值25%
        self.vid_conf_slider.setTickInterval(10)  # 每10%一个刻度
        self.vid_conf_slider.valueChanged.connect(self.update_vid_confidence)

        # 新增温度阈值滑动条
        self.vid_temperature_label = QLabel(f"当前温度阈值: {800:.2f}°C")
        self.vid_temperature_label.setFont(font_main)
        self.vid_temp_slider = QSlider(Qt.Horizontal)
        self.vid_temp_slider.setRange(500, 1200)  # 温度阈值范围500°C到1200°C
        self.vid_temp_slider.setValue(800)  # 默认温度阈值800°C
        self.vid_temp_slider.setTickInterval(50)  # 每50°C一个刻度
        self.vid_temp_slider.valueChanged.connect(self.update_vid_temperature_threshold)

        # 添加组件到视频检测布局
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_layout.addWidget(self.vid_confidence_label, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(self.vid_conf_slider, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(self.vid_temperature_label, alignment=Qt.AlignCenter)  # 添加温度阈值标签
        vid_detection_layout.addWidget(self.vid_temp_slider,alignment=Qt.AlignCenter) # 添加温度阈值滑动条
        vid_detection_widget.setLayout(vid_detection_layout)

        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))

    def update_vid_temperature_threshold(self, value):
        """
        更新火焰检测的温度阈值并在界面上显示
        """
        self.temperature_threshold = value  # 设置当前温度阈值
        self.vid_temperature_label.setText(f"当前温度阈值: {value}°C")

    def update_confidence(self, value):
        """
        更新图片检测的置信度值并在界面上显示
        """
        confidence = value / 100.0
        self.conf_thres = confidence  # 确保conf_thres只更新图片界面的置信度
        self.confidence_label.setText(f"当前置信度: {confidence:.2f}")

    def update_vid_confidence(self, value):
        """
        更新视频检测的置信度值并在界面上显示
        """
        confidence = value / 100.0
        self.vid_conf_thres = confidence  # 确保vid_conf_thres只更新视频界面的置信度
        self.vid_confidence_label.setText(f"当前置信度: {confidence:.2f}")

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    '''
    ***检测图片***
    '''

    def detect_img(self):
        model = self.model  # 模型
        output_size = self.output_size  # 输出大小
        source = self.img2predict  # 要预测的文件/目录/URL/glob模式，0表示摄像头
        imgsz = [640, 640]  # 推理大小（像素）
        conf_thres = self.conf_thres  # 置信度阈值
        iou_thres = 0.45  # NMS IOU阈值
        max_det = 1000  # 每张图片的最大检测数量
        device = self.device  # cuda设备，例如0或0,1,2,3或cpu
        view_img = False  # 显示结果
        save_txt = False  # 将结果保存到*.txt文件
        save_conf = False  # 在--save-txt标签中保存置信度
        save_crop = False  # 保存裁剪的预测框
        nosave = False  # 不保存图像/视频
        classes = None  # 按类别过滤：--class 0，或者--class 0 2 3
        agnostic_nms = False  # 类别不可知的NMS
        augment = False  # 增强推理
        visualize = False  # 可视化特征
        line_thickness = 3  # 边界框厚度（像素）
        hide_labels = False  # 隐藏标签
        hide_conf = False  # 隐藏置信度
        half = False  # 使用FP16半精度推理
        dnn = False  # 使用OpenCV DNN进行ONNX推理
        # print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)  # 选择设备
            webcam = False  # 是否使用摄像头
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx  # 模型参数
            imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
            save_img = not nosave and not source.endswith('.txt')  # 是否保存推理图像
            # 数据加载器
            if webcam:
                view_img = check_imshow()  # 检查是否显示图像
                cudnn.benchmark = True  # 设置为True以加速恒定图像大小推理
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 流数据集
                bs = len(dataset)  # 批量大小
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 图像数据集
                bs = 1  # 批量大小
            vid_path, vid_writer = [None] * bs, [None] * bs  # 视频路径和写入器
            # 运行推理
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # 预热
            dt, seen = [0.0, 0.0, 0.0], 0  # 计时器和已处理图像数量
            for path, im, im0s, vid_cap, s in dataset:  # 数据集迭代
                t1 = time_sync()  # 同步时间
                im = torch.from_numpy(im).to(device)  # 将numpy数组转换为torch张量并移动到设备
                im = im.half() if half else im.float()  # uint8到fp16/32
                im /= 255  # 0 - 255到0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # 扩展批次维度
                t2 = time_sync()  # 同步时间
                dt[0] += t2 - t1  # 计算时间
                # 推理
                pred = model(im, augment=augment, visualize=visualize)  # 模型预测
                t3 = time_sync()  # 同步时间
                dt[1] += t3 - t2  # 计算时间
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 非最大抑制
                dt[2] += time_sync() - t3  # 计算时间
                # 二阶段分类器（可选）
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # 处理预测结果
                for i, det in enumerate(pred):  # 每张图片
                    seen += 1
                    if webcam:  # 批量大小>=1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # 转换为Path
                    s += '%gx%g ' % im.shape[2:]  # 打印字符串
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益whwh
                    imc = im0.copy() if save_crop else im0  # 用于save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 注释器
                    if len(det):
                        # 将框从img_size重新缩放到im0大小
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        # 打印结果
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # 每类的检测数量
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串
                        # 写入结果
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # 写入文件
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            if save_img or save_crop or view_img:  # 添加边界框到图像
                                c = int(cls)  # 整数类别
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                #                  BGR=True)
                    # 打印时间（仅推理）
                    LOGGER.info(f'{s}完成. ({t3 - t2:.3f}s)')
                    # 流结果
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1毫秒
                    # 保存结果（带有检测的图像）
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))  # 设置QPixmap

    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。

    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        # pass
        model = self.model  # 模型
        output_size = self.output_size  # 输出大小
        # source = self.img2predict  # 视频源文件/目录/URL/glob模式，0表示摄像头
        imgsz = [640, 640]  # 推理大小（像素）
        conf_thres = self.vid_conf_thres  # 置信度阈值
        temp_thres = self.temperature_threshold # 温度阈值
        iou_thres = 0.45  # NMS IOU阈值
        max_det = 1000  # 每张图片的最大检测数量
        # device = self.device  # cuda设备，例如0或0,1,2,3或cpu
        view_img = False  # 显示结果
        save_txt = False  # 将结果保存到*.txt文件
        save_conf = False  # 在--save-txt标签中保存置信度
        save_crop = False  # 保存裁剪的预测框
        nosave = False  # 不保存图像/视频
        classes = None  # 按类别过滤：--class 0，或者--class 0 2 3
        agnostic_nms = False  # 类别不可知的NMS
        augment = False  # 增强推理
        visualize = False  # 可视化特征
        line_thickness = 3  # 边界框厚度（像素）
        hide_labels = False  # 隐藏标签
        hide_conf = False  # 隐藏置信度
        half = False  # 使用FP16半精度推理
        dnn = False  # 使用OpenCV DNN进行ONNX推理
        source = str(self.vid_source)  # 视频源
        webcam = self.webcam  # 是否使用摄像头
        device = select_device(self.device)  # 选择设备
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx  # 模型参数
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
        save_img = not nosave and not source.endswith('.txt')  # 是否保存推理图像

        # 加载模拟热成像数据
        temperature_matrix = np.random.uniform(100, 2000, (640, 640))  # 假设温度矩阵，范围100-2000°C

        # 数据加载器
        if webcam:
            view_img = check_imshow()  # 检查是否显示图像
            cudnn.benchmark = True  # 设置为True以加速恒定图像大小推理
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 流数据集
            bs = len(dataset)  # 批量大小
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)  # 图像数据集
            bs = 1  # 批量大小
        vid_path, vid_writer = [None] * bs, [None] * bs  # 视频路径和写入器
        # 运行推理
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # 预热
        dt, seen = [0.0, 0.0, 0.0], 0  # 计时器和已处理图像数量
        for path, im, im0s, vid_cap, s in dataset:  # 数据集迭代
            t1 = time_sync()  # 同步时间
            im = torch.from_numpy(im).to(device)  # 将numpy数组转换为torch张量并移动到设备
            im = im.half() if half else im.float()  # uint8到fp16/32
            im /= 255  # 0 - 255到0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 扩展批次维度
            # 获取最新的置信度阈值
            conf_thres = self.vid_conf_thres  # 每次循环时动态获取最新的置信度阈值
            temp_thres = self.temperature_threshold # 动态更新温度阈值
            t2 = time_sync()  # 同步时间
            dt[0] += t2 - t1  # 计算时间
            # 推理
            pred = model(im, augment=augment, visualize=visualize)  # 模型预测
            t3 = time_sync()  # 同步时间
            dt[1] += t3 - t2  # 计算时间
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 非最大抑制
            dt[2] += time_sync() - t3  # 计算时间
            # 二阶段分类器（可选）
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # 处理预测结果
            for i, det in enumerate(pred):  # 每张图片
                seen += 1
                if webcam:  # 批量大小>=1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # 转换为Path
                s += '%gx%g ' % im.shape[2:]  # 打印字符串
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益whwh
                imc = im0.copy() if save_crop else im0  # 用于save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 注释器
                if len(det):
                    # 将框从img_size重新缩放到im0大小
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # 打印结果
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # 每类的检测数量
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串
                    # 写入结果
                    for *xyxy, conf, cls in reversed(det):

                        # 获取边界框中心
                        x_center = int((xyxy[0] + xyxy[2]) / 2)
                        y_center = int((xyxy[1] + xyxy[3]) / 2)

                        # 获取该位置的温度值
                        temp = temperature_matrix[y_center, x_center]  # 假设这里是热成像图中对应位置的温度值

                        # 格式化温度值和置信度
                        # if temp>=temp_thres:
                        #     temp_label = f'Temperature: {temp:.2f}°C'
                        if save_txt:  # 写入文件
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:  # 添加边界框到图像
                            c = int(cls)  # 整数类别
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f} ')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # 如果是烟雾类别，则不显示温度标签
                            # 获取处理后的图像
                            im0 = annotator.result()
                            if names[c] != "smoke":  # 如果类别不是烟雾
                                if temp>=temp_thres:
                                    temp_label = f'Temperature: {temp:.2f}°C'
                                    # 添加温度标签
                                    im0 = cv2.putText(im0, temp_label, 
                                                                (int(x_center), int(y_center - 25)), 
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                                                (0, 255, 0), 2)  # Green color, thickness=2

                print(f"置信度: {conf_thres:.2f} 温度: {temp:.2f}°C")
                # 打印时间（仅推理）
                LOGGER.info(f'{s}完成. ({t3 - t2:.3f}s)')
                # 流结果
                # 保存结果（带有检测的图像）
                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # self.vid_img
                # if view_img:
                # cv2.imshow(str(p), im0)
                # self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # cv2.waitKey(1)  # 1毫秒
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:  # 检查是否停止
                self.stopEvent.clear()  # 清除停止事件
                self.webcam_detection_btn.setEnabled(True)  # 启用摄像头检测按钮
                self.mp4_detection_btn.setEnabled(True)  # 启用MP4检测按钮
                self.reset_vid()  # 重置视频
                break  # 跳出循环
        self.reset_vid()  # 重置视频

    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
