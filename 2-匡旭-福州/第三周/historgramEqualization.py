#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist--直方图均衡化
函数原型：equalizeHist(src, dst=None)
参数：
src: 图像矩阵（单通道图像）
dst：默认即可
'''
# 读取图像
img = cv2.imread("lenna.png")

def plot_hist(img) :
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

def opencv_hist(img) :
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    plt.plot(hist)
# 全局均衡
def equalHist(img) :
    # 将彩色图转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_hist(gray)
    # opencv_hist(gray)
    cv2.imshow("gray", gray)
    # 将灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)
    plot_hist(dst)
    cv2.imshow("dst", dst)

# 自定义均衡程度
def clahehist(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    plot_hist(dst)
    cv2.imshow("clahe", dst)

equalHist(img)
clahehist(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
