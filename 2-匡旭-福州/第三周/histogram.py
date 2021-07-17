import cv2
from matplotlib import pyplot as plt
import numpy as np

'''
calcHist----计算图像直方图
函数原型
'''
# 灰度图像直方图
img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)  # 读取一副彩色图片,BGR
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)

'''
灰度图像直方图实现
'''
# 方法一，使用 matplotlib 的子库 pyplot 实现
# 其中绘制直方图主要调用 hist() 函数实现
plt.hist(gray.ravel(), 256)
plt.show()

# 方法二，OpenCV绘制灰度图直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()  # 新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")  # x轴标签
plt.ylabel("# of Pixels")  # y轴标签
plt.plot(hist)
plt.xlim([0, 256])  # 设置x坐标轴范围
plt.show()


'''
彩色图像直方图实现
'''

# chans = cv2.split(img)  # 分离出图片的B，G，R颜色通道
# colors = ("b", "g", "r")
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# # zip()函数：用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元祖
# for (chan, color) in zip(chans, colors) :
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 255])
#     plt.plot(hist, color = color)
#     plt.xlim([0, 256])
# plt.show()