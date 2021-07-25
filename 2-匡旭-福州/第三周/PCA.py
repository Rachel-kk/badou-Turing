#!/user/bin/env python
#encoding=gbk

from matplotlib import pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

# 参数return_X_Y：控制输出数据的结构，若选为True，则将因变量和自变量独立导出
X, y = load_iris(return_X_y=True) # 加载数据，X表示数据集中的属性数据，y表示数据标签
pca = dp.PCA(n_components=2) # 加载PCA算法，设置降维后的主成分数目3
reduced_X = pca.fit_transform(X) # 用X来训练PCA模型，同时返回降维后的数据
# print(reduced_X)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)): # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0 :
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
# 画二维散点图
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()



