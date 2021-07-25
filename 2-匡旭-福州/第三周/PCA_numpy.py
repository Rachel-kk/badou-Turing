import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import decomposition

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis = 0)
        self.covariance = np.dot(X.T, X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[: self.n_components]]
        # 对x进行降维
        return np.dot(X, self.components_)

iris = datasets.load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

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
# print('=====newX=====\n', newX)

# pca_sklearn = decomposition.PCA(n_components=2)
# newX_sklearn = pca_sklearn.fit_transform(X)
# print('====newX_sklearn=====\n', newX_sklearn)
