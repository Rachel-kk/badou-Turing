import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 绘制3d散点图


from sklearn import decomposition # 主成分
from sklearn import datasets # 数据集

np.random.seed(5) # 用于生成指定随机数 第5堆种子

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
# data对应样本的4个特征，150行4列
X = iris.data
# print(X.shape)

# target对应样本的类别（目标属性），150行1列
y = iris.target
# print(y.shape)

fig = plt.figure(1, figsize=(4, 3))
plt.clf() # clear figure 清楚所有轴，但是窗口打开，这样它可以被重复使用
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla() # 清楚axes,即当前figure中的活动的axes，但其他axes保持不变
pca = decomposition.PCA(n_components=3)
pca.fit(X) # 用数据X训练PCA模型， 函数返回值：调用fit方法的对象本身
X = pca.transform(X) # 将数据X转换成降维后的数据
# print(X.shape)
# print(pca.explained_variance_ratio_) # 返回所保留n个成分各自的方差百分比

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
