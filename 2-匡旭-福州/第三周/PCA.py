#!/user/bin/env python
#encoding=gbk

from matplotlib import pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

# ����return_X_Y������������ݵĽṹ����ѡΪTrue������������Ա�����������
X, y = load_iris(return_X_y=True) # �������ݣ�X��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
pca = dp.PCA(n_components=2) # ����PCA�㷨�����ý�ά������ɷ���Ŀ3
reduced_X = pca.fit_transform(X) # ��X��ѵ��PCAģ�ͣ�ͬʱ���ؽ�ά�������
# print(reduced_X)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)): # ���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
    if y[i] == 0 :
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
# ����άɢ��ͼ
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()



