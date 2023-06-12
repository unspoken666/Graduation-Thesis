# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 18:49
# @Author  : unspoken.wang
# @Email   : unspoken.wang@dbappsecurity.com.cn
# @File    : mnist_LDA_3.py

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
# from sklearn.datasets.samples_generator import make_classification
# X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,
#                            n_clusters_per_class=1,class_sep =0.5, random_state =10)
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
from sklearn.model_selection import train_test_split

from load import load_mnist_datasets, to_categorical


def labelToNum(label):
  ret = []
  for item in label:
      for i in range(len(item)):
          if item[i] == 1:
              ret.append([i])

  return np.array(ret)

# def selectLabel(data, label0, label1):
#     ret = []
#     for item in data:
#         if item[-1] == label0:
#             item[-1] = 0
#             ret.append(item)
#         if item[-1] == label1:
#             item[-1] = 1
#             ret.append(item)
#     return np.array(ret)

def selectLabel(data, label):
    ret = []
    for item in data:
        if item[-1] == label:
            item[-1] = 0
            ret.append(item)
        else:
            item[-1] = 1
            ret.append(item)
    return np.array(ret)

print("reading dataset...")
X, val_set, test_set = load_mnist_datasets('D:/python/python-svm-sgd-master/test/mnist/data/mnist.pkl.gz')
train_y, val_y, test_y = to_categorical(X[1]), to_categorical(val_set[1]), to_categorical(test_set[1])
X, val_set, test_set = X[0], val_set[0], test_set[0]
# print(train_set[0])
y = labelToNum(train_y)
data = np.hstack((X, y))
data = selectLabel(data, 3)
X = data[:, :-1]
y = np.array(list(data[:, -1]))
print("=============X================", X)
print("=============y================", y)

from sklearn.decomposition import PCA
pca = PCA(n_components=40, whiten=True)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
print("X_new", X_new)
print("X_new.shape", X_new.shape)
print("y.shape", y.shape)
y_tmp = np.array([[i] for i in y])
dataNew = pd.DataFrame(np.concatenate((X_new, y_tmp), axis=1))
# dataNew = pd.DataFrame(X_new)
print(type(dataNew))
dataNew.to_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca.csv", index=False, header=None)

X_Train, X_test = train_test_split(dataNew, test_size=0.3, random_state=100)
X_Train.to_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca_train.csv", index=False, header=None)
X_test.to_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca_test.csv", index=False, header=None)


# plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
# plt.show()

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=1)
# lda.fit(X, y)
# X_new = lda.transform(X)
# print("X_new", X_new)
# print("X_new.shape", X_new.shape)
# plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
# plt.show()


def distance(vec1, vec2):
    ret = 0
    for i in range(len(vec1)):
        ret += abs(vec1[i]-vec2[i])
    return ret


#使用所有的数据
x = np.array(X_Train)[:, :-1]
vector = np.array(x[0])
distanceList = []
for i in range(0, len(x)):
    item = np.array(x[i])

    distanceList.append(distance(item,vector))
# distanceList.sort()
maxIndex = distanceList.index(max(distanceList))
maxValue = max(distanceList)
minIndex = distanceList.index(min(distanceList))
minValue = min(distanceList)
# distanceList.sort()
# index = distanceList.index(distanceList.all())
sorted_id_min = sorted(range(len(distanceList)), key=lambda k: distanceList[k])
sorted_id_max = sorted(range(len(distanceList)), key=lambda k: distanceList[k], reverse=True)
distanceList = np.array(distanceList)
print()

