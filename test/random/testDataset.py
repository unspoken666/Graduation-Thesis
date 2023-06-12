# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 14:12
# @Author  : unspoken.wang
# @Email   : unspoken.wang@dbappsecurity.com.cn
# @File    : testDataset.py

# 生成用于分类的数据集
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1200, n_features=40, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2)

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
dataNew.to_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca.csv", index=False, header=None)

X_Train, X_test = train_test_split(dataNew, test_size=0.3, random_state=100)
X_Train.to_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca_train.csv", index=False, header=None)
X_test.to_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca_test.csv", index=False, header=None)

# print("X", X)
# rng=np.random.RandomState(2)
# X+=2*rng.uniform(size=X.shape)
# print("X_new", X)
# unique_lables=set(labels)
# colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
# for k,col in zip(unique_lables,colors):
#     x_k=X[labels==k]
#     plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",
#              markersize=14)
# plt.title('data by make_classification()')
# plt.show()
