import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

def selectByLabel(data):
    ret0 = []
    ret1 = []
    for i in range(len(data)):
        if(data.iloc[i,-1]==-1):
            ret0.append(data.iloc[i])
        elif(data.iloc[i,-1]==1):
            ret1.append(data.iloc[i])
    ret0 = pd.DataFrame(np.array(ret0))
    ret1 = pd.DataFrame(np.array(ret1))
    return ret0,ret1

def distance(vec1, vec2):
    ret = 0
    for i in range(len(vec1)):
        ret += abs(vec1[i]-vec2[i])
    return ret


# data = pd.read_csv("resource/data/adult/adult.csv")
# train = pd.read_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult_train.csv", header=None)
# test = pd.read_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult_test.csv",  header=None)
# test_x = test.iloc[:, :-1]
# test_y = test.iloc[:, -1]
# train_x = train.iloc[:, :-1]
# train_y = train.iloc[:, -1]
#
# X = np.vstack((train_x, test_x))
# y = np.array(list(train_y)+list(test_y))

data = pd.read_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult.csv", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.decomposition import PCA
# pca = PCA(n_components=14,whiten=True)
pca = PCA(whiten=True, random_state=100)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
print("X_new", X_new)
print("X_new.shape", X_new.shape)
print("y.shape", y.shape)
# ss = StandardScaler()
# std_cps = ss.fit_transform(X_new)
y_tmp = np.array([[i] for i in y])
dataNew = pd.DataFrame(np.concatenate((X_new, y_tmp), axis=1))
# dataNew = pd.DataFrame(X_new)
print(type(dataNew))
dataNew.to_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult_pca.csv", index=False, header=None)



X_Train, X_test = train_test_split(dataNew, test_size=0.3, random_state=100)
X_Train.to_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult_pca_train.csv", index=False, header=None)
X_test.to_csv("D:/python/python-svm-sgd-master/test/adult/resource/adult_pca_test.csv", index=False, header=None)


# train, test = train_test_split(data, test_size = 0.20)
# np.savetxt("resource/data/adult/adult_train.csv", train, delimiter=',', fmt="%.3f")
# np.savetxt("resource/data/adult/adult_test.csv", test, delimiter=',', fmt="%.3f")
#使用所有的数据





#使用所有的数据
train_x = np.array(X_Train)
x = train_x
vector = x[0]
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

