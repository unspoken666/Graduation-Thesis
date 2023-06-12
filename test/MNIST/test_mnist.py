import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def selectByLabel(data):
    ret0 = []
    ret1 = []
    for i in range(len(data)):
        if(data.iloc[i,-1]==0):
            ret0.append(data.iloc[i])
        elif(data.iloc[i,-1]==1):
            ret1.append(data.iloc[i])
    ret0 = pd.DataFrame(np.array(ret0))
    ret1 = pd.DataFrame(np.array(ret1))
    return ret0,ret1

# def balancedRandom(data,num):
#     count0 = 0
#     count1 = 0
#     ret = []
#
#     for i in range(len(data)):
#         if(data.iloc[i,-1]==-1 and count0<num/2):
#             count0 += 1
#             ret.append(data.iloc[i])
#         elif(data.iloc[i,-1]==1 and count1<num/2):
#             count1 += 1
#             ret.append(data.iloc[i])
#     ret = np.array(ret)
#     ret = pd.DataFrame(ret)
#     return ret.iloc[:,:-1],ret.iloc[:,-1]




# data = pd.read_csv("resource/data/covid/covid_normal.csv")
# data0 = pd.read_csv("resource/data/wine/winequality-red.csv",sep=";")
# data1 = pd.read_csv("resource/data/wine/winequality-white.csv",sep=";")
# label0 = np.array([-1 for i in range(len(data0))])
# label1 = np.ones(len(data1))
# label0 = pd.DataFrame(label0)
# label1 = pd.DataFrame(label1)
# data0 = pd.concat([data0, label0], axis=1)
# data1 = pd.concat([data1, label1], axis=1)
# data0 = pd.DataFrame(data0)
# data1 = pd.DataFrame(data1)
# data = pd.concat([data0, data1])
# data_x = data.iloc[:, :-1]
# data_y = data.iloc[:, -1]
# data_y = pd.DataFrame(data_y)
# data_y = np.array(data_y)
# data_x_normalized = MinMaxScaler().fit_transform(data_x.values)
# data = np.hstack((data_x_normalized, data_y))
# data = pd.DataFrame(data)
# train, test = train_test_split(data, test_size = 0.20)

train = pd.read_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca_train.csv", header=None)
test = pd.read_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca_test.csv", header=None)
# np.savetxt("resource/data/wine/wine_normal_train.csv", train, delimiter=',', fmt="%.3f")
# np.savetxt("resource/data/wine/wine_normal_test.csv", test, delimiter=',', fmt="%.3f")

#使用所有的数据
test_x = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
train_x = train.iloc[:, :-1]
train_y = train.iloc[:, -1]




# 用训练集训练：
# model1 = SVC(kernel="linear", C=100, gamma='auto')
# model1.fit(train_x, train_y)
# 用测试集预测：
# prediction = model1.predict(test_x)
# print('所有数据准确率：', round(metrics.accuracy_score(prediction, test_y), 6))


#equal random 520条数据
model2 = SVC(kernel="linear", C=100, gamma='auto')

num = 709

accuracy = 0
count = 1
for i in range(count):

    train_x = train.iloc[num*i:num*(i+1), :-1]
    train_y = train.iloc[num*i:num*(i+1), -1]
    # model = SVC(kernel="linear", C=100, gamma='auto')
    # 用训练集训练：
    model2.fit(train_x, train_y)
    # 用测试集预测：
    prediction = model2.predict(test_x)
    accuracy += metrics.accuracy_score(prediction, test_y)
print('equal random平均准确率：', round(accuracy/count, 6))

#balanced random
model3 = SVC(kernel="linear", C=100, gamma='auto')
accuracy = 0
data0, data1 = selectByLabel(train)
num = int(num/2)
for i in range(count):
    # model = SVC(kernel="linear", C=100, gamma='auto')
    # 用训练集训练：
    train_x0 = data0.iloc[num*i:num*(i+1), :-1]
    train_y0 = data0.iloc[num*i:num*(i+1), -1]
    train_x1 = data1.iloc[num*i:num*(i+1), :-1]
    train_y1 = data1.iloc[num*i:num*(i+1), -1]
    train_x = pd.concat([train_x0, train_x1])
    train_y = pd.concat([train_y0, train_y1])
    model3.fit(train_x, train_y)
    # 用测试集预测：
    prediction = model3.predict(test_x)
    accuracy += metrics.accuracy_score(prediction, test_y)
print('balanced random平均准确率：', round(accuracy/count, 6))


#子集筛选520条数据
model4 = SVC(kernel="linear", C=100, gamma='auto')
data = pd.read_csv("D:/python/python-svm-sgd-master/test/mnist/pca40_3_vs_rest/resource/mnist_pca_train_subset.csv", header=None)
train_x = data.iloc[:, :-1]
train_y = data.iloc[:, -1]
# model = SVC(kernel="linear", C=100, gamma='auto')
# 用训练集训练：
model4.fit(train_x, train_y)
# 用测试集预测：
prediction = model4.predict(test_x)
print('子集准确率：', round(metrics.accuracy_score(prediction, test_y), 6))


