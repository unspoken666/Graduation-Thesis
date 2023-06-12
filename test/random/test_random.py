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
        if(data.iloc[i,-1]==0):
            ret0.append(data.iloc[i])
        elif(data.iloc[i,-1]==1):
            ret1.append(data.iloc[i])
    ret0 = pd.DataFrame(np.array(ret0))
    ret1 = pd.DataFrame(np.array(ret1))
    return ret0,ret1

# data = pd.read_csv("resource/data/random/random.csv", header=None)
train = pd.read_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca_train.csv", header=None)
test = pd.read_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca_test.csv", header=None)
# train, test = train_test_split(data, test_size = 0.20)
# np.savetxt("resource/data/random/random_train.csv", header=None, train, delimiter=',', fmt="%.3f")
# np.savetxt("resource/data/random/random_test.csv", header=None, test, delimiter=',', fmt="%.3f")
#使用所有的数据
test_x = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
train_x = train.iloc[:, :-1]
train_y = train.iloc[:, -1]

def distance(vec1, vec2):
    ret = 0
    for i in range(len(vec1)):
        ret += abs(vec1[i]-vec2[i])
    return ret

x = np.array(train_x)
vector = np.array(x[0])
distanceList = []
for i in range(len(x)):
    item = np.array(x[i])

    distanceList.append(distance(item, vector))
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




model1 = SVC(kernel="linear", C=100, gamma='auto')
# 用训练集训练：
model1.fit(train_x, train_y)
# 用测试集预测：
prediction = model1.predict(test_x)
print('所有数据准确率：', metrics.accuracy_score(prediction, test_y))
# print('所有数据f1-score：', metrics.f1_score(prediction, test_y))


#equal random 520条数据
model2 = SVC(kernel="linear", C=100, gamma='auto')
num = 529
#num = 1125

accuracy = 0
count = 1
for i in range(count):

    train_x = train.iloc[num*i:num*(i+1), :-1]
    train_y = train.iloc[num*i:num*(i+1), -1]
    # model = SVC(kernel="linear", C=98, gamma='auto')
    # 用训练集训练：
    model2.fit(train_x, train_y)
    # 用测试集预测：
    prediction = model2.predict(test_x)
    accuracy += metrics.accuracy_score(prediction, test_y)
print('equal random平均准确率：', accuracy/count)
# print('equal randomf1-score：', metrics.f1_score(prediction, test_y))


#balanced random
model3 = SVC(kernel="linear", C=100, gamma='auto')
accuracy = 0
train0,train1 = selectByLabel(train)
num = int(num/2)
for i in range(count):
    # model = SVC(kernel="linear", C=100, gamma='auto')
    # 用训练集训练：
    train_x0 = train0.iloc[num*i:num*(i+1), :-1]
    train_y0 = train0.iloc[num*i:num*(i+1), -1]
    train_x1 = train1.iloc[num*i:num*(i+1), :-1]
    train_y1 = train1.iloc[num*i:num*(i+1), -1]
    train_x = pd.concat([train_x0, train_x1])
    train_y = pd.concat([train_y0, train_y1])

    model3.fit(train_x, train_y)
    # 用测试集预测：
    prediction = model3.predict(test_x)
    accuracy += metrics.accuracy_score(prediction, test_y)
print('balanced random平均准确率：', accuracy/count)
# print('balanced randomf1-score：', metrics.f1_score(prediction, test_y))




#子集筛选520条数据
model4 = SVC(kernel="linear", C=100, gamma='auto')
data = pd.read_csv("D:/python/python-svm-sgd-master/test/random/resource/random_pca_train_subset.csv", header=None)
train_x = data.iloc[:, :-1]
train_y = data.iloc[:, -1]
# model = SVC(kernel="linear", C=100, gamma='auto')
# 用训练集训练：
model4.fit(train_x, train_y)
# 用测试集预测：
prediction = model4.predict(test_x)
print('子集准确率：', metrics.accuracy_score(prediction, test_y))
# print('子集f1-score：', metrics.f1_score(prediction, test_y))


