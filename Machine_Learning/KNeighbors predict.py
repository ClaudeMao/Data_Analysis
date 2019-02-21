# -*- coding: utf-8 -*-
from time import clock
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.neighbors import KNeighborsClassifier

clock()
trainset = pd.read_excel('纳税评估.xlsx')
df = pd.DataFrame(trainset)
def paint():
    a1 = df[df.监控等级 == 3].入库
    a2 = df[df.监控等级 == 3].营业收入
    b1 = df[df.监控等级 == 2].入库
    b2 = df[df.监控等级 == 2].营业收入
    c1 = df[df.监控等级 == 1].入库
    c2 = df[df.监控等级 == 1].营业收入
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.axis([0, 1e6, 0, 1e8])
    plt.scatter(a1, a2, c='r', marker='.')
    plt.scatter(b1, b2, c='y', marker='.')
    plt.scatter(c1, c2, c='g', marker='.')
    plt.xlabel(u'入库', fontproperties=font)
    plt.ylabel(u'营业收入', fontproperties=font)
    plt.show()
    print('作图运行时间为:{}s'.format(clock()))

def calPrecision(prediction,truth):
    numSamples = len(prediction)
    numCorrect = 0
    for k in range(numSamples):
        if prediction[k] == truth[k]:
            numCorrect += 1
    precision = numCorrect/numSamples
    return precision
def main():
    train_data = df.loc[:, ['入库', '营业收入']][0:-1000]
    train_target = df.监控等级[0:-1000]  # 对前五万多数据进行学习
    knn=KNeighborsClassifier()
    knn.fit(train_data, train_target)  # 生成分类器
    train_data_data = df.loc[:, ['入库', '营业收入']][-1000:]
    train_data_target = list(df.监控等级[-1000:])
    '''
    此处不加list的话calPrecision中的
    if prediction[k] == truth[k]:
    truth数据格式不对，将无法索引也就无法比值
    '''
    forPredict = knn.predict(train_data_data)  #预测剩余数据计算分类器的准确率
    precision = calPrecision(forPredict, train_data_target)
    print('KNN分类器的准确率为：{}'.format(precision * 100))
    data_new = pd.read_excel('新增纳税人.xlsx')
    cf = pd.DataFrame(data_new)
    pred_data = cf.loc[:, ['入库', '营业收入']]
    result = knn.predict(pred_data)
    print(result)
    print('预测运行时间为:{}s'.format(clock()))

train_data = df.loc[:, ['入库', '营业收入']][0:-1000]
print(type(train_data))

