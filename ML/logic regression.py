# -*- coding: utf-8 -*-
from time import clock
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib as plt

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

def main():
    train_data = df.loc[:,['入库']]
    train_target = df.监控等级
    model = LinearRegression()
    model.fit(train_data,train_target)
    predset = pd.read_excel('新增纳税人.xlsx')
    cf = pd.DataFrame(predset)
    new_data = cf.loc[:,['入库']]
    predicted = model.predict(new_data)
    print(predicted)
    print('预测运行时间为:{}s'.format(clock()))

main()
