# -*- coding: utf-8 -*-
from time import clock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.font_manager import FontProperties

clock()
trainset = pd.read_excel('纳税评估.xlsx')
df = pd.DataFrame(trainset)

def main():
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
    x=np.arange(0, 1e6)
    y=np.sin(x)
    plt.plot(x,y)
    print('运行时间为:{}s'.format(clock()))
    plt.show()

x=np.arange(0,1e6)
y=np.arcsin(x)
plt.plot(x,y)
print('运行时间为:{}s'.format(clock()))
plt.show()
