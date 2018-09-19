import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from time import clock
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

clock()
finished = pd.read_excel('附件一：已结束项目任务数据.xls')
df = pd.DataFrame(finished)
vip = pd.read_excel('附件二：会员信息数据.xlsx')
cf = pd.DataFrame(vip)
new = pd.read_excel('附件三：新项目任务数据.xls')
ff = pd.DataFrame(new)

# Paint()按照位置与完成情况进行作图


def paint():
    a1 = df[df.任务执行情况 == 0].任务gps经度
    a2 = df[df.任务执行情况 == 0].任务gps纬度   # 此处修改了源文件，源文件gps后有空格
    b1 = df[df.任务执行情况 == 1].任务gps经度
    b2 = df[df.任务执行情况 == 1].任务gps纬度
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.scatter(a1, a2, c='r', marker='.', label='0')  # 红色代表未完成——0
    plt.scatter(b1, b2, c='g', marker='.', label='1')  # 绿色代表完成——1
    plt.xlabel(u'任务gps经度', fontproperties=font)
    plt.ylabel(u'任务gps纬度', fontproperties=font)
    plt.title(u'附件一位置与完成情况', fontproperties=font)
    plt.legend(loc='upper right')
    plt.show()

# Paint2()按照位置与高于低于地区平均任务标价进行作图


def paint2():
    a1 = df[df.任务标价 > 69].任务gps经度
    a2 = df[df.任务标价 > 69].任务gps纬度   # 此处修改了源文件，源文件gps后有空格
    b1 = df[df.任务标价 <= 69].任务gps经度
    b2 = df[df.任务标价 <= 69].任务gps纬度
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.scatter(a1, a2, c='r', marker='.', label='>69')  # 红色代表高于地区平均任务标价
    plt.scatter(b1, b2, c='g', marker='.', label='<=69')  # 绿色代表低于地区平均任务标价
    plt.xlabel(u'任务gps经度', fontproperties=font)
    plt.ylabel(u'任务gps纬度', fontproperties=font)
    plt.title(u'附件一位置与高低于平均值', fontproperties=font)
    plt.legend(loc='upper right')
    plt.show()

# Paint3()按照位置与各任务区间分布进行作图


def paint3():
    a1 = df[df.任务标价 >= 65][df.任务标价 < 70].任务gps经度
    a2 = df[df.任务标价 >= 65][df.任务标价 < 70].任务gps纬度  # 此处修改了源文件，源文件gps后有空格
    b1 = df[df.任务标价 >= 70][df.任务标价 < 75].任务gps经度
    b2 = df[df.任务标价 >= 70][df.任务标价 < 75].任务gps纬度
    c1 = df[df.任务标价 >= 75][df.任务标价 < 80].任务gps经度
    c2 = df[df.任务标价 >= 75][df.任务标价 < 80].任务gps纬度
    d1 = df[df.任务标价 >= 80][df.任务标价 <= 85].任务gps经度
    d2 = df[df.任务标价 >= 80][df.任务标价 <= 85].任务gps纬度
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.scatter(a1, a2, s=50, c='g', marker='.', label='[65,70)')
    plt.scatter(b1, b2, s=100, c='b', marker='.', label='[70,75)')
    plt.scatter(c1, c2, s=200, c='m', marker='.', label='[75,80)')
    plt.scatter(d1, d2, s=400, c='r', marker='.', label='[80,85]')
    plt.xlabel(u'任务gps经度', fontproperties=font)
    plt.ylabel(u'任务gps纬度', fontproperties=font)
    plt.title(u'附件一位置与各任务标价区间分布', fontproperties=font)
    plt.legend(loc='upper right')
    plt.show()

# Paint4()会员分布位置作图


def paint4():
    a1 = cf.会员gps经度
    a2 = cf.会员gps纬度
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.axis([112.5, 114.5, 22.4, 24.0])
    plt.scatter(a1, a2, c='b', marker='.')
    plt.xlabel(u'会员gps经度', fontproperties=font)
    plt.ylabel(u'会员gps纬度', fontproperties=font)
    plt.title(u'附件二会员分布位置', fontproperties=font)
    plt.show()

# 新项目任务分布作图


def paint5():
    a1 = ff.任务gps经度  # 原标签中gps为GPS
    a2 = ff.任务gps纬度
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.grid()
    plt.axis([112.5, 114.5, 22.4, 24.0])
    plt.scatter(a1, a2, c='g', marker='.')
    plt.xlabel(u'任务GPS经度', fontproperties=font)
    plt.ylabel(u'任务GPS纬度', fontproperties=font)
    plt.title(u'附件三新项目任务分布', fontproperties=font)
    plt.show()

'''计算任务点周围平均任务限额、会员密度、任务密度
将其作为三维数据，存入二维数组，作为train_data
将任务定价作为train_target
用K近邻算法进行学习并对新任务定价进行预测'''

# 生成三维数据 类型为np.ndarray


def func(x):
    a = x.任务gps经度
    b = x.任务gps纬度
    s = 458.5596  # 选中任务点加减0.1度的面积
    fset = np.array([0, 0, 0])
    for i in range(len(a)):
        viplist = cf[cf.会员gps经度 >= a[i]-0.01][cf.会员gps经度 <= a[i]+0.01][cf.会员gps纬度 >= b[i]-0.01]\
            [cf.会员gps纬度 <= b[i]+0.01].预订任务限额
        tasklist = x[x.任务gps经度 >= a[i]-0.01][x.任务gps经度 <= a[i]+0.01][x.任务gps纬度 >= b[i]-0.01]\
            [x.任务gps纬度 <= b[i]+0.01].任务gps经度
        tasklim = sum(viplist)/s
        vipdensity = len(viplist)/s
        taskdensity = len(tasklist)/s
        g = []
        g.append(tasklim)
        g.append(vipdensity)
        g.append(taskdensity)
        g = np.array([g])
        fset = np.row_stack((fset, g))  # add a new list to fset
    vset = fset[1:]
    return vset

# 标准差归一化,并存为DataFrame


def norm(x):
    a = func(x)
    data_col_means = a.mean(axis=0)
    data_col_std = a.std(axis=0)
    a_shape = a.shape
    a_rows = a_shape[0]
    a_columns = a_shape[1]
    for i in range(0, a_rows):
        for j in range(0, a_columns):
            a[i][j] = (a[i][j]-data_col_means[j])/data_col_std[j]
    frame = pd.DataFrame(a)
    frame.columns = ['tasklim', 'vipdensity', 'taskdensity']
    return frame

# 线性归一化,并存为DataFrame


def norm2(x):
    a = func(x)
    data_col_max = a.max(axis=0)
    data_col_min = a.min(axis=0)
    a_shape = a.shape
    a_rows = a_shape[0]
    a_columns = a_shape[1]
    for i in range(0, a_rows):
        for j in range(0, a_columns):
            a[i][j] = (a[i][j] - data_col_min[j]) / (data_col_max[j] - data_col_min[j])
    frame = pd.DataFrame(a)
    frame.columns = ['tasklim', 'vipdensity', 'taskdensity']
    return frame

# 非线性归一化,并存为DataFrame


def norm3(x):
    a = func(x)
    data_col_max = a.max(axis=0)
    b = np.log10(a)
    data_col_max_lg = np.log10(data_col_max)
    b_shape = b.shape
    b_rows = b_shape[0]
    b_columns = b_shape[1]
    for i in range(0, b_rows):
        for j in range(0, b_columns):
            b[i][j] = b[i][j] / data_col_max_lg[j]
    frame = pd.DataFrame(b)
    frame.columns = ['tasklim', 'vipdensity', 'taskdensity']
    return frame


def main():
    a = norm(df)
    train_data = a.loc[:, ['tasklim', 'vipdensity', 'taskdensity']]
    train_target = [int(i) for i in df.任务标价]  # type of target should be int or str
    x_train, x_test, y_train, y_test = train_test_split\
        (train_data, train_target, test_size=0.8, random_state=2)  # x为训练数据,y为训练目标
    knn = KNeighborsClassifier(algorithm='auto')  # 预测任务标价
    knn.fit(x_train, y_train)
    accuracy = knn.score(x_test, y_test)
    print('accuracy：{:.2f}%'.format(accuracy * 100))
    scores = cross_val_score(knn, train_data, train_target, cv=10)
    print('交叉验证后平均accuracy和95%的置信区间:{:.5f}(+/-{:.5f})'.format(scores.mean(), scores.std() * 2))
    b = norm(ff)
    predset = b.loc[:, ['tasklim', 'vipdensity', 'taskdensity']]
    result = knn.predict(predset)
    print(result)

main()
print('Time used:{}s'.format(clock()))
