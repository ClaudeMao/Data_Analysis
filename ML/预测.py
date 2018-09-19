# -*- coding: utf-8 -*-
from time import clock
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

clock()
trainset=pd.read_excel('纳税评估.xlsx') 
df=pd.DataFrame(trainset)
a1=df[df.监控等级==3].入库
a2=df[df.监控等级==3].营业收入
b1=df[df.监控等级==2].入库
b2=df[df.监控等级==2].营业收入
c1=df[df.监控等级==1].入库
c2=df[df.监控等级==1].营业收入


font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
plt.grid()
plt.axis([0,1e6,0,1e8])
plt.scatter(a1,a2,c='r',marker='.')
plt.scatter(b1,b2,c='y',marker='.')
plt.scatter(c1,c2,c='g',marker='.')
plt.xlabel(u'入库',fontproperties=font)
plt.ylabel(u'营业收入',fontproperties=font)
plt.show()
print('作图运行时间为:{}s'.format(clock()))


from sklearn.model_selection import cross_val_predict
train_data=df.loc[:,['入库','营业收入']]
train_target=df.监控等级
x_train,x_test,y_train,y_test=train_test_split\
(train_data,train_target,test_size=0.8,random_state=2)#x为data,y为target
clf=svm.SVC(kernel='linear',C=1)                      #kernel='linear'无法拟合
clf.fit(x_train,y_train)             #训练
accuracy=clf.score(x_test,y_test)    #求准确率
print('准确率为：{:.2f}%'.format(accuracy*100))
print('运行时间为:{}s'.format(clock()))
scores=cross_val_score(clf,train_data,train_target,cv=5)
print('交叉验证后准确率为：',scores)
print('平均值和95%的置信区间:{:.5f}(+/-{:.5f})'.format\
      (scores.mean(),scores.std()*2))#平均值和95%的置信区间
predset=pd.read_excel('新增纳税人.xlsx')
cf=pd.DataFrame(predset)
pred_data=cf.loc[:,['入库','营业收入']]
result=clf.predict(pred_data)
print(result)
print('运行时间为:{}s'.format(clock()))
