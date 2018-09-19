from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import matplotlib.pyplot as plt
dataSet=np.array([[1,2],[2,5],[3,4],[4,5],[5,8],\
                  [10,13],[11,10],[12,11],[13,15],[15,14]])
km=KMeans(n_clusters=2)
km.fit(dataSet)
labels=km.labels_  
#km为k均值法对应的模型，其对dataSet进行无监督学习后已对dataSet进行分类
#后进行有监督学习需要对应的标签，所以用label对其进行认为标签以便进行
#有监督学习
#help(km.labels_)可查看加标签用法
#也可使用标签对其进行人工分类
#labels=np.array([0,0,0,0,1,1,1,1,1,1])
#想取array中的值必须要用切片取
knn=KNeighborsClassifier()
knn.fit(dataSet, labels)
data_new=np.array([[6,9]])
#因为之前学习数据为二维数组，此处使用一维虽然在类型上符合但是分类器不服
label_new=knn.predict(data_new)
plt.figure(facecolor='w')#设置背景色
plt.axis([0, 16, 0, 16])
mark=['or', 'ob']
for i in range(dataSet.shape[0]):
    plt.plot(dataSet[i,0],dataSet[i,1],mark[labels[i]])
    #plot函数分别要x和y的坐标集
plt.plot(data_new[0, 0], data_new[0, 1], mark[label_new[0]], markersize=17)
plt.show()
