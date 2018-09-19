from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
dataSet=np.array([[1,2],[2,5],[3,4],[4,5],[5,8],\
                  [10,13],[11,10],[12,11],[13,15],[15,14]])
km=KMeans(n_clusters=2)
km.fit(dataSet)
plt.figure(facecolor='w')
plt.axis([0,16,0,16])
mark=['or','ob']
for i in range(dataSet.shape[0]):
    plt.plot(dataSet[i,0],dataSet[i,1],mark[km.labels_[i]])
plt.show()
