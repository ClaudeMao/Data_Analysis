from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
dataSet=np.array([[1,2],[2,5],[3,4],[4,5],[5,8],\
                  [10,13],[11,10],[12,11],[13,15],[15,14]])
X=dataSet[:,0].reshape(-1,1)
y=dataSet[:,1]
linear=linear_model.LinearRegression()
linear.fit(X,y)
X_new=np.array([[7]])
plt.figure(facecolor='w')
plt.axis([0,16,0,16])
plt.scatter(X,y,color='black')
mark=['or','ob']
plt.plot(X,linear.predict(X),c='b',linewidth=3)
plt.plot(X_new,linear.predict(X_new),'Dr',markersize=17)
plt.show()
