from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import numpy as nps
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.sans-serif']=['SimHei']
def showCluster(dataSet,k,centers,clusters):
    plt.figure(facecolor='white')
    num,nattr=dataSet.shape
    mark=['or','ob','og','om','oy']#预置聚类颜色，个数溢出会报错
    for i in range(num):
        plt.plot(dataSet[i,0],dataSet[i,1],mark[int(clusters[i])])
    mark=['Dr','Db','Dg','Dm','Dy']
    for i in range(k):
        plt.plot(centers[i,0],centers[i,1],mark[i],markersize=17)
    plt.title('在iris数据集上的K均值聚类')
    plt.xlabel('萼片长度')
    plt.ylabel('萼片宽度')
    plt.show()
def main():
    iris=datasets.load_iris()
    dataSet=iris.data
    k=eval(input('请输入K值：'))
    iterNum=eval(input('请输入最大迭代次数：'))
    model=KMeans(n_clusters=k,init='random',max_iter=iterNum)
    #要生成的质心个数k 初始化质心随机 最大迭代次数iterNum
    model.fit(dataSet)
    showCluster(dataSet,k,model.cluster_centers_,model.labels_)
main()
