import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue
random.seed(1)

class DBSCAN:
    def __init__(self,epsilon=0.11,min_pts=5):
        self.epsilon=epsilon
        self.min_pts=min_pts
        self.labels_=None
        self.C=None
        self.Omega=set()
        self.N_epsilon={}

    # p213 图9.9 DBSCAN算法
    def fit(self,X):
        self.C={}
        for j in range(X.shape[0]):
            dist=np.sqrt(np.sum((X-X[j])**2,axis=1))
            self.N_epsilon[j]=np.where(dist<=self.epsilon)[0]
            if len(self.N_epsilon[j])>=self.min_pts:
                self.Omega.add(j)
        self.k=0
        Gamma=set(range(X.shape[0]))
        while len(self.Omega)>0:
            Gamma_old=set(Gamma)
            o=random.sample(list(self.Omega),1)[0]
            Q=Queue()
            Q.put(o)
            Gamma.remove(o)
            while not Q.empty():
                q=Q.get()
                if len(self.N_epsilon[q])>=self.min_pts:
                    Delta=set(self.N_epsilon[q]).intersection(set(Gamma))
                    for delta in Delta:
                        Q.put(delta)
                        Gamma.remove(delta)
            self.C[self.k]=Gamma_old.difference(Gamma)
            self.Omega=self.Omega.difference(self.C[self.k])
            self.k += 1
        self.labels_=np.zeros((X.shape[0],),dtype=np.int32)
        for i in range(self.k):
            self.labels_[list(self.C[i])]=i


if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    dbscan=DBSCAN()
    dbscan.fit(X)
    print('C:',dbscan.C)
    print(dbscan.labels_)
    plt.figure(12)
    plt.subplot(121)
    plt.scatter(X[:,0],X[:,1],c=dbscan.labels_)
    plt.title('tinyml')

    import sklearn.cluster as cluster
    sklearn_DBSCAN=cluster.DBSCAN(eps=0.11,min_samples=5,metric='l2')
    sklearn_DBSCAN.fit(X)
    print(sklearn_DBSCAN.labels_)
    plt.subplot(122)
    plt.scatter(X[:,0],X[:,1],c=sklearn_DBSCAN.labels_)
    plt.title('sklearn')
    plt.show()

