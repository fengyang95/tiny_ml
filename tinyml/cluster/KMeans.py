import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(1)

class KMeans:
    def __init__(self,k=2):
        self.labels_=None
        self.mu=None
        self.k=k

    # p203图9.2算法流程
    def fit(self,X):
        self.mu=X[random.sample(range(X.shape[0]),self.k)]
        while True:
            C={}
            for i in range(self.k):
                C[i]=[]
            for j in range(X.shape[0]):
                d=np.zeros((self.k,))
                for i in range(self.k):
                    d[i]=np.sqrt(np.sum((X[j]-self.mu[i])**2))
                lambda_j=np.argmin(d)
                C[lambda_j].append(j)
            mu_=np.zeros((self.k,X.shape[1]))
            for i in range(self.k):
                mu_[i]=np.mean(X[C[i]],axis=0)
            if np.sum((mu_-self.mu)**2)<1e-5:
                self.C=C
                break
            else:
                self.mu=mu_
        self.labels_=np.zeros((X.shape[0],),dtype=np.int32)
        for i in range(self.k):
            self.labels_[C[i]]=i

    def predict(self,X):
        preds=[]
        for j in range(X.shape[0]):
            d=np.zeros((self.k,))
            for i in range(self.k):
                d[i]=np.sqrt(np.sum((X[j]-self.mu[i])**2))
            preds.append(np.argmin(d))
        return np.array(preds)

if __name__=='__main__':
    # p202 西瓜数据集4.0
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])
    X_test=X
    kmeans=KMeans(k=3)
    kmeans.fit(X)
    print(kmeans.C)
    print(kmeans.labels_)
    print(kmeans.predict(X))

    plt.scatter(X[np.where(kmeans.labels_==0),0],X[np.where(kmeans.labels_==0),1],c='r')
    plt.scatter(kmeans.mu[0:1,0],kmeans.mu[0:1,1],c='r',marker='+')
    plt.scatter(X[np.where(kmeans.labels_==1),0],X[np.where(kmeans.labels_==1),1],c='g')
    plt.scatter(kmeans.mu[1:2,0],kmeans.mu[1:2,1],c='g',marker='+')
    plt.scatter(X[np.where(kmeans.labels_==2),0],X[np.where(kmeans.labels_==2),1],c='b')
    plt.scatter(kmeans.mu[2:3,0],kmeans.mu[2:3,1],c='b',marker='+')
    plt.show()


