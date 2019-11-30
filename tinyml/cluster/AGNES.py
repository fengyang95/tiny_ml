import numpy as np
import matplotlib.pyplot as plt

class AGNES:
    def __init__(self,k=3,dist_type='AVG'):
        self.k=k
        self.labels_=None
        self.C={}
        self.dist_func=None
        if dist_type=='MIN':
            self.dist_func=self.mindist
        elif dist_type=='MAX':
            self.dist_func=self.maxdist
        else:
            self.dist_func=self.avgdist

    # p215 图9.11 AGNES算法
    def fit(self,X):
        for j in range(X.shape[0]):
            self.C[j]=set()
            self.C[j].add(j)
        M=1e10*np.ones((X.shape[0],X.shape[0]),dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(i+1,X.shape[0]):
                M[i,j]=self.dist_func(X,self.C[i],self.C[j])
                M[j,i]=M[i,j]
        q=X.shape[0]
        while q>self.k:
            index=np.argmin(M)
            i_=index//M.shape[1]
            j_=index%M.shape[1]
            self.C[i_]=set(self.C[i_].union(self.C[j_]))
            #print(self.C[i_])
            for j in range(j_+1,q):
                self.C[j-1]=set(self.C[j])
            del self.C[q-1]
            M=np.delete(M,[j_],axis=0)
            M=np.delete(M,[j_],axis=1)
            for j in range(q-1):
                if i_!=j:
                    M[i_,j]=self.dist_func(X,self.C[i_],self.C[j])
                    M[j,i_]=M[i_,j]
            q-=1
        self.labels_=np.zeros((X.shape[0],),dtype=np.int32)
        for i in range(self.k):
            self.labels_[list(self.C[i])] = i

    @classmethod
    def mindist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        min=1e10
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            dmin=np.min(d)
            if dmin<min:
                min=dmin
        return min

    @classmethod
    def maxdist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        max=0
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            dmax=np.max(d)
            if dmax>max:
                max=dmax
        return max

    @classmethod
    def avgdist(cls,X,Ci,Cj):
        Xi=X[list(Ci)]
        Xj=X[list(Cj)]
        sum=0.
        for i in range(len(Xi)):
            d=np.sqrt(np.sum((Xi[i]-Xj)**2,axis=1))
            sum+=np.sum(d)
        dist=sum/(len(Ci)*len(Cj))
        return dist




if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    X_test=X
    agnes=AGNES()
    agnes.fit(X)
    print('C:', agnes.C)
    print(agnes.labels_)
    plt.figure(12)
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=agnes.labels_)
    plt.title('tinyml')

    from sklearn.cluster.hierarchical import AgglomerativeClustering
    sklearn_agnes=AgglomerativeClustering(n_clusters=7,affinity='l2',linkage='average')
    sklearn_agnes.fit(X)
    print(sklearn_agnes.labels_)
    plt.subplot(122)
    plt.scatter(X[:,0],X[:,1],c=sklearn_agnes.labels_)
    plt.title('sklearn')
    plt.show()





