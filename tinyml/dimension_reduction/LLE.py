import numpy as np
import scipy
"""
Omega的计算参考这篇blog
[局部线性嵌入(LLE)原理总结](https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral)
"""
class LLE:
    def __init__(self,d_=2,k=6,reg=1e-3):
        self.d_=d_
        self.k=k
        self.reg=reg

    # p237 图10.10 LLE算法
    def fit(self,X):
        m=X.shape[0]
        Dist=np.zeros((m,m),dtype=np.float32)
        self.Omega=np.zeros((m,m),dtype=np.float32)
        self.Q={}
        for i in range(m):
            Dist[i,:]=np.sqrt(np.sum((X[i]-X)**2,axis=1))
            self.Q[i]=np.argsort(Dist[i,:])[1:self.k+1]
            self.compute_omega(i,X)

        self.M=np.matmul((np.identity(m)-self.Omega).T,(np.identity(m)-self.Omega))
        w,v=np.linalg.eig(self.M)
        index=np.argsort(w)
        self.Z=v[:,index][:,1:1+self.d_]

    def fit_transform(self,X):
        self.fit(X)
        return self.Z

    def compute_omega(self,i,X):
        Z=(X[i]-X[self.Q[i]]).dot((X[i]-X[self.Q[i]]).T)
        Z += self.reg * np.trace(Z) * np.identity(self.k)
        Ik=np.ones((self.k,))
        Zinv=np.linalg.inv(Z)
        self.Omega[i, self.Q[i]]=np.matmul(Zinv,Ik)/(Ik.T.dot(Zinv).dot(Ik))

if __name__=='__main__':
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
                  [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
                  [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                  [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
                  [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
                  [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])
    X = np.c_[X, X]
    lle = LLE(d_=2, k=5,reg=1e-3)
    Z = lle.fit_transform(X)
    print(Z)

    import sklearn.manifold as manifold
    sklearn_LLE= manifold.LocallyLinearEmbedding(n_components=2,n_neighbors=5,reg=1e-3)
    Z2 = sklearn_LLE.fit_transform(X)
    print(Z2)

    print('check diff:',np.sum((Z2-Z)**2))
