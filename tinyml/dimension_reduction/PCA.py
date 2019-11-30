import numpy as np

class PCA:
    def __init__(self,d_=2):
        self.d_=d_
        self.W=None
        self.mean_x=None
        self.V=None

    # p231 图10.5 PCA算法
    def fit(self,X):
        self.mean_x=np.mean(X,axis=0)
        X_new=X-self.mean_x
        covM=X_new.T.dot(X_new)
        v,w = np.linalg.eig(covM)
        idx = v.argsort()[::-1]
        self.W=w[:,idx][:,:self.d_]
        self.V=v[idx][:self.d_]


    def fit_transform(self,X):
        self.fit(X)
        X=X-self.mean_x
        new_X=X.dot(self.W)
        return new_X


if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])
    X=np.c_[X,X]

    pca=PCA(d_=2)
    Z=pca.fit_transform(X)
    print(Z)

    import sklearn.decomposition as decomposition
    sklearn_PCA=decomposition.PCA(n_components=2,svd_solver='full')
    Z2=sklearn_PCA.fit_transform(X)
    print(Z2)

    print('diff:',np.sum((Z-Z2)**2))
