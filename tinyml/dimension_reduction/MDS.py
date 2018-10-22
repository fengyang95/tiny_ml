import numpy as np
import matplotlib.pyplot as plt
# 不知道如何验证正确性，sklearn中的实现方式和西瓜书中不一致,sklearn中用的smacof方法
# 和自己实现的KernelPCA线性核时结果一致

class MDS:
    def __init__(self,d_=2):
        self.d_=d_
        self.Z=None
        self.values_=None
        self.vectors_=None

    # p229 图10.3 MDS算法
    def fit(self,X):
        m=X.shape[0]
        B=X.dot(X.T)
        Dist_2=np.zeros((m,m),dtype=np.float32)
        for i in range(m):
            for j in range(m):
                Dist_2[i,j]=B[i,i]+B[j,j]-2*B[i,j]
        Dist_i2=np.mean(Dist_2,axis=1).reshape(-1,1)
        Dist_j2=np.mean(Dist_2,axis=0).reshape(1,-1)
        dist_2=np.mean(Dist_2)
        B_new=-0.5*(Dist_2-Dist_i2-Dist_j2+dist_2)

        """
        B_new=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                B_new[i,j]=-0.5*(Dist_2[i,j]-Dist_i2[i,0]-Dist_j2[0,j]+dist_2)
        """
        # 用eig和eigh函数分解出的结果符号位不同
        values,vectors=np.linalg.eig(B_new)
        #values,vectors=np.linalg.eigh(B_new)
        idx=np.argsort(values)[::-1]
        self.values_=values[idx][:self.d_]
        # print('values:',self.values_)
        self.vectors_=vectors[:,idx][:,:self.d_]
        self.Z=self.vectors_.dot(np.diag(np.sqrt(self.values_))).real

    def fit_transform(self,X):
        self.fit(X)
        return self.Z


if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    X=np.c_[X,X]
    mds=MDS(d_=2)
    Z=mds.fit_transform(np.array(X))
    print(Z)

    """
    import sklearn.manifold as manifold
    sklearn_MDS=manifold.MDS(n_components=2,metric=True,random_state=False)
    Z2=sklearn_MDS.fit_transform(X)
    print(Z2)
    print('diff:',np.sum((Z-Z2)**2))
    """




