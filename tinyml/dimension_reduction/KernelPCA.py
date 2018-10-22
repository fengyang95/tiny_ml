import numpy as np
# 线性核 与 sklearn 结果一致
#　其他核都与sklearn结果不一致!!! 还没找到原因

class KernelPCA:
    def __init__(self,d_=2,kernel='linear',gamma=None,coef0=1.,degress=3):
        self.d_=d_
        self.W=None
        self.mean_x=None
        self.V=None
        self.kernel=kernel
        self.coef0=coef0
        self.degress=degress
        if gamma is None:
            self.gamma=1./self.d_
        else:
            self.gamma=gamma

    def kernel_func(self,kernel,x1,x2):
        if kernel=='linear':
            return x1.dot(x2.T)
        elif kernel=='rbf':
            return np.exp(-self.gamma*(np.sum((x1-x2)**2)))
        elif kernel=='poly':
            return np.power(self.gamma*(x1.dot(x2.T)+1)+self.coef0,self.degress)
        elif kernel=='sigmoid':
            return np.tanh(self.gamma*(x1.dot(x2.T))+self.coef0)

    def computeK(self,X,kernel):
        m=X.shape[0]
        K=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i<=j:
                    K[i,j]=self.kernel_func(kernel,X[i],X[j])
                else:
                    K[i,j]=K[j,i]
        return K

    # p233 公式10.24
    def fit(self,X):
        self.mean_x=np.mean(X,axis=0)
        X_new=X-self.mean_x
        K=self.computeK(X_new,kernel=self.kernel)
        # sklearn实现用的eigh分解
        values,vectors = np.linalg.eigh(K)
        idx = values.argsort()[::-1]
        # 这一步不可少
        vectors/=np.sqrt(values)
        self.alphas_= vectors[:, idx][:, :self.d_]
        self.lambdas_= values[idx][:self.d_]

    # 公式 10.25
    def fit_transform(self,X):
        self.fit(X)
        X = X - self.mean_x
        m=X.shape[0]
        self.Z=np.zeros((m,self.d_))
        for k in range(m):
            for j in range(self.d_):
                sum=0.
                for i in range(m):
                    sum+= self.alphas_[i, j] * (self.kernel_func(self.kernel, X[i], X[k]))
                self.Z[k,j]=sum
        return self.Z


if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    X=np.c_[X,X]
    kpca=KernelPCA(d_=2, kernel='linear', gamma=1. / 2)
    Z=kpca.fit_transform(X)
    print('tinyml:')
    #print('lambdas:', kpca.lambdas_)
    #print('alphas:', kpca.alphas_)
    print(Z)

    import sklearn.decomposition as decomposition
    sklearn_KPCA=decomposition.KernelPCA(n_components=2, kernel='linear', gamma=1. / 2, eigen_solver='dense', random_state=False)
    Z2=sklearn_KPCA.fit_transform(X)
    print('sklearn')
    #print('lambdas:',sklearn_KPCA.lambdas_)
    #print('alphas:',sklearn_KPCA.alphas_)
    print(Z2)

    print('Z diff:',np.sum((Z-Z2)**2))



