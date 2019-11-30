import numpy as np
import matplotlib.pyplot as plt

class GaussianMixture:
    def __init__(self,k=3,max_iter=50):
        self.k=k
        self.max_iter=max_iter
        self.labels_=None
        self.C=None
        self.alpha=None
        self.mu=None
        self.cov=None
        self.gamma=None
        pass

    # p210 图9.6 高斯混合聚类算法
    def fit(self,X):
        # p210初始化方法
        self.alpha=np.zeros((self.k,))
        for i in range(self.k):
            self.alpha[i]=1./self.k
        mu_indices=[5,21,26]
        self.mu=X[mu_indices]
        self.cov=np.array([[[0.1,0.],[0.0,0.1]],[[0.1,0.],[0.,0.1]],[[0.1,0.],[0.,0.1]]])

        self.gamma=np.zeros((X.shape[0],self.k))
        for _ in range(self.max_iter):
            for j in range(X.shape[0]):
                alpha_p=np.zeros((self.k,))
                sum=0.
                for i in range(self.k):
                    alpha_p[i]=self.alpha[i]*self._p(X[j],self.mu[i],self.cov[i])
                    sum+=alpha_p[i]
                self.gamma[j,:]=alpha_p/sum

            for i in range(self.k):
                sum_gamma_i=np.sum(self.gamma[:,i])
                self.mu[i]=X.T.dot(self.gamma[:,i])/sum_gamma_i
                numerator=0.
                for j in range(X.shape[0]):
                    numerator+=(self.gamma[j,i]*((X[j]-self.mu[i]).reshape(-1,1).dot((X[j]-self.mu[i]).reshape(1,-1))))
                self.cov[i]=numerator/sum_gamma_i
                self.alpha[i]=sum_gamma_i/X.shape[0]
        self.labels_=np.argmax(self.gamma,axis=1)
        self.C={}
        for i in range(self.k):
            self.C[i]=[]
        for j in range(len(self.labels_)):
            self.C[self.labels_[j]].append(j)

    def predict(self,X):
        gamma = np.zeros((X.shape[0], self.k))
        for j in range(X.shape[0]):
            alpha_p = np.zeros((self.k,))
            sum = 0.
            for i in range(self.k):
                alpha_p[i] = self.alpha[i] * self._p(X[j], self.mu[i], self.cov[i])
                sum += alpha_p[i]
            gamma[j, :] = alpha_p / sum
        return np.argmax(gamma,axis=1)


    # 公式 9.28
    @classmethod
    def _p(cls,x,mu,cov):
        exp_coef=-0.5*((x-mu).T.dot(np.linalg.inv(cov)).dot(x-mu))
        p=np.exp(exp_coef)/(np.power(2*np.pi,mu.shape[0]/2)*np.sqrt(np.linalg.det(cov)))
        return p

if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    X_test=X
    gmm=GaussianMixture(k=3,max_iter=50)
    gmm.fit(X)
    print(gmm.C)
    print(gmm.labels_)
    print(gmm.predict(X_test))
    plt.scatter(X[:, 0], X[:, 1], c=gmm.labels_)
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1],c=range(gmm.k), marker='+')
    plt.title('tinyml')
    plt.show()


    from sklearn.mixture import GaussianMixture

    sklearn_gmm = GaussianMixture(n_components=3, covariance_type='full',
                                  max_iter=50).fit(X)
    labels=sklearn_gmm.predict(X)
    print(labels)
    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.title('sklearn')
    plt.show()





