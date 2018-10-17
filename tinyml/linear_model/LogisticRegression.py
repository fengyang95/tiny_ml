import numpy as np
from sklearn import linear_model

np.random.seed(1)
class LogisticRegression:
    def __init__(self,max_iter=100):
        self.beta=None
        self.n_features=None
        self.max_iter=max_iter

    def fit(self,X,y):
        n_samples=X.shape[0]
        self.n_features=X.shape[1]
        extra=np.ones((n_samples,))
        X=np.c_[X,extra]
        self.beta=np.random.random((X.shape[1],))
        for i in range(self.max_iter):
            dldbeta=self._dldbeta(X,y,self.beta)
            dldldbetadbeta=self._dldldbetadbeta(X,self.beta)
            self.beta-=(1./dldldbetadbeta*dldbeta)

    @staticmethod
    def _dldbeta(X,y,beta):
        '''
        # 分步计算的方法
        m=X.shape[0]
        sum=np.zeros(X.shape[1],).T
        for i in range(m):
            sum+=(y[i]-np.exp(X[i].dot(beta))/(1+np.exp(X[i].dot(beta))))
        '''
        # vectorize的方法
        sum=np.sum(y-np.exp(X.dot(beta))/(1+np.exp(X.dot(beta))),axis=0)
        return -sum

    @staticmethod
    def _dldldbetadbeta(X,beta):
        '''
        # 非向量化的方法
        m=X.shape[0]
        sum=0.
        for i in range(m):
            p1=np.exp(X[i].dot(beta))/(1+np.exp(X[i].dot(beta)))
            sum+=X[i].dot(X[i].T)*p1*(1-p1)

        '''
        # 向量化的方法
        p1=np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
        sum=np.sum(X.dot(X.T)*p1*(1-p1))
        return sum

    def predict_proba(self,X):
        n_samples = X.shape[0]
        extra = np.ones((n_samples,))
        X = np.c_[X, extra]
        if self.beta is None:
            raise RuntimeError('cant predict before fit')
        p1 = np.exp(X.dot(self.beta)) / (1 + np.exp(X.dot(self.beta)))
        p0 = 1 - p1
        return np.c_[p0,p1]

    def predict(self,X):
        p=self.predict_proba(X)
        res=np.argmax(p,axis=1)
        return res


if __name__=='__main__':
    X = np.array([[1.0, 0.5, 0.5], [1.0, 1.0, 0.3], [-0.1, 1.2, 0.5], [1.5, 2.4, 3.2], [1.3, 0.2, 1.4]])
    y = np.array([1, 0, 0, 1, 1])
    lr = LogisticRegression()
    lr.fit(X, y)
    X_test = np.array([[1.3, 1, 3.2], [-1.2, 1.2, 0.8],[1,2,0.4],[1.2,0.23,-0.5]])
    print(lr.beta)
    p=lr.predict_proba(X_test)
    print(p)
    y_pre = lr.predict(X_test)
    print(y_pre)

    
    sklearn_logist=linear_model.LogisticRegression(max_iter=100,solver='newton-cg')
    sklearn_logist.fit(X,y)
    print(sklearn_logist.intercept_)
    print(sklearn_logist.coef_)
    sklearn_y_pre=sklearn_logist.predict(X_test)
    print(sklearn_y_pre)



