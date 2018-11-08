import numpy as np
"""
Gaussian Discriminant Analysis 
https://see.stanford.edu/materials/aimlcs229/cs229-notes2.pdf
https://zhuanlan.zhihu.com/p/37476759
"""
class GDA:
    def __init__(self,n):
        self.Phi=None
        self.mu0=None
        self.mu1=None
        self.Sigma=None
        self.n=n
        pass

    def fit(self, X, y):
        m=X.shape[0]
        bincount=np.bincount(y)
        assert bincount.shape==(2,)
        self.Phi=bincount[1]*1./m
        zeros_indices=np.where(y==0)
        one_indices=np.where(y==1)
        self.mu0=np.mean(X[zeros_indices],axis=0)
        self.mu1=np.mean(X[one_indices],axis=0)
        self.Sigma=(X[zeros_indices]-self.mu0).T.dot(X[zeros_indices]-self.mu0)+(X[one_indices]-self.mu1).T.dot(X[one_indices]-self.mu1)
        self.Sigma=self.Sigma/m

    def predict_proba(self, X):
        probs=[]
        m=X.shape[0]
        p0=1-self.Phi
        p1=self.Phi
        for i in range(m):
            px_y0=np.exp(-0.5*(X[i]-self.mu0).dot(self.Sigma).dot((X[i]-self.mu0).T))/(np.power(2*np.pi,self.n/2)*np.linalg.det(self.Sigma))
            px_y1 = np.exp(-0.5 * (X[i] - self.mu1).dot(self.Sigma).dot((X[i] - self.mu1).T)) / (np.power(2*np.pi,self.n/2)*np.linalg.det(self.Sigma))
            p_y0=px_y0*p0
            p_y1=px_y1*p1
            probs.append([p_y0,p_y1])
        return np.array(probs)

    def predict(self, X):
        p = self.predict_proba(X)
        res = np.argmax(p, axis=1)
        return res


if __name__ == '__main__':
    X = np.array([[1.0, 0.5, 0.5], [1.0, 1.0, 0.3], [-0.1, 1.2, 0.5], [1.5, 2.4, 3.2], [1.3, 0.2, 1.4]])
    y = np.array([1, 0, 0, 1, 1])
    X_test = np.array([[1.3, 1, 3.2], [-1.2, 1.2, 0.8], [1, 2, 0.4], [1.2, 0.23, -0.5]])

    gda = GDA(n=2)
    gda.fit(X, y)
    lda_prob = gda.predict_proba(X_test)
    lda_pred = gda.predict(X_test)
    print('gda_prob:', lda_prob)
    print('gda_pred:', lda_pred)

