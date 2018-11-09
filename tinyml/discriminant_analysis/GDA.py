import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
"""
Gaussian Discriminant Analysis 
https://see.stanford.edu/materials/aimlcs229/cs229-notes2.pdf
https://zhuanlan.zhihu.com/p/37476759
"""
class GDA:
    def __init__(self):
        self.Phi=None
        self.mu0=None
        self.mu1=None
        self.Sigma=None
        self.n=None
        pass

    def fit(self, X, y):
        m=X.shape[0]
        self.n=X.shape[1]
        bincount=np.bincount(y)
        assert bincount.shape==(2,)
        self.Phi=bincount[1]*1./m
        zeros_indices=np.where(y==0)
        one_indices=np.where(y==1)
        self.mu0=np.mean(X[zeros_indices],axis=0)
        self.mu1=np.mean(X[one_indices],axis=0)
        self.Sigma=np.zeros((self.n,self.n))
        for i in range(m):
            if y[i]==0:
                tmp=(X[i]-self.mu0).T.dot((X[i]-self.mu0))
                self.Sigma+=tmp
            else:
                tmp=(X[i]-self.mu1).reshape(-1,1).dot((X[i]-self.mu1).reshape(1,-1))
                self.Sigma+=tmp

        self.Sigma=(X[zeros_indices]-self.mu0).T.dot(X[zeros_indices]-self.mu0)+(X[one_indices]-self.mu1).T.dot(X[one_indices]-self.mu1)
        self.Sigma=self.Sigma/m


    def predict_proba(self, X):
        probs=[]
        m=X.shape[0]
        p0=1-self.Phi
        p1=self.Phi
        denominator=np.power(2*np.pi,self.n/2)*np.sqrt(np.linalg.det(self.Sigma))
        for i in range(m):
            px_y0=np.exp(-0.5*(X[i]-self.mu0).dot(np.linalg.inv(self.Sigma)).dot((X[i]-self.mu0).T))/denominator
            px_y1 = np.exp(-0.5 * (X[i] - self.mu1).dot(np.linalg.inv(self.Sigma)).dot((X[i] - self.mu1).T)) /denominator
            p_y0=px_y0*p0
            p_y1=px_y1*p1
            probs.append([p_y0/(p_y0+p_y1),p_y1/(p_y0+p_y1)])
        return np.array(probs)

    def predict(self, X):
        p = self.predict_proba(X)
        res = np.argmax(p, axis=1)
        return res


if __name__ == '__main__':
    np.random.seed(42)
    breast_data = load_breast_cancer()
    X, y = breast_data.data, breast_data.target
    X=MinMaxScaler().fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    gda = GDA()
    gda.fit(X_train, y_train)
    lda_prob = gda.predict_proba(X_test)
    lda_pred = gda.predict(X_test)
    print('gda_prob:', lda_prob)
    print('gda_pred:', lda_pred)
    print('accuracy:',len(y_test[y_test ==lda_pred]) * 1. / len(y_test))

