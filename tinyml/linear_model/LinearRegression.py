import numpy as np
from sklearn import linear_model


class LinearRegression:
    def __init__(self):
        self.w=None
        self.n_features=None

    def fit(self,X,y):
        """
        w=(X^TX)^{-1}X^Ty
        """
        assert isinstance(X,np.ndarray) and isinstance(y,np.ndarray)
        assert X.ndim==2 and y.ndim==1
        assert y.shape[0]==X.shape[0]
        n_samples = X.shape[0]
        self.n_features=X.shape[1]
        extra=np.ones((n_samples,))
        X=np.c_[X,extra]
        if self.n_features<n_samples:
            self.w=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        else:
            raise ValueError('dont have enough samples')

    def predict(self,X):
        n_samples=X.shape[0]
        extra = np.ones((n_samples,))
        X = np.c_[X, extra]
        if self.w is None:
            raise RuntimeError('cant predict before fit')
        y_=X.dot(self.w)
        return y_

if __name__=='__main__':
    X=np.array([[1.0,0.5,0.5],[1.0,1.0,0.3],[-0.1,1.2,0.5],[1.5,2.4,3.2],[1.3,0.2,1.4]])
    y=np.array([1,0.5,1.5,2,-0.3])
    lr=LinearRegression()
    lr.fit(X,y)
    X_test=np.array([[1.3,1,3.2],[-1.2,1.2,0.8]])
    y_pre=lr.predict(X_test)
    print(y_pre)

    sklearn_lr=linear_model.LinearRegression()
    sklearn_lr.fit(X,y)
    sklearn_y_pre=sklearn_lr.predict(X_test)
    print(sklearn_y_pre)

    ridge_reg = linear_model.Ridge(alpha=0., solver='lsqr')
    ridge_reg.fit(X, y)
    ridge_y_pre=ridge_reg.predict(X_test)
    print(ridge_y_pre)


