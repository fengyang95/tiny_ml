import numpy as np
from sklearn import linear_model

# 采用MSE作为损失函数
# penalty = 'l2' 则为 Ridge Regression
# penalty = 'l1' 则为 Lasso Regression
# penalty = 'l1l2' 则为 Elastic Net
# alpha 为 正则化系数

# https://wwdguu.github.io/2018/09/01/%C2%96HOMLWSLATF-ch4/
np.random.seed(1)
class SGDRegressor:
    def __init__(self,max_iter=100,penalty=None,alpha=1e-3,l1_ratio=0.5):
        self.w = None
        self.n_features = None
        self.penalty=penalty
        self.alpha=alpha
        self.l1_ratio=l1_ratio
        self.max_iter=max_iter

    #
    def fit(self, X, y):
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert y.shape[0] == X.shape[0]
        n_samples = X.shape[0]
        self.n_features = X.shape[1]
        extra = np.ones((n_samples,1))
        X = np.c_[X,extra]
        self.w=np.random.randn(X.shape[1],1)
        for iter in range(self.max_iter):
            for i in range(n_samples):
                sample_index=np.random.randint(n_samples)
                x_sample=X[sample_index:sample_index+1]
                y_sample=y[sample_index:sample_index+1]
                lr=SGDRegressor.learning_schedule(iter*n_samples+i)
                # 求导
                grad=2*x_sample.T.dot(x_sample.dot(self.w)-y_sample)
                if self.penalty is not None:
                    # Ridge
                    if self.penalty=='l2':
                        grad+=self.alpha*self.w
                    # Lasso
                    elif self.penalty=='l1':
                        grad+=self.alpha*np.sign(self.w)
                    # Elastic Net
                    elif self.penalty=='l1l2':
                        grad+=(self.alpha*self.l1_ratio*np.sign(self.w)+
                               (1-self.l1_ratio)*self.alpha*self.w)

                self.w=self.w-lr*grad


    def predict(self, X):

        n_samples = X.shape[0]
        extra = np.ones((n_samples,1))
        X = np.c_[X,extra]
        if self.w is None:
            raise RuntimeError('cant predict before fit')
        y_ = X.dot(self.w)
        return y_

    @staticmethod
    def learning_schedule(t):
        return 5 / (t + 50)


if __name__ == '__main__':
    X = 2 * np.random.rand(100,1)
    y = 4 + 3 * X + np.random.randn(100,1)
    y=y.ravel()
    print(X.shape)
    print(y.shape)
    lr = SGDRegressor(max_iter=200,penalty='l1l2',alpha=1e-3,l1_ratio=0.5)
    lr.fit(X, y)
    print('w:',lr.w)

    sklearn_lr = linear_model.SGDRegressor(max_iter=200,penalty='l1',alpha=1e-3)
    sklearn_lr.fit(X, y)
    print(sklearn_lr.coef_)
    print(sklearn_lr.intercept_)

