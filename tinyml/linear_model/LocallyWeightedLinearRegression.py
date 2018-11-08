import numpy as np
import matplotlib.pyplot as plt
"""
implementation of Locally weighted linear regression in http://cs229.stanford.edu/notes/cs229-notes1.pdf
"""
class LocallyWeightedLinearRegression:
    def __init__(self,tau):
        self.tau=tau
        self.w=None

    def fit_predict(self,X,y,checkpoint_x):
        m = X.shape[0]
        self.n_features = X.shape[1]
        extra = np.ones((m,))
        X = np.c_[X, extra]
        checkpoint_x=np.r_[checkpoint_x,1]
        self.X=X
        self.y=y
        self.checkpoint_x=checkpoint_x
        weight=np.zeros((m,))
        for i in range(m):
            weight[i]=np.exp(-(X[i]-checkpoint_x).dot((X[i]-checkpoint_x).T)/(2*(self.tau**2)))
        weight_matrix=np.diag(weight)
        self.w=np.linalg.inv(X.T.dot(weight_matrix).dot(X)).dot(X.T).dot(weight_matrix).dot(y)
        return checkpoint_x.dot(self.w)

    def fit_transform(self,X,y,checkArray):
        m=len(y)
        preds=[]
        for i in range(m):
            preds.append(self.fit_predict(X,y,checkArray[i]))
        return np.array(preds)


if __name__=='__main__':
    X=np.linspace(0,30,100)
    y=X**2+2
    X=X.reshape(-1,1)
    lr=LocallyWeightedLinearRegression(tau=5)
    y_pred=lr.fit_transform(X,y,X)
    plt.plot(X,y,label='gt')
    plt.plot(X,y_pred,label='pred')
    plt.legend()
    plt.show()

