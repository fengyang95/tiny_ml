import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.random.seed(42)
import torch
from torch import nn,optim


class SGDLogisticRegression:
    class LogisticRegressionModel(nn.Module):
        def __init__(self,n_features):
            super(SGDLogisticRegression.LogisticRegressionModel,self).__init__()
            self.linear=nn.Linear(n_features,1)
            self.sigmoid=nn.Sigmoid()

        def forward(self,X):
            return self.sigmoid(self.linear(X))

    def __init__(self,max_iter=100000,learning_rate=0.005):
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.criterion=nn.BCELoss()
        self.fitted=False

    def fit(self,X,y):
        n_feature=X.shape[1]
        self.model=SGDLogisticRegression.LogisticRegressionModel(n_feature)
        self.optimizer=optim.SGD(self.model.parameters(),lr=self.learning_rate)
        X=torch.from_numpy(X.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        for epoch in range(self.max_iter):
            y_predict=self.model(X)[:,0]
            loss=self.criterion(y_predict,y)
            #print('epoch:',epoch,' loss.item():',loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self,X):
        X = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            y_pred = self.model(X).detach().numpy()
            y_pred[y_pred>0.5]=1
            y_pred[y_pred<=0.5]=0
        return y_pred[:,0]


class LogisticRegression:
    def __init__(self,max_iter=100,use_matrix=True):
        self.beta=None
        self.n_features=None
        self.max_iter=max_iter
        self.use_Hessian=use_matrix

    def fit(self,X,y):
        n_samples=X.shape[0]
        self.n_features=X.shape[1]
        extra=np.ones((n_samples,))
        X=np.c_[X,extra]
        self.beta=np.random.random((X.shape[1],))
        for i in range(self.max_iter):
            if self.use_Hessian is not True:
                dldbeta=self._dldbeta(X,y,self.beta)
                dldldbetadbeta=self._dldldbetadbeta(X,self.beta)
                self.beta-=(1./dldldbetadbeta*dldbeta)
            else:
                dldbeta = self._dldbeta(X, y, self.beta)
                dldldbetadbeta = self._dldldbetadbeta_matrix(X, self.beta)
                self.beta -= (np.linalg.inv(dldldbetadbeta).dot(dldbeta))



    @staticmethod
    def _dldbeta(X,y,beta):
        # 《机器学习》 公式 3.30
        m=X.shape[0]
        sum=np.zeros(X.shape[1],).T
        for i in range(m):
            sum+=X[i]*(y[i]-np.exp(X[i].dot(beta))/(1+np.exp(X[i].dot(beta))))
        return -sum

    @staticmethod
    def _dldldbetadbeta_matrix(X,beta):
        m=X.shape[0]
        Hessian=np.zeros((X.shape[1],X.shape[1]))
        for i in range(m):
            p1 = np.exp(X[i].dot(beta)) / (1 + np.exp(X[i].dot(beta)))
            tmp=X[i].reshape((-1,1))
            Hessian+=tmp.dot(tmp.T)*p1*(1-p1)
        return Hessian

    @staticmethod
    def _dldldbetadbeta(X,beta):
        # 《机器学习》公式 3.31
        m=X.shape[0]
        sum=0.
        for i in range(m):
            p1=np.exp(X[i].dot(beta))/(1+np.exp(X[i].dot(beta)))
            sum+=X[i].dot(X[i].T)*p1*(1-p1)
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
    breast_data = load_breast_cancer()
    X, y = breast_data.data[:,:7], breast_data.target
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    tinyml_logisticreg = LogisticRegression(max_iter=100,use_matrix=True)
    tinyml_logisticreg.fit(X_train, y_train)
    lda_prob = tinyml_logisticreg.predict_proba(X_test)


    lda_pred = tinyml_logisticreg.predict(X_test)
    # print('tinyml logistic_prob:', lda_prob)
    # print('tinyml logistic_pred:', lda_pred)
    print('tinyml accuracy:', len(y_test[y_test == lda_pred]) * 1. / len(y_test))

    sklearn_logsticreg = linear_model.LogisticRegression(max_iter=100,solver='newton-cg')
    sklearn_logsticreg.fit(X_train, y_train)
    sklearn_prob = sklearn_logsticreg.predict_proba(X_test)
    sklearn_pred = sklearn_logsticreg.predict(X_test)
    # print('sklearn prob:',sklearn_prob)
    # print('sklearn pred:',sklearn_pred)
    print('sklearn accuracy:', len(y_test[y_test == sklearn_pred]) * 1. / len(y_test))

    torch_sgd_logisticreg=SGDLogisticRegression(100000,0.01)
    torch_sgd_logisticreg.fit(X_train,y_train)
    torch_pred=torch_sgd_logisticreg.predict(X_test)
    print('torch accuracy:',len(y_test[y_test==torch_pred])/len(y_test))

    # expected output
    """
    tinyml accuracy: 0.9590643274853801
    sklearn accuracy: 0.9298245614035088
    torch accuracy: 0.9532163742690059
    """







