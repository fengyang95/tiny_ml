import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
np.random.seed(0)
import torch
from torch import nn,optim
class SGDFMClassifier:
    class FMClassifier(nn.Module):
        def __init__(self,n_features,loss='logistic',degree=2,n_components=2):
            super(SGDFMClassifier.FMClassifier,self).__init__()
            self.loss=loss
            self.degree=degree
            self.n_components=n_components
            self.linear=nn.Linear(n_features,1)
            self.v=nn.Parameter(torch.Tensor(n_features,self.n_components))
            stdev=1./math.sqrt(self.v.size(1))
            self.v.data.uniform_(-stdev,stdev)
            self.sigmoid=nn.Sigmoid()

        def forward(self,X):
            y=self.linear(X)+0.5*torch.sum(torch.pow(torch.mm(X,self.v),2)-
                                           torch.mm(torch.pow(X,2),torch.pow(self.v,2)))
            return self.sigmoid(y)

    def __init__(self,max_iter=100000,learning_rate=0.005):
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.criterion=nn.BCELoss()
        self.fitted=False

    def fit(self,X,y):
        n_feature=X.shape[1]
        self.model=self.FMClassifier(n_feature)
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

if __name__=='__main__':
    breast_data = load_breast_cancer()
    X, y = breast_data.data[:, :7], breast_data.target
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    torch_mfclassifier = SGDFMClassifier(20000, 0.001)
    torch_mfclassifier.fit(X_train, y_train)
    torch_pred = torch_mfclassifier.predict(X_test)
    print('torch accuracy:', len(y_test[y_test == torch_pred]) / len(y_test))