import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import copy
from sklearn import tree

"""
loss使用均方误差
残差为 y-y_pred
李航《统计学习方法》 p151
"""
class GradientBoostingRegressor:
    def __init__(self,base_estimator=None,n_estimators=10,lr=0.1):
        self.base_estimator=base_estimator
        self.n_esimators=n_estimators
        self.estimators=[]
        self.lr=lr
        self.mean=None

    def fit(self,X,y):
        F0_x=np.ones_like(y)*np.mean(y)
        y_pred=F0_x
        self.mean=np.mean(y)
        for i in range(self.n_esimators):
            hm=copy.deepcopy(self.base_estimator)
            hm.fit(X,y-y_pred)
            self.estimators.append(hm)
            y_pred=y_pred+self.lr*hm.predict(X)

    def predict(self,X):
        y=self.mean*np.ones((X.shape[0],))
        for i in range(self.n_esimators):
            y=y+self.lr*self.estimators[i].predict(X)
        return y


if __name__=='__main__':
    breast_data = datasets.load_boston()
    X, y = breast_data.data, breast_data.target
    print(X.shape)
    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:], y[400:]

    sklearn_decisiontree_reg=tree.DecisionTreeRegressor(min_samples_split=15, min_samples_leaf=5,random_state=False)
    sklearn_decisiontree_reg.fit(X_train, y_train)
    decisiontree_pred=sklearn_decisiontree_reg.predict(X_test)
    print('base estimator:',mean_squared_error(y_test,decisiontree_pred))

    tinyml_gbdt_reg=GradientBoostingRegressor(n_estimators=500, base_estimator=tree.DecisionTreeRegressor(min_samples_split=15, min_samples_leaf=5, random_state=False))
    tinyml_gbdt_reg.fit(X_train, y_train)
    y_pred=tinyml_gbdt_reg.predict(X_test)
    print('tinyml mse:',mean_squared_error(y_test,y_pred))


    sklearn_gbdt_reg=ensemble.GradientBoostingRegressor(n_estimators=500,min_samples_leaf=5,min_samples_split=15,random_state=False)
    sklearn_gbdt_reg.fit(X_train,y_train)
    sklearn_pred=sklearn_gbdt_reg.predict(X_test)
    print('sklearn mse:',mean_squared_error(y_test,sklearn_pred))
