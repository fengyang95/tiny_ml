import numpy as np
from sklearn import datasets,ensemble,tree
from sklearn.metrics import mean_squared_error
from tinyml.tree.DecisionTreeRegressor import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self,base_estimator,n_estimators=10,min_samples_leaf=5,min_samples_split=15):
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.estimators_=[]

    def fit(self,X,y):
        for t in range(self.n_estimators):
            estimator_t=self.base_estimator(random_state=True,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf)
            estimator_t.fit(X,y)
            self.estimators_.append(estimator_t)

    def predict(self,X):
        preds=[]
        for t in range(self.n_estimators):
            preds.append(self.estimators_[t].predict(X))
        return np.mean(np.array(preds),axis=0)


if __name__=='__main__':
    breast_data = datasets.load_boston()
    X, y = breast_data.data, breast_data.target
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    tinyml_decisiontree_reg=tree.DecisionTreeRegressor(min_samples_split=20, min_samples_leaf=5,random_state=True)
    tinyml_decisiontree_reg.fit(X_train, y_train)
    decisiontree_pred=tinyml_decisiontree_reg.predict(X_test)
    print('base estimator:',mean_squared_error(y_test,decisiontree_pred))

    tinyml_rf_reg=RandomForestRegressor(n_estimators=100, base_estimator=tree.DecisionTreeRegressor)
    tinyml_rf_reg.fit(X_train,y_train)
    y_pred=tinyml_rf_reg.predict(X_test)
    print('tinyml rf mse:',mean_squared_error(y_test,y_pred))


    sklearn_rf_reg=ensemble.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, min_samples_split=20, random_state=False)
    sklearn_rf_reg.fit(X_train, y_train)
    sklearn_pred=sklearn_rf_reg.predict(X_test)
    print('sklearn mse:',mean_squared_error(y_test,sklearn_pred))
