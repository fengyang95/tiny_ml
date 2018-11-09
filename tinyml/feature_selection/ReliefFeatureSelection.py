import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
import lightgbm as lgb
# 处理连续型
class ReliefFeatureSelection:
    def __init__(self):
        self.feature_importances_=None

    def fit(self,X,y):
        label_to_indices={}
        labels=np.unique(y)
        for label in labels:
            label_to_indices[label]=list(np.where(y==label)[0])
        m,n=X.shape
        self.feature_importances_= np.zeros((n,))
        for j in range(n):
            for i in range(m):
                label_i=y[i]
                xi_nhs=(X[i,j]-X[label_to_indices[label_i],j])**2
                if len(xi_nhs)==1:
                    xi_nh=0
                else:
                    xi_nh=np.sort(xi_nhs)[1]
                self.feature_importances_[j]-=xi_nh
                for label in labels:
                    if label==label_i:
                        continue
                    xi_nm=np.sort((X[i,j]-X[label_to_indices[label],j])**2)[0]
                    self.feature_importances_[j]+=(xi_nm*len(label_to_indices[label])/m)

    def transform(self,X,k_features):
        choosed_indices=np.argsort(self.feature_importances_)[::-1][:k_features]
        return X[:,choosed_indices]

if __name__=='__main__':
    breast_data = load_breast_cancer()
    X, y = breast_data.data, breast_data.target
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)
    reliefF=ReliefFeatureSelection()
    reliefF.fit(X,y)
    print('relief feature_importances:',reliefF.feature_importances_)






