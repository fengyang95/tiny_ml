import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
import random

# 处理连续型
class ReliefFeatureSelection:
    def __init__(self,sample_ratio=0.5,k=5,seed=None):
        self.feature_importances_=None
        self.k=k
        self.sample_ratio=sample_ratio
        self.seed=seed
        random.seed(self.seed)

    def fit(self,X,y):
        m,n=X.shape
        self.feature_importances_=np.zeros((n,))
        for t in range(self.k):
            indices=random.sample(range(m),int(m*self.sample_ratio))
            subX,suby=X[indices],y[indices]
            self.feature_importances_+=self._fit(subX,suby)
        self.feature_importances_/=self.k


    def transform(self,X,k_features):
        choosed_indices=np.argsort(self.feature_importances_)[::-1][:k_features]
        return X[:,choosed_indices]

    def _fit(self,subX,suby):
        label_to_indices = {}
        labels = np.unique(suby)
        for label in labels:
            label_to_indices[label] = list(np.where(suby == label)[0])
        m, n = subX.shape
        feature_scores_ = np.zeros((n,))
        for j in range(n):
            for i in range(m):
                label_i = suby[i]
                xi_nhs = (subX[i, j] - subX[label_to_indices[label_i], j]) ** 2
                if len(xi_nhs) == 1:
                    xi_nh = 0
                else:
                    xi_nh = np.sort(xi_nhs)[1]
                feature_scores_[j] -= xi_nh
                for label in labels:
                    if label == label_i:
                        continue
                    xi_nm = np.sort((subX[i, j] - subX[label_to_indices[label], j]) ** 2)[0]
                    feature_scores_[j] += (xi_nm * len(label_to_indices[label]) / m)
        return feature_scores_


if __name__=='__main__':
    breast_data = load_breast_cancer()
    subX, suby = breast_data.data, breast_data.target
    scaler=MinMaxScaler()
    subX=scaler.fit_transform(subX)
    reliefF=ReliefFeatureSelection()
    reliefF.fit(subX, suby)
    print('relief feature_importances:',reliefF.feature_importances_)
    print('sorted:',np.argsort(reliefF.feature_importances_))

    import skrebate.relieff as relieff
    skrebate_reliefF=relieff.ReliefF()
    skrebate_reliefF.fit(subX, suby)
    print('skrebate feature_importances_:',skrebate_reliefF.feature_importances_)
    print('sorted:',np.argsort(skrebate_reliefF.feature_importances_))





