from sklearn import discriminant_analysis
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LDA:
    def __init__(self):
        self.omega=None
        self.omiga_mu_0=None
        self.omiga_mu_1=None
        pass

    # 《机器学习》 p61
    def fit(self,X,y):
        n_samples = X.shape[0]
        extra = np.ones((n_samples,))
        X = np.c_[X, extra]
        X_0=X[np.where(y==0)]
        X_1=X[np.where(y==1)]
        mu_0=np.mean(X_0,axis=0)
        mu_1=np.mean(X_1,axis=0)
        S_omega=X_0.T.dot(X_0)+X_1.T.dot(X_1)
        invS_omega=np.linalg.inv(S_omega)
        self.omega=invS_omega.dot(mu_0 - mu_1)
        self.omega_mu_0=self.omega.T.dot(mu_0)
        self.omega_mu_1=self.omega.T.dot(mu_1)
        pass

    # 书上没讲怎么判断分类
    # 采用距离度量，计算X到两个投影中心的L2距离，分类为距离更近的类别。
    def predict_proba(self,X):
        if self.omega is None:
            raise RuntimeError('cant predict before fit')
        n_samples = X.shape[0]
        extra = np.ones((n_samples,))
        X = np.c_[X, extra]
        omega_mu = X.dot(self.omega)
        d1=np.sqrt((omega_mu-self.omega_mu_1)**2)
        d0=np.sqrt((omega_mu-self.omega_mu_0)**2)
        prob_0=d1/(d0+d1)
        prob_1=1-prob_0
        return np.column_stack([prob_0, prob_1])

    def predict(self,X):
        p = self.predict_proba(X)
        res = np.argmax(p, axis=1)
        return res


if __name__=='__main__':
    np.random.seed(42)
    breast_data = load_breast_cancer()
    X, y = breast_data.data, breast_data.target
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lda = LDA()
    lda.fit(X_train, y_train)
    lda_prob = lda.predict_proba(X_test)
    lda_pred = lda.predict(X_test)
    #print('tinyml lda_prob:', lda_prob)
    #print('tinyml lda_pred:', lda_pred)
    print('tinyml accuracy:', len(y_test[y_test == lda_pred]) * 1. / len(y_test))


    sklearn_lda = discriminant_analysis.LinearDiscriminantAnalysis()
    sklearn_lda.fit(X_train,y_train)
    sklearn_prob=sklearn_lda.predict_proba(X_test)
    sklearn_pred=sklearn_lda.predict(X_test)
    #print('sklearn prob:',sklearn_prob)
    #print('sklearn pred:',sklearn_pred)
    print('sklearn accuracy:',len(y_test[y_test==sklearn_pred])*1./len(y_test))
