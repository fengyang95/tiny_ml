from sklearn import discriminant_analysis
import numpy as np

class LDA:
    def __init__(self):
        self.omiga=None
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
        S_omiga=X_0.T.dot(X_0)+X_1.T.dot(X_1)
        invS_omiga=np.linalg.inv(S_omiga)
        self.omiga=invS_omiga.dot(mu_0-mu_1)
        self.omiga_mu_0=self.omiga.T.dot(mu_0)
        self.omiga_mu_1=self.omiga.T.dot(mu_1)
        pass

    # 书上没讲怎么判断分类
    # 采用距离度量，计算X到两个投影中心的L2距离，分类为距离更近的类别。
    def predict_proba(self,X):
        if self.omiga is None:
            raise RuntimeError('cant predict before fit')
        n_samples = X.shape[0]
        extra = np.ones((n_samples,))
        X = np.c_[X, extra]
        omiga_mu = X.dot(self.omiga)
        d1=np.sqrt((omiga_mu-self.omiga_mu_1)**2)
        d0=np.sqrt((omiga_mu-self.omiga_mu_0)**2)
        prob_0=d1/(d0+d1)
        prob_1=1-prob_0
        return np.column_stack([prob_0, prob_1])

    def predict(self,X):
        p = self.predict_proba(X)
        res = np.argmax(p, axis=1)
        return res


if __name__=='__main__':

    X = np.array([[1.0, 0.5, 0.5], [1.0, 1.0, 0.3], [-0.1, 1.2, 0.5], [1.5, 2.4, 3.2], [1.3, 0.2, 1.4]])
    y = np.array([1, 0, 0, 1, 1])
    X_test = np.array([[1.3, 1, 3.2], [-1.2, 1.2, 0.8], [1, 2, 0.4], [1.2, 0.23, -0.5]])

    lda=LDA()
    lda.fit(X,y)
    lda_prob=lda.predict_proba(X_test)
    lda_pred=lda.predict(X_test)
    print('lda_prob:',lda_prob)
    print('lda_pred:',lda_pred)

    sklearn_lda = discriminant_analysis.LinearDiscriminantAnalysis()
    sklearn_lda.fit(X,y)
    sklearn_prob=sklearn_lda.predict_proba(X_test)
    sklearn_pred=sklearn_lda.predict(X_test)
    print('sklearn prob:',sklearn_prob)
    print('sklearn pred:',sklearn_pred)


