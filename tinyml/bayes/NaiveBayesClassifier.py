import numpy as np
# 只考虑离散值
class NaiveBayesClassifier:
    def __init__(self,n_classes=2):
        self.n_classes=n_classes
        self.priori_P={}
        self.conditional_P={}
        self.N={}
        pass

    def fit(self,X,y):
        for i in range(self.n_classes):
            # 公式 7.19
            self.priori_P[i]=(len(y[y==i])+1)/(len(y)+self.n_classes)
        for col in range(X.shape[1]):
            self.N[col]=len(np.unique(X[:,col]))
            self.conditional_P[col]={}
            for row in range(X.shape[0]):
                val=X[row,col]
                if val not in self.conditional_P[col].keys():
                    self.conditional_P[col][val]={}
                    for i in range(self.n_classes):
                        D_xi=np.where(X[:,col]==val)
                        D_c=np.where(y==i)
                        D_cxi=len(np.intersect1d(D_xi,D_c))
                        # 公式 7.20
                        self.conditional_P[col][val][i]=(D_cxi+1)/(len(y[y==i])+self.N[col])
                else:
                    continue

    def predict(self,X):
        pred_y=[]
        for i in range(len(X)):
            p=np.ones((self.n_classes,))
            for j in range(self.n_classes):
                p[j]=self.priori_P[j]
            for col in range(X.shape[1]):
                val=X[i,col]
                for j in range(self.n_classes):
                    p[j]*=self.conditional_P[col][val][j]
            pred_y.append(np.argmax(p))
        return pred_y

if __name__=='__main__':
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                [2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1],
                                [1, 1, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0],
                                [1, 1, 1, 1, 1, 0], [0, 2, 2, 0, 2, 1],
                                [2, 2, 2, 2, 2, 0], [2, 0, 0, 2, 2, 1],
                                [0, 1, 0, 1, 0, 0], [2, 1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                                [0, 0, 1, 1, 1, 0]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    X_test=np.array([[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                     [0, 0, 1, 1, 1, 0],
                     [2, 0, 0, 2, 2, 0],
                     [0, 0, 1, 1, 1, 0]
                     ])

    naive_bayes=NaiveBayesClassifier(n_classes=2)
    naive_bayes.fit(X,y)
    print('self.PrirP:',naive_bayes.priori_P)
    print('self.CondiP:',naive_bayes.conditional_P)
    pred_y=naive_bayes.predict(X_test)
    print('pred_y:',pred_y)


