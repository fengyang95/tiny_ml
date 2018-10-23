# 只针对2分类
# 自己实现的DecisionTreeClassifier没有实现 sample_weight参数
# 重点在AdaBoost， 使用sklearn的DecisionTreeClassifier作为基学习器
import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as sklearnAdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.datasets as datasets
class AdaBoostClassifier:
    def __init__(self, base_estimator=LogisticRegression(solver='newton-cg'), n_estimators=300):
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.hs_=[]
        self.epsilons_=[]
        self.alphas_=[]
        self.Ds_=[]

    def fit(self,X,y):
        m=X.shape[0]
        self.Ds_.append(np.ones((m,))/m)
        for t in range(self.n_estimators):
            ht=self.base_estimator
            ht.fit(X, y,self.Ds_[t])
            y_pred = ht.predict(X).astype(np.int32)
            valid_indices=(y!=y_pred)
            mask=np.ones((len(y),))
            mask[valid_indices]=0
            epsilon_t =1-np.sum(self.Ds_[t]*mask)
            if epsilon_t>0.5:
                break
            self.hs_.append(copy.copy(ht))
            self.epsilons_.append(epsilon_t)
            alpha_t=0.5*np.log((1-epsilon_t)/epsilon_t)
            self.alphas_.append(alpha_t)
            self.Ds_.append(self.Ds_[t]*np.exp(-alpha_t*y*y_pred))
            self.Ds_[t + 1] = self.Ds_[t + 1] / np.sum(self.Ds_[t + 1])
        self.alphas_=np.array(self.alphas_)
        self.Ds_=np.array(self.Ds_)
        self.epsilons_=np.array(self.epsilons_)

    @classmethod
    def calc_epsilon(clf,D,y_target,y_pred):
        return 1-np.sum(D[y_target==y_pred])

    def predict(self,X):
        #H=np.zeros((X.shape[0],))
        #for t in range(len(self.alphas_)):
        #    print('pred:',self.hs_[t].predict(X))
        #    H+=(self.alphas_[t]*self.hs_[t].predict(X))
        #return np.sign(H)
        self.classes_=np.array([-1,1])
        classes = self.classes_[:, np.newaxis]
        pred = None

        pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.hs_,
                                               2*self.alphas_))
        pred /= self.alphas_.sum()
        pred[:, 0] *= -1
        pred=pred.sum(axis=1)
        return self.classes_.take(pred > 0, axis=0)


if __name__=='__main__':

    breast_data=datasets.load_breast_cancer()
    X,y=breast_data.data,breast_data.target
    y=2*y-1
    X_train,y_train=X[:200],y[:200]
    X_test,y_test=X[200:],y[200:]

    sklearn_decision_tree=LogisticRegression(solver='newton-cg')
    sklearn_decision_tree.fit(X_train,y_train)
    y_pred_decison_tree=sklearn_decision_tree.predict(X_test)
    print('single decision tree:',len(y_test[y_pred_decison_tree==y_test])*1.0/len(y_test))

    print('tinyml:')
    adaboost_clf = AdaBoostClassifier(n_estimators=10)
    adaboost_clf.fit(X_train,y_train)
    print('alpha:',adaboost_clf.alphas_)
    print('epsilon:',adaboost_clf.epsilons_)
    y_pred = adaboost_clf.predict(X_test)
    print('adaboost y_pred:', len(y_test[y_pred==y_test])*1./len(y))

    print('sklearn:')
    sklearn_adboost_clf=sklearnAdaBoostClassifier(n_estimators=100,random_state=False,algorithm='SAMME',base_estimator=LogisticRegression(solver='newton-cg'))
    sklearn_adboost_clf.fit(X_train,y_train)
    print('alpha:',sklearn_adboost_clf.estimator_weights_)
    print('epsilon:',sklearn_adboost_clf.estimator_errors_)
    sklearn_y_pred=sklearn_adboost_clf.predict(X_test)
    print('sklearn adaboost y_pred:',len(y_test[y_test==sklearn_y_pred])*1./len(y_test))