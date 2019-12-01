from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_eval(data,classifier):
    train_X, train_y, test_X, test_y=data
    classifier.fit(train_X,train_y)
    preds_y=classifier.predict(test_X)
    accuracy=len(preds_y[preds_y==test_y])/len(preds_y)
    return accuracy

from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.tree as tree

from tinyml.bayes.NaiveBayesClassifier import NaiveBayesClassifierContinuous as tinymlNaiveBayesClassifier
from tinyml.discriminant_analysis.LDA import LDA as tinymlLDA
from tinyml.discriminant_analysis.GDA import GDA as tinymlGDA
from tinyml.ensemble.AdaBoostClassifier import AdaBoostClassifier as tinymlAdaboostClassifier
from tinyml.linear_model.LogisticRegression import LogisticRegression as tinymlLogisticRegression
from tinyml.svm.SVC import SVC as tinymlSVC
from tinyml.tree.DecisionTreeClassifier import DecisionTreeClassifier as tinymlDecsionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier as sklearnAdaboostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearnLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as sklearnGDA
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB as sklearnNaiveBayes
from sklearn.tree import DecisionTreeClassifier as sklearnDecisionTreeClassifier
if __name__=='__main__':


    X, y=load_breast_cancer(return_X_y=True)
    print(X[:,7:])
    X=X[:,:7]
    X = MinMaxScaler().fit_transform(X)

    #y=(2*y-1).astype(np.int)
    n_classes=2

    train_X, test_X, train_y, test_y=train_test_split(X, y, test_size=0.3,random_state=0)

    data= train_X, train_y, test_X, test_y

    acc_tinyml_naivebayes=train_and_eval(data,tinymlNaiveBayesClassifier(n_classes=n_classes))
    print('tinyml accuracy NaiveBayes:',acc_tinyml_naivebayes)
    acc_sklearn_naivebayes=train_and_eval(data,sklearnNaiveBayes())
    print('sklearn accuracy NaiveBayes:',acc_sklearn_naivebayes)



    acc_tinyml_adaboost_classifier=train_and_eval((train_X,(train_y*2-1).astype(np.int),
                                                   test_X,(test_y*2-1).astype(np.int)),tinymlAdaboostClassifier(n_estimators=100,base_estimator=tree.DecisionTreeClassifier(max_depth=1,random_state=False),method='re-weighting'))
    print('tinyml accuracy AdaboostClassifier:',acc_tinyml_adaboost_classifier)
    acc_sklearn_adaboost_classifier=train_and_eval(data,sklearnAdaboostClassifier(n_estimators=100, random_state=False, algorithm='SAMME',
                                                    base_estimator=tree.DecisionTreeClassifier(max_depth=1,random_state=False)))
    print('sklearn accuracy AdaboostClassifier:',acc_sklearn_adaboost_classifier)

    acc_tinyml_lda_classifier=train_and_eval(data,tinymlLDA())
    print('tinyml accuracy LDA:',acc_tinyml_lda_classifier)
    acc_sklearn_lda_classifier=train_and_eval(data,sklearnLDA())
    print('sklearn accuracy LDA:',acc_sklearn_lda_classifier)

    acc_tinyml_gda_classifier=train_and_eval(data,tinymlGDA())
    print('tinyml accuracy GDA:',acc_tinyml_gda_classifier)
    acc_sklearn_gda_classifier=train_and_eval(data,sklearnGDA())
    print('sklearn accuracy QDA:',acc_sklearn_gda_classifier)

    acc_tinyml_logistic=train_and_eval(data,tinymlLogisticRegression(max_iter=100,use_matrix=False))
    print('tinyml accuracy Logistic:',acc_tinyml_logistic)
    acc_sklearn_logistic=train_and_eval(data,sklearnLogisticRegression(max_iter=100,solver='newton-cg'))
    print('sklearn acccuracy Logistic:',acc_sklearn_logistic)

    tinyml_svc=tinymlSVC(max_iter=100,kernel='rbf',C=1)
    tinyml_svc.fit(train_X, (train_y*2-1).astype(int))
    preds_y = np.sign(tinyml_svc.predict(test_X))
    acc_tinyml_SVC = len(preds_y[preds_y == (2*test_y-1).astype(np.int)]) / len(preds_y)
    print('tinyml accuracy SVC:',acc_tinyml_SVC)

    acc_sklearn_SVC=train_and_eval(data,SVC(kernel='rbf'))
    print('sklearn accuracy SVC:',acc_sklearn_SVC)

    """
    acc_tinyml_decision_tree_classifier=train_and_eval(data,tinymlDecsionTreeClassifier(tree_type='ID3',k_classes=2))
    print('tinyml accuracy decision tree:',acc_tinyml_decision_tree_classifier)
    acc_sklearn_decision_tree_classifier=train_and_eval(data,sklearnDecisionTreeClassifier())
    print('sklearn accuracy decison tree:',acc_sklearn_decision_tree_classifier) 
    """
