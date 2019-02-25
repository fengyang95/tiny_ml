import numpy as np
from tinyml.tree import treePlotter
import sklearn.datasets as datasets
from sklearn.metrics import mean_squared_error
import sklearn.tree as tree
import graphviz

class DecisionTreeRegressor:
    """
    《统计学习方法》 p69 最小二乘回归树
    """
    def __init__(self, min_samples_split=3,min_samples_leaf=1,random_state=False):
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.random=random_state
        self.tree = None

    def fit(self, X, y):
        D = {}
        D['X'] = X
        D['y'] = y
        A = np.arange(X.shape[1])
        self.tree = self.TreeGenerate(D, A)

    def predict(self, X):
        if self.tree is None:
            raise RuntimeError('cant predict before fit')
        y_pred = []
        for i in range(X.shape[0]):
            tree = self.tree
            x = X[i]
            while True:
                if not isinstance(tree, dict):
                    y_pred.append(tree)
                    break
                a = list(tree.keys())[0]
                tree = tree[a]
                if isinstance(tree, dict):
                    val = x[a]
                    split_val=float(list(tree.keys())[0][1:])
                    if val<=split_val:
                        tree=tree[list(tree.keys())[0]]
                    else:
                        tree=tree[list(tree.keys())[1]]
                else:
                    y_pred.append(tree)
                    break
        return np.array(y_pred)

    def TreeGenerate(self, D, A):
        X = D['X']
        y = D['y']
        if len(y)<=self.min_samples_split:
            return np.mean(y)
        split_j=None
        split_s=None
        min_val=1.e10
        select_A=A
        if self.random is True:
            d=len(A)
            select_A=np.random.choice(A,size=int(d//2),replace=False)
        for j in select_A:
            for s in np.unique(X[:,j]):
                left_indices=np.where(X[:,j]<=s)[0]
                right_indices=np.where(X[:,j]>s)[0]
                if len(left_indices)<self.min_samples_leaf or len(right_indices)<self.min_samples_leaf:
                    continue
                val=np.sum((y[left_indices]-np.mean(y[left_indices]))**2)+np.sum((y[right_indices]-np.mean(y[right_indices]))**2)
                if val<min_val:
                    split_j=j
                    split_s=s
                    min_val=val

        if split_j is None:
            return np.mean(y)
        tree = {split_j: {}}
        left_indices=np.where(X[:,split_j]<=split_s)[0]
        right_indices=np.where(X[:,split_j]>split_s)[0]
        D_left, D_right = {},{}
        D_left['X'],D_left['y'] = X[left_indices],y[left_indices]
        D_right['X'],D_right['y']=X[right_indices],y[right_indices]
        tree[split_j]['l'+str(split_s)]=self.TreeGenerate(D_left,A)
        tree[split_j]['r'+str(split_s)]=self.TreeGenerate(D_right,A)
        # 当前节点值
        tree[split_j]['val']=np.mean(y)
        return tree


if __name__=='__main__':
    breast_data = datasets.load_boston()
    X, y = breast_data.data, breast_data.target
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]


    decisiontree_reg=DecisionTreeRegressor(min_samples_split=20,min_samples_leaf=5)
    decisiontree_reg.fit(X_train,y_train)
    print(decisiontree_reg.tree)
    treePlotter.createPlot(decisiontree_reg.tree)
    y_pred=decisiontree_reg.predict(X_test)
    print('tinyml mse:',mean_squared_error(y_test,y_pred))


    sklearn_reg=tree.DecisionTreeRegressor(min_samples_split=20,min_samples_leaf=5,random_state=False)
    sklearn_reg.fit(X_train,y_train)
    print(sklearn_reg.feature_importances_)
    sklearn_pred=sklearn_reg.predict(X_test)
    print('sklearn mse:',mean_squared_error(y_test,sklearn_pred))
    dot_data=tree.export_graphviz(sklearn_reg,out_file=None)
    graph=graphviz.Source(dot_data)
