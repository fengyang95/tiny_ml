import numpy as np
import abc
from sklearn import datasets,tree,ensemble
from sklearn.metrics import mean_squared_error
import xgboost as xgb
np.random.seed(1)

# 使用MSELoss
# 不考虑缺失值
# 只考虑连续值
# 参考 陈天奇 xgboost slides实现

class LossBase(object):
    def __init__(self,y_target,y_pred):
        self.y_target=y_target
        self.y_pred=y_pred
        pass

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def g(self):
        raise NotImplementedError

    @abc.abstractmethod
    def h(self):
        raise NotImplementedError

class MSELoss(LossBase):
    def __init__(self,y_target,y_pred):
        super(MSELoss,self).__init__(y_target,y_pred)

    def forward(self):
        return (self.y_target-self.y_pred)**2

    def g(self):
        return 2*(self.y_pred-self.y_target)

    def h(self):
        return 2*np.ones_like(self.y_target)

class CART:

    def __init__(self, reg_lambda=1, gamma=0., max_depth=3,col_sample_ratio=0.6,row_sample_ratio=1.):
        self.reg_lambda=reg_lambda
        self.gamma=gamma
        self.max_depth=max_depth
        self.tree = None
        self.leaf_nodes=0
        self.obj_val=0.
        self.col_sample_ratio=col_sample_ratio
        self.row_sample_ratio=row_sample_ratio

    def fit(self, X, y,g,h):
        D = {}
        D['X'] = X
        D['y'] = y
        A = np.arange(X.shape[1])
        m=len(y)
        self.tree = self.TreeGenerate(D,A,g,h,np.array(range(m)),0)
        self.obj_val=-0.5*self.obj_val+self.gamma*self.leaf_nodes

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

    def TreeGenerate(self, D, A,g,h,indices,depth):
        X = D['X']
        y = D['y']
        if depth>self.max_depth:
            G=np.sum(g[indices])
            H=np.sum(h[indices])
            w=-(G/(H+self.reg_lambda))
            self.obj_val+=(G**2/(H+self.reg_lambda))
            self.leaf_nodes+=1
            #print('w:',w)
            return w
        split_j=None
        split_s=None
        max_gain=0.

        col_sample_indices=np.random.choice(A,size=int(len(A)*self.col_sample_ratio))
        indices=np.random.choice(indices,size=int(len(indices)*self.row_sample_ratio))

        for j in A:
            if j not in col_sample_indices:
                continue
            for s in np.unique(X[:,j]):
                tmp_left=np.where(X[indices,j]<=s)[0]
                tmp_right=np.where(X[indices,j]>s)[0]
                if len(tmp_left)<1 or len(tmp_right)<1:
                    continue
                left_indices=indices[tmp_left]
                right_indices=indices[tmp_right]
                #print('left_indices:',left_indices)
                #print('right_indices:',right_indices)
                G_L=np.sum(g[left_indices])
                G_R=np.sum(g[right_indices])
                H_L=np.sum(h[left_indices])
                H_R=np.sum(h[right_indices])
                gain=  (G_L ** 2 / (H_L + self.reg_lambda) + G_R ** 2 / (H_R + self.reg_lambda) - (G_L + G_R) ** 2 / (H_L + H_R + self.reg_lambda)) - self.gamma
                if gain>max_gain:
                    split_j=j
                    split_s=s
        if split_j is None:
            G = np.sum(g[indices])
            H = np.sum(h[indices])
            w = -(G / (H + self.reg_lambda))
            self.obj_val += (G ** 2 / (H + self.reg_lambda))
            self.leaf_nodes += 1
            return w

        tree = {split_j: {}}
        left_indices=indices[np.where(X[indices,split_j]<=split_s)[0]]
        right_indices=indices[np.where(X[indices,split_j]>split_s)[0]]
        tree[split_j]['l'+str(split_s)]=self.TreeGenerate(D,A,g,h,left_indices,depth+1)
        tree[split_j]['r'+str(split_s)]=self.TreeGenerate(D,A,g,h,right_indices,depth+1)
        # 当前节点值
        tree[split_j]['val']= -(np.sum(g[indices]) / (np.sum(h[indices]) + self.reg_lambda))
        return tree


class XGBRegressor:
    def __init__(self, reg_lambda=1, gamma=0., max_depth=5, n_estimators=250, eta=.1):
        self.reg_lambda=reg_lambda
        self.gamma=gamma
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.eta=eta
        self.mean=None
        self.estimators_=[]

    def fit(self,X,y):
        self.mean=np.mean(y)
        #self.mean=0
        y_pred = np.ones_like(y)*self.mean
        loss = MSELoss(y, y_pred)
        g, h = loss.g(), loss.h()
        for t in range(self.n_estimators):
            estimator_t=CART(self.reg_lambda, self.gamma, self.max_depth)
            y_target=y-y_pred
            estimator_t.fit(X,y_target,g,h)
            #print(estimator_t.tree)
            #print('leaf_nodes:',estimator_t.leaf_nodes)
            #print('obj_val:',estimator_t.obj_val)
            self.estimators_.append(estimator_t)
            y_pred+=(self.eta*estimator_t.predict(X))
            loss=MSELoss(y,y_pred)
            print('t:',t,' loss:',np.mean(loss.forward()))
            g,h=loss.g(),loss.h()

    def predict(self,X):
        y_pred=np.ones((X.shape[0],))*self.mean
        for t in range(self.n_estimators):
            y_pred+=(self.eta*self.estimators_[t].predict(X))
        return y_pred

if __name__=='__main__':
    breast_data = datasets.load_boston()
    X, y = breast_data.data, breast_data.target

    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    sklearn_decisiontree_reg=tree.DecisionTreeRegressor(min_samples_split=15, min_samples_leaf=5,random_state=False)
    sklearn_decisiontree_reg.fit(X_train, y_train)
    decisiontree_pred=sklearn_decisiontree_reg.predict(X_test)
    print('base estimator:',mean_squared_error(y_test,decisiontree_pred))

    tinyml_gbdt_reg=XGBRegressor(n_estimators=100,max_depth=10,gamma=0.)
    tinyml_gbdt_reg.fit(X_train, y_train)
    y_pred=tinyml_gbdt_reg.predict(X_test)
    print('tinyml mse:',mean_squared_error(y_test,y_pred))

    xgb_reg=xgb.sklearn.XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=100,gamma=0,reg_lambda=1)
    xgb_reg.fit(X_train,y_train)
    xgb_pred=xgb_reg.predict(X_test)
    print('xgb  mse:',mean_squared_error(y_test,xgb_pred))

