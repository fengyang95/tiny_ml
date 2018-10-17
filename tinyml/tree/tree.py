import numpy as np

# 只处理离散值，不考虑缺失值
from tinyml.tree.treePlotter import createPlot
np.random.seed(100)
class DecisionTree:
    def __init__(self,tree_type='ID3',k_classes=2):
        self.tree_type=tree_type
        self.k_classes=k_classes
        if tree_type=='ID3':
            self.gain_func=self.Gain
        elif tree_type=='CART':
            self.gain_func=self.GiniIndex
        self.tree=None

    def fit(self,X,y):
        D={}
        D['X']=X
        D['y']=y
        A=np.arange(X.shape[1])
        aVs={}
        for a in A:
            aVs[a]=np.unique(X[:,a])
        self.tree=self.TreeGenerate(D,A,aVs)

    def predict(self,X):
        if self.tree is None:
            raise RuntimeError('cant predict before fit')
        y_pred=[]
        for i in range(X.shape[0]):
            tree = self.tree
            x=X[i]
            while True:
                if not isinstance(tree,dict):
                    y_pred.append(tree)
                    break
                a=list(tree.keys())[0]
                tree=tree[a]
                if isinstance(tree,dict):
                    val = x[a]
                    tree = tree[val]
                else:
                    y_pred.append(tree)
                    break
        return np.array(y_pred)


    def TreeGenerate(self,D,A,aVs):
        X=D['X']
        y=D['y']
        # 情形1
        unique_classes=np.unique(y)
        if len(unique_classes)==1:
            return unique_classes[0]
        flag=True
        for a in A:
            if(len(np.unique(X[:,a]))>1):
                flag=False
                break
        # 情形2
        if flag:
            return np.argmax(np.bincount(y))

        gains=np.zeros((len(A),))
        for i in range(len(A)):
            gains[i]=self.gain_func(D,A[i])
        #print(gains)
        if self.tree_type=='CART':
            a_best=A[np.argmin(gains)]
        elif self.tree_type=='ID3':
            a_best=A[np.argmax(gains)]
        subA=np.delete(A,np.argmax(gains))
        tree={a_best:{}}

        for av in aVs[a_best]:
            indices=np.where(X[:,a_best]==av)
            Dv={}
            Dv['X']=X[indices]
            Dv['y']=y[indices]
            if len(Dv['y'])==0:
                tree[a_best][av]=np.argmax(np.bincount(y))
            else:
                tree[a_best][av]=self.TreeGenerate(Dv,subA,aVs)
        return tree


    # 《机器学习》 公式4.1 信息熵
    @classmethod
    def Ent(cls,D):
        y=D['y']
        bin_count=np.bincount(y)
        total=len(y)
        ent=0.
        for k in range(len(bin_count)):
            p_k=bin_count[k]/total
            if p_k!=0:
                 ent+=p_k*np.log2(p_k)
        return -ent

    # 《机器学习》 公式4.2 信息增益
    # a表示属性 列 index
    @classmethod
    def Gain(cls,D,a):
        X=D['X']
        y=D['y']
        aV=np.unique(X[:,a])
        sum=0.
        for v in range(len(aV)):
            Dv={}
            indices=np.where(X[:,a]==aV[v])
            Dv['X']=X[indices]
            Dv['y']=y[indices]
            ent=cls.Ent(Dv)
            sum+=(len(Dv['y'])/len(y)*ent)
        gain=cls.Ent(D)-sum
        return gain

    # 《机器学习》 公式4.5
    @classmethod
    def Gini(cls,D):
        y = D['y']
        bin_count = np.bincount(y)
        total = len(y)
        ent = 0.
        for k in range(len(bin_count)):
            p_k = bin_count[k] / total
            ent+=p_k**2
        return 1-ent

    # 公式4.6
    @classmethod
    def GiniIndex(cls,D,a):
        X = D['X']
        y = D['y']
        aV = np.unique(X[:, a])
        sum = 0.
        for v in range(len(aV)):
            Dv = {}
            indices = np.where(X[:, a] == aV[v])
            Dv['X'] = X[indices]
            Dv['y'] = y[indices]
            ent = cls.Gini(Dv)
            sum += (len(Dv['y']) / len(y) * ent)
        gain = sum
        return gain






if __name__=='__main__':
    watermelon_data = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                [2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1],
                                [1, 1, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0],
                                [1, 1, 1, 1, 1, 0], [0, 2, 2, 0, 2, 1],
                                [2, 2, 2, 2, 2, 0], [2, 0, 0, 2, 2, 1],
                                [0, 1, 0, 1, 0, 0], [2, 1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                                [0, 0, 1, 1, 1, 0]])
    label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #train_indices=np.array([0,1,2,5,6,9,13,14,15,16])
    #watermelon_data=watermelon_data[train_indices]
    #label=label[train_indices]

    X_test=np.array([[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                        [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                     [0, 0, 1, 1, 1, 0]])

    decision_clf=DecisionTree(tree_type='CART')
    decision_clf.fit(watermelon_data,label)
    print(decision_clf.tree)
    createPlot(decision_clf.tree)

    y_pred=decision_clf.predict(X_test)
    print('y_pred:',y_pred)









