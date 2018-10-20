import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(10)
class LVQ:
    def __init__(self,t,eta=0.1,max_iter=400):
        # t[i]表示第i个原型向量对应的类别
        self.t=t
        # p[i]表示第i个原型向量的值
        self.p=None
        self.c=len(np.unique(t))
        self.q=len(t)
        self.eta=eta
        self.max_iter=max_iter
        self.C=None
        self.labels_=None

    # p205 图9.4 学习向量量化算法
    def fit(self,X,y):
        C={}
        for i in range(self.q):
            C[i]=[]
        self.p=np.zeros((len(self.t),X.shape[1]))

        # 初始化原型向量 从p类别标记对应的X中随机选择1个作为初始原型向量
        for i in range(self.q):
            candidate_indices=np.where(y==self.t[i])[0]
            target_indice=random.sample(list(candidate_indices),1)
            self.p[i]=X[target_indice]
        """
        # 书上p的选取
        indices=[4,11,17,22,28]
        self.p=X[indices]
        """
        for _ in range(self.max_iter):
            j=random.sample(list(range(len(y))),1)
            d=np.sqrt(np.sum((X[j]-self.p)**2,axis=1))
            i_=np.argmin(d)
            old_p=self.p
            if y[j]==t[i_]:
                self.p[i_]=self.p[i_]+self.eta*(X[j]-self.p[i_])
            else:
                self.p[i_]=self.p[i_]-self.eta*(X[j]-self.p[i_])

        for j in range(X.shape[0]):
            d=np.sqrt(np.sum((X[j]-self.p)**2,axis=1))
            i_=np.argmin(d)
            C[i_].append(j)
        self.C=C
        self.labels_ = np.zeros((X.shape[0],), dtype=np.int32)
        for i in range(self.q):
            self.labels_[C[i]] = i


    def predict(self,X):
        preds_y=[]
        for j in range(X.shape[0]):
            d=np.sqrt(np.sum((X[j]-self.p)**2,axis=1))
            i_=np.argmin(d)
            preds_y.append(self.t[i_])
        return np.array(preds_y)


if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])
    y=np.zeros((X.shape[0],),dtype=np.int32)
    y[range(9,21)]=1
    t=np.array([0,1,1,0,0],dtype=np.int32)

    print(y)
    X_test=X
    lvq=LVQ(t)
    lvq.fit(X,y)
    print(lvq.C)
    print(lvq.labels_)
    print(lvq.predict(X))
    plt.scatter(X[np.where(lvq.labels_ == 0), 0], X[np.where(lvq.labels_ == 0), 1], c='r')
    plt.scatter(lvq.p[0:1, 0], lvq.p[0:1, 1], c='r', marker='+')
    plt.scatter(X[np.where(lvq.labels_ == 1), 0], X[np.where(lvq.labels_ == 1), 1], c='g')
    plt.scatter(lvq.p[1:2, 0], lvq.p[1:2, 1], c='g', marker='+')
    plt.scatter(X[np.where(lvq.labels_ == 2), 0], X[np.where(lvq.labels_ == 2), 1], c='b')
    plt.scatter(lvq.p[2:3, 0], lvq.p[2:3, 1], c='b', marker='+')
    plt.scatter(X[np.where(lvq.labels_==3),0],X[np.where(lvq.labels_==3),1],c='m')
    plt.scatter(lvq.p[3:4,0],lvq.p[3:4,1],c='m',marker='+')
    plt.scatter(X[np.where(lvq.labels_ == 4), 0], X[np.where(lvq.labels_ == 4), 1], c='y')
    plt.scatter(lvq.p[4:5, 0], lvq.p[4:5, 1], c='y',marker='+')
    plt.show()

