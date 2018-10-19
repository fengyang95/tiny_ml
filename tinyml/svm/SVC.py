import numpy as np

"""
[知乎专栏：支持向量机(SVM)——SMO算法](https://zhuanlan.zhihu.com/p/32152421)
[SMO算法](https://blog.csdn.net/c406495762/article/details/78072313#2-smo%E7%AE%97%E6%B3%95)
"""
np.random.seed(1)
class SVC:
    def __init__(self,max_iter=100,C=1,kernel='rbf',sigma=1):
        self.b=0.
        self.alpha=None
        self.max_iter=max_iter
        self.C=C
        self.kernel=kernel
        self.K=None
        self.X=None
        self.y=None
        if kernel=='rbf':
            self.sigma=sigma
        pass

    def kernel_func(self,kernel,x1,x2):
        if kernel=='linear':
            return x1.dot(x2.T)
        elif kernel=='rbf':
            return np.exp(-(np.sum((x1-x2)**2))/(2*self.sigma*self.sigma))


    def computeK(self,X,kernel):
        m=X.shape[0]
        K=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i<=j:
                    K[i,j]=self.kernel_func(kernel,X[i],X[j])
                else:
                    K[i,j]=K[j,i]
        return K

    def compute_u(self,X,y):
        u = np.zeros((X.shape[0],))
        for j in range(X.shape[0]):
            u[j]=np.sum(y*self.alpha*self.K[:,j])+self.b
        return u

    def checkKKT(self,u,y,i):
        if self.alpha[i]<self.C and y[i]*u[i]<=1:
            return False
        if self.alpha[i]>0 and y[i]*u[i]>=1:
            return False
        if (self.alpha[i]==0 or self.alpha[i]==self.C) and y[i]*u[i]==1:
            return False
        return True


    def fit(self,X,y):
        self.X=X
        self.y=y
        self.K=self.computeK(X,self.kernel)
        self.alpha=np.random.random((X.shape[0],))
        self.omiga=np.zeros((X.shape[0],))

        for _ in range(self.max_iter):
            u = self.compute_u(X, y)
            finish=True
            for i in range(X.shape[0]):
                if not self.checkKKT(u,y,i):
                    finish=False
                    y_indices=np.delete(np.arange(X.shape[0]),i)
                    j=y_indices[int(np.random.random()*len(y_indices))]
                    E_i=np.sum(self.alpha*y*self.K[:,i])+self.b-y[i]
                    E_j=np.sum(self.alpha*y*self.K[:,j])+self.b-y[j]
                    if y[i]!=y[j]:
                        L=max(0,self.alpha[j]-self.alpha[i])
                        H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
                    else:
                        L=max(0,self.alpha[j]+self.alpha[i]-self.C)
                        H=min(self.C,self.alpha[j]+self.alpha[i])
                    eta=self.K[i,i]+self.K[j,j]-2*self.K[i,j]
                    alpha2_new_unc=self.alpha[j]+y[j]*(E_i-E_j)/eta
                    alpha2_old=self.alpha[j]
                    alpha1_old=self.alpha[i]
                    if alpha2_new_unc>H:
                        self.alpha[j]=H
                    elif alpha2_new_unc<L:
                        self.alpha[j]=L
                    else:
                        self.alpha[j]=alpha2_new_unc
                    self.alpha[i]=alpha1_old+y[i]*y[j]*(alpha2_old-self.alpha[j])
                    b1_new=-E_i-y[i]*self.K[i,i]*(self.alpha[i]-alpha1_old)-y[j]*self.K[j,i]*(self.alpha[j]-alpha2_old)+self.b
                    b2_new=-E_j-y[i]*self.K[i,j]*(self.alpha[i]-alpha1_old)-y[j]*self.K[j,j]*(self.alpha[j]-alpha2_old)+self.b
                    if self.alpha[i]>0 and self.alpha[i]<self.C:
                        self.b=b1_new
                    elif self.alpha[j]>0 and self.alpha[j]<self.C:
                        self.b=b2_new
                    else:
                        self.b=(b1_new+b2_new)/2
            if finish:
                break



    def predict(self,X):
        y_preds=[]
        for i in range(X.shape[0]):
            K=np.zeros((len(self.y),))
            support_indices=np.where(self.alpha>0)[0]
            for j in support_indices:
                K[j]=self.kernel_func(self.kernel,self.X[j],X[i])
            y_pred=np.sum(self.y[support_indices]*self.alpha[support_indices]*K[support_indices].T)
            y_pred+=self.b
            y_preds.append(y_pred)
        return np.array(y_preds)


if __name__=='__main__':

    # 测试 线性核
    X = np.array([[2, -1], [3, -2], [1, 0], [0,1],[-2,1],[-1.3,0.3],[-0.2,-0.8],[2.3,-3.3],[-2,-4],[7,8]])
    y = np.array([1, 1, 1, 1,-1,-1,-1,-1,-1,1])
    svc=SVC(max_iter=100,kernel='linear',C=10000)

    """
    # 测试rbf核
    X=np.array([[1,0],[-1,0],[0,-1],[0,1],[2,np.sqrt(5)],[2,-np.sqrt(5)],[-2,np.sqrt(5)],[-2,-np.sqrt(5)],[300,400]])
    y=np.array([-1,-1,-1,-1,1,1,1,1,1])
    svc=SVC(max_iter=100,kernel='rbf',C=1)
    """
    svc.fit(X,y)
    print('alpha:',svc.alpha)
    print('b:',svc.b)
    pred_y=svc.predict(np.array([[1,0],[-0.2,-0.1],[0,1]]))
    print('pred_y1:',pred_y)
    pred_y=np.sign(pred_y)
    print('pred_y:',pred_y)




