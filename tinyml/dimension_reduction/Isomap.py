import numpy as np
# 用Floyd_Warshall算法算出的dist和sklearn有差异
# MDS也有差异
class Isomap:
    def __init__(self,k=5,d_=2):
        self.d_=d_
        self.k=k
        self.dist_matrix_=None

    @staticmethod
    def Floyd_Warshall(Dist):
        m = Dist.shape[0]
        for k in range(m):
            for i in range(m):
                for j in range(m):
                    Dist[i, j] = min(Dist[i,j],Dist[i, k] + Dist[k, j])
        return Dist

    def fit(self,X):
        m = X.shape[0]
        Dist = np.zeros((m, m), dtype=np.float32)
        self.Omega = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            Dist[i, :] = np.sqrt(np.sum((X[i] - X) ** 2, axis=1))
            inf_index=np.argsort(Dist[i,:])[self.k+1:]
            Dist[i,inf_index]=float('inf')
        Dist=Isomap.Floyd_Warshall(Dist)
        self.dist_matrix_=Dist
        # 使用MDS中的步骤
        Dist_i2 = np.mean(Dist, axis=1).reshape(-1, 1)
        Dist_j2 = np.mean(Dist, axis=0).reshape(1, -1)
        dist_2 = np.mean(Dist)
        B_new = -0.5 * (Dist - Dist_i2 - Dist_j2 + dist_2)
        # 用eig和eigh函数分解出的结果符号位不同
        #values, vectors = np.linalg.eig(B_new)
        values,vectors=np.linalg.eigh(B_new)
        idx = np.argsort(values)[::-1]
        self.values_ = values[idx][:self.d_]
        # print('values:',self.values_)
        self.vectors_ = vectors[:, idx][:, :self.d_]
        self.Z = self.vectors_.dot(np.diag(np.sqrt(self.values_))).real


    def fit_transform(self,X):
        self.fit(X)
        return self.Z
        pass

if __name__=='__main__':
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    X=np.c_[X,X]
    isomap=Isomap(k=5,d_=2)
    Z=isomap.fit_transform(X)
    print('tinyml:')
    print(Z)

    import sklearn.manifold as manifold
    sklearn_Isomap=manifold.Isomap(n_neighbors=5, n_components=2,path_method='auto')
    Z2=sklearn_Isomap.fit_transform(X)
    print('sklearn')
    print(Z2)

    print('dist_matrix_diff:',np.sum((isomap.dist_matrix_-sklearn_Isomap.dist_matrix_)**2))
    print('Z diff:',np.sum((Z-Z2)**2))