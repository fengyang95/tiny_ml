from tinyml.dimension_reduction.PCA import PCA as tinymlPCA
from tinyml.dimension_reduction.KernelPCA import KernelPCA as tinymlKernalPCA
from tinyml.dimension_reduction.LLE import LLE as tinymlLLE
from tinyml.dimension_reduction.Isomap import Isomap as tinymlIsomap
from tinyml.dimension_reduction.MDS import MDS as tinymlMDS

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKernalPCA
from sklearn.manifold import LocallyLinearEmbedding as sklearnLLE
from sklearn.manifold import MDS as sklearnMDS
from sklearn.manifold import Isomap as sklearnIsomap

from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    from sklearn.datasets import make_s_curve

    X, y = make_s_curve(n_samples=500,
                        noise=0.1,
                        random_state=0)

    # PCA
    tinyml_pca = tinymlPCA(d_=2)
    X_=tinyml_pca.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('tinyml PCA')
    plt.savefig('./dimension_reduction_result/tinyml_PCA.jpg')
    plt.show()

    sklearn_pca=sklearnPCA(n_components=2,svd_solver='full')
    X_=sklearn_pca.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('sklearn PCA')
    plt.savefig('./dimension_reduction_result/sklearn_PCA.jpg')
    plt.show()

    # KPCA
    tinyml_kpca = tinymlKernalPCA(d_=2, kernel='rbf',gamma=0.5)
    X_ = tinyml_kpca.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('tinyml KernalPCA')
    plt.savefig('./dimension_reduction_result/tinyml_KernalPCA.jpg')
    plt.show()

    sklearn_kpca = sklearnKernalPCA(n_components=2, kernel='rbf', gamma=0.5)
    X_ = sklearn_kpca.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('sklearn KernalPCA')
    plt.savefig('./dimension_reduction_result/sklearn_KernalPCA.jpg')
    plt.show()

    # LLE

    tinyml_lle = tinymlLLE(d_=2, k=30,reg=1e-3)
    X_ = tinyml_lle.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('tinyml LLE')
    plt.savefig('./dimension_reduction_result/tinyml_LLE.jpg')
    plt.show()

    sklearn_lle= sklearnLLE(n_components=2,n_neighbors=30,reg=1e-3)
    X_ = sklearn_lle.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('sklearn LLE')
    plt.savefig('./dimension_reduction_result/sklearn_LLE.jpg')
    plt.show()

    # MDS

    tinyml_mds = tinymlMDS(d_=2)
    X_ = tinyml_mds.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('tinyml MDS')
    plt.savefig('./dimension_reduction_result/tinyml_MDS.jpg')
    plt.show()

    sklearn_mds = sklearnMDS(n_components=2,metric=True,random_state=False)
    X_ = sklearn_mds.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('sklearn MDS')
    plt.savefig('./dimension_reduction_result/sklearn_MDS.jpg')
    plt.show()

    """
    # Isomap
    tinyml_isomap = tinymlIsomap(k=5,d_=2)
    X_ = tinyml_isomap.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('tinyml Isomap')
    plt.savefig('./dimension_reduction_result/tinyml_Isomap.jpg')
    plt.show()

    sklearn_isomap = sklearnIsomap(n_neighbors=5, n_components=2,path_method='auto')
    X_ = sklearn_isomap.fit_transform(X)
    plt.scatter(X_[:, 0], X_[:, 1], c=y)
    plt.title('sklearn Isomap')
    plt.savefig('./dimension_reduction_result/sklearn_Isomap.jpg')
    plt.show()
    """









