from tinyml.cluster.KMeans import KMeans as tinymlKMeans
from tinyml.cluster.AGNES import AGNES as tinymlAGNES
from tinyml.cluster.DBSCAN import DBSCAN as tinymlDBSCAN
from tinyml.cluster.GaussianMixture import GaussianMixture as tinymlGaussianMixture
from tinyml.cluster.LVQ import LVQ as tinymlLVQ

from sklearn.cluster.hierarchical import AgglomerativeClustering as sklearnAGNES
from sklearn.cluster import DBSCAN as sklearnDBSCAN
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.mixture import GaussianMixture as sklearnGaussianMixture

import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    # p202 西瓜数据集4.0
    X = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
                  [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
                  [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                  [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
                  [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
                  [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

    # KMeans
    tinyml_kmeans = tinymlKMeans(k=3)
    tinyml_kmeans.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=tinyml_kmeans.labels_)
    plt.scatter(tinyml_kmeans.mu[:, 0], tinyml_kmeans.mu[:, 1], c=range(tinyml_kmeans.k), marker='+')
    plt.title('tinyml KMeans')
    plt.savefig('./cluster_result/tinyml_KMeans.jpg')
    plt.show()

    sklearn_kmeans = sklearnKMeans(n_clusters=3)
    sklearn_kmeans.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)
    plt.scatter(sklearn_kmeans.cluster_centers_[:,0],sklearn_kmeans.cluster_centers_[:,1],c=range(sklearn_kmeans.n_clusters),marker='+')
    plt.title('sklearn KMeans')
    plt.savefig('./cluster_result/sklearn_KMeans.jpg')
    plt.show()

    # DBSCAN
    tinyml_dbscan = tinymlDBSCAN()
    tinyml_dbscan.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=tinyml_dbscan.labels_)
    plt.title('tinyml DBSCAN')
    plt.savefig('./cluster_result/tinyml_DBSCAN.jpg')
    plt.show()

    sklearn_DBSCAN =sklearnDBSCAN(eps=0.11, min_samples=5, metric='l2')
    sklearn_DBSCAN.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_DBSCAN.labels_)
    plt.title('sklearn DBSCAN')
    plt.savefig('./cluster_result/sklearn_DBSCAN.jpg')
    plt.show()

    # GMM
    tinyml_gmm = tinymlGaussianMixture(k=3, max_iter=50)
    tinyml_gmm.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=tinyml_gmm.labels_)
    plt.scatter(tinyml_gmm.mu[:, 0], tinyml_gmm.mu[:, 1], c=range(tinyml_gmm.k), marker='+')
    plt.title('tinyml GMM')
    plt.savefig('./cluster_result/tinyml_GMM.jpg')
    plt.show()

    sklearn_gmm = sklearnGaussianMixture(n_components=3, covariance_type='full',
                                  max_iter=50).fit(X)
    labels = sklearn_gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(sklearn_gmm.means_[:,0],sklearn_gmm.means_[:,1],c=range(sklearn_gmm.n_components),marker='+')
    plt.title('sklearn GMM')
    plt.savefig('./cluster_result/sklearn_GMM.jpg')
    plt.show()

    # AGNES
    tinyml_agnes = tinymlAGNES(k=3)
    tinyml_agnes.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=tinyml_agnes.labels_)
    plt.title('tinyml AGNES')
    plt.savefig('./cluster_result/tinyml_AGNES.jpg')
    plt.show()

    sklearn_agnes = sklearnAGNES(n_clusters=3, affinity='l2', linkage='average')
    sklearn_agnes.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_agnes.labels_)
    plt.title('sklearn AGNES')
    plt.savefig('./cluster_result/sklearn_AGNES.jpg')
    plt.show()


