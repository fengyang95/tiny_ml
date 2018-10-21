import numpy
from sklearn import metrics, datasets, manifold
from scipy import optimize
from matplotlib import pyplot
import pandas
import collections


def generate_circle_data():
    xx = numpy.zeros((1200, 3))
    x1 = numpy.ones((400,)) + 0.5 * numpy.random.rand(400) - 0.5
    r1 = numpy.linspace(0, 2 * numpy.pi, 20)
    r2 = numpy.linspace(0, numpy.pi, 20)
    r1, r2 = numpy.meshgrid(r1, r2)
    r1 = r1.ravel()
    r2 = r2.ravel()
    xx[0:400, 0] = x1 * numpy.sin(r1) * numpy.sin(r2)
    xx[0:400, 1] = x1 * numpy.cos(r1) * numpy.sin(r2)
    xx[0:400, 2] = x1 * numpy.cos(r2)
    x1 = 3 * numpy.ones((400,)) + 0.6 * numpy.random.rand(400) - 0.6
    xx[400:800, 0] = x1 * numpy.sin(r1) * numpy.sin(r2)
    xx[400:800, 1] = x1 * numpy.cos(r1) * numpy.sin(r2)
    xx[400:800, 2] = x1 * numpy.cos(r2)
    x1 = 6 * numpy.ones((400,)) + 1.1 * numpy.random.rand(400) - 0.6
    xx[800:1200, 0] = x1 * numpy.sin(r1) * numpy.sin(r2)
    xx[800:1200, 1] = x1 * numpy.cos(r1) * numpy.sin(r2)
    xx[800:1200, 2] = x1 * numpy.cos(r2)
    target = numpy.zeros((1200,))
    target[0:400] = 0
    target[400:800] = 1
    target[800:1200] = 2
    target = target.astype('int')
    return xx, target


def get_data():
    data = datasets.load_iris()
    return data.data, data.target


def calculate_distance(x, y):
    d = numpy.sqrt(numpy.sum((x - y) ** 2))
    return d


def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = numpy.square(D)
    Di = numpy.sum(DD, axis=1) / n1
    Dj = numpy.sum(DD, axis=0) / n1
    Dij = numpy.sum(DD) / (n1 ** 2)
    B = numpy.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)
    return B


def MDS(data, n=2):
    D = calculate_distance_matrix(data, data)
    B = cal_B(D)
    Be, Bv = numpy.linalg.eigh(B)
    # print numpy.sum(B-numpy.dot(numpy.dot(Bv,numpy.diag(Be)),Bv.T))
    Be_sort = numpy.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:, Be_sort]
    Bez = numpy.diag(Be[0:n])
    # print Bez
    Bvz = Bv[:, 0:n]
    Z = numpy.dot(numpy.sqrt(Bez), Bvz.T).T
    return Z


def t_iris():
    data, target = get_data()
    Z = MDS(data)

    figure1 = pyplot.figure()
    pyplot.subplot(1, 3, 1)
    pyplot.plot(Z[target == 0, 0], Z[target == 0, 1], 'r*', markersize=20)
    pyplot.plot(Z[target == 1, 0], Z[target == 1, 1], 'bo', markersize=20)
    pyplot.plot(Z[target == 2, 0], Z[target == 2, 1], 'gx', markersize=20)
    pyplot.title('CUSTOM')

    pyplot.subplot(1, 3, 2)
    Z1 = manifold.MDS(n_components=2).fit_transform(data)
    pyplot.plot(Z1[target == 0, 0], Z1[target == 0, 1], 'r*', markersize=20)
    pyplot.plot(Z1[target == 1, 0], Z1[target == 1, 1], 'bo', markersize=20)
    pyplot.plot(Z1[target == 2, 0], Z1[target == 2, 1], 'gx', markersize=20)
    pyplot.title('SKLEARN')


    import tinyml.dimension_reduction.MDS as IMDS
    Z2=IMDS.MDS(2).fit_transform(data)
    pyplot.subplot(1, 3, 3)
    pyplot.plot(Z2[target == 0, 0], Z2[target == 0, 1], 'r*', markersize=20)
    pyplot.plot(Z2[target == 1, 0], Z2[target == 1, 1], 'bo', markersize=20)
    pyplot.plot(Z2[target == 2, 0], Z2[target == 2, 1], 'gx', markersize=20)
    pyplot.title('CUSTOM')
    pyplot.show()




def t_ball():
    data, target = generate_circle_data()
    Z = MDS(data)
    figure1 = pyplot.figure()
    pyplot.subplot(1, 2, 1)
    pyplot.plot(Z[target == 0, 0], Z[target == 0, 1], 'r*', markersize=10)
    pyplot.plot(Z[target == 1, 0], Z[target == 1, 1], 'bo', markersize=10)
    pyplot.plot(Z[target == 2, 0], Z[target == 2, 1], 'gx', markersize=10)
    pyplot.title('CUSTOM')
    pyplot.subplot(1, 2, 2)
    Z1 = manifold.MDS(n_components=2).fit_transform(data)
    pyplot.plot(Z1[target == 0, 0], Z1[target == 0, 1], 'r*', markersize=10)
    pyplot.plot(Z1[target == 1, 0], Z1[target == 1, 1], 'bo', markersize=10)
    pyplot.plot(Z1[target == 2, 0], Z1[target == 2, 1], 'gx', markersize=10)
    pyplot.title('SKLEARN')
    pyplot.show()


if __name__ == '__main__':
    #t_ball()
    t_iris()
