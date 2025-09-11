import numpy as np
import matplotlib.pyplot as plt
import sklearn
from collections import Counter

from sklearn.datasets import make_blobs


def distance_L2(a,b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)

def distance_L2d(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

assert distance_L2((3, 4, 0), (0, 0, 12)) == 13


def majority(a):
    counts = Counter(a)
    return counts.most_common()[0][0]

assert majority([0,1,1,1,0]) == 1

"""
Plots based on input points
"""
def show(X, y, marker='.'):
    labels = set(y)
    cl = {lab : [] for lab in labels}
    # cl[lab] shall contain the datapoints labeled lab
    for (a, b) in zip(X, y):
        cl[b].append(a)
    for lab in labels:
        plt.plot([a[0] for a in cl[lab]], [a[1] for a in cl[lab]],
                 marker, label="class {}".format(lab))
    plt.legend()
    plt.show()


class PyClassifier():
    """Common methods to all python classifiers --- if any

    Nothing here yet"""


class PykNNClassifier(PyClassifier):
    """kNN classifier using pure python representations"""

    def __init__(self, k=3, dist=distance_L2d):
        self.k = k
        self.dist = dist

    def fit(self, X_train, t_train):
        self.X_train = X_train
        self.t_train = t_train

    def predict(self, a):
        """
        1. Finds distance between a and all training (x_train) points
        2. finds k-closest points (sort)
        3. Gets majority labels from these k-closest
        """

        # labels = 0 / 1
        [print(float(x)) for x in self.t_train]

        # distances from 'a' to all data points
        distances = [(self.dist(x,a), i) for i, x in enumerate(self.X_train)]

        # sort distance
        distances.sort()

        # get k closest
        k_closest = distances[:self.k]

        print(k_closest)

        return 0


X_np, t_np = make_blobs(n_samples=200, centers=[[0,0],[1,2]],
                  n_features=2, random_state=2024)
X1 = [(X_np[i,0], X_np[i,1]) for i in range(X_np.shape[0])]
t1 = [t_np[i] for i in range(X_np.shape[0])]

#show(X1, t1)


knn = PykNNClassifier()
knn.fit(X1, t1)
p = (2, 2)
knn.predict(p)