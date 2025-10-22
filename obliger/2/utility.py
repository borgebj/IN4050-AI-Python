import numpy as np


class NumpyClassifier:
    """Common methods to all Numpy classifiers --- if any"""


def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    #  Example:
    # [x1, x2]     [1, x1, x2]
    # [x3, x4]  -> [1, x3, x4]
    # [x5, x5]     [1, x5, x6]
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias  # Make an N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis=1)


def accuracy(predicted, gold):
    """Compares predicted to actual (gold)"""
    return np.mean(predicted == gold)