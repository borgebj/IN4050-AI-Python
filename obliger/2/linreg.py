import numpy as np
from data import X_train, t2_train, X_val, t2_val
from plotter import plot_decision_regions


def mse(y_true, y_pred):
    """MSE loss to present losses across epochs
    Step 1-2 includes loss for single sample"""
    # 1. calculates error (y - p)
    # 2. squares errors ^2
    # 3. sums errors        (numpy internal)
    # 4. averages           (numpy internal)
    return np.mean((y_true - y_pred) ** 2)


def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias  # Make an N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis=1)


class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""


class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias = bias

    def fit(self, X_train, t_train, lr=0.1, epochs=10):
        """
        X_train is a NxM matrix, N data points, M features
            - training data

        t_train is a vector of length N,
            - labels for data

        lr is our learning rate
            - how fast models learns

        epochs
            - over how many epochs the model trains

        the target class values for the training data
        """

        if self.bias:
            X_train = add_bias(X_train, self.bias)

        (N, M) = X_train.shape

        self.weights = weights = np.zeros(M)

        for epoch in range(epochs):

            # parts of weight update
            prediction = X_train @ weights      # Y = X * W
            error = (prediction - t_train)      # L = (Y - T)
            gradient = (X_train.T @ error) / N  # gradient avg. over all samples

            # weight update using gradient
            weights -= lr * gradient

            # original
            # weights -= lr / N * X_train.T @ (X_train @ weights - t_train)

            # loss
            loss = mse(y_true=t_train, y_pred=prediction)

            if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3} - Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""

        if self.bias:
            X = add_bias(X, self.bias)

        # computes predictions
        ys = X @ self.weights

        return ys > threshold


def accuracy(predicted, gold):
    return np.mean(predicted == gold)


cl = NumpyLinRegClass()
cl.fit(X_train, t2_train, lr=0.5, epochs=10)     # lav lr, høy epoch (prøv)
print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))

plot_decision_regions(X_train, t2_train, cl)
