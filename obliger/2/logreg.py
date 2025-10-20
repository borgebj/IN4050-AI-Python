import numpy as np
from plotter import plot_decision_regions


# ============== NEW FUNCTIONS ===================
def standard(X):
    """Standard scaler aka Z-score"""
    # do axis=0 > column, due to per-feature
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def bce(y_true, y_pred):
    """BCE loss for binary classification"""
    # 1. sample formula:   -[ylog(p) + (1-y)log(1-p)]
    # 2. sum over samples
    # 3. avg. sum for mean loss
    # eps is a tiny value added to counter division by zero
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def sigmoid(ys):
    """Simple sigmoid for logreg using numpy"""
    return 1 / (1 + np.exp(-ys))

# ================================================


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


class NumpyClassifier:
    """Common methods to all Numpy classifiers --- if any"""


class NumpyLogRegClass(NumpyClassifier):
    """Logistic regression using sigmoid + BCE"""

    def __init__(self, bias=-1):
        self.bias = bias
        self.losses = []        # stores losses
        self.accuracies = []    # store accuracies

    def fit(self, X_train, t_train, validation=None, lr=0.1, epochs=10):
        """
        X_train is a NxM matrix, N data points, M features
            - training data

        t_train is a vector of length N,
            - labels for data

        lr is our learning rate
            - how fast models learns

        epochs
            - over how many epochs the model trains

        validation
            - optional validation set for loss and accuracies(X_val, t_val)

        the target class values for the training data
        """

        if self.bias:
            X_train = add_bias(X_train, self.bias)

        if validation and self.bias:
            (X_val, t_val) = validation
            X_val = add_bias(X_val, self.bias)

        (N, M) = X_train.shape

        self.weights = weights = np.zeros(M)     # weights are all 0 = [0.0, 0.0, ..., 0.0]

        for epoch in range(epochs):

            # parts of weight update
            Z = X_train @ weights                # Z = X * W
            activation = sigmoid(Z)              # A = sigmoid(Z)                  (NEW - logreg)
            error = (activation - t_train)       # L = (Y - T)                     (BCE derivative w/ sigmoid)
            gradient = (X_train.T @ error) / N  # gradient avg. over all samples  (Y.der. * MSE.der.)

            # weight update using gradient
            weights -= lr * gradient

            # loss calculation + store it (validation data)
            if validation:
                Z_val = X_val @ weights
                pred_val = sigmoid(Z_val)

                loss = bce(y_true=t_val, y_pred=pred_val)
                acc = accuracy(predicted=(pred_val > 0.5), gold=t_val)

                self.losses.append(loss)
                self.accuracies.append(acc)

            # print occasionally
            if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                if validation:
                    print(f"Epoch {epoch+1:3} - Loss: {loss:.4f}")
                else:
                    print(f"Epoch {epoch+1:3}")

    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X
        # gives = [0, 1, 0, ...]
        """

        if self.bias:
            X = add_bias(X, self.bias)

        # computes predictions
        ys = X @ self.weights

        # compute activation
        ys = sigmoid(ys)

        return ys > threshold

    def predict_probability(self, X):
        """Predicts probabilities, not classes
        (predict without threshold)"""

        if self.bias:
            X = add_bias(X, self.bias)

        # compute predictions
        ys = X @ self.weights

        # compute + return activation
        return sigmoid(ys)


def main():
    from data import X_train, t2_train, X_val, t2_val

    # task 1 part 2 - scaling data using standard scaler
    norm_train = standard(X_train)
    X_train = norm_train

    cl = NumpyLogRegClass()
    cl.fit(
        X_train=X_train,
        t_train=t2_train,
        lr=0.1, epochs=3,
        validation=(X_val, t2_val))               # training   (seen data)
    predictions = cl.predict(X_val)               # predicting (unseen data)

    print("Accuracy on the validation set:", accuracy(predictions, t2_val))

    probabilities = cl.predict_probability(X_val)

    print("\nPredictions")
    print(predictions[:5])
    print("\nProbabilities")
    print(probabilities[:5])

    print()
    print(cl.accuracies)
    print(cl.losses)

    # plot_decision_regions(X_train, t2_train, cl)


if __name__ == "__main__":
    main()
