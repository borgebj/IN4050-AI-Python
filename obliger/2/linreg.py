import numpy as np
from plotter import plot_decision_regions


# ============== NEW FUNCTIONS ===================
def standard(X, mean, std):
    """Standard scaler aka Z-score
    Usses passed mean and std (must use same as training!)"""
    return (X - mean) / std


def mse(y_true, y_pred):
    """MSE loss to present losses across epochs
    Step 1-2 includes loss for single sample"""
    # 1. calculates error (y - p)
    # 2. squares errors ^2
    # 3. sums errors        (numpy internal)
    # 4. averages           (numpy internal)
    return np.mean((y_true - y_pred) ** 2)

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


class NumpyLinRegClass(NumpyClassifier):
    """Logistic regression using MSE"""

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
            error = (prediction - t_train)      # L = (Y - T)                     (MSE derivative)
            gradient = (X_train.T @ error) / N  # gradient avg. over all samples  (Y.der. * MSE.der.)

            # weight update using gradient
            weights -= lr * gradient


            # loss and accuracy for training data       (used for manual testing)
            loss = mse(y_true=t_train, y_pred=prediction)
            acc = accuracy(predicted=(prediction > 0.5), gold=t_train)

            # print occationally
            if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3} - Loss: {loss:.4f}, Accuracy: {(acc*100):.2f}%\t(train)")


    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""

        if self.bias:
            X = add_bias(X, self.bias)

        # computes predictions
        ys = X @ self.weights

        return ys > threshold


def main():
    from data import X_train, t2_train, X_val, t2_val
    print("\n"*5)

    # ----------------- 1. normalization ---------------- 
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    
    # task 1 part 2 - scaling data using standard scaler
    norm_train = standard(X_train, train_mean, train_std)
    X_train = norm_train
    # ---------------- ---------------- ---------------- 


    # ---------------- 2. Regression ---------------- --
    cl = NumpyLinRegClass()
    cl.fit(
        X_train=X_train, 
        t_train=t2_train, 
        lr=1, epochs=3)                         # training   (seen data)
    predictions = cl.predict(X_val)             # predicting (unseen data)

    print("\nAccuracy on the validation set:", accuracy(predictions, t2_val))

    # plot_decision_regions(X_train, t2_train, cl)

    print("\n"*5)


if __name__ == "__main__":
    main()
