import numpy as np
from plotter import plot_decision_regions


# ============== NEW FUNCTIONS ===================
def standard(X, mean, std):
    """Standard scaler aka Z-score
    Usses passed mean and std (must use same as training!)"""
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
        self.loss_train = []        # loss for training data   
        self.accuracies_train = []  # accuarcies for training data

        self.loss_dev = []          # loss for validation data
        self.accuracies_dev = []    # accuracies for validation data

        self._epochs_trained = 0    # keep track of training duration


    def fit(self, X_train, t_train, tol=0, n_epochs_no_update=5, validation=None, lr=0.1, epochs=10):
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

        # if validation provided, add bias before running
        if validation and self.bias:
            (X_val, t_val) = validation
            X_val = add_bias(X_val, self.bias)

        (N, M) = X_train.shape

        self.weights = weights = np.zeros(M)     # weights are all 0 = [0.0, 0.0, ..., 0.0]

        # keep track of lowest loss and epoch with improvements
        lowest_val_loss = np.inf
        epoch_no_improvement = 0

        for epoch in range(epochs):

            # parts of weight update
            Z = X_train @ weights                # Z = X * W
            activation = sigmoid(Z)              # A = sigmoid(Z)                  (NEW - logreg)
            error = (activation - t_train)       # L = (Y - T)                     (BCE derivative w/ sigmoid)
            gradient = (X_train.T @ error) / N  # gradient avg. over all samples   (Y.der. * BCE+sigmoid.der.)

            # weight update using gradient
            weights -= lr * gradient


            # NEW
            # loss and accuracy for training data       (used for manual testing)
            train_loss = bce(y_true=t_train, y_pred=activation)
            train_acc = accuracy(predicted=(activation>0.5), gold=t_train)
            self.loss_train.append(float(train_loss))
            self.accuracies_train.append(float(train_acc))


            # loss and accuracy for validation data
            if validation:
                Z_val = X_val @ weights
                pred_val = sigmoid(Z_val)

                dev_loss = bce(y_true=t_val, y_pred=pred_val)
                dev_acc = accuracy(predicted=(pred_val>0.5), gold=t_val)
                self.loss_dev.append(dev_loss)
                self.accuracies_dev.append(dev_acc)

                # measuring loss based on tolerance
                if tol is not None:
                    if (lowest_val_loss - dev_loss) > tol:
                        lowest_val_loss = dev_loss
                        epoch_no_improvement = 0
                    else:
                        epoch_no_improvement += 1

                # stopping early (tol and n_epochs)
                if epoch_no_improvement >= n_epochs_no_update:
                    print(f"== Early stopping ==\nEpoch {epoch+1:3} - Loss: {dev_loss:.4f}, Accuracy: {(dev_acc*100):.2f}%\t(dev)")
                    self._epochs_trained = epoch + 1
                    break



            # print occasionally
            if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                if validation:
                    print(f"Epoch {epoch+1:3} - Loss: {dev_loss:.4f}, Accuarcy: {(dev_loss*100):.2f}%\t(dev)")
                else:
                    print(f"Epoch {epoch+1:3} - Loss: {train_loss:.4f}, Accuracy: {(train_acc*100):.2f}%\t(train))")


    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X
        # gives = [0, 1, 0, ...]
        """

        if self.bias:
            X = add_bias(X, self.bias)

        # computes predictions
        ys = X @ self.weights

        # compute logistic function (sigmoid)
        ys = sigmoid(ys)

        if threshold is not None:
            return ys > threshold
        else:
            return ys


    def predict_probability(self, X):
        """Predicts probabilities, not classes
        (predict without threshold)"""
        return self.predict(X, threshold=None)


def main():
    from data import X_train, t2_train, X_val, t2_val
    print("\n"*5)

    # ----------------- 1. normalization ---------------- 
    # do axis=0 > column, due to per-feature
    # we extract mean and std from TRAINING, ensuring others use same scale as trained on
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Normalizing test data
    norm_train = standard(X_train, train_mean, train_std)
    # X_train = norm_train

    # Normalizing validation data
    norm_val = standard(X_val, train_mean, train_std)
    # X_val = norm_val
    # ---------------- ---------------- ---------------- 


    # ---------------- 2. Regression ---------------- --

    cl = NumpyLogRegClass()
    
    # hyperparameters
    learning_rate = 1.0
    epochs = 1000
    tolerance = 1.0
    patience = 5

    print(f"__Hyperparameters__\n"  + 
          f"- Learning:   [{learning_rate}]\n" +
          f"- Epochs:     [{epochs}]\n" +
          f"- Tolerance:  [{tolerance}]\n"+
          f"- Patience:   [{patience}]\n\n"
    )
    cl.fit(
        X_train=X_train,
        t_train=t2_train,
        tol=tolerance, n_epochs_no_update=patience, # hyperparameters (1)
        lr=learning_rate, epochs=epochs,            # hyperparameters (2)
        validation=(X_val, t2_val))                 # training   (seen data)
    predictions = cl.predict(X_val)                 # predicting (unseen data)

    print("\nAccuracy on the validation set:", accuracy(predictions, t2_val))

    # probabilities = cl.predict_probability(X_val)
    # print("\n\nPredictions")
    # print(predictions[:5])
    # print("\nProbabilities")
    # print(probabilities[:5])

    # print("\nTraining loss")
    # print(cl.loss_train[:5])
    # print("\nTraining accuracy")
    # print(cl.accuracies_train[:5])

    # plot_decision_regions(X_train, t2_train, cl)

    print("\n"*5)


if __name__ == "__main__":
    main()
