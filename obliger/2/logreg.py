from utility import NumpyClassifier, add_bias, accuracy, parse_args
from plotter import plot_curves, plot_decision_regions
import numpy as np


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


def sigmoid(y):
    """Simple sigmoid for logreg using numpy"""
    return 1 / (1 + np.exp(-y))

# ================================================


class NumpyLogRegClass(NumpyClassifier):
    """Logistic regression using sigmoid + BCE"""

    def __init__(self, bias=-1, verbose=False):
        self.bias = bias
        self.loss_train = []        # loss for training data
        self.accuracies_train = []  # accuracies for training data

        self.loss_val = []          # loss for validation data
        self.accuracies_val = []    # accuracies for validation data

        self.epochs_trained = 0     # keep track of training duration
        self.verbose = verbose      # optional printing


    def fit(self, X_train, t_train, tol=0.0, n_epochs_no_update=5, validation=None, lr=0.1, epochs=10):
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

        tol, n_epochs_no_update
            - decides when to stop early
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

            # forward pass
            Z = X_train @ weights                # Z = XW
            prediction = sigmoid(Z)              # A = sigmoid(Z)

            # gradient (sigmoid derivative w/ bce)
            error = (prediction - t_train)       # L = (Y - T)
            gradient = (X_train.T @ error) / N  # gradient avg. over samples

            # weight update using gradient
            weights -= lr * gradient

            # training loss
            train_loss = bce(y_true=t_train, y_pred=prediction)
            self.loss_train.append(float(train_loss))

            # training accuracy
            train_acc = accuracy(predicted=(prediction>0.5), gold=t_train)
            self.accuracies_train.append(float(train_acc))


            # loss and accuracy for validation data
            if validation:
                pred_val = sigmoid(X_val @ weights)

                # loss + accuracy calculation
                val_loss = bce(y_true=t_val, y_pred=pred_val)
                val_acc = accuracy(predicted=(pred_val>0.5), gold=t_val)
                self.loss_val.append(float(val_loss))
                self.accuracies_val.append(float(val_acc))

                # measuring loss based on tolerance
                if tol is not None:
                    if (lowest_val_loss - val_loss) > tol:
                        lowest_val_loss = val_loss
                        epoch_no_improvement = 0
                    else:
                        epoch_no_improvement += 1

                # stopping early (tol and n_epochs)
                if epoch_no_improvement >= n_epochs_no_update:
                    if self.verbose: print(f"Epoch {epoch+1:3} - Loss: {val_loss:.4f}, Accuracy: {(val_acc*100):.2f}%\t(dev)")
                    self._epochs_trained = epoch + 1
                    break


            # print occasionally
            if self.verbose:
                if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                    if validation:
                        print(f"Epoch {epoch+1:3} - Loss: {val_loss:.4f}, Accuarcy: {(val_acc*100):.2f}%\t(dev)")
                    else:
                        print(f"Epoch {epoch+1:3} - Loss: {train_loss:.4f}, Accuracy: {(train_acc*100):.2f}%\t(train))")


    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X
        """
        if self.bias:
            X = add_bias(X, self.bias)

        # computes predictions
        ys = X @ self.weights

        # compute logistic function (sigmoid)
        ys = sigmoid(ys)

        # modified to allow probabilities AND classes
        if threshold is not None:
            return ys > threshold       # classes [0, 1, 0, ..]
        else:
            return ys                   # probabilities [0.2, 0.7, ...]


    def predict_probability(self, X):
        """Predicts probabilities, not classes"""
        return self.predict(X, threshold=None)


def main():
    from data import X_train, t2_train, X_val, t2_val
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1  (best: 1.0)
    epochs = args.epochs                # default: 3    (best: ~100)
    tolerance = args.tolerance          # default 1.0   (best: 1.0)
    patience = args.patience            # default: 10   (best: 2)
    verbose = args.verbose              # default: False
    #  ---------------- ---------------- ----------------


    # ----------------- 1. normalization ----------------
    # do axis=0 > column, due to per-feature
    # we extract mean and std from TRAINING, ensuring others use same scale as trained on
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Normalizing test data
    norm_train = standard(X_train, train_mean, train_std)
    X_train = norm_train

    # Normalizing validation data
    norm_val = standard(X_val, train_mean, train_std)
    X_val = norm_val
    # ---------------- ---------------- ----------------


    # ---------------- 2. Regression ---------------- --
    cl = NumpyLogRegClass(verbose=verbose)


    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" +
        f"- Tolerance:  [{tolerance}]\n" +
        f"- Patience:   [{patience}]\n"
    )

    # training   (seen data)
    cl.fit(
        X_train=X_train,
        t_train=t2_train,
        tol=tolerance, n_epochs_no_update=patience, # hyperparameters (1)
        lr=learning_rate, epochs=epochs,            # hyperparameters (2)
        validation=(X_val, t2_val)
    )

    predictions = cl.predict(X_val)                 # predicting (unseen data)
    print("\nAccuracy on the validation set:", accuracy(predictions, t2_val))
    # ---------------- ---------------- ---------------- 


    # ---------------- 3. Plotting ---------------- ----
    # accuracy curve
    acc_train = cl.accuracies_train
    acc_val = cl.accuracies_val
    plot_curves(res_train=acc_train, res_dev=acc_val, label="Accuracy")

    # loss curve
    loss_train = cl.loss_train
    loss_val = cl.loss_val
    plot_curves(res_train=loss_train, res_dev=loss_val, label="Loss")

    plot_decision_regions(X_train, t2_train, cl)
    # ---------------- ---------------- ---------------- 
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()
