from linreg import NumpyClassifier, accuracy, standard, add_bias
from plotter import plot_decision_regions, plot_curves
from utility import parse_args
import numpy as np


# ============== NEW FUNCTIONS ===================
def softmax(X):
    """Softmax activation function for multi-class output"""
    # shifting (x-max(x)) ensures numerical stability
    # axis=1 ensures function applied across each row
    # keepdims ensures original dimensions maintained
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / exp_X.sum(axis=1, keepdims=True)


def cce(y_true, y_pred):
    """Calculates loss with categorical Cross Entropy (cce)"""
    # (1/N) * sum(y * log(p))       (sums over classes)
    eps = 1e-15
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


def onehot(labels, classes):
    C = len(classes)        # no. classes
    N = labels.shape[0]     # no. samples
    label_idx = {label: i for i, label in enumerate(classes)}

    t_onehot = np.zeros((N, C))  # [0,...,0] w/ dimension sample X class

    # marks appropriate class as 1 : [0,...,1,...,0]
    for i, label in enumerate(labels):
        t_onehot[i, label_idx[label]] = 1

    return t_onehot


class NumpySoftmax(NumpyClassifier):
    """One-vs-rest multi-class logistic regression"""

    def __init__(self, bias=-1, verbose=False):
        self.bias = bias

        # loss and accuracies
        self.loss_train, self.loss_val = [], []
        self.accuracies_train, self.accuracies_val = [], []

        self._epochs_trained = 0
        self.verbose = verbose      # optional printing


    def fit(self, X_train, t_train, lr=0.1, epochs=10, tol=0.0, n_epochs_no_update=5, validation=None):
        """
        X_train is a NxM matrix, N data points, M features
            - training data

        t_train is a vector of length N,
            = t_multi_train
            - labels for data (includes multiple classes)

        lr is our learning rate
            - how fast models learns

        epochs
            - over how many epochs the model trains

        validation
            - optional validation set for loss and accuracies(X_val, t_val)

        tol, n_epochs_no_update
            - decides when to stop early, used in logreg
        """

        # all unique classes [0,1,2,3,4]
        classes = np.unique(t_train)

        # encodes labels with onehot
        t_onehot = onehot(t_train, classes)

        if self.bias:
            X_train = add_bias(X_train, self.bias)

        (N, M) = X_train.shape      # samples X features
        C = len(classes)            # no. classes

        # weights updated w/ classes
        self.weights = weights = np.zeros((M, C))  # samples X classes

        # # keep track of lowest loss and epoch with improvements
        lowest_val_loss = np.inf
        epoch_no_improvement = 0


        for epoch in range(epochs):
            # forward pass
            Z = X_train @ weights       # Z = XW
            prediction = softmax(Z)     # A = softmax(Z)

            # gradient
            error = (prediction - t_onehot)     # derivative of softmax w/ cce
            gradient = (X_train.T @ error) / N  # gradient avg. over samples

            # weight update using gradient
            weights -= lr * gradient

            # training loss
            train_loss = cce(y_true=t_onehot, y_pred=prediction)
            self.loss_train.append(float(train_loss))

            # training accuracy
            top_pred = np.argmax(prediction, axis=1)
            train_acc = accuracy(predicted=top_pred, gold=t_train)
            self.accuracies_train.append(float(train_acc))


            # loss and accuracy for validation data
            if validation:
                X_val, t_val = validation
                if self.bias:
                    X_val = add_bias(X_val, self.bias)

                # onehot validation labels
                t_val_onehot = onehot(t_val, classes)

                # validation prediction
                pred_val = softmax(X_val @ weights)

                # validation loss
                val_loss = cce(y_true=t_val_onehot, y_pred=pred_val)
                self.loss_val.append(float(val_loss))

                # validation accuracy
                val_pred_classes = np.argmax(pred_val, axis=1)
                val_acc = accuracy(predicted=val_pred_classes, gold=t_val)
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
                        print(f"Epoch {epoch+1:3} - Loss: {train_loss:.4f}, Accuracy: {(train_acc*100):.2f}%\t(train)")




    def predict(self, X):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X using OVR
        """
        if self.bias:
            X = add_bias(X, self.bias)

        # compute predictions
        ys = X @ self.weights

        # compute softmax
        ys = softmax(ys)

        # idx of highest prob. class
        pred_class = np.argmax(ys, axis=1)

        return pred_class


def main():
    from data import X_train, t_multi_train, X_val, t_multi_val
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1
    epochs = args.epochs                # default: 3
    tolerance = args.tolerance          # default: 1.0
    patience = args.patience            # default: 10
    verbose = args.verbose              # default: False
    #  ---------------- ---------------- ----------------


    # ----------------- 1. normalization ----------------
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
    cl = NumpySoftmax(verbose=verbose)

    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" +
        f"- Tolerance:  [{tolerance}]\n" +
        f"- Patience:   [{patience}]\n"
    )

    # training (seen data)
    cl.fit(
        X_train=X_train, t_train=t_multi_train,
        lr=learning_rate, epochs=epochs,             # hyperparameters (1)
        tol=tolerance, n_epochs_no_update=patience,  # hyperparameters (2)
        validation=(X_val, t_multi_val)
    )

    predictions = cl.predict(X_val)                 # predicting (unseen data)
    print("\nAccuracy on the validation set:", accuracy(predictions, t_multi_val))
    # ---------------- ---------------- ----------------


    # ---------------- 3. Plotting ---------------- ----
    plot_decision_regions(X_train, t_multi_train, cl)
    # ---------------- ---------------- ----------------
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()
