from utility import normalize_data, select_eval, onehot, softmax, cce, accuracy
from linreg import NumpyClassifier, add_bias
from plotter import plot_decision_regions
from argparser import parse_args
import numpy as np


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

            # training metrics
            train_loss = cce(y_true=t_onehot, y_pred=prediction)
            top_pred = np.argmax(prediction, axis=1)  # gets class w/ highest probability 
            train_acc = accuracy(predicted=top_pred, gold=t_train)
            self.loss_train.append(float(train_loss))
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

                # validation metrics
                val_loss = cce(y_true=t_val_onehot, y_pred=pred_val)
                val_pred_classes = np.argmax(pred_val, axis=1)  # gets class w/ highest probability
                val_acc = accuracy(predicted=val_pred_classes, gold=t_val)
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
                    if self.verbose:
                        print(
                            f"Epoch {epoch + 1:3} - "
                            f"Loss: {val_loss:.4f}, "
                            f"Accuracy: {val_acc:.3f} (dev) {train_acc:.3f} (train)")
                    self._epochs_trained = epoch + 1
                    break

            # print occasionally
            if self.verbose:
                if (epoch + 1) % max(1, epochs//5) == 0 or epoch == 0:
                    if validation:
                        print(
                            f"Epoch {epoch + 1:3} - "
                            f"Loss: {val_loss:.4f}, "
                            f"Accuarcy: {val_acc:.3f} (dev) {train_acc:.3f} (train)")
                    else:
                        print(f"Epoch {epoch + 1:3} - "
                              f"Loss: {train_loss:.4f}, "
                              f"Accuracy: {train_acc:.3f} (train)")

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
    from data import (
        X_train, t_multi_train,  # train data
        X_val, t_multi_val,      # validation data
        X_test, t_multi_test     # test data
    )
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1
    epochs = args.epochs                # default: 3
    tolerance = args.tolerance          # default: 1.0
    patience = args.patience            # default: 10
    verbose = args.verbose              # default: False
    eval_set = args.eval_set            # default: validation
    #  ---------------- ---------------- ----------------


    # ----------------- 1. normalization ----------------
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    # ---------------------------------------------------


    # ----------------- 2. evaluation set ----------------
    (X_eval, t_eval) = select_eval(X_train, t_multi_train, X_val, t_multi_val, X_test, t_multi_test, eval_set)
    # ----------------------------------------------------


    # ---------------- 3. Regression -------------------
    cl = NumpySoftmax(verbose=verbose)

    print(
        f"___Hyperparameters___\n"
        f"- Learning:   [{learning_rate}]\n"
        f"- Epochs:     [{epochs}]\n"
        f"- Tolerance:  [{tolerance}]\n"
        f"- Patience:   [{patience}]\n"
        f"\n________Info________\n"
        f"- Evaluation: [{eval_set}]\n\n"
    )

    # training (seen data)
    cl.fit(
        X_train=X_train, t_train=t_multi_train,
        lr=learning_rate, epochs=epochs,             # hyperparameters (1)
        tol=tolerance, n_epochs_no_update=patience,  # hyperparameters (2)
        validation=(X_eval, t_eval)
    )

    predictions = cl.predict(X_eval)                 # predicting (unseen data)
    print("\nAccuracy on the validation set:", accuracy(predictions, t_eval))
    # ---------------- ---------------- ----------------


    # ---------------- 3. Plotting ---------------- ----
    plot_decision_regions(X_train, t_multi_train, cl)
    # ---------------- ---------------- ----------------
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()
