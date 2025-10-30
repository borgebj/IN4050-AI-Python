from utility import normalize_data, select_eval, accuracy
from plotter import plot_decision_regions
from logreg import NumpyLogRegClass
from linreg import NumpyClassifier
from argparser import parse_args
import numpy as np


class NumpyOneVsRest(NumpyClassifier):
    """One-vs-rest multi-class logistic regression"""

    def __init__(self, bias=-1, verbose=False):
        self.bias = bias

        # loss and accuracies
        self.loss_train, self.loss_val = [], []
        self.accuracies_train, self.accuracies_val = [], []

        self._epochs_trained = 0
        self.verbose = verbose      # optional printing


    def fit(self, X_train, t_train,lr=0.1, epochs=10, tol=0.0, n_epochs_no_update=5, validation=None):
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

        self.classifiers = {}

        # mark class C in training data
        for c in classes:

            # marks label C:  3: [0, 3, 2] -> [False, True, False] -> [0, 1, 0]
            t_class = (t_train == c).astype('int')

            # one classifier each class - train and save
            ccl = NumpyLogRegClass(self.bias, self.verbose)

            # if validation provided
            if validation:
                (X_val, t_val) = validation
                t_class_val = (t_val == c).astype('int')

                if self.verbose: print(f"\nclass{c}")
                ccl.fit(X_train, t_class, lr, epochs, tol, n_epochs_no_update, (X_val, t_class_val))

            # run with default
            else:
                ccl.fit(X_train, t_class, lr, epochs)

            # save each classifier
            self.classifiers[c] = ccl



    def predict(self, X):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X using OVR
        """
        # turns (class x sample) to (sample x class), making row = sample
        probs = np.column_stack([
            ccl.predict_probability(X)
            for c, ccl in self.classifiers.items()
        ])
        # list of samples with their probs.   ->    Sample 1 = [class1, class2, ...]
        pred_idx = np.argmax(probs, axis=1)  # <-- highest prob. idx across rows ↑

        # get pred. class from classifiers using idx  e.g:  idx 2 from [0,2,4] is class 4
        class_keys = list(self.classifiers.keys())
        pred_class = np.array(class_keys)[pred_idx]

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
    epochs = args.epochs                # default: 10
    tolerance = args.tolerance          # default: 1.0
    patience = args.patience            # default: 10
    verbose = args.verbose              # default: False
    eval_set = args.eval_set            # default: validation
    #  --------------------------------------------------


    # ----------------- 1. normalization ----------------
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    # ---------------------------------------------------


    # ----------------- 2. evaluation set ----------------
    (X_eval, t_eval) = select_eval(X_train, t_multi_train, X_val, t_multi_val, X_test, t_multi_test, eval_set)
    # ----------------------------------------------------


    # ---------------- 3. Regression -------------------
    cl = NumpyOneVsRest(verbose=verbose)

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
    # --------------------------------------------------


    # ---------------- 3. Plotting ---------------------
    plot_decision_regions(X_train, t_multi_train, cl)
    # --------------------------------------------------
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()
