from utility import NumpyClassifier, accuracy, parse_args
from logreg import NumpyLogRegClass, standard
from plotter import plot_decision_regions, plot_curves
import numpy as np


class NumpyOneVsRest(NumpyClassifier):
    """One-vs-rest multi-class logistic regression"""

    def __init__(self, bias=-1, verbose=False):
        self.bias = bias

        self.loss_train = []
        self.accuracies_train = []

        self.loss_val = []
        self.accuracies_val = []

        self._epochs_trained = 0
        self.verbose = verbose      # optional printing


    def fit(self, X_train, t_train, tol=0.0, n_epochs_no_update=5, validation=None, lr=0.1, epochs=10):
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
            t_class = (t_train == c).astype('int')

            # one classifier each class - train and save
            ccl = NumpyLogRegClass(self.bias, self.verbose)

            # if validation provided, for logreg
            if validation:
                (X_val, t_val) = validation
                t_class_val = (t_val == c).astype('int')

                if self.verbose: print(f"\nclass{c}")
                ccl.fit(
                    X_train, t_class,
                    lr=lr, epochs=epochs,                            # hyperparameters (1)
                    tol=tol, n_epochs_no_update=n_epochs_no_update,  # hyperparameters (2)
                    validation=(X_val, t_class_val)
                )
            # run with default
            else:
                ccl.fit(
                    X_train, t_class,
                    lr=lr, epochs=epochs
                )

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
    from data import X_train, t_multi_train, X_val, t_multi_val
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1  (best: 1.0)
    epochs = args.epochs                # default: 3    (best: 1000)
    tolerance = args.tolerance          # default 1.0   (best: 1.0)
    patience = args.patience            # default: 10   (best: 885)
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
    cl = NumpyOneVsRest(verbose=verbose)

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
        tol=tolerance, n_epochs_no_update=patience, # hyperparameters (1)
        lr=learning_rate, epochs=epochs,            # hyperparameters (2)
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
