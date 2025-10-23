from utility import NumpyClassifier, add_bias, accuracy
from logreg import NumpyLogRegClass, standard
from plotter import plot_decision_regions, plot_curves
import numpy as np


class NumpyOneVsRest(NumpyClassifier):
    """One-vs-rest multi-class logistic regression"""

    def __init__(self, bias=-1):
        self.bias = bias

        self.loss_train = []
        self.accuracies_train = []

        self.loss_dev = []
        self.accuracies_dev = []

        self._epochs_trained = 0
        self.classifiers = {}       # keep track of binary classifiers




    def fit(self, X_train, t_train, lr=0.1, epochs=10):
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

        the target class values for the training data
        """

        classes = np.unique(t_train)
        print("Classes")
        print(classes, end="\n\n")

        X_train = X_train[:5]
        t_train = t_train[:5]

        for ((f1, f2), lab) in zip(X_train, t_train):
            print(f"[{f1:7.2f}  {f2:7.2f}]  ->  {lab:>2}")

        for c in classes:
            # mark class C in training data
            t_class = (t_train == c).astype('int')

            # one classifier each class
            ccl = NumpyLogRegClass()
            ccl.fit(
                X_train, t_class,
                lr=lr, epochs=epochs
            )
            self.classifiers[c] = ccl

            print(f"{c} -> {t_class}")

            # for each classifier
            # get prediction
            # get highest prediction

        print("\nClassifiers")
        print(self.classifiers)


        # binary classes (t_class)
        # ([0, 1, 3 ...]
        # -> [False, True, False, ...]
        # -> [0, 1, 0, ...]

        # cl.fit(X_train, t_class)
        # -> get probability (predict_probability)
        # collect all probabilities (one per class)
        # choose class with the highest probability

        # probs = [
        #     clf_0.predict_probability(x),
        #     clf_1.predict_probability(x),
        #     clf_2.predict_probability(x),
        #     clf_3.predict_probability(x),
        #     clf_4.predict_probability(x),
        # ]
        # predicted_class = np.argmax(probs)



    def predict(self, X):
        return None


def main():
    from data import X_train, t_multi_train, X_val, t_multi_val
    print("="*40+"\n\n")

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
    cl = NumpyOneVsRest()

    # hyperparameters
    learning_rate = 1.0
    epochs = 1000
    tolerance =1.0
    patience = 10

    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" +
        f"- Tolerance:  [{tolerance}]\n" +
        f"- Patience:   [{patience}]\n\n"
    )

    # training (seen data)
    cl.fit(
        X_val,
        t_multi_val
    )

    predictions = cl.predict(X_val)                 # predicting (unseen data)
    print("\nAccuracy on the validation set:", accuracy(predictions, t_multi_val))
    # ---------------- ---------------- ----------------


    # ---------------- 3. Plotting ---------------- ----

    # accuracy curve
    acc_train = cl.accuracies_train
    acc_dev = cl.accuracies_dev
    # plot_curves(res_train=acc_train, res_dev=acc_dev, label="Accuracy")

    # loss curve
    loss_train = cl.loss_train
    loss_dev = cl.loss_dev
    # plot_curves(res_train=loss_train, res_dev=loss_dev, label="Loss")

    # plot_decision_regions(X_train, t_multi_train, cl)

    # ---------------- ---------------- ----------------
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()
