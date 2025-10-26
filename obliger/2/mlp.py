import time

from linreg import standard, NumpyClassifier, add_bias, accuracy
from plotter import plot_decision_regions, plot_curves
from utility import parse_args
from logreg import bce
import numpy as np


# First, we define the logistic function and its derivative:
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_diff(y):
    return y * (1 - y)


class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""

    def __init__(self, bias=-1, dim_hidden=6,  verbose=False):
        """Initialize the hyperparameters"""

        self.bias = bias

        # Dimensionality of the hidden layer
        # (no. neurons in hidden layer)
        self.dim_hidden = dim_hidden

        # activation function (sigmoid)
        self.activ = logistic
        self.activ_diff = logistic_diff

        # loss and accuracies
        self.loss_train, self.loss_val = [], []
        self.accuracies_train, self.accuracies_val = [], []

        self._epochs_trained = 0
        self.verbose = verbose      # optional printing


    def forward(self, X):
        """
        Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""

        Z_hidden = X @ self.weights1            # Z1 = XW1 (+b)
        A_hidden = self.activ(Z_hidden)         # A2 = logistic(Z1)

        hidden_outs = add_bias(
            A_hidden, self.bias)                # each layer has its own bias

        Z_out = hidden_outs @ self.weights2     # Z2 = XW2 (+b)
        outputs = self.activ(Z_out)             # A2 = logistic(z1)

        return hidden_outs, outputs


    def fit(self, X_train, t_train, lr=0.001, epochs=100, tol=0.0, n_epochs_no_update=5, validation=None):
        """Initialize the weights. Train *epochs* many epochs.

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
        self.lr = lr

        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1, 1)

        dim_in = X_train.shape[1]       # how many features (column)
        dim_out = T_train.shape[1]      # how many output neurons

        # ---------- Add weights for the layers ----------

        # weights - (input to hidden)
        self.weights1 = (np.random.rand(
            dim_in + 1,
            self.dim_hidden) * 2 - 1) / np.sqrt(dim_in)     # ~[ -0.408,  0.408 ] range for 6

        # weights - (hidden to output)
        self.weights2 = (np.random.rand(
            self.dim_hidden + 1,
            dim_out) * 2 - 1) / np.sqrt(self.dim_hidden)    # ~[ -0.408,  0.408 ] range for 6

        # ---------- ---------------------- ----------

        # adding bias column to data
        X_train_bias = add_bias(X_train, self.bias)

        # keep track of lowest loss and epoch with improvements
        lowest_val_loss = np.inf
        epoch_no_improvement = 0

        for epoch in range(epochs):
            # One epoch

            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)   # predictions (hidden and output)

            # The delta term on the output node:
            out_deltas = (outputs - T_train)                    # Loss = (Y - T) sigmoid + bce

            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T

            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))

            # Update the weights:
            self.weights2 -= self.lr * (hidden_outs.T @ out_deltas)         # gradient 1
            self.weights1 -= self.lr * (X_train_bias.T @ hiddenact_deltas)  # gradient 2


            # if epoch % 100 == 0 or epoch == epochs-1:
            # training loss
            train_loss = bce(y_true=T_train, y_pred=outputs)
            self.loss_train.append(float(train_loss))

            # training accuracy
            train_acc = accuracy(predicted=(outputs>0.5), gold=T_train)
            self.accuracies_train.append(float(train_acc))


            # loss and accuracy for validation data
            if validation:
                X_val, t_val = validation
                X_val_bias = add_bias(X_val, self.bias)  # add bias
                T_val = t_val.reshape(-1, 1)             # reshape
                val_hidden_out, val_out = self.forward(X_val_bias)

                # loss + accuracy calculation
                val_loss = bce(y_true=T_val, y_pred=val_out)
                val_acc = accuracy(predicted=(val_out>0.5), gold=T_val)
                self.loss_val.append(float(val_loss))
                self.accuracies_val.append(float(val_acc))

                if tol is not None:
                    if (lowest_val_loss - val_loss) > tol:
                        lowest_val_loss = val_loss
                        epoch_no_improvement = 0
                    else:
                        epoch_no_improvement += 1

                # stopping early
                if epoch_no_improvement >= n_epochs_no_update:
                    if self.verbose: print(f"Epoch {epoch+1} - Loss: {val_loss:.4f}, Accuracy: {(val_acc*100):.2f}%\t(dev)")
                    self._epochs_trained = epoch + 1
                    break

            # print occasionally
            if (epoch+1) % max(1, epochs//5) == 0 or epoch == 0:
                if self.verbose:
                    if validation:
                        print(f"Epoch {epoch+1:3} - Loss: {val_loss:.4f}, Accuarcy: {(val_acc*100):.2f}%\t(dev)")
                    else:
                        print(f"Epoch {epoch+1:3} - Loss: {train_loss:.4f}, Accuracy: {(train_acc*100):.2f}%\t(train)")



    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)

        forw = self.forward(Z)[1]
        score = forw[:, 0]

        return (score > 0.5)


def repeated_run(model_args, train_data, eval_data, n_runs=10, **fit_args):
    """Repeatedly run 'n_runs' times and measure each.
    Save best, standard deviation, and average."""
    (dim_hidden, verbose) = model_args
    (X_train, t_train) = train_data
    (X_val, t_val) = eval_data

    best_acc = 0
    all_accuracies = []
    best_cl = None

    for run in range(n_runs):
        # new classifier each run
        cl = MLPBinaryLinRegClass(dim_hidden=dim_hidden, verbose=verbose)

        # training   (seen data)
        cl.fit(X_train=X_train, t_train=t_train, **fit_args)    # training   (seen data)
        predictions = cl.predict(X_val)                         # predicting (unseen data)
        acc = accuracy(predictions, t_val)

        all_accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_cl = cl

        print(f"Run {run + 1}/{n_runs}: accuracy = {acc:.4f}")

    return best_cl, best_acc, all_accuracies


def main():
    from data import X_train, t2_train, X_val, t2_val
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1
    epochs = args.epochs                # default: 3
    tolerance = args.tolerance          # default: 1.0
    patience = args.patience            # default: 10
    dim_hidden = args.hidden_dim        # default: 6
    verbose = args.verbose              # default: False
    # ---------------------------------------------------


    # ----------------- 1. normalization ----------------
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Normalizing test data
    norm_train = standard(X_train, train_mean, train_std)
    X_train = norm_train

    # Normalizing validation data
    norm_val = standard(X_val, train_mean, train_std)
    X_val = norm_val
    # --------------------------------------------------


    # ---------------- 2. Regression -------------------
    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" +
        f"- Tolerance:  [{tolerance}]\n" +
        f"- Patience:   [{patience}]\n" +
        f"- Hidden dim: [{dim_hidden}]\n"
    )

    # repeated run parameters
    n_runs = 10
    train_params = {
        "lr": learning_rate,
        "epochs": epochs,
        "tol": tolerance,
        "n_epochs_no_update": patience,
        "validation": (X_val, t2_val)
    }


    # run (n_runs=10) times, get mean, std and best
    print("="*11+f" Starting {n_runs} runs "+"="*11)
    start = time.time()
    cl, best_acc, all_acc = repeated_run(
        model_args=(dim_hidden, verbose),   # passed to model initialization (verbose / dim_hidden)
        n_runs=n_runs,                      # train and measure x times
        train_data=(X_train, t2_train),     # training data
        eval_data=(X_val, t2_val),          # evaluation
        **train_params                      # lr, epochs, patience, tolerance
    )
    end = (time.time() - start)

    # standard deviation and mean
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)

    print("\n"+"-"*40)
    print(f"Total runtime:   {end:.2f}s")
    print(f"Best accuracy:   {best_acc:.4f}")
    print(f"Mean accuracy:   {mean_acc:.4f}")
    print(f"Std deviation:   {std_acc:.4f}")
    print("-"*40)
    # --------------------------------------------------


    # ---------------- 3. Plotting ---------------------
    # accuracy curve
    acc_train = cl.accuracies_train
    acc_val = cl.accuracies_val
    plot_curves(res_train=acc_train, res_dev=acc_val, label="Accuracy")

    # loss curve
    loss_train = cl.loss_train
    loss_val = cl.loss_val
    plot_curves(res_train=loss_train, res_dev=loss_val, label="Loss")

    plot_decision_regions(X_train, t2_train, cl)
    # --------------------------------------------------
    print("\n"+"="*40)


if __name__ == "__main__":
    main()

