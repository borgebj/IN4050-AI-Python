from linreg import standard, NumpyClassifier, add_bias, accuracy
from plotter import plot_decision_regions
from utility import parse_args
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


    def fit(self, X_train, t_train, lr=0.001, epochs=100):
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

        for e in range(epochs):
            # One epoch

            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)

            # The delta term on the output node:
            out_deltas = (outputs - T_train)                    # Loss = (Y - T)

            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T

            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))

            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas


    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)

        forw = self.forward(Z)[1]
        score = forw[:, 0]

        return (score > 0.5)



def main():
    from data import X_train, t2_train, X_val, t2_val
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1  (best: 1.0)
    epochs = args.epochs                # default: 3    (best: ~100)
    # tolerance = args.tolerance          # default 1.0   (best: 1.0)
    # patience = args.patience            # default: 10   (best: 2)
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
    cl = MLPBinaryLinRegClass(verbose=verbose)

    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" #+
        # f"- Tolerance:  [{tolerance}]\n" +
        # f"- Patience:   [{patience}]\n"
    )

    # training   (seen data)
    cl.fit(
        X_train=X_train,
        t_train=t2_train,
        lr=learning_rate, epochs=epochs,            # hyperparameters (1)
    )

    predictions = cl.predict(X_val)                 # predicting (unseen data)
    print("\nAccuracy on the validation set:", accuracy(predictions, t2_val))
    # ---------------- ---------------- ----------------


    # ---------------- 3. Plotting ---------------- ----
    plot_decision_regions(X_train, t2_train, cl)
    # ---------------- ---------------- ----------------
    print("\n\n"+"="*40)


if __name__ == "__main__":
    main()

