from linreg import standard, NumpyClassifier, add_bias, accuracy
from plotter import plot_decision_regions, plot_curves
from utility import parse_args
import numpy as np
import time

from softmax import softmax, onehot, cce    # for multiclass
from mlp import logistic, logistic_diff     # for binary    
from logreg import bce                      # for binary


class MLP(NumpyClassifier):
    """Neural network base for use with binary and multi-class"""

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


    def initialize_weights(self, dim_in, dim_out):
        # weights - (input to hidden)
        self.weights1 = (np.random.rand(
            dim_in + 1,
            self.dim_hidden) * 2 - 1) / np.sqrt(dim_in)     # ~[ -0.408,  0.408 ] range for 6

        # weights - (hidden to output)
        self.weights2 = (np.random.rand(
            self.dim_hidden + 1,
            dim_out) * 2 - 1) / np.sqrt(self.dim_hidden)    # ~[ -0.408,  0.408 ] range for 6


    def forward_hidden(self, X):
        Z_hidden = X @ self.weights1            # Z1 = XW1 (+b)
        A_hidden = self.activ(Z_hidden)         # A2 = logistic(Z1) - always sigmoid

        hidden_outs = add_bias(
            A_hidden, self.bias)                # bias for hidden layer

        return hidden_outs
    
    
    def forward(self, X):
        hidden_outs = self.forward_hidden(X)
        Z_out = hidden_outs @ self.weights2      # Z2 = XW2 (+b)
        outputs = self.output_activation(Z_out)  # A2 = activ(Z2) - sigmoid / softmax

        return hidden_outs, outputs
    
    
    def fit(self, X_train, t_train, lr=0.001, epochs=100, tol=0.0, n_epochs_no_update=5, validation=None):
        self.lr = lr

        # Turn t_train into a column vector, a N*1 matrix:
        T_train, dim_out = self.process_labels(t_train)

        dim_in = X_train.shape[1]       # how many features (column)
        dim_out = dim_out               # how many output neurons/classes

        # initialize weights for each layer
        self.initialize_weights(dim_in, dim_out)
        
        # adding bias column to data (X)
        X_train_bias = add_bias(X_train, self.bias)

        # keep track of lowest loss and epoch with improvements
        lowest_val_loss = np.inf
        epochs_no_improvements = 0

        for epoch in range(epochs):
            # One epoch

            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)   # prediction (hidden and output)

            # The delta term on the output node:
            out_deltas = self.loss_diff(outputs, T_train)

            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T

            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))

            # Update the weights:
            self.weights2 -= self.lr * (hidden_outs.T @ out_deltas)         # gradient 1
            self.weights1 -= self.lr * (X_train_bias.T @ hiddenact_deltas)  # gradient 2

            # training metrics
            train_loss = self.loss(T_train, outputs)

            if isinstance(self, MLPMultiClass):
                # outputs.shape = (N,C) -> rows = samples, columns = class prob.
                # t_train.shape = (N, ) -> each samples label as index
                train_acc = self.accuracy(outputs, t_train) # pass outputs and raw labels
            else:
                # outputs.shape = (N,1) -> sigmoid output for each sample
                # T_train.shape = (N,1) -> column vector of labels
                train_acc = self.accuracy(outputs, T_train) # pass outputs and reshaped labels

            self.loss_train.append(float(train_loss))
            self.accuracies_train.append(float(train_acc))

            # validation
            if validation:                
                X_val, t_val = validation
                X_val_bias = add_bias(X_val, self.bias)  # add bias
                T_val, _ = self.process_labels(t_val)
                _, val_out = self.forward(X_val_bias)
                
                # validation metrics
                val_loss = self.loss(y_true=T_val, y_pred=val_out)

                if isinstance(self, MLPMultiClass):
                    val_acc = self.accuracy(val_out, t_val)
                else:
                    val_acc = self.accuracy(val_out, T_val)

                self.loss_val.append(float(val_loss))
                self.accuracies_val.append(float(val_acc))

                if tol is not None:
                    if (lowest_val_loss - val_loss) > tol:
                        lowest_val_loss = val_loss
                        epochs_no_improvements = 0
                    else:
                        epochs_no_improvements += 1

                # Early stopping
                if epochs_no_improvements >= n_epochs_no_update:
                    self._epochs_trained = epoch + 1
                    break

            # print occasionally (verbose)
            if self.verbose:
                if (epoch+1) % max(1, epochs//5) == 0 or epoch == 0:
                    if validation:
                        print(f"Epoch {epoch+1:3} - Loss: {val_loss:.4f}, Accuarcy: {(val_acc*100):.2f}%\t(dev)")
                    else:
                        print(f"Epoch {epoch+1:3} - Loss: {train_loss:.4f}, Accuracy: {(train_acc*100):.2f}%\t(train)")

    
    # implemented in respective binary/multclass classes
    def output_activation(self, x): raise NotImplementedError
    def loss(self, y_true, y_pred): raise NotImplementedError
    def loss_diff(self, outputs, T_train): raise NotImplementedError
    def accuracy(self, predicted, gold): raise NotImplementedError
    def process_labels(self, t_train): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class MLPBinary(MLP):
    """Neural Network for binary regression"""

    # Sigmoid as output activation
    def output_activation(self, x):
        return logistic(x)
    
    # labels treated as column vector Nx1
    def process_labels(self, t_train):
        return t_train.reshape(-1, 1), 1
    
    # Binary Cross Entropy for binary loss
    def loss(self, y_true, y_pred):
        return bce(y_true=y_true, y_pred=y_pred)
    
    # Simplified derivative of Cross Entropy using sigmoid
    def loss_diff(self, outputs, T_train):
        return (outputs - T_train)
    
    def accuracy(self, predicted, gold):
        predicted = (predicted > 0.5)
        return accuracy(predicted=predicted, gold=gold)
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)

        forw = self.forward(Z)[1]  # [1] gets output layer results
        score = forw[:, 0]

        return (score > 0.5)
    
    
class MLPMultiClass(MLP):
    """Neural Network for multiclass regression"""
    
    def __init__(self, bias=-1, dim_hidden=6, verbose=False):
        super().__init__(bias, dim_hidden, verbose)
        self.classes = None

    # Softmax as output activation
    def output_activation(self, x):
        return softmax(x)
    
    # labels encoded using onehot   (from softmax.py)
    def process_labels(self, t_train):
        self.classes = np.unique(t_train)
        return onehot(t_train, self.classes), len(self.classes)
    
    # Categorical Cross Entropy for multiclass loss   (from softmax.py)
    def loss(self, y_true, y_pred):
        return cce(y_true=y_true, y_pred=y_pred)
    
    # Simplified derivative of Cross Entropy using softmax  (same as bce+sigmoid!)
    def loss_diff(self, outputs, t_train):
        return (outputs - t_train)
    
    def accuracy(self, predicted, gold):
        # argmax converts predicted.shape (N,C) to (N,) - 1D vector same as gold

        # if ndim == 1, then predicted = final output from predict()
        if predicted.ndim > 1:
            top_pred = np.argmax(predicted, axis=1)
        else:
            top_pred = predicted
        return accuracy(predicted=top_pred, gold=gold)
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)

        outs = self.forward(Z)[1]  # [1] gets output layer results

        pred_classes = np.argmax(outs,axis=1)

        return pred_classes



def repeated_run(model_args, train_data, eval_data, task, n_runs=10, **fit_args):
    """Repeatedly run 'n_runs' times and measure each.
    Save best, standard deviation, and average."""
    (dim_hidden, verbose) = model_args
    (X_train, t_train) = train_data
    (X_val, t_val) = eval_data

    best_acc = 0
    all_accuracies = []
    best_cl = None

    for run in range(n_runs):
        # classifier based on arg.task
        if task == "binary":
            cl = MLPBinary(dim_hidden=dim_hidden, verbose=verbose)
        elif task == "multiclass":
            cl = MLPMultiClass(dim_hidden=dim_hidden, verbose=verbose)

        # training   (seen data)
        cl.fit(X_train=X_train, t_train=t_train, **fit_args)    # training   (seen data)
        predictions = cl.predict(X_val)                         # predicting (unseen data)
        acc = cl.accuracy(predictions, t_val)

        all_accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_cl = cl
        
        width = len(str(n_runs))
        print(f"Run {run + 1:{width}}/{n_runs:{width}}: accuracy = {acc:.4f}")

    return best_cl, best_acc, all_accuracies


def main():
    from data import (
        X_train, X_val,             # input data
        t2_train, t2_val,           # binary labels
        t_multi_train, t_multi_val  # multiclass labels
    )
    print("="*40+"\n\n")
    # ---------------- 0. Command-line-args -------------
    args = parse_args()
    learning_rate = args.learning_rate  # default: 0.1
    epochs = args.epochs                # default: 3
    tolerance = args.tolerance          # default: 1.0
    patience = args.patience            # default: 10
    dim_hidden = args.hidden_dim        # default: 6
    task = args.task                    # default: binary
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

    # choose task
    if task == "binary":
        t_train, t_val = t2_train, t2_val
    elif task == "multiclass":
        t_train, t_val = t_multi_train, t_multi_val

    print(
        f"__Hyperparameters__\n" +
        f"- Learning:   [{learning_rate}]\n" +
        f"- Epochs:     [{epochs}]\n" +
        f"- Tolerance:  [{tolerance}]\n" +
        f"- Patience:   [{patience}]\n" +
        f"- Hidden dim: [{dim_hidden}]\n" +
        f"- Task:       [{task}]\n"
    )

    # repeated run parameters
    n_runs = 10
    train_params = {
        "lr": learning_rate,
        "epochs": epochs,
        "tol": tolerance,
        "n_epochs_no_update": patience,
        "validation": (X_val, t_val)
    }


    # run (n_runs=10) times, get mean, std and best
    print("="*11+f" Starting {n_runs} runs "+"="*11)
    start = time.time()
    cl, best_acc, all_acc = repeated_run(
        model_args=(dim_hidden, verbose),   # passed to model initialization (verbose / dim_hidden)
        task=task,                          # either Binary of Multiclass regression
        n_runs=n_runs,                      # train and measure x times
        train_data=(X_train, t_train),      # training data
        eval_data=(X_val, t_val),          # evaluation
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
    plot_curves(res_train=acc_train, res_dev=acc_val, label="Accuracy", log_x=True)

    # loss curve
    loss_train = cl.loss_train
    loss_val = cl.loss_val
    plot_curves(res_train=loss_train, res_dev=loss_val, label="Loss", log_x=True)

    plot_decision_regions(X_train, t_train, cl)
    # --------------------------------------------------
    print("\n"+"="*40)


if __name__ == "__main__":
    main()

