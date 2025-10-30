import numpy as np


# ============== RUN SETUP ===================
def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (standard(X_train, mean, std),
            standard(X_val, mean, std),
            standard(X_test, mean, std))


def select_eval(X_train, t_train, X_val, t_val, X_test, t_test, eval_set):
    if eval_set == "train":
        return X_train, t_train
    elif eval_set == "test":
        return X_test, t_test
    else:  # "validation"
        return X_val, t_val


# ============== NORMALIZATION ===================
def standard(X, mean, std):
    """Standard scaler aka Z-score
    Uses passed mean and std (must use same as training!)"""
    return (X - mean) / std


# ============== LOSS ===================
def mse(y_true, y_pred):
    """MSE loss to present losses across epochs
    Step 1-2 includes loss for single sample"""
    # 1. calculates error (y - p)
    # 2. squares errors ^2
    # 3. sums errors        (numpy internal)
    # 4. averages           (numpy internal)
    return np.mean((y_true - y_pred) ** 2)


def bce(y_true, y_pred):
    """BCE loss for binary classification"""
    # 1. sample formula:   -[ylog(p) + (1-y)log(1-p)]
    # 2. sum over samples
    # 3. avg. sum for mean loss
    # eps is a tiny value added to counter division by zero
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def cce(y_true, y_pred):
    """Calculates loss with categorical Cross Entropy (cce)"""
    # (1/N) * sum(y * log(p))       (sums over classes)
    eps = 1e-15
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


# ============== ACTIVATION ===================
def logistic(x):
    """Simple sigmoid for logreg using numpy"""
    return 1 / (1 + np.exp(-x))


def logistic_diff(y):
    return y * (1 - y)


def softmax(X):
    """Softmax activation function for multi-class output"""
    # shifting (x-max(x)) ensures numerical stability
    # axis=1 ensures function applied across each row
    # keepdims ensures original dimensions maintained
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / exp_X.sum(axis=1, keepdims=True)


# ============== ENCODING ===================
def onehot(labels, classes):
    C = len(classes)        # no. classes
    N = labels.shape[0]     # no. samples
    label_idx = {label: i for i, label in enumerate(classes)}

    t_onehot = np.zeros((N, C))  # [0,...,0] w/ dimension sample X class

    # marks appropriate class as 1 : [0,...,1,...,0]
    for i, label in enumerate(labels):
        t_onehot[i, label_idx[label]] = 1

    return t_onehot


# ============== Evaluation ===================
def accuracy(predicted, gold):
    """Compares predicted to actual (gold)"""
    return np.mean(predicted == gold)


def precision_recall(predicted, gold, positive=1):
    """Calculates precision and recall based on predictions and true labels
    Used in task 3"""

    # true positive - predictive positive and is true
    tp = np.sum((predicted == positive) & (gold == positive))   # both true

    # false positive - predicted positive but is false
    fp = np.sum((predicted == positive) & (gold != positive))   # pred. true, gold false

    # false negative - predicted negative but is true
    fn = np.sum((predicted != positive) & (gold == positive))   # pred. false, gold true

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0

    return precision, recall
