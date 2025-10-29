import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Generating the dataset
X, t_multi = make_blobs(
    n_samples=[400, 400, 400, 400, 400],    # 5 classes, 400 each
    centers=[
        [0, 1],
        [4, 2],
        [8, 1],                             # Center for 5 clusters
        [2, 0],
        [6, 0]],
    n_features=2,                           # 2 features = 2 coordinates (x,y)
    random_state=2024,
    cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5]
)

# Shuffling the dataset
indices = np.arange(X.shape[0])
seed = 2024
rng = np.random.RandomState(seed)           # seed for reproducibility
rng.shuffle(indices)                        # shuffles data before splitting

# Splitting into train, dev and test
X_train = X[indices[:1000], :]              # 50% train
X_val = X[indices[1000:1500], :]            # 25% dev/validation
X_test = X[indices[1500:], :]               # 25% test

# labels for data (multi-class)
t_multi_train = t_multi[indices[:1000]]     # labels for train
t_multi_val = t_multi[indices[1000:1500]]   # labels for dev/validation
t_multi_test = t_multi[indices[1500:]]      # labels for test

# converting 5-class labels to binary labels
t2_train = (t_multi_train >= 3).astype('int')
t2_val = (t_multi_val >= 3).astype('int')   # classes 3,4  ->  1 ; classes 0,1,2  ->  0
t2_test = (t_multi_test >= 3).astype('int')


# ========== Plotting ==========
def plot_multiclass():
    plt.figure(figsize=(8,6)) # You may adjust the size
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
    plt.title("Multi-class set")
    plt.show()


def plot_binary():
    plt.figure(figsize=(8,6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
    plt.title("Binary set")
    plt.show()
