import numpy as np
from matplotlib import pyplot as plt


# Plotter from precode
def plot_decision_regions(X, t, clf=[], size=(8, 6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size)  # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='Paired')

    plt.scatter(X[:, 0], X[:, 1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.savefig(f"{clf.__class__.__name__}.png")
    plt.show()


# plotter for Loistic regression g)
def plot_curves(res_train, res_dev, label, log_x=False):
    """Plots curve for given result (loss / accuracy) in same figure as function of epochs"""

    epochs = range(len(res_train))

    plt.figure(figsize=(8, 6))

    # optional scaling for x-axis (epochs)
    if log_x:
        plt.xscale("log")

    plt.plot(epochs, res_train, label="Training")
    plt.plot(epochs, res_dev, label="validation")
    plt.title(f"{label} over epochs")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.savefig(f"{label}.png")
    plt.show()
