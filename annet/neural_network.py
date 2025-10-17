import numpy as np


# ========= Activation functions ==================
def relu(x):
    """ReLU activation function in hidden layer"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU used in backpropagation"""
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """Sigmoid activation function in output layer"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Not used due to BCE loss"""
    fx = sigmoid(x)
    return fx * (1 - fx)


# ========== Data Normalization ==================
def normalize(x, min, max):
    """Min-Max Normalization (0-1) with numpy array x"""
    return (x - min) / (max - min)


# ========== Loss function ==================
def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy loss"""
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


def print_prediction(nn, test, true):
    print("\n========== Testing Predictions ==========")
    norm = normalize(test, train_min, train_max)    # norm
    prediction = nn.predict(norm)                   # predict

    for i in range(test.shape[0]):
        features = test[i]
        label = true[i, 0]

        pred_prob = prediction[i, 0]
        pred_class = "Spam" if pred_prob >= 0.5 else "Not Spam"

        # Combine features and label in one line
        print(f"{features} = [{label}]  -->  Pred: {pred_prob*100:.2f}% ({pred_class})")


"""
==[ 3-2-1 Neural Network Structure ]==
Goal: given counts/presence of the words "free", "win", "offer" in an email, predict probability of spam

Input layer:
    * 3 neurons (features)
    * Inputs:
        x1 = count/presence of "free"
        x2 = count/presence of "win"
        x3 = count/presence of "offer"
    * shape for single sample: (1,3)
    * shape for batches: (n,3)

Hidden layer:
    * 2 neurons
    * Activation: ReLU
    * Weights: W1, shape: (2,3)
    * Bias: b1, shape: (2,)

Output layer:
    * 1 neuron
    * Activation: Sigmoid
    * Weights: W2, shape: (1,2)
    * Bias: b2, shape: (1,)

Layer summary:

  Input Layer     Hidden Layer   Output Layer
  (3 neurons)     (2 neurons)    (1 neuron)

   [X1] ──────┐     [H1]───┐
   [X2] ──────┼─►────┤     ├──►──[O1]
   [X3] ──────┘     [H2]───┘        

"""


class SimpleNN:

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr
        # input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        # hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        # input to hidden
        Z1 = X @ self.W1 + self.b1      # Z = XW + b
        A1 = relu(Z1)                   # Activation

        # hidden to output
        Z2 = A1 @ self.W2 + self.b2     # Z = A1W2 + b2
        A2 = sigmoid(Z2)                # Activation = prediction

        return Z1, A1, A2

    def backward(self, X, y_true, Z1, A1, A2):
        N = y_true.shape[0]  # number of samples

        # output layer
        dZ2 = A2 - y_true                # L/Z  (sigmoid + BCE)
        dW2 = A1.T @ dZ2 / N             # L/W  (weight gradients)
        dB2 = np.sum(dZ2, axis=0) / N    # L/b  (bias gradients) L/b

        # hidden layer
        dA2 = dZ2 @ self.W2.T            # L/A
        dZ1 = dA2 * relu_derivative(Z1)  # L/Z
        dW1 = X.T @ dZ1 / N              # L/W
        dB1 = np.sum(dZ1, axis=0) / N    # L/b

        # output update
        self.W2 -= self.lr * dW2         # weight update
        self.b2 -= self.lr * dB2         # bias update

        # hidden update
        self.W1 -= self.lr * dW1         # weight update
        self.b1 -= self.lr * dB1         # bias update

    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            Z1, A1, A2 = self.forward(X)
            self.backward(X, Y, Z1, A1, A2)

            # training loop
            if (epoch + 1) % 50 == 0 or epoch == 0:
                loss = binary_cross_entropy(Y, A2)
                print(f"Epoch {epoch + 1:3}, Loss: {loss:.4f}")

    def predict(self, X):
        _, _, A2 = self.forward(X)
        return A2


np.set_printoptions(precision=4, suppress=True)

# Input
X_orig = np.array([
    [2, 0, 1],  # "free" and "offer"
    [0, 1, 0],  # "win"
    [1, 0, 0],  # "free"
    [3, 1, 2],  # "free", "win", and "offer"
    [0, 0, 0],  # no keywords
])

# Normalize
train_min = X_orig.min(axis=0)  # column-wise min  (min for training data)
train_max = X_orig.max(axis=0)  # column-wise max  (max for training data)
X = normalize(X_orig, train_min, train_max)

# Labels: spam = 1, not spam = 0
y_true = np.array([
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
])

nn = SimpleNN(input_size=3, hidden_size=2, output_size=1, lr=0.1)
nn.train(X, y_true, epochs=500)
np.savez("spam_model.npz", W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2,  # save model
         train_min=train_min, train_max=train_max)

# New test samples (unseen during training)
X_test = np.array([
    [1, 1, 1],  # all keywords present -> likely spam
    [0, 1, 1],  # "win" and "offer" -> probably spam
    [1, 0, 1],  # "free" and "offer" -> likely spam
    [0, 0, 0],  # no keywords -> not spam
    [0, 1, 0],  # "win" only -> spam? borderline
    [1, 0, 0],  # "free" only -> borderline
    [0, 0, 1],  # "offer" only -> borderline
])
y_true_test = np.array([
    [1],  # spam
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
    [0],  # not spam
])

# (normalization done inside)
print("\nSeen data")
print_prediction(nn, X_orig, y_true)        # training data predictions
print("\nUnseen data")
print_prediction(nn, X_test, y_true_test)   # test data predictions

"""
Usage for loading the model  (name "spam_model.npz")

data = np.load("spam_model.npz")
nn.W1 = data["W1"]
nn.b1 = data["b1"]
nn.W2 = data["W2"]
nn.b2 = data["b2"]
train_min = data["train_min"]
train_max = data["train_max"]
"""