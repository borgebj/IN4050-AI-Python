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
def normalize(x):
    """Min-Max Normalization (0-1) with numpy array x"""
    x_min = x.min(axis=0)  # column-wise min
    x_max = x.max(axis=0)  # column-wise max
    return (x - x_min) / (x_max - x_min)  # modifies in place


# ========== Loss function ==================
def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy loss"""
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


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

    def train(self, X, y_true, epochs=100):
        for epoch in range(epochs):
            Z1, A1, A2 = self.forward(X)
            self.backward(X, y_true, Z1, A1, A2)

            # training loop
            if (epoch + 1) % 50 == 0 or epoch == 0:
                loss = binary_cross_entropy(y_true, A2)
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
X = normalize(X_orig)

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

# prediction / testing
print("\n========== Testing Predictions ==========")

predictions = nn.predict(X)

for i in range(X_orig.shape[0]):
    features = X_orig[i]
    label = y_true[i, 0]
    pred_prob = predictions[i, 0]
    pred_class = "Spam" if pred_prob >= 0.5 else "Not Spam"

    # Combine features and label in one line
    print(f"{features} = [{label}]  --> Pred: {pred_prob:.4f} ({pred_class})")
