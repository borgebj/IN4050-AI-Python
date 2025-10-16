import numpy as np


# ========= Activation functions ==================
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# ========== Data Normalization ==================
def normalize(x):
    """Min-Max Normalization (0-1) with numpy array x"""
    x_min = x.min(axis=0)  # column-wise min
    x_max = x.max(axis=0)  # column-wise max
    x[:] = (x - x_min) / (x_max - x_min)  # modifies in place


# ========== Loss function ==================
def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    bce = -np.mean(y_true * np.log(y_pred) * (1 - y_true) * np.log(1 - y_pred))
    return bce


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy loss"""
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


"""
==[ 2-2-2 Neural Network Structure ]==
Goal:  given 'temperature' and 'humidity', predict the probability of:
    * Rain  weather (o1)
    * Windy weather (o2)  

Input layer:
    * 2 neurons (features)
    * Inputs:
        x1 = temperature    (°C)    0-40
        x2 = humidity       (%)     0-100
    * shape for single sample: (1,2)
    * Shape for batches: (n, 2)

Hidden layer:
    * 2 neurons
    * Activation: ReLU
    * Weights: W1, shape: (2,2)

Output layer:
    * 2 neurons
    * Activation: Sigmoid
    * Weights: W2, shape: (2,2)

    Input Layer    Hidden Layer     Output Layer
    (2 neurons)    (2 neurons)      (2 neurons)
    
       [X1] ────┐──── [H1] ────┐──── [O1]
             W1 │           W2 │
       [X2] ────┘──── [H2] ────┘──── [O2]
"""

np.set_printoptions(precision=2, suppress=True)

# truth-labels
y_true = np.array([
    [0, 1],  # sample 1 - no rain, windy
    [1, 1],  # sample 2 - rain, windy
    [0, 0]   # sample 3 - no rain, no windy
])

# ========== Input layer ==========
print(f"\n========== Input layer 0 ==========")
# 1 sample, 2 features
# feature 1 = 0.5
# feature 2 = 0.9
sample1 = [25, 40]  # moderate temp and humidity
sample2 = [30, 80]  # high temp and humidity
sample3 = [10, 20]  # low temp and humidity
X = np.array(
    [
        sample1,
        sample2,
        sample3
    ], dtype=float
)
print(f"\nPre norm:\n{X}")
normalize(X)
print(f"\nPost norm:\n{X}")

# ========== Hidden layer ==========
print(f"\n========== Hidden layer 1 ==========")
# 2 neurons in the hidden layer
W1 = np.array([[0.2, 0.8],
               [0.5, 0.1]])

# calculate Z values for hidden layer
# z1[0, 0] = z for neuron 1, sample 1
# z1[0, 1] = z for neuron 2, sample 1
# each row = each sample
Z1 = X @ W1

# Apply activation
# A1[0, 0] = a for neuron 1, sample 1
# A1[0, 1] = a for neuron 2, sample 1
A1 = relu(Z1)

print("\nWeights\n", W1)
print("\nZ1\n", Z1)
print("\nA1\n", A1)

# ========== Output layer ==========
print(f"\n========== Output layer 2 ==========")
# 2 neurons in the output layer
W2 = np.array([[0.3, 0.4],
               [0.6, 0.9]])

# linear transformation (calculate z)
Z2 = A1 @ W2

# activation
A2 = sigmoid(Z2)  # final predictions

print(f"\nResults:\n{A2}")
