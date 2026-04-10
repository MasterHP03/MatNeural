import numpy as np

class Layer:
    """Parent class for all layers"""
    # Magic method: Automatically call forward(X) when an object is called like method
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, dZ):
        pass

class LinearLayer(Layer):
    """Fully connected layer that computes linear transformation with weights and biases"""
    def __init__(self, n_input, n_output):
        # He Initialization: prevent vanishing gradient by adjusting variance
        # W Shape: (n_output, n_input)
        self.W = np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)
        # B Shape: (n_output, 1) - n_col = 1 for broadcasting on matrix addition
        self.B = np.zeros((n_output, 1))

        self.cache_X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        # Cache X for backpropagation
        self.cache_X = X
        # Linear Transformation (Z = WX + B)
        # X Shape: (n_input, batch) -> Z Shape: (n_output, batch)
        Z = self.W @ X + self.B
        return Z

    def backward(self, dZ):
        # Differentiation of weight: dW = dZ * X^T
        self.dW = dZ @ self.cache_X.T

        # dB = sum of dZ in the batch (axis=1)
        # keepdims=True for not collapsing the dimension (n_output, 1)
        self.dB = np.sum(dZ, axis=1, keepdims=True)

        # dX = W^T * dZ for back propagation
        dX = self.W.T @ dZ
        return dX

class ReLULayer(Layer):
    """Leaky ReLU activation function for non-linearity by filtering negative values"""
    def __init__(self, coeff_leaky=0.01):
        self.cache_A = None
        # coeff_leaky: prevent dying ReLU (gradient being 0 for negative)
        self.coeff_leaky = coeff_leaky

    def forward(self, Z):
        # Z if Z > 0, 0.01 * Z if Z < 0 (Leaky)
        A = np.maximum(Z * self.coeff_leaky, Z)
        self.cache_A = A
        return A

    def backward(self, dA):
        A = self.cache_A
        # diff(ReLU) = 1 for positive, 0.01 for negative
        dZ = dA * ((A >= 0) + self.coeff_leaky * (A < 0))
        return dZ

class SigmoidLayer(Layer):
    """Activation function that compresses output into a probability between 0 and 1"""
    def __init__(self):
        self.cache_A = None

    def forward(self, Z):
        # Sigmoid: A = 1 / (1 + e^(-Z))
        A = 1 / (1 + np.exp(-Z))
        self.cache_A = A
        return A

    def backward(self, dA):
        A = self.cache_A
        # diff(Sigmoid) = A * (1 - A)
        dZ = dA * (A * (1 - A))
        return dZ

class TanhLayer(Layer):
    """Activation function that maps output into smooth curve that ranges (-1, 1)"""
    def __init__(self):
        self.cache_A = None

    def forward(self, Z):
        # Tanh: A = tanh(Z)
        A = np.tanh(Z)
        self.cache_A = A
        return A

    def backward(self, dA):
        A = self.cache_A
        # diff(Tanh) = 1 - A^2
        dZ = dA * (1 - A ** 2)
        return dZ

class SoftmaxLayer(Layer):
    """Layer that transforms output into the probability distribution (of which sum is 1) for classification"""
    def __init__(self):
        self.cache_A = None

    def forward(self, Z):
        # Prevent overflow of e^Z by subtracting with max value
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        # Proportion (Z / sum(Z))
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        self.cache_A = A
        return A

    def backward(self, dA):
        # If coupled with CrossEntropyLoss, diff(Softmax) * diff(CrossEntropy) = (prediction - real)
        # Therefore, here we just pass the error
        return dA