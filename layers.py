import numpy as np

class Layer:
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, dZ):
        pass

class LinearLayer(Layer):
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)
        self.B = np.zeros((n_output, 1))

        self.cache_X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.cache_X = X
        Z = self.W @ X + self.B
        return Z

    def backward(self, dZ):
        self.dW = dZ @ self.cache_X.T
        self.dB = np.sum(dZ, axis=1, keepdims=True)

        dX = self.W.T @ dZ
        return dX

class SigmoidLayer(Layer):
    def __init__(self):
        self.cache_A = None

    def forward(self, Z):
        A = 1 / (1 + np.exp(-Z))
        self.cache_A = A
        return A

    def backward(self, dA):
        A = self.cache_A
        dZ = dA * (A * (1 - A))
        return dZ

class ReLULayer(Layer):
    def __init__(self, coeff_leaky):
        self.cache_A = None
        self.coeff_leaky = coeff_leaky

    def forward(self, Z):
        A = np.maximum(Z * self.coeff_leaky, Z)
        self.cache_A = A
        return A

    def backward(self, dA):
        A = self.cache_A
        dZ = dA * ((A >= 0) + self.coeff_leaky * (A < 0))
        return dZ