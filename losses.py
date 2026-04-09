import numpy as np

class Loss:
    """Parent class for all loss functions"""
    def __call__(self, predictions, targets):
        # Magic method: auto call forward()
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        pass

    def backward(self):
        pass

class MSELoss(Loss):
    """
    Mean Squared Error
    - Usually used for Regression
    """
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        # Caches predictions and targets (real values) for differentiation on back propagation
        self.predictions = predictions
        self.targets = targets

        # MSE: mean(square(error))
        # Squared in order to measure magnitude of negative errors and give more penalty for larger error
        return np.mean((predictions - targets) ** 2)

    def backward(self):
        # diff(MSE): mean(2 * error)
        N = self.targets.shape[1] # Number of data (batch size)
        return 2 * (self.predictions - self.targets) / N

class CrossEntropyLoss(Loss):
    """
    Cross Entropy
    - Usually used for classification (where we guess probabilities)
    - Usually coupled with Softmax
    """
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        # Cross Entropy: mean(-(target * log(prediction))
        # epsilon to prevent log(0) = -inf
        return -np.mean(np.sum(targets * np.log(predictions + 1e-8), axis=0))

    def backward(self):
        # Coupled with Softmax, diff(Cross Entropy) * diff(Softmax) becomes just a simple subtraction
        # So, dLoss = (pred - target) / N
        N = self.targets.shape[1]
        return (self.predictions - self.targets) / N