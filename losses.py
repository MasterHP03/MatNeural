import numpy as np

class Loss:
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        pass

    def backward(self):
        pass

class MSELoss(Loss):
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        return np.mean((predictions - targets) ** 2)

    def backward(self):
        N = self.targets.shape[1]
        return 2 * (self.predictions - self.targets) / N

class CrossEntropyLoss(Loss):
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        return -np.mean(np.sum(targets * np.log(predictions + 1e-8), axis=0))

    def backward(self):
        N = self.targets.shape[1]
        return (self.predictions - self.targets) / N