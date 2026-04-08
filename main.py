import numpy as np

class Layer:
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, dZ):
        pass

    def update(self, learning_rate=0.1, beta_v=0.9, beta_s=0.999, epsilon=1e-8):
        pass

class LinearLayer(Layer):
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)
        self.B = np.zeros((n_output, 1))

        self.v_W = np.zeros_like(self.W)
        self.s_W = np.zeros_like(self.W)
        self.v_B = np.zeros_like(self.B)
        self.s_B = np.zeros_like(self.B)

        self.t = 0

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

    def update(self, learning_rate=0.1, beta_v=0.9, beta_s=0.999, epsilon=1e-8):
        self.t += 1

        self.v_W = beta_v * self.v_W + (1 - beta_v) * self.dW
        self.s_W = beta_s * self.s_W + (1 - beta_s) * self.dW ** 2
        v_W_hat = self.v_W / (1 - beta_v ** self.t)
        s_W_hat = self.s_W / (1 - beta_s ** self.t)
        self.W = self.W - learning_rate * v_W_hat / (np.sqrt(s_W_hat) + epsilon)

        self.v_B = beta_v * self.v_B + (1 - beta_v) * self.dB
        self.s_B = beta_s * self.s_B + (1 - beta_s) * self.dB ** 2
        v_B_hat = self.v_B / (1 - beta_v ** self.t)
        s_B_hat = self.s_B / (1 - beta_s ** self.t)
        self.B = self.B - learning_rate * v_B_hat / (np.sqrt(s_B_hat) + epsilon)

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

class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, X):
        return self.forward(X)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dOut):
        dout = dOut
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, learning_rate=0.1, beta_v=0.9, beta_s=0.999, epsilon=1e-8):
        for layer in self.layers:
            layer.update(learning_rate, beta_v, beta_s, epsilon)

tX = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

tY = np.array([[0, 1, 1, 0]])

success = 0

for trial in range(100):
    model = Sequential(
        LinearLayer(2, 4),
        ReLULayer(coeff_leaky=0.01),
        LinearLayer(4, 1),
        SigmoidLayer()
    )

    predictions = None
    for epoch in range(10000):
        predictions = model(tX)
        dLoss = (predictions - tY)

        model.backward(dLoss)
        model.step(learning_rate=0.1, beta_v=0.9, beta_s=0.999, epsilon=1e-8)

    print(f"Prediction {trial}\n", np.round(predictions, 2))
    if np.allclose(predictions, tY, atol=0.1):
        success += 1

print(success)