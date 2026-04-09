import numpy as np

class Adam:
    def __init__(self, model, learning_rate=0.1, beta_v=0.9, beta_s=0.999, epsilon=1e-8):
        self.model = model
        self.lr = learning_rate
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.epsilon = epsilon
        self.t = 0

        self.v_W = [np.zeros_like(l.W) if hasattr(l, 'W') else None for l in model.layers]
        self.s_W = [np.zeros_like(l.W) if hasattr(l, 'W') else None for l in model.layers]
        self.v_B = [np.zeros_like(l.B) if hasattr(l, 'B') else None for l in model.layers]
        self.s_B = [np.zeros_like(l.B) if hasattr(l, 'B') else None for l in model.layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'W'):
                self.v_W[i] = self.beta_v * self.v_W[i] + (1 - self.beta_v) * layer.dW
                self.s_W[i] = self.beta_s * self.s_W[i] + (1 - self.beta_s) * layer.dW ** 2
                v_W_hat = self.v_W[i] / (1 - self.beta_v ** self.t)
                s_W_hat = self.s_W[i] / (1 - self.beta_s ** self.t)

                layer.W -= self.lr * v_W_hat / (np.sqrt(s_W_hat) + self.epsilon)

                self.v_B[i] = self.beta_v * self.v_B[i] + (1 - self.beta_v) * layer.dB
                self.s_B[i] = self.beta_s * self.s_B[i] + (1 - self.beta_s) * layer.dB ** 2
                v_B_hat = self.v_B[i] / (1 - self.beta_v ** self.t)
                s_B_hat = self.s_B[i] / (1 - self.beta_s ** self.t)

                layer.B -= self.lr * v_B_hat / (np.sqrt(s_B_hat) + self.epsilon)