import numpy as np

class Adam:
    """
    Adam optimizer\n
    - Momentum v: 1st moment. "Previous directions"\n
    - RMSProp s: 2nd moment. Effectively adjusts step size. "Update less updated ones more"
        * Squared to remember magnitude of steps and then normalized by sqrt.
        * Its realization is more like "update more updated ones less", so it goes to the denominator.
        * Intuitively, it is like tracking the steep way cautiously and driving at full speed on flat region
    """
    def __init__(self, model, learning_rate=1e-3, beta_v=0.9, beta_s=0.999, epsilon=1e-8):
        self.model = model
        self.lr = learning_rate
        # beta_v: determines how much we keep the momentum
        self.beta_v = beta_v
        # beta_s: determines how sensitively we adjust the step size
        self.beta_s = beta_s
        # Prevents division by 0
        self.epsilon = epsilon
        # Epoch tracker for bias correction (Initially we don't have momentum; "Go faster on early updates")
        self.t = 0

        # Prepares zero matrices matching the shape of W and B for all layers
        self.v_W = [np.zeros_like(l.W) if hasattr(l, 'W') else None for l in model.layers]
        self.s_W = [np.zeros_like(l.W) if hasattr(l, 'W') else None for l in model.layers]
        self.v_B = [np.zeros_like(l.B) if hasattr(l, 'B') else None for l in model.layers]
        self.s_B = [np.zeros_like(l.B) if hasattr(l, 'B') else None for l in model.layers]

    def step(self):
        """Update W and B with gradients dW, dB"""
        # Step count++ (should be first added to prevent division by zero on bias correction)
        self.t += 1

        # Enumerate: for + foreach. gives both index and the item
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'W'): # choose only the layers having matrix W
                # 1. Update weights
                # 1) Next momentum: Prev Momentum : Current Gradient = Beta_v : 1 - Beta_v
                self.v_W[i] = self.beta_v * self.v_W[i] + (1 - self.beta_v) * layer.dW
                # 2) Next rate: Prev Step : (Curr Grad)^2 = Beta_s : 1 - Beta_s
                self.s_W[i] = self.beta_s * self.s_W[i] + (1 - self.beta_s) * layer.dW ** 2

                # 3) Bias Correction: Boosts initial v and s (initially they tend to stay near 0)
                #   as t gets larger, denominator approaches 1, removing the effect of this correction
                v_W_hat = self.v_W[i] / (1 - self.beta_v ** self.t)
                s_W_hat = self.s_W[i] / (1 - self.beta_s ** self.t)

                # 4) Finally, update weight
                # W = W - (lr * momentum) / (sqrt(step) + epsilon)
                layer.W -= self.lr * v_W_hat / (np.sqrt(s_W_hat) + self.epsilon)

                # 2. Update biases (same logic and math)
                self.v_B[i] = self.beta_v * self.v_B[i] + (1 - self.beta_v) * layer.dB
                self.s_B[i] = self.beta_s * self.s_B[i] + (1 - self.beta_s) * layer.dB ** 2
                v_B_hat = self.v_B[i] / (1 - self.beta_v ** self.t)
                s_B_hat = self.s_B[i] / (1 - self.beta_s ** self.t)

                layer.B -= self.lr * v_B_hat / (np.sqrt(s_B_hat) + self.epsilon)