import pickle


class Sequential:
    """
    Container that sequentially connects several layers
    """
    def __init__(self, *layers):
        # Save layers as a list
        self.layers = list(layers)

    def __call__(self, X):
        # Magic method: auto forward propagation
        return self.forward(X)

    def add(self, layer):
        """Adds a layer to the sequential dynamically"""
        self.layers.append(layer)

    def forward(self, X):
        """
        Forward propagation
        - Sequentially pass the input X through the layers
        - Pipeline structure where the output of previous layer is passed to the next layer
        :param X: input to the sequential model
        :return: output of the sequential model
        """
        out = X
        for layer in self.layers:
            # Polymorphism: forward propagation whatever the layer is
            out = layer.forward(out)
        return out

    def backward(self, dOut):
        """
        Backward propagation
        - Pass the final error through the layers in reversed order
        - Gradient is passed throughout the layers, grounding on the Chain Rule
        :param dOut: error (loss) of the prediction
        """
        dout = dOut
        # Reverse the layer to propagate backwards
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def save(self, filename):
        params = [(l.W, l.B) if hasattr(l, 'W') else None for l in self.layers]
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        print(f"Saved weights and biases to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if params[i] is not None:
                layer.W, layer.B = params[i]
        print(f"Loaded weights and biases from {filename}")
