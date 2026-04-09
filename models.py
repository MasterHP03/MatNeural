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