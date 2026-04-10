import numpy as np
from core.layers import LinearLayer, ReLULayer
from core.models import Sequential
from core.optimizers import Adam
import matplotlib.pyplot as plt

tX = np.linspace(0, 2 * np.pi, 100).reshape(1, 100)
tX_norm = (tX - np.mean(tX)) / np.std(tX)

tY = np.sin(tX)

predictions = None

model = Sequential(
    LinearLayer(1, 16),
    ReLULayer(),
    LinearLayer(16, 16),
    ReLULayer(),
    LinearLayer(16, 1)
)

optimizer = Adam(model)

for epoch in range(20000):
    predictions = model(tX_norm)
    dLoss = 2 * (predictions - tY) / tY.shape[1]

    model.backward(dLoss)
    optimizer.step()

    if epoch % 1000 == 0:
        loss = np.mean((predictions - tY) ** 2)
        print(f"Epoch [{epoch:5d}/20000] | Loss {loss:.6f}")

plt.plot(tX[0], tY[0], label="Real sin(x)", color="blue")
plt.plot(tX[0], predictions[0], label="Predicted sin(x)", color="red", linestyle="--")
plt.legend()
plt.title("AI Sine Wave Regression")
plt.show()