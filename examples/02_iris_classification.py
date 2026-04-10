import numpy as np
from core.layers import LinearLayer, ReLULayer, SoftmaxLayer
from core.losses import CrossEntropyLoss
from core.models import Sequential
from core.optimizers import Adam
from sklearn.datasets import load_iris

iris = load_iris()
tX = iris.data.T
tX = (tX - np.mean(tX, axis=1, keepdims=True)) / np.std(tX, axis=1, keepdims=True)

tY_labels = iris.target
tY = np.zeros((3, 150))
tY[tY_labels, np.arange(150)] = 1

predictions = None

model = Sequential(
    LinearLayer(4, 16),
    ReLULayer(),
    LinearLayer(16, 3),
    SoftmaxLayer()
)

criterion = CrossEntropyLoss()
optimizer = Adam(model, learning_rate=0.01)
n_loop = 5000

for epoch in range(n_loop):
    predictions = model(tX)
    loss_val = criterion(predictions, tY)

    dLoss = criterion.backward()
    model.backward(dLoss)
    optimizer.step()

    if epoch % 500 == 0 or epoch == n_loop - 1:
        pred_labels = np.argmax(predictions, axis=0)
        acc = np.mean(pred_labels == tY_labels) * 100

        print(f"Epoch [{epoch:4d}/{n_loop}] | Loss {loss_val:.4f} | Acc:[{acc:.2f}%]")
