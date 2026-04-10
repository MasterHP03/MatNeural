import numpy as np
import matplotlib.pyplot as plt

from layers import LinearLayer, ReLULayer, SoftmaxLayer
from losses import CrossEntropyLoss
from models import Sequential
from optimizers import Adam
import time

np.random.seed(42)
N = 300
K = 2
X = np.zeros((N * K, 2))
y = np.zeros(N * K, dtype='uint8')

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
    y[ix] = j

tX = X.T
tY = np.zeros((K, N * K))
tY[y, np.arange(N * K)] = 1

model = Sequential(
    LinearLayer(2, 64),
    ReLULayer(),
    LinearLayer(64, 64),
    ReLULayer(),
    LinearLayer(64, 2),
    SoftmaxLayer(),
)

criterion = CrossEntropyLoss()
optimizer = Adam(model, learning_rate=0.01)

print("Training model...")
n_loop = 1000
start_time = time.time()
for epoch in range(n_loop):
    predictions = model(tX)

    loss_val = criterion(predictions, tY)
    dLoss = criterion.backward()

    model.backward(dLoss)
    optimizer.step()

    if epoch % (n_loop / 10) == 0 or epoch == n_loop - 1:
        acc = np.mean(np.argmax(predictions, axis=0) == y) * 100
        print(f"Epoch [{epoch:4d}/{n_loop}] | Loss {loss_val:.4f} | Acc: {acc:.1f}%")

end_time = time.time()
print(f"Training complete! (Time elapsed: {end_time - start_time:.2f}s)")

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid_X = np.c_[xx.ravel(), yy.ravel()].T
Z = np.argmax(model(grid_X), axis=0)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
plt.title("Decision Boundary")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
