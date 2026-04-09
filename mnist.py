import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from layers import LinearLayer, ReLULayer, SoftmaxLayer
from models import Sequential
from optimizers import Adam
from losses import CrossEntropyLoss

print("Loading data...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

N = 1000
tX = mnist['data'][:N].T
tX = tX / 255.0

tY_labels = mnist.target[:N].astype(int)
tY = np.zeros((10, N))
tY[tY_labels, np.arange(N)] = 1

model = Sequential(
    LinearLayer(784, 64),
    ReLULayer(),
    LinearLayer(64, 32),
    ReLULayer(),
    LinearLayer(32, 10),
    SoftmaxLayer()
)

criterion = CrossEntropyLoss()
optimizer = Adam(model, learning_rate=0.01)

print("Training model...")
n_loop = 1000
for epoch in range(n_loop):
    predictions = model(tX)

    loss_val = criterion(predictions, tY)
    dLoss = criterion.backward()

    model.backward(dLoss)
    optimizer.step()

    if epoch % 100 == 0 or epoch == n_loop - 1:
        pred_labels = np.argmax(predictions, axis=0)
        acc = np.mean(pred_labels == tY_labels) * 100

        print(f"Epoch [{epoch:4d}/{n_loop}] | Loss {loss_val:.4f} | Acc:[{acc:.2f}%]")

print("Training complete!")
idx = np.random.randint(0, N)
test_img = tX[:, idx].reshape(784, 1)
pred_prob = model(test_img)
pred_num = np.argmax(pred_prob)

plt.imshow(test_img.reshape(28, 28), cmap='gray')
plt.title(f"AI Prediction: {pred_num} (Real: {tY_labels[idx]})")
plt.axis('off')
plt.show()
