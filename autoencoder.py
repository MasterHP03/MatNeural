import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from layers import LinearLayer, ReLULayer, SoftmaxLayer, SigmoidLayer
from models import Sequential
from optimizers import Adam
from losses import CrossEntropyLoss, MSELoss
import time

print("Loading data...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

N = 1000
tX = mnist['data'][:N].T
tX = tX / 255.0

tY = tX

model = Sequential(
    LinearLayer(784, 128),
    ReLULayer(),
    LinearLayer(128, 32),
    ReLULayer(),

    LinearLayer(32, 128),
    ReLULayer(),
    LinearLayer(128, 784),
    SigmoidLayer()
)

criterion = MSELoss()
optimizer = Adam(model, learning_rate=0.01)

print("Training model...")
n_loop = 1500
start_time = time.time()
for epoch in range(n_loop):
    predictions = model(tX)

    loss_val = criterion(predictions, tY)
    dLoss = criterion.backward()

    model.backward(dLoss)
    optimizer.step()

    if epoch % 100 == 0 or epoch == n_loop - 1:
        print(f"Epoch [{epoch:4d}/{n_loop}] | Loss {loss_val:.4f}")

end_time = time.time()
print(f"Training complete! (Time elapsed: {end_time - start_time:.2f}s)")
idx = np.random.randint(0, N)
test_img = tX[:, idx].reshape(784, 1)

reconstructed_img = model(test_img)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(test_img.reshape(28, 28), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(reconstructed_img.reshape(28, 28), cmap='gray')
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

plt.show()
