import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
import numpy as np

print("Loading data...")
mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
N = 1000

tX = torch.tensor(mnist.data[:N] / 255.0, dtype=torch.float32)
tY = torch.tensor(mnist.target[:N].astype(int), dtype=torch.long)

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training model...")
n_epoch = 100
start_time = time.time()
for epoch in range(n_epoch):
    optimizer.zero_grad()
    outputs = model(tX)
    loss = criterion(outputs, tY)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == tY).float().mean() * 100
        print(f"Epoch {epoch:3d}/{n_epoch}, Loss {loss.item():.4f}, Accuracy {acc.item():.2f}")

end_time = time.time()
print(f"Training complete!\nTime elapsed: {end_time - start_time:.2f}s")
