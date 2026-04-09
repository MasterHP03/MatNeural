import numpy as np
from layers import LinearLayer, ReLULayer, SigmoidLayer
from models import Sequential
from optimizers import Adam

tX = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

tY = np.array([[0, 1, 1, 0]])

success = 0

for trial in range(1000):
    model = Sequential(
        LinearLayer(2, 8),
        ReLULayer(coeff_leaky=0.01),
        LinearLayer(8, 1),
        SigmoidLayer()
    )

    optimizer = Adam(model)

    predictions = None
    for epoch in range(10000):
        predictions = model(tX)
        dLoss = (predictions - tY)

        model.backward(dLoss)
        optimizer.step()

    if np.allclose(predictions, tY, atol=0.1):
        success += 1
    print(f"Prediction ({success:4d} / {trial + 1:4d})\n", np.round(predictions, 2))

print(success)