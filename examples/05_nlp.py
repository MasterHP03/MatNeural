import numpy as np
from core.layers import LinearLayer, ReLULayer, SoftmaxLayer
from core.models import Sequential
from core.optimizers import Adam
from core.losses import CrossEntropyLoss
import time

vocab = ['movie', 'really', 'good', 'fun', 'wasted', 'time', 'worst', 'boring', 'money', 'evaporated']
word_to_idx = {word: i for i, word in enumerate(vocab)}
V = len(vocab)

reviews = [
    ["really", "good", "fun", "movie"],
    ["totally", "wasted", "time", "worst"],
    ["movie", "really", "best"],
    ["money", "evaporated", "really", "boring"]
]
labels = [1, 0, 1, 0]

tX = np.zeros((V, len(reviews)))
for i, review in enumerate(reviews):
    for word in review:
        if word in word_to_idx:
            tX[word_to_idx[word], i] += 1

tY = np.zeros((2, len(reviews)))
tY[labels, np.arange(len(reviews))] = 1

model = Sequential(
    LinearLayer(V, 16),
    ReLULayer(),
    LinearLayer(16, 2),
    SoftmaxLayer()
)

criterion = CrossEntropyLoss()
optimizer = Adam(model, learning_rate=0.05)

print("Training model...")
n_loop = 100
start_time = time.time()
for epoch in range(n_loop):
    predictions = model(tX)

    loss_val = criterion(predictions, tY)
    dLoss = criterion.backward()

    model.backward(dLoss)
    optimizer.step()

    if epoch % (n_loop / 10) == 0 or epoch == n_loop - 1:
        acc = np.mean(np.argmax(predictions, axis=0) == labels) * 100
        print(f"Epoch [{epoch:4d}/{n_loop}] | Loss {loss_val:.4f} | Acc: {acc:.1f}%")

end_time = time.time()
print(f"Training complete! (Time elapsed: {end_time - start_time:.2f}s)")

test_sentence = ["really", "time", "wasted", "worst"]
test_vector = np.zeros((V, 1))
for word in test_sentence:
    if word in word_to_idx:
        test_vector[word_to_idx[word], 0] += 1

pred_prob = model(test_vector)
result = "Positive" if np.argmax(pred_prob) == 1 else "Negative"

print(f"Input sentence: {' '.join(test_sentence)}")
print(f"Prediction: {result} (Probability: {np.max(pred_prob) * 100:.2f})")