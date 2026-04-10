import torch
import torch.nn as nn
from core_torch.encoder_block import TransformerEncoderBlock
from core_torch.positional import PositionalEncoding

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, num_classes):
        super().__init__()

        # 1. Embedding Layer (Word -> Num)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 2. Positional information
        self.pos_encoding = PositionalEncoding(embed_size)

        # 3. Stack TransformerEncoderBlock
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_size, heads) for _ in range(num_layers)
        ])

        # 4. Final head for classification
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # Global Average Pooling
        # Averaging all semantics in overall sentence
        # Make a vector, mixture of context of every word
        x = x.mean(dim=1)

        return self.fc_out(x)

vocab = {"movie": 0, "best": 1, "boring": 2, "time": 3, "waste": 4, "<PAD>": 5}
# "best movie": Positive(1), "waste (of) time boring": Negative(1)
train_data = torch.tensor([
    [1, 0, 5, 5],
    [4, 3, 2, 5]
])
train_labels = torch.tensor([1, 0])

model = TransformerClassifier(vocab_size=6, embed_size=16, heads=2, num_layers=2, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(101):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("Training complete")
