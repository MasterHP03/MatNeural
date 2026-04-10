import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Max length of sentence * embedding dimension
        pe = torch.zeros(max_len, d_model)

        # Vertical matrix for position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Horizontal matrix for frequency
        # e^(log(10000) / d_model * 2i)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sine for even columns, Cosine for odd columns
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Constant, not a weight learned. To prevent update, register as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Input word matrix -> Add the 'Barcode' (long as length of sentence)
        # x shape: (Batch, Seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == "__main__":
    d_model = 16 # n_dim
    seq_len = 4 # n_word in sentence

    # Assume words with same shape
    words = torch.zeros(1, seq_len, d_model)

    pos_encoder = PositionalEncoding(d_model=d_model)
    words_with_time = pos_encoder(words)

    print("Positional barcode of the first word:")
    print(torch.round(words_with_time[0, 0, :] * 100) / 100)
    print("Positional barcode of the second word:")
    print(torch.round(words_with_time[0, 1, :] * 100) / 100)
