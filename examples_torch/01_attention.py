import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

        # 3 Linear Layers for Q, K, V
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        # x shape: (batch size, n_word, embed_dimension)
        # ex: (1, 4, 16) -> 1 sentence, 4 words, each word is number with 16 dims

        # 1. Every word has their own Q, K, V
        Q = self.W_q(x) # Context I want to find out is...
        K = self.W_k(x) # My nametag is...
        V = self.W_v(x) # My real semantic is...

        # 2. Calculate Attention score (Q dot K)
        # Dot product = Similarity
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. Scaling for mathematical stability
        # Dot product value gets larger as dimension grows
        scores = scores / math.sqrt(self.embed_size)

        # 4. Transform into probabilities using Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 5. Complete final context (Weight * V)
        # Large value if words are closer, small if not close
        context = torch.matmul(attention_weights, V)

        return context, attention_weights

# Test Code
if __name__ == "__main__":
    torch.manual_seed(42)

    dummy_sentence = torch.randn(1, 4, 16)
    attention_layer = SelfAttention(embed_size=16)
    output_context, weights = attention_layer(dummy_sentence)

    print("Attention Weights (How close each word is to each other):")
    print(torch.round(weights * 100) / 100)
    print("\nFinal output shape (New word vector with Context):", output_context.shape)
