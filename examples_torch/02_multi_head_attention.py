import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=64, num_heads=8):
        super().__init__()
        self.embed_size = embed_size # Total dimension of each word
        self.num_heads = num_heads # Number of heads (cores of attention)
        self.head_dim = embed_size // num_heads # Dimension for each head

        assert (self.head_dim * num_heads == self.embed_size), "embed_size must be divisible by num_heads"

        # We don't make separate 8 layers
        # 64-dim in one shot and slice it afterward
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)

        # Final layer that combines results of 8 heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1] # Length of sentence; num_words

        # 1. Q, K, V. Shape: (Batch, Seq, 64)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Slicing dimension for each head (Reshape)
        # (Batch, Seq, 64) -> (Batch, Seq, 8, 8) -> Move head for computation: (Batch, 8, Seq, 8)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Simultaneous score calculation of 8 heads
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Combine results (V) of each head
        # Shape: (Batch, 8, Seq, 8)
        context = torch.matmul(attention_weights, V)

        # 5. Concatenate heads to recover 64-dim
        # Shape: (Batch, Seq, 64)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)

        # 6. Final process (tuning)
        output = self.fc_out(context)

        return output

# Test Code
if __name__ == '__main__':
    # 1 batch, 4 words, 64-dim sentence
    dummy_sentence = torch.randn(1, 4, 64)

    mha = MultiHeadAttention(embed_size=64, num_heads=8)
    output_context = mha(dummy_sentence)

    print("Shape of the original sentence:", dummy_sentence.shape)
    print("Shape of the output context:", output_context.shape)
    print("Same shape, context compressed")
