import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4):
        super().__init__()

        # 1. Multihead Attention
        # batch_first=True -> (batch, words, dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)

        # 2. Normalization to prevent too large/small value
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 3. Feed-forward network (More process for each word)
        # 64-dim -> 256-dim -> 64-dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, x):
        # Attention + Add & norm
        # Q, K, V all self -> Self-Attention
        attention_out, _ = self.attention(x, x, x)

        # "Add" original value then "Normalize"
        # Prevents loss of learning signal (vanishing)
        x = self.norm1(attention_out + x)

        # Feed Forward + Add & norm
        forward_out = self.feed_forward(x)

        x = self.norm2(forward_out + x)

        return x

if __name__ == "__main__":
    torch.manual_seed(42)

    dummy_sentence = torch.randn(1, 4, 64)

    encoder_block = TransformerEncoderBlock(embed_size=64, heads=8)

    output = encoder_block(dummy_sentence)

    print("Shape of input: ", dummy_sentence.shape)
    print("Shape of output: ", output.shape)
    print("Same shape implies that we can stack this block indefinitely")
