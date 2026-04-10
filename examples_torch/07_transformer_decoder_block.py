import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4):
        super().__init__()

        # 1. Self Attention (Analysis until the words generated)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)

        # 2. Cross Attention (Look up the query from User)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_size)

        # 3. Feed Forward (Deep analysis)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, value, key, tgt_mask):
        # x: the words generated so far
        # value, key: encoder(query)'s context (source)
        # tgt_mask: look-ahead mask (don't see the future)

        # 1. Context comprehension for already generated words
        # Q, K, V all self x
        attention_out, _ = self.attention(query=x, key=x, value=x, attn_mask=tgt_mask)
        x = self.norm1(attention_out + x)

        # 2. Look up the query from encoder
        # Q from decoder, K, V from encoder
        cross_out, _ = self.cross_attention(query=x, key=key, value=value)
        x = self.norm2(cross_out + x)

        # 3. Final process
        forward_out = self.feed_forward(x)
        x = self.norm3(x + forward_out)

        return x

if __name__ == "__main__":
    embed_size = 64

    # query interpreted by encoder
    encoder_out = torch.randn(1, 4, embed_size)

    # answer generated so far by decoder
    decoder_input = torch.randn(1, 2, embed_size)

    # Mask (length = seq length) for "Learning!"
    # We toss the whole answer sheet and make AI to learn.
    # We need mask for covering the next word.
    # If not, AI is literally cheating by peeking the answer before it makes a guess!
    # (Teacher forcing: Ground Truth is tossed to next input, regardless of whether if guess was correct)
    from core_torch.mask import create_lookahead_mask
    mask = create_lookahead_mask(seq_len=2)

    decoder_block = TransformerDecoderBlock(embed_size=embed_size, heads=8)

    output = decoder_block(decoder_input, encoder_out, encoder_out, mask)

    print("Cross-Attention Test")
    print("Shape of encoder context:", encoder_out.shape)
    print("Shape of decoder input:", decoder_input.shape)
    print("Shape of decoder output:", output.shape)
    print("Shape of output is maintained -> If passed to Linear, we get probability of the next word")
