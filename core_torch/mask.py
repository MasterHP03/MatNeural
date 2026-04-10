import torch

def create_lookahead_mask(seq_len):
    # 1. First one-matrix, then tril to make it lower triangle
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # 2. Change numbers to make it easy to add to attention score
    # Future(0) -> -inf, Past/Present(1) -> 0.0
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask

if __name__ == "__main__":
    seq_len = 5
    mask = create_lookahead_mask(seq_len)

    print("Look-ahead mask:")
    print(mask)
    print("-inf is for Softmax (e^(-inf) = 0, making weight(interest) for future words 0%)")
