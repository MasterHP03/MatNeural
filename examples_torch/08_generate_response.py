import torch

from core_torch.mask import create_lookahead_mask


def generate_response(encoder, decoder, vocab, src_input, max_length=10):
    print(f"Query: {src_input}")
    print("Generating...")

    # 1. Encode query to extract context
    memory = encoder(src_input)

    # 2. Decoder starts with the start signal "<START>"
    # Shape: (1 batch, 1 word)
    target_seq = torch.tensor([[vocab["<START>"]]])

    # 3. Autoregressive Generation
    for step in range(max_length):
        # Every step, make a mask for current seq_len
        # (in actual use, it acts as matrix shape matcher rather than a mask)
        seq_len = target_seq.size(1)
        mask = create_lookahead_mask(seq_len)

        # Decode so far generated words
        out = decoder(target_seq, memory, memory, mask)

        # Probability distribution of "the last word" from result of decoder
        next_word_logits = out[:, -1, :]

        # Pick the index of the most probable word
        next_word_idx = torch.argmax(next_word_logits, dim=-1).item()

        # 4. If AI generates "<END>", break the loop
        if next_word_idx == vocab["<END>"]:
            break

        # 5. Autoregressive process
        # Append predicted word and go to next loop stage
        target_seq = torch.cat([target_seq, torch.tensor([[next_word_idx]])], dim=1)

    # Translate index to readable letters
    idx_to_word = {v: k for k, v in vocab.items()}

    # Slice out <START> token
    result_sentence = [idx_to_word[idx.item()] for idx in target_seq[0]][1:]

    print(f"Result: {' '.join(result_sentence)}")
    return result_sentence
