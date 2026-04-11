import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        vocab = set()
        for sentence in all_sentences:
            vocab.update(sentence.split())
        vocab = sorted(vocab)
        word_to_id = {word: idx + 1 for idx, word in enumerate(vocab)}  # start from 1

        # 2. Encode each sentence by replacing words with their IDs
        encoded_tensors = []
        for sentence in all_sentences:
            encoded = [word_to_id[word] for word in sentence.split()]
            encoded_tensors.append(torch.tensor(encoded, dtype=torch.float32))

        # 3. Combine positive + negative into one list of tensors
        # (already combined as all_sentences → encoded_tensors)

        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        padded = nn.utils.rnn.pad_sequence(encoded_tensors, batch_first=True)

        return padded