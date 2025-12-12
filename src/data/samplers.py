import torch
import random

class TokenBatchSampler:
    def __init__(self, dataset, max_tokens, shuffle=True):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle

        self.lengths = [
            len(dataset[i]["src"]) + len(dataset[i]["tgt_inp"])
            for i in range(len(dataset))
        ]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        current_tokens = 0

        for idx in indices:
            tokens = self.lengths[idx]

            if current_tokens + tokens > self.max_tokens:
                if batch:
                    yield batch
                batch = [idx]
                current_tokens = tokens
            else:
                batch.append(idx)
                current_tokens += tokens

        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset)
