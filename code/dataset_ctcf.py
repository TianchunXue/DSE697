# dataset_ctcf.py

import torch
from torch.utils.data import Dataset
import json

class CTCFDatasetMultiTask(Dataset):
    def __init__(self, jsonl_path, model, max_length=128):
        self.samples = []
        self.model = model

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                seq = item["sequence"]
                labels = item["labels"]

                # derive token-level labels from base labels using k-mer
                k = model.k
                token_labels = []
                for i in range(len(seq) - k + 1):
                    kmer_label = labels[i:i + k]
                    token_labels.append(1 if all(x == 1 for x in kmer_label) else 0)

                # pad to match token sequence length
                pad_len = max_length - len(token_labels) - 1
                token_labels = [0] + token_labels + [0] * pad_len

                encoded = model.preprocess([seq], max_length=max_length)

                self.samples.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.tensor(token_labels, dtype=torch.float)
                })

    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):
        return self.samples[idx]
