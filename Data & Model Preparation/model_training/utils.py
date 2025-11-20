import math
import json
import torch
import random
import datetime

import argparse
from torch.utils.data import DataLoader, Dataset

class My_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        with open(data_dir, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        text = record["text"]
        label = record["label"]   # số nguyên 0..3

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def get_loader(mode, path, tokenizer, batch_size):
    dataset = My_Dataset(path, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode=="train"),
        num_workers=2
    )
    return loader

# def get_answer_loss( batch, model):
#     input_ids, labels = (
#         batch["input_ids"],
#         batch["target_ids"],
#     )
#     outputs = model(input_ids=input_ids, labels=labels)
#     loss = outputs.loss

#     return loss