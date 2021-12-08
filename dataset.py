import json
import os
import pickle

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

        if os.path.exists("cache.pickle"):
            with open("cache.pickle", "rb") as f:
                self.train_dataset = pickle.load(f)
        else:
            with open("train.json") as f:
                train_dataset_raw = json.load(f)
            self.train_dataset = []
            iterator = tqdm(enumerate(train_dataset_raw.items()), total=len(train_dataset_raw))
            for i, (k, v) in iterator:
                for val in v:
                    input = val["title"] + ". " + val["summary"]
                    tokens = self.tokenizer(input, add_special_tokens=False, padding=True, return_tensors="pt")
                    with torch.no_grad():
                        embeddings = self.bert(tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
                    self.train_dataset.append((embeddings, i))
            with open("cache.pickle", "wb") as f:
                pickle.dump(self.train_dataset, f)

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, item):
        return self.train_dataset[item]