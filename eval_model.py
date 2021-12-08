import json

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from model import ClassificationModel


def eval(eval_dataset):
        iterator = tqdm(enumerate(eval_dataset.items()), total=len(eval_dataset))
        preds, labels = [], []
        for i, (k, v) in iterator:
            for val in v:
                input = val["title"] + ". " + val["summary"]
                tokens = tokenizer(input, add_special_tokens=False, padding=True, return_tensors="pt")
                embeddings = bert(tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
                logits = model(embeddings)
                preds.append(torch.argmax(logits).item())
                labels.append(i)

        print(classification_report(labels, preds))
        with open("metrics.txt", "w") as f:
            f.write(classification_report(labels, preds))


if __name__ == "__main__":
    with open("test.json") as f:
        eval_dataset = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

    state_dict = torch.load("./model.pt")
    model = ClassificationModel()
    model.load_state_dict(state_dict)
    model.eval()

    eval(eval_dataset)