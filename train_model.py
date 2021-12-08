import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainDataset
from model import ClassificationModel


def train(train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loss = None
    for epoch in range(epochs):
        losses = []
        iterator = tqdm(train_dataloader, total=len(train_dataloader))
        for samples, labels in iterator:
            iterator.set_description(f"Running Epoch: {epoch + 1} with Loss: {sum(losses) / len(losses) if len(losses) > 0 else None}")
            logits = model(samples.squeeze(1))

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()


if __name__ == "__main__":
    lr = 0.0005
    epochs = 100

    torch.manual_seed(0)
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = ClassificationModel()

    train(train_dataloader)
    torch.save(model.state_dict(), "./model.pt")