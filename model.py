from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(768, 8)

    def forward(self, x):
        logits = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return logits