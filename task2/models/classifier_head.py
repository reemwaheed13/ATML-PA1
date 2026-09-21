import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
