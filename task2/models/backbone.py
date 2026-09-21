import torch.nn as nn
import torchvision.models as models

FEATURE_DIM = 512


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.net = net

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        return self.net(x)
