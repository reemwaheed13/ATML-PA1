import torch.nn as nn

from task2.models.backbone import ResNet18Backbone, FEATURE_DIM
from task2.models.classifier_head import ClassifierHead


class BaseMethod(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.head = ClassifierHead(FEATURE_DIM, num_classes)

    def logits(self, x):
        return self.head(self.backbone(x))

    def compute_loss(self, xs, ys, xt=None, progress=0.0):
        raise NotImplementedError
