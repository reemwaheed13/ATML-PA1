import torch.nn as nn

from task4.models.resnet_cifar import ResNetCIFAR


class OSRMethod(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = ResNetCIFAR(num_classes)

    # 10 known-class logits z(x) -> used for selection, CSA, and post-hoc scores
    def logits(self, x):
        return self.net(x)

    # penultimate feature f(x)
    def features(self, x):
        return self.net.features(x)

    def optimize(self, x, y, opt):
        raise NotImplementedError
