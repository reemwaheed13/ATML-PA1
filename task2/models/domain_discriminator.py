import math
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return _GradReverse.apply(x, alpha)


def grl_alpha(progress, gamma=10.0, max_lambda=1.0):
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)
