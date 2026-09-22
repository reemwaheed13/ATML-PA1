import itertools

import torch.nn.functional as F

from task2.methods.dan import mmd_loss
from task3.methods.base import DGMethod


class DANDG(DGMethod):
    def __init__(self, num_classes=7, lambda_dg=1.0, n_per_source=8):
        super().__init__(num_classes)
        self.lambda_dg = lambda_dg
        self.n_per_source = n_per_source

    def loss(self, xs, ys, progress=0.0):
        f = self.backbone(xs)
        cls = F.cross_entropy(self.head(f), ys)
        n = self.n_per_source
        groups = [f[i * n:(i + 1) * n] for i in range(len(xs) // n)]
        pairs = list(itertools.combinations(groups, 2))
        mmd = sum(mmd_loss(a, b) for a, b in pairs) / len(pairs)
        loss = cls + self.lambda_dg * mmd
        return loss, {'cls': cls.item(), 'mmd': mmd.item()}
