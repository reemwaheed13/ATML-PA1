import itertools

import torch.nn.functional as F

from task2.methods.dan import mmd_loss, mmd_warmup_scale
from task3.methods.base import DGMethod


class DANDG(DGMethod):
    def __init__(self, num_classes=7, lambda_dg=1.0, n_per_source=8, warmup_frac=0.0):
        super().__init__(num_classes)
        self.lambda_dg = lambda_dg
        self.n_per_source = n_per_source
        self.warmup_frac = warmup_frac

    def loss(self, xs, ys, progress=0.0):
        f = self.backbone(xs)
        cls = F.cross_entropy(self.head(f), ys)
        n = self.n_per_source
        groups = [f[i * n:(i + 1) * n] for i in range(len(xs) // n)]
        pairs = list(itertools.combinations(groups, 2))
        mmd = sum(mmd_loss(a, b) for a, b in pairs) / len(pairs)
        lam = self.lambda_dg * mmd_warmup_scale(progress, self.warmup_frac)
        loss = cls + lam * mmd
        return loss, {'cls': cls.item(), 'mmd': mmd.item()}
