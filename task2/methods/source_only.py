import torch.nn.functional as F

from task2.methods.base import BaseMethod


class SourceOnly(BaseMethod):
    def compute_loss(self, xs, ys, xt=None, progress=0.0):
        cls = F.cross_entropy(self.logits(xs), ys)
        return cls, {'cls': cls.item()}
