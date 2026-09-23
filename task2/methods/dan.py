import torch
import torch.nn.functional as F

from task2.methods.base import BaseMethod


def _sq_dists(a, b):
    return (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)


def mmd_loss(fs, ft, muls=(0.5, 1.0, 2.0)):
    z = torch.cat([fs, ft], 0)
    median = _sq_dists(z, z).detach().median().clamp_min(1e-8)
    xx, yy, xy = _sq_dists(fs, fs), _sq_dists(ft, ft), _sq_dists(fs, ft)
    loss = 0.0
    for m in muls:
        bw = m * median
        loss = loss + torch.exp(-xx / bw).mean() + torch.exp(-yy / bw).mean() \
                    - 2.0 * torch.exp(-xy / bw).mean()
    return loss


def mmd_warmup_scale(progress, warmup_frac):
    """Ramp the alignment weight 0->1 linearly over the first `warmup_frac` of
    training progress, then hold at 1. warmup_frac<=0 returns 1.0 (no-op), so
    runs without the knob behave exactly as before."""
    if warmup_frac and warmup_frac > 0:
        return min(1.0, progress / warmup_frac)
    return 1.0


class DAN(BaseMethod):
    def __init__(self, num_classes=7, lambda_mmd=1.0, warmup_frac=0.0):
        super().__init__(num_classes)
        self.lambda_mmd = lambda_mmd
        self.warmup_frac = warmup_frac

    def compute_loss(self, xs, ys, xt, progress=0.0):
        fs, ft = self.backbone(xs), self.backbone(xt)
        cls = F.cross_entropy(self.head(fs), ys)
        mmd = mmd_loss(fs, ft)
        lam = self.lambda_mmd * mmd_warmup_scale(progress, self.warmup_frac)
        loss = cls + lam * mmd
        return loss, {'cls': cls.item(), 'mmd': mmd.item()}
