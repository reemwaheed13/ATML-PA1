import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from task4.methods.base import OSRMethod
from task4.models.resnet_cifar import FEATURE_DIM
from task4.methods.manifold_mixup import manifold_mixup


class PROSER(OSRMethod):
    def __init__(self, num_classes=10, num_dummy=5, beta=1.0, gamma=0.1):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.num_dummy = num_dummy
        self.beta = beta
        self.gamma = gamma
        self.dummy = nn.Linear(FEATURE_DIM, num_dummy)
        self.mix_dist = Beta(2.0, 2.0)

    # strongest dummy response, aggregated over the num_dummy heads
    def _dummy_score(self, feat):
        return self.dummy(feat).max(dim=1, keepdim=True).values

    # augmented vector [known(10), strongest-dummy(1)] from a precomputed feature
    def _augment(self, feat):
        known = self.net.fc(feat)
        return torch.cat([known, self._dummy_score(feat)], dim=1)

    # --- API for Stage D: known-only vs augmented are cleanly separated ---
    # logits(x)  -> 10 known logits           (selection, CSA, MLS row)  [inherited]
    # augmented_logits(x) -> [known(10), dummy(1)]  (placeholder detection score)
    def augmented_logits(self, x):
        return self._augment(self.features(x))

    def optimize(self, x, y, opt):
        n = x.shape[0] // 2
        x_cls, y_cls = x[:n], y[:n]
        x_mix, y_mix = x[n:], y[n:]
        K = self.num_classes

        # classifier placeholders (first half)
        aug = self._augment(self.features(x_cls))          # (n, K+1)
        l1 = F.cross_entropy(aug, y_cls)                   # true class stays largest
        aug_masked = aug.clone()
        aug_masked[torch.arange(n, device=x.device), y_cls] = float('-inf')
        dummy_tgt = torch.full((n,), K, dtype=torch.long, device=x.device)
        l2 = F.cross_entropy(aug_masked, dummy_tgt)        # dummy wins once true class masked

        # data placeholders (second half): manifold mixup -> toward dummy
        feat_mix = manifold_mixup(self.net, x_mix, y_mix, self.mix_dist)
        aug_mix = self._augment(feat_mix)
        dummy_tgt_mix = torch.full((x_mix.shape[0],), K, dtype=torch.long, device=x.device)
        l_data = F.cross_entropy(aug_mix, dummy_tgt_mix)

        loss = l1 + self.beta * l2 + self.gamma * l_data
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'loss': float(loss.item()), 'l1': float(l1.item()),
                'l2': float(l2.item()), 'data': float(l_data.item())}
