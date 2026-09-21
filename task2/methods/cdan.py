import torch
import torch.nn.functional as F

from task2.methods.base import BaseMethod
from task2.models.backbone import FEATURE_DIM
from task2.models.domain_discriminator import DomainDiscriminator, grad_reverse, grl_alpha
from task2.methods.dann import domain_labels


class CDAN(BaseMethod):
    def __init__(self, num_classes=7, max_lambda=1.0):
        super().__init__(num_classes)
        self.discriminator = DomainDiscriminator(FEATURE_DIM * num_classes)
        self.max_lambda = max_lambda

    def compute_loss(self, xs, ys, xt, progress=0.0):
        fs, ft = self.backbone(xs), self.backbone(xt)
        ls, lt = self.head(fs), self.head(ft)
        cls = F.cross_entropy(ls, ys)
        f = torch.cat([fs, ft], 0)
        p = F.softmax(torch.cat([ls, lt], 0), dim=1)
        g = torch.bmm(f.unsqueeze(2), p.unsqueeze(1)).flatten(1)
        alpha = grl_alpha(progress, max_lambda=self.max_lambda)
        d_out = self.discriminator(grad_reverse(g, alpha))
        dloss = F.cross_entropy(d_out, domain_labels(len(fs), len(ft), f.device))
        loss = cls + dloss
        return loss, {'cls': cls.item(), 'domain': dloss.item(), 'alpha': alpha}
