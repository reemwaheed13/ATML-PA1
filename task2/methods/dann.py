import torch
import torch.nn.functional as F

from task2.methods.base import BaseMethod
from task2.models.backbone import FEATURE_DIM
from task2.models.domain_discriminator import DomainDiscriminator, grad_reverse, grl_alpha


def domain_labels(n_s, n_t, device):
    return torch.cat([torch.zeros(n_s), torch.ones(n_t)]).long().to(device)


class DANN(BaseMethod):
    def __init__(self, num_classes=7, max_lambda=1.0):
        super().__init__(num_classes)
        self.discriminator = DomainDiscriminator(FEATURE_DIM)
        self.max_lambda = max_lambda

    def compute_loss(self, xs, ys, xt, progress=0.0):
        fs, ft = self.backbone(xs), self.backbone(xt)
        cls = F.cross_entropy(self.head(fs), ys)
        alpha = grl_alpha(progress, max_lambda=self.max_lambda)
        f = torch.cat([fs, ft], 0)
        d_out = self.discriminator(grad_reverse(f, alpha))
        d_lab = domain_labels(len(fs), len(ft), f.device)
        dloss = F.cross_entropy(d_out, d_lab)
        loss = cls + dloss
        disc_acc = (d_out.argmax(1) == d_lab).float().mean().item()
        return loss, {'cls': cls.item(), 'domain': dloss.item(),
                      'alpha': alpha, 'disc_acc': disc_acc}
