import copy

import torch
import torch.nn.functional as F

from task3.methods.base import DGMethod


def assert_bn_frozen(model):
    assert all(not m.training for m in model.modules()
               if isinstance(m, torch.nn.BatchNorm2d)), "BN leaked into train mode"


class SAM(DGMethod):
    def __init__(self, num_classes=7, rho=0.05):
        super().__init__(num_classes)
        self.rho = rho

    def loss(self, xs, ys, progress=0.0):
        cls = F.cross_entropy(self.logits(xs), ys)
        return cls, {'cls': cls.item()}

    @torch.no_grad()
    def _grad_norm(self):
        # global L2 over all trainable grads, not per-parameter.
        return torch.norm(torch.stack(
            [p.grad.norm(2) for p in self.parameters() if p.grad is not None]), 2)

    @torch.no_grad()
    def _ascent(self):
        scale = self.rho / self._grad_norm().clamp_min(1e-12)
        eps = {}
        for p in self.parameters():
            if p.grad is None:
                continue
            e = p.grad * scale
            p.add_(e)                      # in-place: same tensor, Adam state keyed by it survives
            eps[p] = e
        return eps

    @torch.no_grad()
    def _restore(self, eps):
        for p, e in eps.items():
            p.sub_(e)                      # back to theta; never reload state_dict / rebuild opt

    def optimize(self, xs, ys, opt, progress=0.0, max_grad_norm=None):
        assert_bn_frozen(self)
        loss, info = self.loss(xs, ys, progress)   # true objective at theta -> reported
        opt.zero_grad()
        loss.backward()
        eps = self._ascent()                       # theta -> theta + rho * g/||g||
        opt.zero_grad()                            # drop ascent grad so pass 2 is clean g'

        assert_bn_frozen(self)
        loss2, _ = self.loss(xs, ys, progress)     # loss at theta + eps
        loss2.backward()
        self._restore(eps)                         # theta restored; step() acts from theta
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        opt.step()                                 # AdamW update at theta using grad @ theta+eps
        return info

    def self_check(self, xs, ys, make_opt, max_grad_norm=None):
        # Step-0 invariant: rho=0 must reduce to plain AdamW, so a rho=0.05 step and a
        # rho=0 step from the SAME weights on the SAME batch must move the parameters
        # differently. Equal updates mean the ascent perturbation is a silent no-op
        # (the SAM-restore or global-grad-norm trap). make_opt builds the throwaway
        # optimizer from the SAME lr/wd as the real one, so the deltas are representative.
        # Runs on deepcopies -> the live model, optimizer moments, and RNG are untouched.
        def step_delta(rho):
            m = copy.deepcopy(self)                 # inherits frozen-BN eval state of self
            m.rho = rho
            before = [p.detach().clone() for p in m.parameters()]
            m.optimize(xs, ys, make_opt(m.parameters()), max_grad_norm=max_grad_norm)
            return torch.cat([(p.detach() - b).flatten()
                              for p, b in zip(m.parameters(), before)])
        d_real, d_zero = step_delta(self.rho), step_delta(0.0)
        assert not torch.allclose(d_real, d_zero), (
            "SAM self-check: rho=0 and rho=0.05 give identical updates; "
            "ascent perturbation is a no-op")
