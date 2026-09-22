import torch

from task2.methods.base import BaseMethod


class DGMethod(BaseMethod):
    def loss(self, xs, ys, progress=0.0):
        raise NotImplementedError

    def optimize(self, xs, ys, opt, progress=0.0, max_grad_norm=None):
        loss, info = self.loss(xs, ys, progress)
        opt.zero_grad()
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        opt.step()
        return info
