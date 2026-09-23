import torch.nn.functional as F

from task4.methods.base import OSRMethod


class Vanilla(OSRMethod):
    def optimize(self, x, y, opt):
        logits = self.logits(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'loss': float(loss.item()), 'cls': float(loss.item())}
