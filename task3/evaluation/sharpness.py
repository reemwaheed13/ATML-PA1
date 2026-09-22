import torch
import torch.nn.functional as F

from shared.pacs_protocol import SOURCE_DOMAINS

SEED = 6304
RADIUS = 0.05
N_PER_SOURCE = 32


# One fixed source batch, built once and reused across ALL models so the sharpness proxy
# compares the same points. Drawn from source-val (eval_transform => deterministic images)
# with a seeded generator, 32 per source. Never touches the target domain.
def make_sharpness_batch(srcs, device, seed=SEED, n_per_source=N_PER_SOURCE):
    g = torch.Generator().manual_seed(seed)
    xs_parts, ys_parts = [], []
    for d in SOURCE_DOMAINS:
        ds = srcs[d]['val']
        idx = torch.randperm(len(ds), generator=g)[:n_per_source].tolist()
        for i in idx:
            x, y = ds[i]
            xs_parts.append(x)
            ys_parts.append(int(y))
    xs = torch.stack(xs_parts).to(device)
    ys = torch.tensor(ys_parts, device=device)
    return xs, ys


@torch.no_grad()
def _global_grad_norm(params):
    return torch.norm(torch.stack(
        [p.grad.norm(2) for p in params if p.grad is not None]), 2)


# Sharpness proxy: delta = L(theta + eps) - L(theta), with eps = radius * g/||g|| in the
# ascent (worst-case) direction, GLOBAL grad-norm, radius 0.05 -- the SAM eps at rho=0.05.
# L is the plain classification loss for every method so the comparison is method-agnostic.
# The perturbation is applied and then restored in-place on the same tensors (exact undo).
def sharpness(model, xs, ys, radius=RADIUS):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model.logits(xs), ys)
    loss.backward()
    scale = radius / _global_grad_norm(params).clamp_min(1e-12)

    eps = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                eps.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            eps.append(e)
        loss_perturbed = F.cross_entropy(model.logits(xs), ys).item()
        for p, e in zip(params, eps):
            if e is not None:
                p.sub_(e)

    model.zero_grad(set_to_none=True)
    base = float(loss.item())
    return {
        'loss': base,
        'loss_perturbed': float(loss_perturbed),
        'delta_sharpness': float(loss_perturbed - base),
    }
