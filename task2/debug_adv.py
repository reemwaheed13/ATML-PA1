import argparse
import math

import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from shared.pacs_protocol import (
    load_splits, source_datasets, target_train_dataset, SOURCE_DOMAINS,
)
from task2.train import build_method, load_config, resolve_root, infinite


def backbone_grad_norm(model):
    total = 0.0
    for p in model.backbone.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


@torch.no_grad()
def probe(model, xs, xt):
    fs, ft = model.backbone(xs), model.backbone(xt)
    f = torch.cat([fs, ft], 0)
    feat_l2 = f.norm(dim=1).mean().item()
    max_logit = float('nan')
    if hasattr(model, 'discriminator'):
        in_dim = model.discriminator.net[0].in_features
        if in_dim == f.shape[1]:                       # DANN: raw features
            d_in = f
        else:                                          # CDAN: vec(f (x) p)
            p = torch.softmax(model.head(f), dim=1)
            d_in = torch.bmm(f.unsqueeze(2), p.unsqueeze(1)).flatten(1)
        max_logit = model.discriminator(d_in).abs().max().item()
    return feat_l2, max_logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='task2/configs/dann.yaml')
    ap.add_argument('--base', default='task2/configs/base.yaml')
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--iters', type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.base, args.config)
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = resolve_root(cfg, args.pacs_root)

    splits = load_splits(cfg['split_path'])
    srcs = source_datasets(root, splits)
    gen = torch.Generator()
    gen.manual_seed(cfg['seed'])

    # Same loaders / batching as train.py.
    src_iters = {
        d: infinite(DataLoader(srcs[d]['train'], batch_size=cfg['batch_per_source'],
                               shuffle=True, drop_last=True,
                               num_workers=cfg['num_workers'], generator=gen))
        for d in SOURCE_DOMAINS
    }
    tgt_iter = infinite(DataLoader(target_train_dataset(root),
                                   batch_size=cfg['batch_target'], shuffle=True,
                                   drop_last=True, num_workers=cfg['num_workers'],
                                   generator=gen))

    n_src_train = sum(len(srcs[d]['train']) for d in SOURCE_DOMAINS)
    iters_per_epoch = math.ceil(
        n_src_train / (cfg['batch_per_source'] * len(SOURCE_DOMAINS)))
    total_iters = cfg['max_epochs'] * iters_per_epoch

    model = build_method(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])

    print(f"run={cfg['run']}  iters_per_epoch={iters_per_epoch}  "
          f"total_iters={total_iters}  (epoch 1 covers iters 0..{iters_per_epoch-1})")
    print("iter    p       alpha    cls        domain        feat_L2   max|dlogit|   bb_grad")

    model.train()  # backbone re-freezes BatchNorm, exactly as train.py
    for it in range(args.iters):
        progress = it / max(total_iters, 1)

        xs_parts, ys_parts = [], []
        for d in SOURCE_DOMAINS:
            x, y = next(src_iters[d])
            xs_parts.append(x)
            ys_parts.append(y)
        xs = torch.cat(xs_parts).to(device)
        ys = torch.cat(ys_parts).to(device)
        xt = next(tgt_iter)[0].to(device)

        loss, info = model.compute_loss(xs, ys, xt, progress)
        opt.zero_grad()
        loss.backward()
        gnorm = backbone_grad_norm(model)
        opt.step()

        if it % 10 == 0:
            feat_l2, max_logit = probe(model, xs, xt)
            print(f"{it:4d}  {progress:.4f}  {info.get('alpha', 0):.4f}  "
                  f"{info.get('cls', 0):8.3f}  {info.get('domain', 0):11.3f}  "
                  f"{feat_l2:7.2f}  {max_logit:11.2f}  {gnorm:8.2f}")


if __name__ == '__main__':
    main()
