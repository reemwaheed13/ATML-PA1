import os
import csv
import math
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from common.metrics import macro_f1, top1_accuracy
from shared.pacs_protocol import (
    load_splits, source_datasets, target_train_dataset, SOURCE_DOMAINS,
)
from task2.methods.source_only import SourceOnly
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.cdan import CDAN


def build_method(cfg):
    n = cfg['num_classes']
    m = cfg['method']
    if m == 'source_only':
        return SourceOnly(n)
    if m == 'dan':
        return DAN(n, lambda_mmd=cfg['lambda_mmd'])
    if m == 'dann':
        return DANN(n, max_lambda=cfg['max_lambda'])
    if m == 'cdan':
        return CDAN(n, max_lambda=cfg['max_lambda'])
    raise ValueError(f"unknown method {m!r}")


def load_config(base_path, method_path):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(method_path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def resolve_root(cfg, cli_root):
    root = cli_root or cfg.get('pacs_root') or os.environ.get('PACS_ROOT')
    if not root:
        raise SystemExit("PACS root not set: pass --pacs_root or set PACS_ROOT.")
    return root


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_sources(model, val_loaders, device):
    model.eval()
    per_domain = {}
    for domain, loader in val_loaders.items():
        logits_all, labels_all = [], []
        for x, y in loader:
            logits_all.append(model.logits(x.to(device)).cpu().numpy())
            labels_all.append(y.numpy())
        logits_all = np.concatenate(logits_all)
        labels_all = np.concatenate(labels_all)
        per_domain[domain] = {
            'f1': float(macro_f1(logits_all, labels_all)),
            'acc': float(top1_accuracy(logits_all, labels_all)),
        }
    mean_f1 = float(np.mean([per_domain[d]['f1'] for d in SOURCE_DOMAINS]))
    return per_domain, mean_f1


def train(cfg, cli_root, resume):
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = resolve_root(cfg, cli_root)

    splits = load_splits(cfg['split_path'])
    srcs = source_datasets(root, splits)

    gen = torch.Generator()
    gen.manual_seed(cfg['seed'])

    src_train_loaders = {
        d: DataLoader(srcs[d]['train'], batch_size=cfg['batch_per_source'],
                      shuffle=True, drop_last=True,
                      num_workers=cfg['num_workers'], generator=gen)
        for d in SOURCE_DOMAINS
    }
    val_loaders = {
        d: DataLoader(srcs[d]['val'], batch_size=64, shuffle=False,
                      num_workers=cfg['num_workers'])
        for d in SOURCE_DOMAINS
    }
    src_iters = {d: infinite(src_train_loaders[d]) for d in SOURCE_DOMAINS}

    # One epoch = one pass over the source training images (24 per update).
    n_src_train = sum(len(srcs[d]['train']) for d in SOURCE_DOMAINS)
    iters_per_epoch = math.ceil(
        n_src_train / (cfg['batch_per_source'] * len(SOURCE_DOMAINS)))
    total_iters = cfg['max_epochs'] * iters_per_epoch

    target_iter = None
    if cfg['uses_target']:
        tgt_loader = DataLoader(target_train_dataset(root),
                                batch_size=cfg['batch_target'], shuffle=True,
                                drop_last=True, num_workers=cfg['num_workers'],
                                generator=gen)
        target_iter = infinite(tgt_loader)

    model = build_method(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])

    ckpt_dir = os.path.join(os.environ.get('CKPT_DIR', 'checkpoints'),
                            'task2', cfg['run'])
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, 'last.pt')
    best_path = os.path.join(ckpt_dir, 'best.pt')

    log_dir = os.path.join('results', 'task2', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{cfg['run']}.csv")
    fields = (['epoch', 'cls_loss', 'align', 'disc_acc']
              + [f'f1_{d}' for d in SOURCE_DOMAINS]
              + [f'acc_{d}' for d in SOURCE_DOMAINS]
              + ['mean_val_f1', 'best'])

    start_epoch, best_f1, best_epoch, bad, global_iter = 0, -1.0, -1, 0, 0
    resumed = False
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        start_epoch = state['epoch'] + 1
        best_f1, best_epoch = state['best_f1'], state['best_epoch']
        bad, global_iter = state['bad'], state['global_iter']
        resumed = True
        print(f"[resume] continuing from epoch {start_epoch} (best_f1={best_f1:.4f})")
    # Fresh run truncates any old log; a resume keeps its log but re-writes the
    # header if the file was lost (e.g. re-cloned Colab session, checkpoint on Drive).
    if not resumed or not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(fields)

    for epoch in range(start_epoch, cfg['max_epochs']):
        model.train()  # backbone re-freezes BatchNorm here (see ResNet18Backbone.train)
        cls_sum = align_sum = disc_sum = 0.0
        for _ in range(iters_per_epoch):
            progress = global_iter / max(total_iters, 1)

            xs_parts, ys_parts = [], []
            for d in SOURCE_DOMAINS:
                x, y = next(src_iters[d])          # 8 images from this source domain
                xs_parts.append(x)
                ys_parts.append(y)
            xs = torch.cat(xs_parts).to(device)     # 24 source images
            ys = torch.cat(ys_parts).to(device)

            xt = None
            if cfg['uses_target']:
                xt = next(target_iter)[0].to(device)  # 24 unlabelled target images

            loss, info = model.compute_loss(xs, ys, xt, progress)
            opt.zero_grad()
            loss.backward()
            if cfg.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['max_grad_norm'])
            opt.step()

            cls_sum += info.get('cls', 0.0)
            align_sum += info.get('mmd', info.get('domain', 0.0))
            disc_sum += info.get('disc_acc', 0.0)
            global_iter += 1

        per_domain, mean_f1 = evaluate_sources(model, val_loaders, device)
        n = iters_per_epoch
        improved = mean_f1 > best_f1
        row = ([epoch, cls_sum / n, align_sum / n, disc_sum / n]
               + [per_domain[d]['f1'] for d in SOURCE_DOMAINS]
               + [per_domain[d]['acc'] for d in SOURCE_DOMAINS]
               + [mean_f1, int(improved)])
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        print(f"[{cfg['run']}] epoch {epoch:2d}  cls={cls_sum/n:.4f}  "
              f"align={align_sum/n:.4f}  mean_val_f1={mean_f1:.4f}"
              + ("  *best*" if improved else ""))

        if improved:
            best_f1, best_epoch, bad = mean_f1, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'mean_val_f1': mean_f1, 'per_domain': per_domain,
                        'config': cfg}, best_path)
        else:
            bad += 1

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch,
                    'bad': bad, 'global_iter': global_iter, 'config': cfg}, last_path)

        if bad >= cfg['patience']:
            print(f"[{cfg['run']}] early stop at epoch {epoch} "
                  f"(best {best_f1:.4f} @ epoch {best_epoch})")
            break

    print(f"[{cfg['run']}] done. best mean source-val macro-F1 "
          f"{best_f1:.4f} @ epoch {best_epoch} -> {best_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='method yaml (extends base.yaml)')
    ap.add_argument('--base', default='task2/configs/base.yaml')
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--resume', action='store_true',
                    help='resume from last.pt (Colab disconnects)')
    args = ap.parse_args()
    cfg = load_config(args.base, args.config)
    train(cfg, args.pacs_root, args.resume)


if __name__ == '__main__':
    main()
