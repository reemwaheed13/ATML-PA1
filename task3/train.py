import os
import csv
import json
import math
import argparse

import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from shared.pacs_protocol import load_splits, source_datasets, SOURCE_DOMAINS
# Reuse Task 2's exact source-validation selection + loader helpers so the ERM
# consistency check compares the same computation, not just the same intent.
from task2.train import evaluate_sources, infinite, load_config, resolve_root
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM


def build_optimizer(params, cfg):
    # Single source of truth for optimizer hyperparameters; also used by SAM.self_check.
    return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def build_method(cfg):
    n = cfg['num_classes']
    m = cfg['method']
    if m == 'source_only':
        raise SystemExit(
            "ERM is the reused Task 2 source_only checkpoint and is NOT trained in "
            "Task 3. evaluate_sketch reads checkpoints/task2/source_only/best.pt "
            "directly via erm.yaml's reuse_checkpoint.")
    if m == 'dan_dg':
        # n_per_source shares cfg['batch_per_source'] with the batch-balance assert
        # below, so the slice width and the invariant can't drift apart.
        return DANDG(n, lambda_dg=cfg['lambda_dg'], n_per_source=cfg['batch_per_source'])
    if m == 'sam':
        return SAM(n, rho=cfg['rho'])
    raise ValueError(f"unknown method {m!r}")


def write_logs(run, rows, csv_path, json_path):
    fields = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log = {k: [r[k] for r in rows] for k in fields}   # columnar, matches Task 2 logs_json
    with open(json_path, 'w') as f:
        json.dump({'run': run, 'log': log}, f, indent=1)


def load_prior_rows(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        log = json.load(f)['log']
    fields = list(log.keys())
    return [{k: log[k][i] for k in fields} for i in range(len(log['epoch']))]


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

    # One epoch = one pass over the source training images (batch_per_source per domain).
    n_src_train = sum(len(srcs[d]['train']) for d in SOURCE_DOMAINS)
    iters_per_epoch = math.ceil(
        n_src_train / (cfg['batch_per_source'] * len(SOURCE_DOMAINS)))
    total_iters = cfg['max_epochs'] * iters_per_epoch

    model = build_method(cfg).to(device)
    opt = build_optimizer(model.parameters(), cfg)
    make_opt = lambda params: build_optimizer(params, cfg)   # same lr/wd for self_check

    ckpt_dir = os.path.join(os.environ.get('CKPT_DIR', 'checkpoints'),
                            'task3', cfg['run'])
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, 'last.pt')
    best_path = os.path.join(ckpt_dir, 'best.pt')

    log_dir = os.path.join('results', 'task3', 'logs')
    json_dir = os.path.join('results', 'task3', 'logs_json')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{cfg['run']}.csv")
    json_path = os.path.join(json_dir, f"{cfg['run']}.json")

    start_epoch, best_f1, best_epoch, bad, global_iter = 0, -1.0, -1, 0, 0
    rows = []
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        start_epoch = state['epoch'] + 1
        best_f1, best_epoch = state['best_f1'], state['best_epoch']
        bad, global_iter = state['bad'], state['global_iter']
        rows = load_prior_rows(json_path)
        print(f"[resume] continuing from epoch {start_epoch} (best_f1={best_f1:.4f})")

    for epoch in range(start_epoch, cfg['max_epochs']):
        model.train()  # backbone re-freezes BatchNorm here (see ResNet18Backbone.train)
        cls_sum = mmd_sum = 0.0
        for _ in range(iters_per_epoch):
            progress = global_iter / max(total_iters, 1)

            xs_parts, ys_parts = [], []
            for d in SOURCE_DOMAINS:
                x, y = next(src_iters[d])           # batch_per_source images from this source
                xs_parts.append(x)
                ys_parts.append(y)
            xs = torch.cat(xs_parts).to(device)      # sources concatenated as per-domain blocks
            ys = torch.cat(ys_parts).to(device)

            # DAN-DG slices xs into contiguous per-domain blocks; make the balance a loud
            # invariant, not a silent assumption (a short/last batch would mis-group MMD).
            assert xs.shape[0] == len(SOURCE_DOMAINS) * cfg['batch_per_source'], (
                f"expected {len(SOURCE_DOMAINS) * cfg['batch_per_source']} balanced "
                f"source images, got {xs.shape[0]}")

            # Step-0 self-check: must run AFTER model.train() (so the deepcopy inherits the
            # frozen-BN state) but BEFORE the first optimizer step mutates the live weights.
            # Do not move below model.optimize(...). No-op for every method except SAM.
            if global_iter == 0:
                model.self_check(xs, ys, make_opt, cfg.get('max_grad_norm'))

            info = model.optimize(xs, ys, opt, progress,
                                  max_grad_norm=cfg.get('max_grad_norm'))
            cls_sum += info.get('cls', 0.0)
            mmd_sum += info.get('mmd', 0.0)
            global_iter += 1

        per_domain, mean_f1 = evaluate_sources(model, val_loaders, device)
        n = iters_per_epoch
        improved = mean_f1 > best_f1
        row = {'epoch': epoch, 'cls_loss': cls_sum / n, 'mmd': mmd_sum / n}
        for d in SOURCE_DOMAINS:
            row[f'f1_{d}'] = per_domain[d]['f1']
        for d in SOURCE_DOMAINS:
            row[f'acc_{d}'] = per_domain[d]['acc']
        row['mean_val_f1'] = mean_f1
        row['best'] = int(improved)
        rows.append(row)
        write_logs(cfg['run'], rows, csv_path, json_path)

        print(f"[{cfg['run']}] epoch {epoch:2d}  cls={cls_sum/n:.4f}  "
              f"mmd={mmd_sum/n:.4f}  mean_val_f1={mean_f1:.4f}"
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
    ap.add_argument('--base', default='task3/configs/base.yaml')
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--resume', action='store_true',
                    help='resume from last.pt (Colab disconnects)')
    args = ap.parse_args()
    cfg = load_config(args.base, args.config)
    train(cfg, args.pacs_root, args.resume)


if __name__ == '__main__':
    main()
