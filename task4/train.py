import os
import csv
import json
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from common.metrics import top1_accuracy
from task4.data.make_splits import load_splits
from task4.data.cifar10 import cifar10_train_val, TRAIN_TRANSFORMS
from task4.methods.vanilla import Vanilla
from task4.methods.gcsc import GCSC


def load_config(base_path, method_path):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(method_path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def resolve_root(cli_root):
    return cli_root or os.environ.get('DATA_ROOT', 'data')


def resolve_ckpt(rel):
    # config stores 'checkpoints/task4/<run>/best.pt'; strip prefix to honor CKPT_DIR
    root = os.environ.get('CKPT_DIR', 'checkpoints')
    return os.path.join(root, rel.split('checkpoints/', 1)[-1])


def build_method(cfg):
    n = cfg['num_classes']
    m = cfg['method']
    if m == 'vanilla':
        return Vanilla(n)
    if m == 'gcsc':
        return GCSC(n)
    if m == 'proser':
        from task4.methods.proser import PROSER
        return PROSER(n, num_dummy=cfg['num_dummy'], beta=cfg['beta'], gamma=cfg['gamma'])
    raise ValueError(f"unknown method {m!r}")


def init_from_checkpoint(model, cfg, device):
    rel = cfg.get('init_checkpoint')
    if not rel:
        return
    path = resolve_ckpt(rel)
    state = torch.load(path, map_location=device)['model']
    net_state = {k[len('net.'):]: v for k, v in state.items() if k.startswith('net.')}
    model.net.load_state_dict(net_state, strict=True)
    print(f"[init] loaded backbone from {path}")


@torch.no_grad()
def evaluate_known(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        logits_all.append(model.logits(x.to(device)).cpu().numpy())
        labels_all.append(y.numpy())
    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    return float(top1_accuracy(logits_all, labels_all))


def write_logs(run, rows, csv_path, json_path):
    fields = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log = {k: [r[k] for r in rows] for k in fields}
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
    root = resolve_root(cli_root)

    splits = load_splits(cfg['split_path'])
    train_tf = TRAIN_TRANSFORMS[cfg['train_transform']]
    ds = cifar10_train_val(root, splits, train_tf)

    gen = torch.Generator()
    gen.manual_seed(cfg['seed'])
    train_loader = DataLoader(ds['train'], batch_size=cfg['batch_size'], shuffle=True,
                              drop_last=True, num_workers=cfg['num_workers'], generator=gen)
    val_loader = DataLoader(ds['val'], batch_size=256, shuffle=False,
                            num_workers=cfg['num_workers'])

    model = build_method(cfg).to(device)
    init_from_checkpoint(model, cfg, device)

    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['max_epochs'])

    ckpt_dir = os.path.join(os.environ.get('CKPT_DIR', 'checkpoints'), 'task4', cfg['run'])
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, 'last.pt')
    best_path = os.path.join(ckpt_dir, 'best.pt')

    log_dir = os.path.join('results', 'task4', 'logs')
    json_dir = os.path.join('results', 'task4', 'logs_json')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{cfg['run']}.csv")
    json_path = os.path.join(json_dir, f"{cfg['run']}.json")

    start_epoch, best_acc, best_epoch = 0, -1.0, -1
    rows = []
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        start_epoch = state['epoch'] + 1
        best_acc, best_epoch = state['best_acc'], state['best_epoch']
        rows = load_prior_rows(json_path)
        print(f"[resume] continuing from epoch {start_epoch} (best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, cfg['max_epochs']):
        model.train()
        cur_lr = opt.param_groups[0]['lr']
        loss_sum, n_iter = {}, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            info = model.optimize(x, y, opt)
            for k, v in info.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + v
            n_iter += 1
        sched.step()

        val_acc = evaluate_known(model, val_loader, device)
        improved = val_acc > best_acc
        row = {'epoch': epoch, 'lr': cur_lr}
        for k in loss_sum:
            row[k] = loss_sum[k] / n_iter
        row['val_acc'] = val_acc
        row['best'] = int(improved)
        rows.append(row)
        write_logs(cfg['run'], rows, csv_path, json_path)

        loss_str = '  '.join(f"{k}={loss_sum[k] / n_iter:.4f}" for k in loss_sum)
        print(f"[{cfg['run']}] epoch {epoch:3d}  lr={cur_lr:.4f}  {loss_str}  "
              f"val_acc={val_acc:.4f}" + ("  *best*" if improved else ""))

        if improved:
            best_acc, best_epoch = val_acc, epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_acc': val_acc, 'config': cfg}, best_path)

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': epoch,
                    'best_acc': best_acc, 'best_epoch': best_epoch,
                    'config': cfg}, last_path)

    print(f"[{cfg['run']}] done. best CIFAR-10 val acc "
          f"{best_acc:.4f} @ epoch {best_epoch} -> {best_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='method yaml (extends base.yaml)')
    ap.add_argument('--base', default='task4/configs/base.yaml')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--resume', action='store_true', help='resume from last.pt')
    args = ap.parse_args()
    cfg = load_config(args.base, args.config)
    train(cfg, args.data_root, args.resume)


if __name__ == '__main__':
    main()
