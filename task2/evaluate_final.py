import os
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from common.metrics import macro_f1, top1_accuracy
from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import (
    load_splits, source_datasets, target_eval_dataset, SOURCE_DOMAINS,
)
from task2.train import build_method
from task2.evaluation.domain_separability import domain_separability
from task2.evaluation.class_analysis import (
    per_class_accuracy, confusion_matrix, top_confusions_for_class,
    largest_gain_and_drop,
)

SEED = 6304
MAIN_RUNS = ['source_only', 'dan', 'dann', 'cdan']
DESIGN_RUNS = ['dan_lambda0.1', 'dan', 'dan_lambda10']


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        f = model.backbone(x.to(device))
        feats.append(f.cpu().numpy())
        logits.append(model.head(f).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


def load_run(run, ckpt_dir, device):
    best = os.path.join(ckpt_dir, 'task2', run, 'best.pt')
    if not os.path.exists(best):
        print(f"[skip] {run}: no checkpoint at {best}")
        return None
    state = torch.load(best, map_location=device)
    model = build_method(state['config']).to(device)
    model.load_state_dict(state['model'])
    return model


def evaluate_run(model, src_val_loaders, tgt_loader, device, num_classes):
    per_domain, src_feats = {}, []
    for d, loader in src_val_loaders.items():
        f, logits, labels = extract(model, loader, device)
        src_feats.append(f)
        per_domain[d] = {'acc': float(top1_accuracy(logits, labels)),
                         'f1': float(macro_f1(logits, labels))}
    src_feats = np.concatenate(src_feats)
    mean_src_acc = float(np.mean([per_domain[d]['acc'] for d in SOURCE_DOMAINS]))
    mean_src_f1 = float(np.mean([per_domain[d]['f1'] for d in SOURCE_DOMAINS]))

    tgt_feats, tgt_logits, tgt_labels = extract(model, tgt_loader, device)
    tgt_preds = tgt_logits.argmax(1)
    sep = domain_separability(src_feats, tgt_feats, seed=SEED)

    return {
        'per_domain': per_domain,
        'mean_src_acc': mean_src_acc,
        'mean_src_f1': mean_src_f1,
        'target_acc': float(top1_accuracy(tgt_logits, tgt_labels)),
        'target_f1': float(macro_f1(tgt_logits, tgt_labels)),
        'separability': sep['separability'],
        'sep_n_per_domain': sep['n_per_domain'],
        'per_class_acc': per_class_accuracy(tgt_preds, tgt_labels, num_classes),
        'confusion': confusion_matrix(tgt_preds, tgt_labels, num_classes),
        'labels': tgt_labels,
        'preds': tgt_preds,
    }


def write_curves(runs, log_root, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_cls, ax_align) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        path = os.path.join(log_root, f'{run}.csv')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        epochs = [int(r['epoch']) for r in rows]
        ax_cls.plot(epochs, [float(r['cls_loss']) for r in rows], marker='.', label=run)
        if run != 'source_only':
            ax_align.plot(epochs, [float(r['align']) for r in rows], marker='.', label=run)
    ax_cls.set(title='Classification loss', xlabel='epoch', ylabel='loss')
    ax_align.set(title='Alignment / domain loss', xlabel='epoch', ylabel='loss')
    ax_cls.legend(); ax_align.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[curves] -> {out_path}")


def write_separability_scatter(summary, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for run, m in summary['methods'].items():
        ax.scatter(m['separability'], m['target_acc'])
        ax.annotate(run, (m['separability'], m['target_acc']),
                    textcoords='offset points', xytext=(5, 5))
    ax.axvline(0.5, ls='--', color='gray', lw=1)
    ax.set(title='Domain separability vs target accuracy',
           xlabel='domain separability (0.5 = chance)',
           ylabel='target (Sketch) accuracy')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[scatter] -> {out_path}")


def write_design_study_plot(summary, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r for r in DESIGN_RUNS if r in summary['design_study']]
    if not runs:
        return
    order = sorted(runs, key=lambda r: summary['design_study'][r]['lambda_mmd'])
    lam = [summary['design_study'][r]['lambda_mmd'] for r in order]
    tgt = [summary['design_study'][r]['target_acc'] for r in order]
    sep = [summary['design_study'][r]['separability'] for r in order]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(lam, tgt, marker='o', label='target accuracy')
    ax.plot(lam, sep, marker='s', label='separability')
    ax.set_xscale('log')
    ax.set(title='DAN alignment-strength study',
           xlabel='lambda_mmd (log scale)', ylabel='value')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[design] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm_frozen', action='store_true',
                    help='REQUIRED. Confirms all checkpoints and settings are frozen '
                         'before target labels are read.')
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--ckpt_dir', default=os.environ.get('CKPT_DIR', 'checkpoints'))
    ap.add_argument('--split_path', default='shared/splits/pacs_sketch_seed6304.json')
    ap.add_argument('--out', default='results/task2_final.json')
    args = ap.parse_args()

    if not args.confirm_frozen:
        raise SystemExit(
            "Refusing to read target labels. Re-run with --confirm_frozen only after "
            "every checkpoint, hyper-parameter and setting is frozen.")

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = args.pacs_root or os.environ.get('PACS_ROOT')
    if not root:
        raise SystemExit("PACS root not set: pass --pacs_root or set PACS_ROOT.")

    splits = load_splits(args.split_path)
    srcs = source_datasets(root, splits)
    src_val_loaders = {
        d: DataLoader(srcs[d]['val'], batch_size=64, shuffle=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }
    tgt_loader = DataLoader(target_eval_dataset(root, final_eval=True),
                            batch_size=64, shuffle=False, num_workers=2)
    num_classes = len(PACS_CLASSES)

    all_runs = MAIN_RUNS + [r for r in DESIGN_RUNS if r not in MAIN_RUNS]
    results = {}
    for run in all_runs:
        model = load_run(run, args.ckpt_dir, device)
        if model is None:
            continue
        results[run] = evaluate_run(model, src_val_loaders, tgt_loader,
                                    device, num_classes)
        r = results[run]
        print(f"[{run}] mean_src_f1={r['mean_src_f1']:.4f}  "
              f"target_acc={r['target_acc']:.4f}  target_f1={r['target_f1']:.4f}  "
              f"separability={r['separability']:.4f}")

    if 'source_only' not in results:
        raise SystemExit("source_only checkpoint missing; it is the reference baseline.")
    base = results['source_only']

    summary = {'classes': PACS_CLASSES, 'methods': {}, 'design_study': {}}
    for run in MAIN_RUNS:
        if run not in results:
            continue
        r = results[run]
        pc_delta = [a - b for a, b in zip(r['per_class_acc'], base['per_class_acc'])]
        entry = {
            'per_domain': r['per_domain'],
            'mean_src_acc': r['mean_src_acc'], 'mean_src_f1': r['mean_src_f1'],
            'target_acc': r['target_acc'], 'target_f1': r['target_f1'],
            'target_acc_change_vs_source_only': r['target_acc'] - base['target_acc'],
            'separability': r['separability'],
            'per_class_acc': r['per_class_acc'],
            'per_class_change_vs_source_only': pc_delta,
        }
        if run != 'source_only':
            gd = largest_gain_and_drop(pc_delta, PACS_CLASSES)
            cm = r['confusion']
            gi = PACS_CLASSES.index(gd['largest_gain']['class'])
            di = PACS_CLASSES.index(gd['largest_drop']['class'])
            entry['largest_gain'] = {**gd['largest_gain'],
                                     'confusions': top_confusions_for_class(cm, PACS_CLASSES, gi)}
            entry['largest_drop'] = {**gd['largest_drop'],
                                     'confusions': top_confusions_for_class(cm, PACS_CLASSES, di)}
        summary['methods'][run] = entry

    for run in DESIGN_RUNS:
        if run in results:
            r = results[run]
            summary['design_study'][run] = {
                'lambda_mmd': {'dan_lambda0.1': 0.1, 'dan': 1.0, 'dan_lambda10': 10.0}[run],
                'mean_src_acc': r['mean_src_acc'], 'mean_src_f1': r['mean_src_f1'],
                'separability': r['separability'], 'target_acc': r['target_acc'],
            }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[json] -> {args.out}")

    with open('results/task2_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method'] + [f'src_{d}_f1' for d in SOURCE_DOMAINS]
                   + ['mean_src_f1', 'mean_src_acc', 'target_acc', 'target_f1',
                      'target_acc_change', 'separability'])
        for run in MAIN_RUNS:
            if run not in summary['methods']:
                continue
            m = summary['methods'][run]
            w.writerow([run] + [m['per_domain'][d]['f1'] for d in SOURCE_DOMAINS]
                       + [m['mean_src_f1'], m['mean_src_acc'], m['target_acc'],
                          m['target_f1'], m['target_acc_change_vs_source_only'],
                          m['separability']])

    with open('results/task2_design_study.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'lambda_mmd', 'mean_src_f1', 'separability', 'target_acc'])
        for run in DESIGN_RUNS:
            if run in summary['design_study']:
                d = summary['design_study'][run]
                w.writerow([run, d['lambda_mmd'], d['mean_src_f1'],
                            d['separability'], d['target_acc']])

    write_curves(MAIN_RUNS, os.path.join('results', 'task2', 'logs'),
                 os.path.join('report', 'figures', 'task2_curves.png'))
    write_separability_scatter(
        summary, os.path.join('report', 'figures', 'task2_separability.png'))
    write_design_study_plot(
        summary, os.path.join('report', 'figures', 'task2_design_study.png'))
    print("done.")


if __name__ == '__main__':
    main()
