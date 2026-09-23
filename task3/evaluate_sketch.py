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
from task2.train import load_config
from task2.methods.source_only import SourceOnly
from task2.evaluation.class_analysis import (
    per_class_accuracy, confusion_matrix, top_confusions_for_class,
    largest_gain_and_drop,
)
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM
from task3.evaluation.domain_metrics import extract, per_domain_metrics, summarize
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.evaluation.sharpness import make_sharpness_batch, sharpness
from task3.evaluation.compare_task2 import write_comparison

SEED = 6304
MAIN_RUNS = ['erm', 'dan_dg', 'sam', 'dan_dg_warmup']
DESIGN_RUNS = ['dan_dg_lambda0.1', 'dan_dg', 'dan_dg_lambda10']
LAMBDA_DG = {'dan_dg_lambda0.1': 0.1, 'dan_dg': 1.0, 'dan_dg_lambda10': 10.0}
RUN_CONFIGS = {
    'erm': 'task3/configs/erm.yaml',
    'dan_dg': 'task3/configs/dan_dg.yaml',
    'sam': 'task3/configs/sam.yaml',
    'dan_dg_warmup': 'task3/configs/dan_dg_warmup.yaml',
    'dan_dg_lambda0.1': 'task3/configs/dan_dg_lambda0.1.yaml',
    'dan_dg_lambda10': 'task3/configs/dan_dg_lambda10.yaml',
}


# Checkpoint-agnostic: the model class is chosen from the config saved INSIDE the
# checkpoint, not from the run name, so any best.pt reconstructs its own architecture.
def build_eval_method(cfg):
    m = cfg['method']
    n = cfg['num_classes']
    if m == 'source_only':
        return SourceOnly(n)
    if m == 'dan_dg':
        return DANDG(n, lambda_dg=cfg['lambda_dg'], n_per_source=cfg['batch_per_source'],
                     warmup_frac=cfg.get('lambda_warmup_frac', 0.0))
    if m == 'sam':
        return SAM(n, rho=cfg['rho'])
    raise ValueError(f"unknown method {m!r}")


# ERM reuses the Task 2 source_only checkpoint (reuse_checkpoint in erm.yaml); the leading
# 'checkpoints/' is stripped so CKPT_DIR (Drive on Colab) is honored. Everything else lives
# under <ckpt_dir>/task3/<run>/best.pt.
def checkpoint_path(cfg, run, ckpt_dir):
    reuse = cfg.get('reuse_checkpoint')
    if reuse:
        reuse = reuse.replace('\\', '/')
        rel = reuse[len('checkpoints/'):] if reuse.startswith('checkpoints/') else reuse
        return os.path.join(ckpt_dir, rel)
    return os.path.join(ckpt_dir, 'task3', run, 'best.pt')


def load_run(run, ckpt_dir, device):
    cfg = load_config('task3/configs/base.yaml', RUN_CONFIGS[run])
    path = checkpoint_path(cfg, run, ckpt_dir)
    if not os.path.exists(path):
        print(f"[skip] {run}: no checkpoint at {path}")
        return None
    state = torch.load(path, map_location=device)
    model = build_eval_method(state['config']).to(device)
    model.load_state_dict(state['model'])
    return model


def evaluate_run(model, src_val_loaders, sharp_batch, tgt_loader, device, num_classes):
    per_domain, feats = per_domain_metrics(model, src_val_loaders, device)
    summ = summarize(per_domain, SOURCE_DOMAINS)
    sep = source_domain_separability(feats)
    sharp = sharpness(model, *sharp_batch)

    tgt_feats, tgt_logits, tgt_labels = extract(model, tgt_loader, device)
    tgt_preds = tgt_logits.argmax(1)
    return {
        'per_domain': per_domain,
        'source_summary': summ,
        'source_separability': sep['separability'],
        'sep_n_per_domain': sep['n_per_domain'],
        'sep_chance': sep['chance'],
        'sharpness': sharp,
        'target_acc': float(top1_accuracy(tgt_logits, tgt_labels)),
        'target_f1': float(macro_f1(tgt_logits, tgt_labels)),
        'per_class_acc': per_class_accuracy(tgt_preds, tgt_labels, num_classes),
        'confusion': confusion_matrix(tgt_preds, tgt_labels, num_classes),
    }


# Results are organized per task (commit ffed9d1): canonical Task 2 final lives at
# results/task2/task2_final.json; the flat path is kept only as a fallback.
def task2_final_path():
    for p in ('results/task2/task2_final.json', 'results/task2_final.json'):
        if os.path.exists(p):
            return p
    return 'results/task2/task2_final.json'


# The Task 3 ERM Sketch row is the SAME frozen checkpoint as Task 2 source_only, so its
# target numbers must match Task 2's source_only row (a silent-regression tripwire).
def check_erm_consistency(erm_result):
    ref_path = task2_final_path()
    if not os.path.exists(ref_path):
        return {'checked': False, 'reason': f'{ref_path} not found'}
    with open(ref_path) as f:
        ref = json.load(f)
    so = ref.get('methods', {}).get('source_only')
    if not so:
        return {'checked': False, 'reason': 'source_only missing in task2_final.json'}
    d_acc = abs(erm_result['target_acc'] - so['target_acc'])
    d_f1 = abs(erm_result['target_f1'] - so['target_f1'])
    ok = d_acc < 1e-6 and d_f1 < 1e-6
    if not ok:
        print(f"[WARN] ERM Sketch row differs from Task 2 source_only: "
              f"d_acc={d_acc:.2e} d_f1={d_f1:.2e}")
    else:
        print("[check] ERM Sketch row matches Task 2 source_only exactly.")
    return {'checked': True, 'match': ok,
            'task2_target_acc': so['target_acc'], 'task2_target_f1': so['target_f1']}


def write_curves(log_root, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_cls, ax_mmd) = plt.subplots(1, 2, figsize=(11, 4))
    for run in DESIGN_RUNS + ['dan_dg_warmup', 'sam']:
        path = os.path.join(log_root, f'{run}.csv')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        epochs = [int(r['epoch']) for r in rows]
        ax_cls.plot(epochs, [float(r['cls_loss']) for r in rows], marker='.', label=run)
        if run != 'sam':
            ax_mmd.plot(epochs, [float(r['mmd']) for r in rows], marker='.', label=run)
    ax_cls.set(title='Classification loss', xlabel='epoch', ylabel='loss')
    ax_mmd.set(title='Pairwise source MMD (DAN-DG)', xlabel='epoch', ylabel='mmd')
    ax_cls.legend(); ax_mmd.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[curves] -> {out_path}")


def write_design_study_plot(summary, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r for r in DESIGN_RUNS if r in summary['design_study']]
    if not runs:
        return
    order = sorted(runs, key=lambda r: summary['design_study'][r]['lambda_dg'])
    lam = [summary['design_study'][r]['lambda_dg'] for r in order]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(lam, [summary['design_study'][r]['target_acc'] for r in order],
            marker='o', label='target (Sketch) acc')
    ax.plot(lam, [summary['design_study'][r]['worst_src_f1'] for r in order],
            marker='s', label='worst-source F1')
    ax.plot(lam, [summary['design_study'][r]['source_separability'] for r in order],
            marker='^', label='source separability')
    ax.axhline(1.0 / len(SOURCE_DOMAINS), ls='--', color='gray', lw=1,
               label='separability chance')
    ax.set_xscale('log')
    ax.set(title='DAN-DG alignment-strength study',
           xlabel='lambda_dg (log scale)', ylabel='value')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[design] -> {out_path}")


def write_sharpness_bar(summary, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r for r in MAIN_RUNS if r in summary['methods']]
    vals = [summary['methods'][r]['sharpness']['delta_sharpness'] for r in runs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(runs, vals)
    ax.set(title='Sharpness proxy  (L(theta+eps) - L(theta), rho=0.05)',
           ylabel='delta sharpness')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[sharpness] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm_frozen', action='store_true',
                    help='REQUIRED. Confirms all checkpoints and settings are frozen '
                         'before the Sketch target labels are read.')
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--ckpt_dir', default=os.environ.get('CKPT_DIR', 'checkpoints'))
    ap.add_argument('--split_path', default='shared/splits/pacs_sketch_seed6304.json')
    ap.add_argument('--out', default='results/task3/task3_final.json')
    args = ap.parse_args()

    if not args.confirm_frozen:
        raise SystemExit(
            "Refusing to load Sketch / read target labels. Re-run with --confirm_frozen "
            "only after every checkpoint, hyper-parameter and setting is frozen. Sketch "
            "must never influence Task 3 training, selection, or hyper-parameters.")

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
    sharp_batch = make_sharpness_batch(srcs, device)
    tgt_loader = DataLoader(target_eval_dataset(root, final_eval=True),
                            batch_size=64, shuffle=False, num_workers=2)
    num_classes = len(PACS_CLASSES)

    all_runs = MAIN_RUNS + [r for r in DESIGN_RUNS if r not in MAIN_RUNS]
    results = {}
    for run in all_runs:
        model = load_run(run, args.ckpt_dir, device)
        if model is None:
            continue
        results[run] = evaluate_run(model, src_val_loaders, sharp_batch,
                                    tgt_loader, device, num_classes)
        r = results[run]
        s = r['source_summary']
        print(f"[{run}] mean_src_f1={s['mean_f1']:.4f}  "
              f"worst_src_f1={s['worst_f1']:.4f} ({s['worst_f1_domain']})  "
              f"target_acc={r['target_acc']:.4f}  target_f1={r['target_f1']:.4f}  "
              f"source_sep={r['source_separability']:.4f}  "
              f"delta_sharp={r['sharpness']['delta_sharpness']:.4f}")

    if 'erm' not in results:
        raise SystemExit("erm checkpoint missing; it is the reference baseline.")
    base = results['erm']

    summary = {
        'classes': PACS_CLASSES,
        'separability_chance': 1.0 / len(SOURCE_DOMAINS),
        'erm_consistency': check_erm_consistency(base),
        'methods': {},
        'design_study': {},
    }
    for run in MAIN_RUNS:
        if run not in results:
            continue
        r = results[run]
        pc_delta = [a - b for a, b in zip(r['per_class_acc'], base['per_class_acc'])]
        entry = {
            'per_domain': r['per_domain'],
            'source_summary': r['source_summary'],
            'source_separability': r['source_separability'],
            'sep_n_per_domain': r['sep_n_per_domain'],
            'sharpness': r['sharpness'],
            'target_acc': r['target_acc'],
            'target_f1': r['target_f1'],
            'target_acc_change_vs_erm': r['target_acc'] - base['target_acc'],
            'per_class_acc': r['per_class_acc'],
            'per_class_change_vs_erm': pc_delta,
        }
        if run != 'erm':
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
        if run not in results:
            continue
        r = results[run]
        summary['design_study'][run] = {
            'lambda_dg': LAMBDA_DG[run],
            'mean_src_f1': r['source_summary']['mean_f1'],
            'worst_src_f1': r['source_summary']['worst_f1'],
            'source_separability': r['source_separability'],
            'delta_sharpness': r['sharpness']['delta_sharpness'],
            'target_acc': r['target_acc'],
        }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[json] -> {args.out}")

    with open('results/task3/task3_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method']
                   + [f'src_{d}_f1' for d in SOURCE_DOMAINS]
                   + [f'src_{d}_acc' for d in SOURCE_DOMAINS]
                   + ['mean_src_f1', 'worst_src_f1', 'worst_src_f1_domain',
                      'mean_src_acc', 'worst_src_acc', 'worst_src_acc_domain',
                      'target_acc', 'target_f1', 'target_acc_change_vs_erm',
                      'source_separability', 'delta_sharpness'])
        for run in MAIN_RUNS:
            if run not in summary['methods']:
                continue
            m = summary['methods'][run]
            s = m['source_summary']
            w.writerow([run]
                       + [m['per_domain'][d]['f1'] for d in SOURCE_DOMAINS]
                       + [m['per_domain'][d]['acc'] for d in SOURCE_DOMAINS]
                       + [s['mean_f1'], s['worst_f1'], s['worst_f1_domain'],
                          s['mean_acc'], s['worst_acc'], s['worst_acc_domain'],
                          m['target_acc'], m['target_f1'],
                          m['target_acc_change_vs_erm'],
                          m['source_separability'],
                          m['sharpness']['delta_sharpness']])

    with open('results/task3/task3_design_study.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'lambda_dg', 'mean_src_f1', 'worst_src_f1',
                    'source_separability', 'delta_sharpness', 'target_acc'])
        for run in DESIGN_RUNS:
            if run in summary['design_study']:
                d = summary['design_study'][run]
                w.writerow([run, d['lambda_dg'], d['mean_src_f1'], d['worst_src_f1'],
                            d['source_separability'], d['delta_sharpness'],
                            d['target_acc']])

    write_curves(os.path.join('results', 'task3', 'logs'),
                 os.path.join('report', 'figures', 'task3_curves.png'))
    write_design_study_plot(
        summary, os.path.join('report', 'figures', 'task3_design_study.png'))
    write_sharpness_bar(
        summary, os.path.join('report', 'figures', 'task3_sharpness.png'))
    write_comparison(summary, task2_path=task2_final_path())
    print("done.")


if __name__ == '__main__':
    main()
