import os
import csv
import json
import argparse

import numpy as np
import torch

from common.metrics import top1_accuracy
from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import fit_mahalanobis, mahalanobis_score
from task4.evaluation.thresholds import osr_row, calibrate
from task4.evaluation.failure_analysis import confident_failures, plot_failures

CACHE_DIR = 'task4/cache'
OUT_DIR = 'results/task4'
FIG_DIR = 'report/figures'


def load_cache(model):
    p = os.path.join(CACHE_DIR, f'{model}.pt')
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location='cpu', weights_only=False)


def vanilla_scorer(cache):
    tr = cache['splits']['train']
    means, var = fit_mahalanobis(tr['feat'], tr['labels'])

    def compute(split, name):
        s = cache['splits'][split]
        if name == 'MSP':
            return msp_score(s['logits'])
        if name == 'MLS':
            return mls_score(s['logits'])
        if name == 'Energy':
            return energy_score(s['logits'])
        if name == 'Mahalanobis':
            return mahalanobis_score(s['feat'], means, var)
        raise ValueError(name)
    return compute


def placeholder_u(cache, split):
    a = cache['splits'][split]['augmented']  # [known(10), strongest-dummy(1)]
    m = a.max(axis=1, keepdims=True)
    p = np.exp(a - m)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, -1]  # dummy probability = unknownness


def write_csv(path, rows):
    fields = ['name'] + list(next(iter(rows.values())).keys())
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name, row in rows.items():
            w.writerow([name] + [row[k] for k in fields[1:]])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    van = load_cache('vanilla')
    if van is None:
        raise SystemExit("no vanilla cache; run extract_outputs --model vanilla first")
    compute = vanilla_scorer(van)

    # --- Table 1: four post-hoc scores on the frozen vanilla model ---
    table1 = {}
    for name in ['MSP', 'MLS', 'Energy', 'Mahalanobis']:
        table1[name] = osr_row(compute('val', name), compute('test', name),
                               compute('near', name), compute('far', name))

    # --- Table 2: Vanilla / GCSC / PROSER via MLS + PROSER placeholder row ---
    table2 = {}
    for model in ['vanilla', 'gcsc', 'proser']:
        c = load_cache(model)
        if c is None:
            continue
        row = osr_row(mls_score(c['splits']['val']['logits']),
                      mls_score(c['splits']['test']['logits']),
                      mls_score(c['splits']['near']['logits']),
                      mls_score(c['splits']['far']['logits']))
        row['csa'] = float(top1_accuracy(c['splits']['test']['logits'],
                                         c['splits']['test']['labels']))
        row['score'] = 'MLS'
        table2[model] = row

    pc = load_cache('proser')
    if pc is not None and 'augmented' in pc['splits']['val']:
        row = osr_row(placeholder_u(pc, 'val'), placeholder_u(pc, 'test'),
                      placeholder_u(pc, 'near'), placeholder_u(pc, 'far'))
        row['csa'] = float(top1_accuracy(pc['splits']['test']['logits'],
                                         pc['splits']['test']['labels']))
        row['score'] = 'placeholder'
        table2['proser_placeholder'] = row

    # --- Figure: score distributions for MSP / MLS / Mahalanobis (known vs unknown) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, name in zip(axes, ['MSP', 'MLS', 'Mahalanobis']):
        u_test = compute('test', name)
        u_unk = np.concatenate([compute('near', name), compute('far', name)])
        ax.hist(u_test, bins=50, alpha=0.6, density=True, label='known')
        ax.hist(u_unk, bins=50, alpha=0.6, density=True, label='unknown')
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'task4_scores.png'), dpi=150)
    plt.close(fig)

    # --- Failure analysis: confidently accepted unknowns under vanilla MLS tau ---
    tau_mls = calibrate(mls_score(van['splits']['val']['logits']), 0.95)
    near_recs, near_idx = confident_failures(van, 'near',
                                             mls_score(van['splits']['near']['logits']), tau_mls)
    far_recs, far_idx = confident_failures(van, 'far',
                                           mls_score(van['splits']['far']['logits']), tau_mls)
    plot_failures(van, near_recs, near_idx, far_recs, far_idx,
                  os.path.join(FIG_DIR, 'task4_failures.png'))

    out = {
        'table1_vanilla_scores': table1,
        'table2_model_comparison': table2,
        'vanilla_mls_tau': tau_mls,
        'failures': {'near': near_recs, 'far': far_recs},
    }
    with open(os.path.join(OUT_DIR, 'task4_final.json'), 'w') as f:
        json.dump(out, f, indent=2)
    write_csv(os.path.join(OUT_DIR, 'task4_vanilla_scores.csv'), table1)
    write_csv(os.path.join(OUT_DIR, 'task4_model_comparison.csv'), table2)

    print("wrote results/task4/{task4_final.json,task4_vanilla_scores.csv,"
          "task4_model_comparison.csv} and report/figures/task4_{scores,failures}.png")


if __name__ == '__main__':
    main()
