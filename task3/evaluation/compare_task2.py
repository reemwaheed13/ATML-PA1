import os
import csv
import json

from shared.pacs import PACS_CLASSES

# task3 method -> the Task 2 method it is the natural counterpart of.
# ERM (target-free baseline) is literally the Task 2 source_only checkpoint; DAN-DG
# (target-FREE pairwise source alignment) is the DG analogue of DAN (target-AWARE
# source-target alignment). SAM has no Task 2 counterpart, so it is not paired here.
PAIRS = [('erm', 'source_only'), ('dan_dg', 'dan')]


def _load_task2(path):
    if not os.path.exists(path):
        print(f"[compare] {path} not found; skipping Task 2 comparison.")
        return None
    with open(path) as f:
        return json.load(f)


# NOTE ON SEPARABILITY: the two tasks report DIFFERENT diagnostics under this name --
# Task 2 = binary source-vs-target(Sketch) separability (alignment to the target); Task 3
# = 3-way source-domain separability (source invariance). They are placed in distinctly
# named columns and must NOT be read as the same axis. Only Sketch acc/F1 and the per-class
# Sketch deltas are like-for-like (same frozen Sketch eval set, same class order).
def write_comparison(task3_summary,
                     task2_path='results/task2/task2_final.json',
                     out_csv='results/task3/task3_vs_task2.csv',
                     out_perclass='results/task3/task3_vs_task2_perclass.csv',
                     out_fig='report/figures/task3_vs_task2.png'):
    t2 = _load_task2(task2_path)
    if t2 is None:
        return
    t2m = t2.get('methods', {})
    t3m = task3_summary.get('methods', {})

    rows = []
    for t3_name, t2_name in PAIRS:
        if t3_name not in t3m or t2_name not in t2m:
            print(f"[compare] skip pair {t3_name}<->{t2_name}: missing result")
            continue
        a, b = t3m[t3_name], t2m[t2_name]
        rows.append({
            'task3_method': t3_name,
            'task2_method': t2_name,
            't3_target_acc': a['target_acc'],
            't2_target_acc': b['target_acc'],
            'd_target_acc': a['target_acc'] - b['target_acc'],
            't3_target_f1': a['target_f1'],
            't2_target_f1': b['target_f1'],
            'd_target_f1': a['target_f1'] - b['target_f1'],
            't3_source_sep_3way': a['source_separability'],
            't2_sep_src_vs_tgt': b['separability'],
        })
    if not rows:
        return

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[compare] -> {out_csv}")

    # Per-class Sketch deltas: DAN-DG (vs ERM) beside DAN (vs source_only). ERM and
    # source_only are the SAME frozen checkpoint, so both deltas share one baseline and
    # are directly comparable class-by-class.
    have_perclass = ('dan_dg' in t3m and 'dan' in t2m
                     and 'per_class_change_vs_erm' in t3m['dan_dg']
                     and 'per_class_change_vs_source_only' in t2m['dan'])
    if have_perclass:
        dg = t3m['dan_dg']['per_class_change_vs_erm']
        dan = t2m['dan']['per_class_change_vs_source_only']
        with open(out_perclass, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class', 'dan_dg_delta_vs_erm',
                        'dan_delta_vs_source_only', 'dg_minus_dan'])
            for c, cls in enumerate(PACS_CLASSES):
                w.writerow([cls, dg[c], dan[c], dg[c] - dan[c]])
        print(f"[compare] -> {out_perclass}")

    _write_fig(rows, t3m, t2m, out_fig)


def _write_fig(rows, t3m, t2m, out_fig):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    have_perclass = 'dan_dg' in t3m and 'dan' in t2m
    ncols = 2 if have_perclass else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    x = np.arange(len(rows))
    labels = [f"{r['task3_method']}\nvs {r['task2_method']}" for r in rows]
    ax.bar(x - 0.2, [r['t3_target_acc'] for r in rows], 0.4, label='Task 3 (DG, target-free)')
    ax.bar(x + 0.2, [r['t2_target_acc'] for r in rows], 0.4, label='Task 2 (UDA, target-aware)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(title='Sketch accuracy: Task 3 vs Task 2', ylabel='target (Sketch) accuracy')
    ax.legend()

    if have_perclass:
        dg = t3m['dan_dg']['per_class_change_vs_erm']
        dan = t2m['dan']['per_class_change_vs_source_only']
        ax = axes[1]
        xc = np.arange(len(PACS_CLASSES))
        ax.bar(xc - 0.2, dg, 0.4, label='DAN-DG (vs ERM)')
        ax.bar(xc + 0.2, dan, 0.4, label='DAN (vs source_only)')
        ax.axhline(0, color='gray', lw=1)
        ax.set_xticks(xc)
        ax.set_xticklabels(PACS_CLASSES, rotation=45, ha='right')
        ax.set(title='Per-class Sketch change (shared ERM/source_only baseline)',
               ylabel='delta accuracy')
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig) or '.', exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"[compare] -> {out_fig}")
