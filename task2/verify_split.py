import argparse
import os

import torch
from torch.utils.data import DataLoader

from shared.pacs_protocol import load_splits, source_datasets, SOURCE_DOMAINS
from task2.train import build_method, resolve_root, evaluate_sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pacs_root', default=None)
    ap.add_argument('--ckpt_dir', default=os.environ.get('CKPT_DIR', 'checkpoints'))
    ap.add_argument('--split_path', default='shared/splits/pacs_sketch_seed6304.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best = os.path.join(args.ckpt_dir, 'task2', 'source_only', 'best.pt')
    if not os.path.exists(best):
        raise SystemExit(f"checkpoint not found: {best}")
    state = torch.load(best, map_location=device)
    cfg = state['config']

    root = resolve_root(cfg, args.pacs_root)
    splits = load_splits(args.split_path)
    srcs = source_datasets(root, splits)
    val_loaders = {
        d: DataLoader(srcs[d]['val'], batch_size=64, shuffle=False,
                      num_workers=cfg.get('num_workers', 2))
        for d in SOURCE_DOMAINS
    }

    model = build_method(cfg).to(device)
    model.load_state_dict(state['model'])

    per_domain, mean_f1 = evaluate_sources(model, val_loaders, device)
    for d in SOURCE_DOMAINS:
        print(f"{d:14s}  macro-F1={per_domain[d]['f1']:.4f}  acc={per_domain[d]['acc']:.4f}")
    mean_acc = sum(per_domain[d]['acc'] for d in SOURCE_DOMAINS) / len(SOURCE_DOMAINS)
    print(f"{'mean':14s}  macro-F1={mean_f1:.4f}  acc={mean_acc:.4f}")


if __name__ == '__main__':
    main()
