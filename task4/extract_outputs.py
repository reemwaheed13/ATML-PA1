import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed
from task4.data.make_splits import load_splits
from task4.data.cifar10 import cifar10_eval_subset, cifar10_test
from task4.data.cifar100_unknowns import build_cifar100_unknowns
from task4.train import build_method, resolve_root

CACHE_DIR = 'task4/cache'


def ckpt_path(model):
    root = os.environ.get('CKPT_DIR', 'checkpoints')
    return os.path.join(root, 'task4', model, 'best.pt')


def load_model(model_name, device):
    state = torch.load(ckpt_path(model_name), map_location=device)
    cfg = state['config']
    model = build_method(cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_split(model, ds, device, want_augmented):
    # every split runs through the eval (unaugmented) transform -> identical pipeline
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    feats, logits, labels, augs = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        f = model.features(x)
        z = model.net.fc(f)  # 10 known logits, from the SAME feature
        feats.append(f.cpu().numpy())
        logits.append(z.cpu().numpy())
        labels.append(np.asarray(y))
        if want_augmented:
            augs.append(model.augmented_logits(x).cpu().numpy())
    out = {
        'feat': np.concatenate(feats),
        'logits': np.concatenate(logits),
        'labels': np.concatenate(labels),
    }
    if want_augmented:
        out['augmented'] = np.concatenate(augs)  # [known(10), strongest-dummy(1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['vanilla', 'gcsc', 'proser'])
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--confirm_unknowns', action='store_true',
                    help='CIFAR-100 is evaluation-only; required to load the unknown groups')
    args = ap.parse_args()
    if not args.confirm_unknowns:
        raise SystemExit(
            "CIFAR-100 unknowns are evaluation-only; pass --confirm_unknowns to proceed.")

    set_seed(6304)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = resolve_root(args.data_root)

    model, cfg = load_model(args.model, device)
    splits = load_splits(cfg['split_path'])
    want_aug = (cfg['method'] == 'proser')

    datasets = {
        'train': cifar10_eval_subset(root, splits, 'train'),  # Mahalanobis mu_c/Sigma
        'val':   cifar10_eval_subset(root, splits, 'val'),    # threshold calibration
        'test':  cifar10_test(root),                          # known evaluation
        'near':  build_cifar100_unknowns(root, 'near'),
        'far':   build_cifar100_unknowns(root, 'far'),
    }

    cache = {'model': args.model, 'method': cfg['method'], 'splits': {}}
    for name, ds in datasets.items():
        d = extract_split(model, ds, device, want_aug)
        if name in ('near', 'far'):
            # carry class names + raw images so evaluate_osr stays cache-only
            classes = ds.dataset.classes
            targets = ds.dataset.targets
            d['class_names'] = [classes[targets[i]] for i in ds.indices]
            d['images'] = ds.dataset.data[ds.indices]
        cache['splits'][name] = d
        print(f"[{args.model}] {name}: feat={d['feat'].shape} logits={d['logits'].shape}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f"{args.model}.pt")
    torch.save(cache, out)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
