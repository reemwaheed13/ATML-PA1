import os
import json
import argparse
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from common.seed import set_seed
from shared.pacs import (
    PACS_CLASSES, build_domain, train_transform, eval_transform,
)

SEED = 6304
SOURCE_DOMAINS = ['photo', 'art_painting', 'cartoon']
TARGET_DOMAIN = 'sketch'
DEFAULT_SPLIT_PATH = 'shared/splits/pacs_sketch_seed6304.json'


class MaskedLabels(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        return img, -1


def make_splits(root, out_path=DEFAULT_SPLIT_PATH):
    set_seed(SEED)
    splits = {
        'seed': SEED,
        'source_domains': SOURCE_DOMAINS,
        'target_domain': TARGET_DOMAIN,
        'classes': PACS_CLASSES,
        'domains': {},
    }
    for domain in SOURCE_DOMAINS:
        targets = np.array(build_domain(root, domain, eval_transform).targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
        splits['domains'][domain] = {
            'n_total': int(len(targets)),
            'train': sorted(int(i) for i in train_idx),
            'val':   sorted(int(i) for i in val_idx),
        }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"saved -> {out_path}")
    return splits


def load_splits(path=DEFAULT_SPLIT_PATH):
    with open(path) as f:
        return json.load(f)


def source_datasets(root, splits):
    out = {}
    for domain in SOURCE_DOMAINS:
        idx = splits['domains'][domain]
        out[domain] = {
            'train': Subset(build_domain(root, domain, train_transform), idx['train']),
            'val':   Subset(build_domain(root, domain, eval_transform),  idx['val']),
        }
    return out


def target_train_dataset(root):
    return MaskedLabels(build_domain(root, TARGET_DOMAIN, train_transform))


def target_eval_dataset(root, final_eval=False):
    if not final_eval:
        raise RuntimeError(
            "target labels are for final evaluation only; pass final_eval=True to confirm")
    return build_domain(root, TARGET_DOMAIN, eval_transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out_path', default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()
    make_splits(args.root, args.out_path)
