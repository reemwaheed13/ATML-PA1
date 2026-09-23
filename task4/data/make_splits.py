import os
import json
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from common.seed import set_seed
from task4.data.cifar10 import build_cifar10, eval_transform, CIFAR10_CLASSES

SEED = 6304
DEFAULT_SPLIT_PATH = 'task4/data/splits/cifar10_seed6304.json'


def make_splits(root, out_path=DEFAULT_SPLIT_PATH):
    set_seed(SEED)
    ds = build_cifar10(root, True, eval_transform)
    targets = np.array(ds.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    splits = {
        'seed': SEED,
        'classes': CIFAR10_CLASSES,
        'n_total': int(len(targets)),
        'train': sorted(int(i) for i in train_idx),
        'val':   sorted(int(i) for i in val_idx),
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(splits, f)
    print(f"saved -> {out_path}  train={len(splits['train'])} val={len(splits['val'])}")
    return splits


def load_splits(path=DEFAULT_SPLIT_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.environ.get('DATA_ROOT', 'data'))
    parser.add_argument('--out_path', default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()
    make_splits(args.root, args.out_path)
