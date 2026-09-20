import os
import json
import numpy as np
from torchvision.datasets import STL10
from sklearn.model_selection import StratifiedShuffleSplit
from common.seed import set_seed

SEED = 6304

def create_stl10_splits(data_dir="./data"):
    set_seed(SEED)
    tr_ds = STL10(root=data_dir, split="train", download=True)
    t_ds = STL10(root=data_dir, split="test", download=True)

    targets = np.array(tr_ds.labels)
    tr_splitting = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(tr_splitting.split(np.zeros(len(targets)), targets))

    test_targets = np.array(t_ds.labels)
    te_splitting = StratifiedShuffleSplit(n_splits=1, train_size=500, random_state=SEED)
    test_idx, _ = next(te_splitting.split(np.zeros(len(test_targets)), test_targets))

    os.makedirs("results", exist_ok=True)
    split_data = [SEED, train_idx.tolist(), val_idx.tolist(), test_idx.tolist()]

    with open("results/task1_splits.json", "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Saved to results/task1_splits.json")


if __name__ == "__main__":
    create_stl10_splits()
