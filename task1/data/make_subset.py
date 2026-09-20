import os
import json
import numpy as np
from torchvision.datasets import STL10
from sklearn.model_selection import StratifiedShuffleSplit #split data w equal eg from each class
from common.seed import set_seed

SEED = 6304

def create_stl10_splits(data_dir="./data"):
    set_seed(SEED)
    tr_ds = STL10(root=data_dir, split="train", download=True)
    t_ds = STL10(root=data_dir, split="test", download=True) #now we have the test and train datasets

    targets = np.array(tr_ds.labels)
    tr_splitting = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(tr_splitting.split(np.zeros(len(targets)),targets)) #this will be placeholder for all of task1 so no data rn

    # actually select the 500 test images.
    test_targets = np.array(t_ds.labels)
    te_splitting = StratifiedShuffleSplit(n_splits=1, train_size=500, random_state=SEED)
    test_idx, _ = next(te_splitting.split(np.zeros(len(test_targets)), test_targets)) 
    #we want the selected 500 test images, not the rest

    os.makedirs("results", exist_ok=True)
    seed = [SEED]
    train_indices= train_idx.tolist()
    val_indices= val_idx.tolist()
    eval_500_indices= test_idx.tolist()

    split_data = [seed, train_indices, val_indices, eval_500_indices]

    #save the json file here
    output_path = "results/task1_splits.json"

    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"saved to {output_path}")


if __name__ == "__main__":
    create_stl10_splits()


