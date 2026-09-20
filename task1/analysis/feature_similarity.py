import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm

from common.seed import set_seed
from task1.models.backbone import (
    ResNet50Backbone, ViTBackbone, CLIPBackbone,
)
from task1.data.transforms import (
    resize_224, normalize_for, to_grayscale, rotate_hue, translate, patch_shuffle,
)

SEED = 6304


class TransformedSubset(Dataset):
    def __init__(self, base_ds, indices, transform_fn, normalize):
        self.base = base_ds
        self.indices = indices
        self.transform_fn = transform_fn
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        img = self.transform_fn(resize_224(img))   # intervene on common 224 canvas
        return self.normalize(img), label


def get_transformed_feats(transform_fn, backbone, te_ds, test_idx, normalize, device):
    loader = DataLoader(TransformedSubset(te_ds, test_idx, transform_fn, normalize),
                        batch_size=64, shuffle=False, num_workers=2)
    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, leave=False):
            feats.append(backbone(imgs.to(device)).cpu())
    return torch.cat(feats).numpy()


def cosine_stability(clean, transformed):
    # average cosine sim between each clean/transformed feature pair
    a = clean       / np.linalg.norm(clean,       axis=1, keepdims=True)
    b = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
    return float((a * b).sum(axis=1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--cache_dir',   default='./cache/features')
    parser.add_argument('--results_dir', default='./results')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('results/task1_splits.json') as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds = STL10(root=args.data_dir, split='test', download=False)

    configs = [
        ('resnet', ResNet50Backbone),
        ('vit',    ViTBackbone),
        ('clip',   CLIPBackbone),
    ]

    results = {}

    for name, BackboneClass in configs:
        print(f"\n--- {name} ---")
        backbone = BackboneClass().to(device)
        norm = normalize_for(name)

        # clean features already cached from run_task1.py
        clean_feats = np.load(
            os.path.join(args.cache_dir, name, f'{name}_test.npz')
        )['feats']

        r = {}

        # single-shot interventions
        for label, fn in [('grayscale',    to_grayscale),
                           ('hue_rotation', rotate_hue),
                           ('patch_shuffle', patch_shuffle)]:
            print(f"  {label}")
            trans = get_transformed_feats(fn, backbone, te_ds, test_idx, norm, device)
            r[label] = cosine_stability(clean_feats, trans)

        # translation: average stability across 4 directions per delta
        r['translation'] = {}
        for delta in [8, 16, 32]:
            stabs = []
            for direction in ['right', 'left', 'up', 'down']:
                fn = lambda img, d=delta, dr=direction: translate(img, d, dr)
                trans = get_transformed_feats(fn, backbone, te_ds, test_idx, norm, device)
                stabs.append(cosine_stability(clean_feats, trans))
            r['translation'][delta] = float(np.mean(stabs))

        results[name] = r
        print(r)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, 'task1_feature_similarity.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
