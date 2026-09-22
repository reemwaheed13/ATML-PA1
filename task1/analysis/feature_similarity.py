import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from PIL import Image
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
    def __init__(self, base_ds, indices, transform_fn, normalize, pass_idx=False):
        self.base = base_ds
        self.indices = indices
        self.transform_fn = transform_fn
        self.normalize = normalize
        self.pass_idx = pass_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        img = resize_224(img)
        img = self.transform_fn(img, i) if self.pass_idx else self.transform_fn(img)
        return self.normalize(img), label


def get_transformed_feats(transform_fn, backbone, te_ds, test_idx, normalize, device,
                          pass_idx=False):
    loader = DataLoader(TransformedSubset(te_ds, test_idx, transform_fn, normalize, pass_idx),
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
    parser.add_argument('--results_dir', default='./results/task1')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(args.results_dir, 'task1_splits.json')) as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds = STL10(root=args.data_dir, split='test', download=False)

    _cc_path = os.path.join(args.results_dir, 'task1_cue_conflicts_metadata.json')
    if os.path.exists(_cc_path):
        with open(_cc_path) as f:
            _cc = json.load(f)['conflicts']
        _cc_dir  = os.path.join(args.results_dir, 'cue_conflicts')
        _cc_pils = [Image.open(os.path.join(_cc_dir, c['filename'])).convert('RGB') for c in _cc]
    else:
        _cc = None

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


        for label, fn, use_idx in [('grayscale',    to_grayscale,  False),
                                   ('hue_rotation', rotate_hue,    False),
                                   ('patch_shuffle', patch_shuffle, True)]:
            print(f"  {label}")
            trans = get_transformed_feats(fn, backbone, te_ds, test_idx, norm, device,
                                          pass_idx=use_idx)
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

        # cue conflict — pair each conflict with the clean feature of its content image
        if _cc is not None:
            conf_feats = []
            with torch.no_grad():
                for start in range(0, len(_cc_pils), 64):
                    batch = torch.stack([norm(img) for img in _cc_pils[start:start+64]]).to(device)
                    conf_feats.append(backbone(batch).cpu().numpy())
            conf_feats   = np.concatenate(conf_feats, axis=0)
            clean_paired = np.array([clean_feats[c['content_pos']] for c in _cc])
            r['cue_conflict'] = cosine_stability(clean_paired, conf_feats)
            print(f"  cue_conflict cosine_stability={r['cue_conflict']:.4f}")

        results[name] = r
        print(r)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, 'task1_feature_similarity.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
