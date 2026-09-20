import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from PIL import Image
from common.seed import set_seed
from task1.models.backbone import ResNet50Backbone, ViTBackbone, CLIPBackbone
from task1.data.transforms import (
    resize_224, normalize_for, to_grayscale, rotate_hue, translate, patch_shuffle,
)

SEED = 6304
STL10_CLASSES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']


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


def plot_tsne(clean_feats, trans_feats, labels, title, out_path):
    # fit tsne on clean + transformed together so they share the same 2d space
    combined = np.concatenate([clean_feats, trans_feats], axis=0)
    emb = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(combined)

    n = len(clean_feats)
    clean_emb = emb[:n]
    trans_emb  = emb[n:]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in range(10):
        mask = labels == cls
        # circles = clean, x = transformed
        ax.scatter(clean_emb[mask, 0], clean_emb[mask, 1],
                   c=[colors[cls]], marker='o', s=15, alpha=0.7, label=STL10_CLASSES[cls])
        ax.scatter(trans_emb[mask, 0], trans_emb[mask, 1],
                   c=[colors[cls]], marker='x', s=15, alpha=0.7)

    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--cache_dir',   default='./cache/features')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--figures_dir', default='./report/figures')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(args.results_dir, 'task1_splits.json')) as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds = STL10(root=args.data_dir, split='test', download=False)
    os.makedirs(args.figures_dir, exist_ok=True)

    _cc_path = os.path.join(args.results_dir, 'cue_conflicts_metadata.json')
    if os.path.exists(_cc_path):
        with open(_cc_path) as f:
            _cc = json.load(f)['conflicts']
        _cc_dir  = os.path.join(args.results_dir, 'cue_conflicts')
        _cc_pils = [Image.open(os.path.join(_cc_dir, c['filename'])).convert('RGB') for c in _cc]
        _cc_lbls = np.array([c['content_label'] for c in _cc])
    else:
        _cc = None

    configs = [
        ('resnet', ResNet50Backbone),
        ('vit',    ViTBackbone),
        ('clip',   CLIPBackbone),
    ]

    interventions = [
        ('grayscale',     to_grayscale,                              False),
        ('hue_rotation',  rotate_hue,                               False),
        ('patch_shuffle', patch_shuffle,                            True),
        ('translation32', lambda img: translate(img, 32, 'right'),  False),
    ]

    for name, BackboneClass in configs:
        print(f"\n--- {name} ---")
        backbone = BackboneClass().to(device)
        norm = normalize_for(name)

        d = np.load(os.path.join(args.cache_dir, name, f'{name}_test.npz'))
        clean_feats, labels = d['feats'], d['labels']

        for label, fn, use_idx in interventions:
            print(f"  {label}")
            trans_feats = get_transformed_feats(fn, backbone, te_ds, test_idx, norm, device,
                                                pass_idx=use_idx)
            out_path = os.path.join(args.figures_dir, f'tsne_{name}_{label}.png')
            plot_tsne(clean_feats, trans_feats, labels,
                      title=f't-SNE: {name} — {label}', out_path=out_path)

        if _cc is not None:
            print("  cue_conflict")
            conf_feats = []
            with torch.no_grad():
                for start in range(0, len(_cc_pils), 64):
                    batch = torch.stack([norm(img) for img in _cc_pils[start:start+64]]).to(device)
                    conf_feats.append(backbone(batch).cpu().numpy())
            conf_feats   = np.concatenate(conf_feats, axis=0)
            clean_paired = np.array([clean_feats[c['content_pos']] for c in _cc])
            out_path = os.path.join(args.figures_dir, f'tsne_{name}_cue_conflict.png')
            plot_tsne(clean_paired, conf_feats, _cc_lbls,
                      title=f't-SNE: {name} — cue_conflict', out_path=out_path)


if __name__ == '__main__':
    main()
