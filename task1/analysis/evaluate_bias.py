import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from PIL import Image
from tqdm import tqdm

from common.seed import set_seed
from common.metrics import top1_accuracy, macro_f1, prediction_consistency
from task1.models.backbone import (
    ResNet50Backbone, ViTBackbone, CLIPBackbone, LinearHead,
    RESNET50_DIM, VIT_B16_DIM, CLIP_VIT_B32_DIM,
)
from task1.data.transforms import (
    resize_224, normalize_for, to_grayscale, rotate_hue, translate, patch_shuffle,
)

SEED = 6304

#basically we apply the transformation on the common 224 canvas, normalize and evaluate
class TransformedSubset(Dataset):
    def __init__(self, base_ds, indices, transform_fn, normalize, pass_idx=False):
        self.base = base_ds
        self.indices = indices
        self.transform_fn = transform_fn  # pil → pil (the intervention, on 224)
        self.normalize = normalize        # pil → tensor (backbone normalization)
        self.pass_idx = pass_idx          # True only for patch_shuffle (per-image perm)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        img = resize_224(img)
        img = self.transform_fn(img, i) if self.pass_idx else self.transform_fn(img)
        return self.normalize(img), label


def eval_transform(transform_fn, backbone, head, te_ds, test_idx, normalize, device,
                   pass_idx=False):
    loader = DataLoader(TransformedSubset(te_ds, test_idx, transform_fn, normalize, pass_idx),
                        batch_size=64, shuffle=False, num_workers=2)
    logits, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False):
            feats = backbone(imgs.to(device)).cpu()
            logits.append(head(feats))
            labels.append(lbls)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--cache_dir',   default='./cache/features')
    parser.add_argument('--ckpt_dir',    default='./cache/checkpoints')
    parser.add_argument('--results_dir', default='./results')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('results/task1_splits.json') as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds = STL10(root=args.data_dir, split='test', download=False)

    # pre-load conflict images once; per-backbone norm applied inside the loop
    _cc_path = os.path.join(args.results_dir, 'cue_conflicts_metadata.json')
    if os.path.exists(_cc_path):
        with open(_cc_path) as f:
            _cc = json.load(f)['conflicts']
        _cc_dir  = os.path.join(args.results_dir, 'cue_conflicts')
        _cc_pils = [Image.open(os.path.join(_cc_dir, c['filename'])).convert('RGB') for c in _cc]
        _cc_c    = np.array([c['content_label'] for c in _cc])
        _cc_s    = np.array([c['style_label']   for c in _cc])
        print(f"loaded {len(_cc)} cue-conflict images")
    else:
        _cc = None
        print("cue_conflicts_metadata.json not found — run make_cue_conflicts.py first")

    configs = [
        ('resnet', ResNet50Backbone, RESNET50_DIM),
        ('vit',    ViTBackbone,      VIT_B16_DIM),
        ('clip',   CLIPBackbone,     CLIP_VIT_B32_DIM),
    ]

    results = {}

    for name, BackboneClass, dim in configs:
        print(f"\n--- {name} ---")
        backbone = BackboneClass().to(device)
        norm = normalize_for(name)

        head = LinearHead(dim)
        head.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'{name}_head.pt'),
                                        map_location='cpu'))
        head.eval()

        # load clean test features to get clean predictions without re-running backbone
        d = np.load(os.path.join(args.cache_dir, name, f'{name}_test.npz'))
        clean_logits = head(torch.tensor(d['feats'])).detach().numpy()
        clean_preds  = clean_logits.argmax(1)
        labels       = d['labels']

        r = {}

        # grayscale
        gs_logits, _ = eval_transform(to_grayscale, backbone, head, te_ds, test_idx, norm, device)
        r['grayscale'] = {
            'acc':         float(top1_accuracy(gs_logits, labels)),
            'f1':          float(macro_f1(gs_logits, labels)),
            'consistency': float(prediction_consistency(clean_preds, gs_logits.argmax(1))),
        }

        # hue rotation (our chosen second color intervention)
        hr_logits, _ = eval_transform(rotate_hue, backbone, head, te_ds, test_idx, norm, device)
        r['hue_rotation'] = {
            'acc':         float(top1_accuracy(hr_logits, labels)),
            'f1':          float(macro_f1(hr_logits, labels)),
            'consistency': float(prediction_consistency(clean_preds, hr_logits.argmax(1))),
        }

        # translation: average across 4 directions at each delta
        r['translation'] = {}
        for delta in [0, 8, 16, 32]:
            accs, conss = [], []
            for direction in ['right', 'left', 'up', 'down']:
                fn = lambda img, d=delta, dr=direction: translate(img, d, dr)
                t_logits, _ = eval_transform(fn, backbone, head, te_ds, test_idx, norm, device)
                accs.append(top1_accuracy(t_logits, labels))
                conss.append(prediction_consistency(clean_preds, t_logits.argmax(1)))
            r['translation'][delta] = {
                'acc':         float(np.mean(accs)),
                'consistency': float(np.mean(conss)),
            }

        # patch shuffle — per-image permutation seeded by position in test subset
        ps_logits, _ = eval_transform(patch_shuffle, backbone, head, te_ds, test_idx, norm, device,
                                      pass_idx=True)
        r['patch_shuffle'] = {
            'acc':         float(top1_accuracy(ps_logits, labels)),
            'f1':          float(macro_f1(ps_logits, labels)),
            'consistency': float(prediction_consistency(clean_preds, ps_logits.argmax(1))),
        }

        # cue conflict — shape bias and coverage
        if _cc is not None:
            preds = []
            with torch.no_grad():
                for start in range(0, len(_cc_pils), 64):
                    batch = torch.stack([norm(img) for img in _cc_pils[start:start+64]]).to(device)
                    preds.append(head(backbone(batch).cpu()).argmax(1).numpy())
            preds     = np.concatenate(preds)
            n_shape   = int((preds == _cc_c).sum())
            n_texture = int((preds == _cc_s).sum())
            n_total   = len(preds)
            denom     = n_shape + n_texture
            r['cue_conflict'] = {
                'n_shape':    n_shape,
                'n_texture':  n_texture,
                'n_other':    n_total - denom,
                'shape_bias': round(100.0 * n_shape / denom, 2) if denom > 0 else None,
                'coverage':   round(100.0 * denom / n_total,  2),
            }
            print('  cue_conflict', r['cue_conflict'])

        results[name] = r
        print(r)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, 'task1_interventions.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
