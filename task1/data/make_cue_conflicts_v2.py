"""Cue-conflict generator with control over how the STYLE image is fed to the stylizer.

Every helper (VGG features, AdaIN target, Gram/TV losses, optimizer, rejection rule) is reused
UNCHANGED from make_cue_conflicts.py. Differences from that script's main():
  * --style_mode: how the style photo is presented
      up     : bilinear resize to 224x224 (identical to make_cue_conflicts.py)
      native : the 96x96 photo as is (no upsampling, so its fine texture is not blurred)
      center : central 64x64 crop at native resolution (mostly the object, less background)
  * --out_dir and --meta_path are honoured, and the run settings are stored in the metadata.
Content/style image selection makes the same RNG calls as make_cue_conflicts.py, so for the same
--n_per_dir the same image pairs are used.
"""
import os, sys, json, argparse
import numpy as np
import torch
from torchvision.datasets import STL10
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cue_conflicts as base          # reused unchanged
from common.seed import set_seed


def style_input(pil96, mode):
    if mode == 'up':
        return pil96.resize((224, 224), Image.BILINEAR)
    if mode == 'native':
        return pil96
    if mode == 'center':
        return pil96.crop((16, 16, 80, 80))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',    default='./data')
    ap.add_argument('--results_dir', default='./results')
    ap.add_argument('--out_dir',     default='results/cue_conflicts')
    ap.add_argument('--meta_path',   default='results/cue_conflicts_metadata.json')
    ap.add_argument('--n_per_dir',   type=int, default=base.N_PER_DIR)
    ap.add_argument('--steps',       type=int, default=base.OPT_STEPS)
    ap.add_argument('--style_scale', type=float, required=True)
    ap.add_argument('--style_mode',  choices=['up', 'native', 'center'], required=True)
    a = ap.parse_args()

    set_seed(base.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} | {a.n_per_dir} attempts/dir | {a.steps} steps | "
          f"style_scale {a.style_scale} | style_mode {a.style_mode}")

    with open(os.path.join(a.results_dir, 'task1_splits.json')) as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds  = STL10(root=a.data_dir, split='test', download=False)
    labels = np.array(te_ds.labels)

    class_to_pos = {}
    for pos, gidx in enumerate(test_idx):
        class_to_pos.setdefault(int(labels[gidx]), []).append((pos, int(gidx)))

    vgg = base.VGGFeatures().to(device).eval()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.meta_path) or '.', exist_ok=True)
    rng = np.random.default_rng(base.SEED)

    metadata, n_attempted, n_rejected = [], 0, 0
    for cls_a, cls_b in base.CLASS_PAIRS:
        for content_cls, style_cls in [(cls_a, cls_b), (cls_b, cls_a)]:
            c_pool, s_pool = class_to_pos[content_cls], class_to_pos[style_cls]
            n     = min(a.n_per_dir, len(c_pool), len(s_pool))
            c_sel = rng.choice(len(c_pool), size=n, replace=False)
            s_sel = rng.choice(len(s_pool), size=n, replace=False)
            tag   = f"{base.STL10_CLASSES[content_cls]}_shape_{base.STL10_CLASSES[style_cls]}_tex"
            print(f"\n{tag}")

            accepted = 0
            for i in tqdm(range(n)):
                c_pos, c_gidx = c_pool[c_sel[i]]
                s_pos, s_gidx = s_pool[s_sel[i]]

                c_pil = te_ds[c_gidx][0].resize((224, 224), Image.BILINEAR)
                s_pil = style_input(te_ds[s_gidx][0], a.style_mode)

                out_t   = base.stylize(base._to_normed(c_pil, device), base._to_normed(s_pil, device),
                                       vgg, device, a.steps, a.style_scale)
                out_pil = base._to_pil(out_t)
                arr     = np.array(out_pil)

                n_attempted += 1
                if base._reject(arr):
                    n_rejected += 1
                    continue

                fname = f"{tag}_{i:03d}.png"
                out_pil.save(os.path.join(a.out_dir, fname))
                metadata.append({
                    'filename':      fname,
                    'content_label': int(content_cls),
                    'style_label':   int(style_cls),
                    'content_pos':   int(c_pos),
                    'style_pos':     int(s_pos),
                })
                accepted += 1
            print(f"  {accepted}/{n} accepted")

    print(f"\nattempted {n_attempted} | rejected {n_rejected} | saved {len(metadata)}")
    if len(metadata) < 200:
        print(f"WARNING: only {len(metadata)} saved (need >= 200 for the full run)")

    with open(a.meta_path, 'w') as f:
        json.dump({
            'rejection_rule': 'pixel_std<8 or mean<15 or mean>240',
            'attempted': n_attempted,
            'rejected':  n_rejected,
            'saved':     len(metadata),
            'settings': {
                'script': 'make_cue_conflicts_v2.py', 'steps': a.steps,
                'style_scale': a.style_scale, 'style_mode': a.style_mode,
                'n_per_dir': a.n_per_dir, 'seed': base.SEED, 'lr': base.LR,
                'weights': {'content': 1.0, 'style': base.STYLE_W * a.style_scale, 'tv': base.TV_W},
                'class_pairs': [[base.STL10_CLASSES[x], base.STL10_CLASSES[y]] for x, y in base.CLASS_PAIRS],
            },
            'conflicts': metadata,
        }, f, indent=2)
    print(f"metadata -> {a.meta_path}")


if __name__ == '__main__':
    main()
