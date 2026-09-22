import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision.datasets import STL10
from PIL import Image
from tqdm import tqdm

from common.seed import set_seed

SEED = 6304

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

CLASS_PAIRS = [
    (0, 1),   # airplane ↔ bird
    (2, 9),   # car ↔ truck
    (3, 5),   # cat ↔ dog
    (4, 6),   # deer ↔ horse
    (7, 8),   # monkey ↔ ship
]

N_PER_DIR = 25
OPT_STEPS = 300
LR        = 0.05
STYLE_W   = 1e4
TV_W      = 1e-5

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_normed(pil_img, device):
    t = TF.to_tensor(pil_img)
    return ((t - _MEAN) / _STD).to(device)


def _to_pil(t):
    return TF.to_pil_image((t.cpu() * _STD + _MEAN).clamp(0, 1))


class VGGFeatures(nn.Module):
    _ENDS = [2, 7, 12, 21]   # relu1_1, relu2_1, relu3_1, relu4_1

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        layers = list(vgg.children())
        prev, slices = 0, []
        for end in self._ENDS:
            slices.append(nn.Sequential(*layers[prev:end]))
            prev = end
        self.slices = nn.ModuleList(slices)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


def _adain(cf, sf):
    c_mean, c_std = cf.mean(dim=[2,3], keepdim=True), cf.std(dim=[2,3], keepdim=True) + 1e-5
    s_mean, s_std = sf.mean(dim=[2,3], keepdim=True), sf.std(dim=[2,3], keepdim=True) + 1e-5
    return s_std * (cf - c_mean) / c_std + s_mean


def _gram(f):
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def _tv(x):
    return ((x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean() +
            (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean())


def stylize(content_t, style_t, vgg, device, n_steps, style_scale=1.0):
    mean_d, std_d = _MEAN.to(device), _STD.to(device)
    with torch.no_grad():
        c_feats = vgg(content_t.unsqueeze(0))
        s_feats = vgg(style_t.unsqueeze(0))

    # AdaIN at relu4_1 preserves semantic shape; Gram at relu1-3 transfers texture
    target  = _adain(c_feats[3], s_feats[3]).detach()
    s_grams = [_gram(s_feats[i]).detach() for i in range(3)]

    x = content_t.unsqueeze(0).clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)

    for _ in range(n_steps):
        opt.zero_grad()
        xf = vgg(x)
        loss = (F.mse_loss(xf[3], target) +
                style_scale * STYLE_W * sum(F.mse_loss(_gram(xf[i]), s_grams[i]) for i in range(3)) +
                TV_W * _tv(x))
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.data = ((x.data * std_d + mean_d).clamp_(0, 1) - mean_d) / std_d

    return x.detach().squeeze(0)


def _reject(arr):
    std, mean = float(arr.std()), float(arr.mean())
    return std < 8.0 or mean < 15 or mean > 240


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')


    parser.add_argument('--results_dir', default='./results/task1')
    parser.add_argument('--out_dir',     default='results/task1/cue_conflicts')
    parser.add_argument('--meta_path',   default='results/task1/task1_cue_conflicts_metadata.json')
    parser.add_argument('--n_per_dir',   type=int,   default=N_PER_DIR)
    parser.add_argument('--steps',       type=int,   default=OPT_STEPS)
    parser.add_argument('--style_scale', type=float, default=1.0)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}  |  {args.n_per_dir} attempts/dir  |  {args.steps} steps/image  |  style_scale {args.style_scale}")

    with open(os.path.join(args.results_dir, 'task1_splits.json')) as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds  = STL10(root=args.data_dir, split='test', download=False)
    labels = np.array(te_ds.labels)

    class_to_pos = {}
    for pos, gidx in enumerate(test_idx):
        lbl = int(labels[gidx])
        class_to_pos.setdefault(lbl, []).append((pos, int(gidx)))

    vgg = VGGFeatures().to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_path) or '.', exist_ok=True)
    rng = np.random.default_rng(SEED)

    metadata, n_attempted, n_rejected = [], 0, 0

    for cls_a, cls_b in CLASS_PAIRS:
        for content_cls, style_cls in [(cls_a, cls_b), (cls_b, cls_a)]:
            c_pool, s_pool = class_to_pos[content_cls], class_to_pos[style_cls]
            n      = min(args.n_per_dir, len(c_pool), len(s_pool))
            c_sel  = rng.choice(len(c_pool), size=n, replace=False)
            s_sel  = rng.choice(len(s_pool), size=n, replace=False)
            tag    = f"{STL10_CLASSES[content_cls]}_shape_{STL10_CLASSES[style_cls]}_tex"
            print(f"\n{tag}")

            accepted = 0
            for i in tqdm(range(n)):
                c_pos, c_gidx = c_pool[c_sel[i]]
                _,     s_gidx = s_pool[s_sel[i]]

                # te_ds[gidx] returns a correct RGB PIL (STL10.data is channels-first,
                # so indexing the dataset avoids a manual transpose); matches run_task1.py
                c_pil = te_ds[c_gidx][0].resize((224, 224), Image.BILINEAR)
                s_pil = te_ds[s_gidx][0].resize((224, 224), Image.BILINEAR)

                out_t   = stylize(_to_normed(c_pil, device), _to_normed(s_pil, device),
                                  vgg, device, args.steps, args.style_scale)
                out_pil = _to_pil(out_t)
                arr     = np.array(out_pil)

                n_attempted += 1
                if _reject(arr):
                    n_rejected += 1
                    continue

                fname = f"{tag}_{i:03d}.png"
                out_pil.save(os.path.join(args.out_dir, fname))
                metadata.append({
                    'filename':      fname,
                    'content_label': int(content_cls),
                    'style_label':   int(style_cls),
                    'content_pos':   int(c_pos),
                })
                accepted += 1

            print(f"  {accepted}/{n} accepted")

    print(f"\nattempted {n_attempted} | rejected {n_rejected} | saved {len(metadata)}")
    if len(metadata) < 200:
        print(f"WARNING: only {len(metadata)} saved — increase --n_per_dir")

    meta_path = os.path.join(args.results_dir, 'task1_cue_conflicts_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'rejection_rule': 'pixel_std<8 or mean<15 or mean>240',
            'attempted': n_attempted,
            'rejected':  n_rejected,
            'saved':     len(metadata),
            'conflicts': metadata,
        }, f, indent=2)
    print(f"metadata → {meta_path}")


if __name__ == '__main__':
    main()
