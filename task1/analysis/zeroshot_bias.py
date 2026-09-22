"""CLIP zero-shot under the same interventions as evaluate_bias.py.

Purpose (Task 1, Research Question 3): compare CLIP's *zero-shot* decisions with
those of its *trained* linear head. evaluate_bias.py already produces the trained
head's behaviour for every intervention; this script produces the zero-shot
counterpart on the exact same images, so the two can be compared directly.

Nothing is trained. We reuse TransformedSubset and the transforms unchanged; the
only new logic is the zero-shot scoring (scaled cosine similarity to text
prompts + softmax), kept identical to run_task1.py's clean zero-shot block.
"""
import os, json, argparse
import numpy as np
import torch
import open_clip
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from PIL import Image
from tqdm import tqdm

from common.seed import set_seed
from common.metrics import top1_accuracy, macro_f1, mean_max_confidence, prediction_consistency
from task1.models.backbone import CLIPBackbone
from task1.data.transforms import (
    normalize_for, to_grayscale, rotate_hue, translate, patch_shuffle,
)
from task1.analysis.evaluate_bias import TransformedSubset   # reuse, do not reimplement

SEED = 6304
STL10_CLASSES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

# clean acc/confidence are recomputed here and asserted against the value stored
# in task1_clean_baseline.json; this tolerance only absorbs cross-run GPU
# floating-point non-determinism. A real pipeline mismatch (wrong prompt,
# normalization, or scale) shifts these by far more than TOL.
TOL = 5e-3


def build_text_features(clip_bb, device):
    # identical prompt + tokenizer + normalization to run_task1.py
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text_tok = tokenizer([f"a photo of a {c}." for c in STL10_CLASSES]).to(device)
    with torch.no_grad():
        text_f = clip_bb.clip.encode_text(text_tok)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
    return text_f


def zeroshot_logits_from_loader(clip_bb, text_f, scale, loader, device):
    logits, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False):
            img_f = clip_bb(imgs.to(device))          # forward() already L2-normalizes
            logits.append((scale * img_f @ text_f.T).cpu())   # same scaled cosine sim as run_task1
            labels.append(lbls)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def eval_zs(transform_fn, clip_bb, text_f, scale, te_ds, test_idx, norm, device, pass_idx=False):
    loader = DataLoader(TransformedSubset(te_ds, test_idx, transform_fn, norm, pass_idx),
                        batch_size=64, shuffle=False, num_workers=2)
    return zeroshot_logits_from_loader(clip_bb, text_f, scale, loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--results_dir', default='./results')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    with open(os.path.join(args.results_dir, 'task1_splits.json')) as f:
        _, _, _, test_idx = json.load(f)
    test_idx = np.array(test_idx)

    te_ds = STL10(root=args.data_dir, split='test', download=False)

    clip_bb = CLIPBackbone().to(device)
    scale   = clip_bb.clip.logit_scale.exp().item()   # CLIP's learned temperature (~100)
    text_f  = build_text_features(clip_bb, device)
    norm    = normalize_for('clip')

    results = {}

    # --- clean (identity transform through the same resize+normalize path) ---
    identity = lambda img: img
    clean_logits, labels = eval_zs(identity, clip_bb, text_f, scale, te_ds, test_idx, norm, device)
    clean_preds = clean_logits.argmax(1)
    results['clean'] = {
        'acc':        float(top1_accuracy(clean_logits, labels)),
        'macro_f1':   float(macro_f1(clean_logits, labels)),
        'confidence': float(mean_max_confidence(clean_logits)),
    }
    print('clean', results['clean'])

    # --- grayscale ---
    gs_logits, _ = eval_zs(to_grayscale, clip_bb, text_f, scale, te_ds, test_idx, norm, device)
    results['grayscale'] = {
        'acc':         float(top1_accuracy(gs_logits, labels)),
        'f1':          float(macro_f1(gs_logits, labels)),
        'consistency': float(prediction_consistency(clean_preds, gs_logits.argmax(1))),
    }

    # --- hue rotation ---
    hr_logits, _ = eval_zs(rotate_hue, clip_bb, text_f, scale, te_ds, test_idx, norm, device)
    results['hue_rotation'] = {
        'acc':         float(top1_accuracy(hr_logits, labels)),
        'f1':          float(macro_f1(hr_logits, labels)),
        'consistency': float(prediction_consistency(clean_preds, hr_logits.argmax(1))),
    }

    # --- translation: average over 4 directions at each delta ---
    results['translation'] = {}
    for delta in [0, 8, 16, 32]:
        accs, conss = [], []
        for direction in ['right', 'left', 'up', 'down']:
            fn = lambda img, d=delta, dr=direction: translate(img, d, dr)
            t_logits, _ = eval_zs(fn, clip_bb, text_f, scale, te_ds, test_idx, norm, device)
            accs.append(top1_accuracy(t_logits, labels))
            conss.append(prediction_consistency(clean_preds, t_logits.argmax(1)))
        results['translation'][delta] = {
            'acc':         float(np.mean(accs)),
            'consistency': float(np.mean(conss)),
        }

    # --- patch shuffle: per-image permutation keyed by subset position ---
    ps_logits, _ = eval_zs(patch_shuffle, clip_bb, text_f, scale, te_ds, test_idx, norm, device,
                           pass_idx=True)
    results['patch_shuffle'] = {
        'acc':         float(top1_accuracy(ps_logits, labels)),
        'f1':          float(macro_f1(ps_logits, labels)),
        'consistency': float(prediction_consistency(clean_preds, ps_logits.argmax(1))),
    }

    # --- cue conflict: shape bias + coverage + per-image categories ---
    cue_examples = {}
    _cc_path = os.path.join(args.results_dir, 'task1_cue_conflicts_metadata.json')
    if os.path.exists(_cc_path):
        with open(_cc_path) as f:
            _cc = json.load(f)['conflicts']
        _cc_dir  = os.path.join(args.results_dir, 'cue_conflicts')
        _cc_pils = [Image.open(os.path.join(_cc_dir, c['filename'])).convert('RGB') for c in _cc]
        _cc_c    = np.array([c['content_label'] for c in _cc])
        _cc_s    = np.array([c['style_label']   for c in _cc])
        print(f"loaded {len(_cc)} cue-conflict images")

        preds = []
        with torch.no_grad():
            for start in range(0, len(_cc_pils), 64):
                batch = torch.stack([norm(img) for img in _cc_pils[start:start+64]]).to(device)
                img_f = clip_bb(batch)
                preds.append((scale * img_f @ text_f.T).argmax(1).cpu().numpy())
        preds     = np.concatenate(preds)
        n_shape   = int((preds == _cc_c).sum())
        n_texture = int((preds == _cc_s).sum())
        n_total   = len(preds)
        denom     = n_shape + n_texture
        results['cue_conflict'] = {
            'n_shape':    n_shape,
            'n_texture':  n_texture,
            'n_other':    n_total - denom,
            'shape_bias': round(100.0 * n_shape / denom, 2) if denom > 0 else None,
            'coverage':   round(100.0 * denom / n_total,  2),
        }
        print('  cue_conflict', results['cue_conflict'])

        cue_examples['clip_zeroshot'] = [{
            'filename':      _cc[k]['filename'],
            'content_label': int(_cc_c[k]),
            'style_label':   int(_cc_s[k]),
            'pred':          int(preds[k]),
            'category':      'shape'   if preds[k] == _cc_c[k]
                             else 'texture' if preds[k] == _cc_s[k]
                             else 'other',
        } for k in range(n_total)]
    else:
        print("task1_cue_conflicts_metadata.json not found — skipping cue conflict")

    # --- assert clean parity with the stored baseline (read, never hard-code) ---
    with open(os.path.join(args.results_dir, 'task1_clean_baseline.json')) as f:
        base_zs = json.load(f)['clip_zeroshot']
    assert np.isclose(results['clean']['acc'], base_zs['acc'], atol=TOL), (
        f"clean zero-shot acc {results['clean']['acc']} != baseline {base_zs['acc']}")
    assert np.isclose(results['clean']['confidence'], base_zs['confidence'], atol=TOL), (
        f"clean zero-shot confidence {results['clean']['confidence']} != baseline {base_zs['confidence']}")
    print(f"clean parity ok (acc/confidence match baseline within {TOL})")

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, 'task1_clip_zeroshot.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out}")

    if cue_examples:
        cc_out = os.path.join(args.results_dir, 'task1_clip_zeroshot_cue_examples.json')
        with open(cc_out, 'w') as f:
            json.dump(cue_examples, f, indent=2)
        print(f"saved → {cc_out}")


if __name__ == '__main__':
    main()
