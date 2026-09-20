import os, json, argparse
import numpy as np
import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm

from common.seed import set_seed
from common.metrics import top1_accuracy, macro_f1, mean_max_confidence
from task1.models.backbone import (
    ResNet50Backbone, ViTBackbone, CLIPBackbone, LinearHead,
    RESNET50_DIM, VIT_B16_DIM, CLIP_VIT_B32_DIM,
)
from task1.data.transforms import resize_224, normalize_for

SEED = 6304
STL10_CLASSES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']


class STL10Subset(Dataset):
    def __init__(self, base_dataset, indices, normalize):
        self.base = base_dataset
        self.indices = indices
        self.normalize = normalize  #use only this subset of images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]  # returns pil image + int label
        return self.normalize(resize_224(img)), label  # common 224 canvas then normalize


def extract_features(backbone, base_ds, indices, normalize, device, batch_size=64):
    loader = DataLoader(STL10Subset(base_ds, indices, normalize),
                        batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], [] #creates batches and collects feature vectors and labels
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False):
            feats.append(backbone(imgs.to(device)).cpu())
            labels.append(lbls) #run through the backbone, and then concat batches
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def get_or_extract(name, split, backbone, base_ds, indices, normalize, device, cache_dir):
    path = os.path.join(cache_dir, f"{name}_{split}.npz")
    if os.path.exists(path):  # skip extraction if already cached on drive
        d = np.load(path)
        return d["feats"], d["labels"]
    feats, labels = extract_features(backbone, base_ds, indices, normalize, device)
    os.makedirs(cache_dir, exist_ok=True) #this is for the benefit of colab time optimization
    np.savez_compressed(path, feats=feats, labels=labels)
    print(f"cached {name}/{split} → {path}")
    return feats, labels


def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                      in_dim, num_classes=10, epochs=50, patience=5, ckpt_path=None):
    set_seed(SEED)
    head = LinearHead(in_dim, num_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    #training the linear head now using the hyperparams specified in manual
    X_tr = torch.tensor(train_feats)
    y_tr = torch.tensor(train_labels)
    X_val = torch.tensor(val_feats)
    y_val = torch.tensor(val_labels)

    best_acc, no_improve = 0.0, 0

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 256):
            idx = perm[i:i+256]
            loss = loss_fn(head(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            acc = (head(X_val).argmax(1) == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            if ckpt_path:
                torch.save(head.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop at epoch {epoch+1}, best val acc {best_acc:.4f}")
                break

    if ckpt_path:
        head.load_state_dict(torch.load(ckpt_path))  # load best weights back
    return head


def main():
    # --- block 1: setup ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--cache_dir',   default='./cache/features')
    parser.add_argument('--ckpt_dir',    default='./cache/checkpoints')
    parser.add_argument('--results_dir', default='./results')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    # --- block 2: load split indices ---
    with open('results/task1_splits.json') as f:
        _, train_idx, val_idx, test_idx = json.load(f)
    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)

    # base datasets with no transform — we apply resize+normalize in the subset class
    tr_ds = STL10(root=args.data_dir, split='train', download=False)
    te_ds = STL10(root=args.data_dir, split='test',  download=False)

    # --- block 3: loop over backbones ---
    configs = [
        ('resnet', ResNet50Backbone, RESNET50_DIM),
        ('vit',    ViTBackbone,      VIT_B16_DIM),
        ('clip',   CLIPBackbone,     CLIP_VIT_B32_DIM),
    ]

    results = {}
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    for name, BackboneClass, dim in configs:
        print(f"\n--- {name} ---")
        backbone = BackboneClass().to(device)
        normalize = normalize_for(name)

        cache = os.path.join(args.cache_dir, name)
        ckpt  = os.path.join(args.ckpt_dir,  f'{name}_head.pt')

        tr_f, tr_l = get_or_extract(name, 'train', backbone, tr_ds, train_idx, normalize, device, cache)
        va_f, va_l = get_or_extract(name, 'val',   backbone, tr_ds, val_idx,   normalize, device, cache)
        te_f, te_l = get_or_extract(name, 'test',  backbone, te_ds, test_idx,  normalize, device, cache)

        head = train_linear_head(tr_f, tr_l, va_f, va_l, dim, ckpt_path=ckpt)
        head.eval()
        with torch.no_grad():
            logits = head(torch.tensor(te_f)).numpy()

        results[name] = {
            'acc':        float(top1_accuracy(logits, te_l)),
            'macro_f1':   float(macro_f1(logits, te_l)),
            'confidence': float(mean_max_confidence(logits)),
        }
        print(results[name])

    # --- block 4: clip zero-shot ---
    # need raw similarity scores (not just argmax) to compute confidence
    print("\n--- clip zero-shot ---")
    clip_bb = CLIPBackbone().to(device)
    scale = clip_bb.clip.logit_scale.exp().item()  # clip's learned temperature (~100)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text_tok  = tokenizer([f"a photo of a {c}" for c in STL10_CLASSES]).to(device)
    with torch.no_grad():
        text_f = clip_bb.clip.encode_text(text_tok)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)

    zs_loader = DataLoader(STL10Subset(te_ds, test_idx, normalize_for('clip')),
                           batch_size=64, shuffle=False)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(zs_loader):
            img_f = clip_bb(imgs.to(device))
            all_logits.append((scale * img_f @ text_f.T).cpu())  # scaled similarity to each class
            all_labels.append(lbls)

    zs_logits = torch.cat(all_logits).numpy()
    zs_labels = torch.cat(all_labels).numpy()
    results['clip_zeroshot'] = {
        'acc':        float(top1_accuracy(zs_logits, zs_labels)),
        'macro_f1':   float(macro_f1(zs_logits, zs_labels)),
        'confidence': float(mean_max_confidence(zs_logits)),
    }
    print(results['clip_zeroshot'])

    # --- block 5: save ---
    out = os.path.join(args.results_dir, 'task1_clean_baseline.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
