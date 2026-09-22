import numpy as np
import torch

from common.metrics import macro_f1, top1_accuracy


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        f = model.backbone(x.to(device))
        feats.append(f.cpu().numpy())
        logits.append(model.head(f).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


# One pass over every source-val loader. Returns the per_domain metric dict AND the
# raw features per domain (reused for source separability) so features are extracted once.
def per_domain_metrics(model, val_loaders, device):
    per_domain, feats = {}, {}
    for d, loader in val_loaders.items():
        f, logits, labels = extract(model, loader, device)
        per_domain[d] = {'acc': float(top1_accuracy(logits, labels)),
                         'f1': float(macro_f1(logits, labels))}
        feats[d] = f
    return per_domain, feats


# Mean- and worst-source are BOTH derived from the same per_domain dict at eval time.
# worst-by-acc and worst-by-f1 can be different domains, so each is reported with its
# own argmin domain rather than assuming one worst domain.
def summarize(per_domain, domains):
    accs = {d: per_domain[d]['acc'] for d in domains}
    f1s = {d: per_domain[d]['f1'] for d in domains}
    worst_acc_domain = min(accs, key=accs.get)
    worst_f1_domain = min(f1s, key=f1s.get)
    return {
        'mean_acc': float(np.mean(list(accs.values()))),
        'mean_f1': float(np.mean(list(f1s.values()))),
        'worst_acc': accs[worst_acc_domain],
        'worst_acc_domain': worst_acc_domain,
        'worst_f1': f1s[worst_f1_domain],
        'worst_f1_domain': worst_f1_domain,
    }
