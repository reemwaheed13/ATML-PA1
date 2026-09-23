import numpy as np

from task4.data.cifar10 import CIFAR10_CLASSES


# under the given tau, the most confidently ACCEPTED unknowns (lowest u = most known-like)
def confident_failures(cache, split, u, tau, k=3):
    s = cache['splits'][split]
    accepted = np.where(u <= tau)[0]
    order = accepted[np.argsort(u[accepted])]  # most known-like first
    picks = order[:k]
    recs = []
    for i in picks:
        pred = int(np.argmax(s['logits'][i]))
        recs.append({
            'split': split,
            'index': int(i),
            'unknown_class': s['class_names'][i],
            'predicted_known': CIFAR10_CLASSES[pred],
            'u': float(u[i]),
            'tau': float(tau),
        })
    return recs, picks


def plot_failures(cache, near_recs, near_idx, far_recs, far_idx, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [('near', near_recs, near_idx), ('far', far_recs, far_idx)]
    k = max(len(near_idx), len(far_idx), 1)
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 4.6), squeeze=False)
    for r, (split, recs, idx) in enumerate(rows):
        imgs = cache['splits'][split]['images']
        for c in range(k):
            ax = axes[r][c]
            ax.axis('off')
            if c < len(idx):
                ax.imshow(imgs[idx[c]])
                rec = recs[c]
                ax.set_title(f"{rec['unknown_class']}\n->{rec['predicted_known']}",
                             fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
