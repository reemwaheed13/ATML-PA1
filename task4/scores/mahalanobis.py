import numpy as np


# fit class means mu_c and ONE shared diagonal covariance from unaugmented
# CIFAR-10 training features; add 1e-6 to every diagonal entry
def fit_mahalanobis(train_feat, train_labels, num_classes=10, eps=1e-6):
    train_labels = np.asarray(train_labels)
    means = np.stack([train_feat[train_labels == c].mean(axis=0)
                      for c in range(num_classes)])
    centered = train_feat - means[train_labels]
    var = centered.var(axis=0) + eps
    return means, var


# u_Mah = min_c (f - mu_c)^T Sigma^-1 (f - mu_c), Sigma diagonal -> elementwise / var
def mahalanobis_score(feat, means, var):
    inv = 1.0 / var
    N, C = feat.shape[0], means.shape[0]
    d2 = np.empty((N, C), dtype=np.float64)
    for c in range(C):
        diff = feat - means[c]
        d2[:, c] = (diff * diff * inv).sum(axis=1)
    return d2.min(axis=1)
