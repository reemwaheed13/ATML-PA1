import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


# 3-way source separability (distinct from Task 2's binary source-vs-target probe):
# how linearly separable the three SOURCE domains are in feature space. Chance = 1/3.
# Balanced by subsampling every domain to the smallest source-val count, so class_weight
# is redundant but kept for symmetry with the Task 2 probe. Multinomial LogReg C=1 is the
# lbfgs default for >2 classes, so multi_class is left implicit to stay forward-compatible.
def source_domain_separability(feats_by_domain, seed=SEED):
    domains = list(feats_by_domain.keys())
    rng = np.random.RandomState(seed)
    n = min(len(feats_by_domain[d]) for d in domains)

    X_parts, y_parts = [], []
    for label, d in enumerate(domains):
        f = np.asarray(feats_by_domain[d])
        idx = rng.choice(len(f), n, replace=False)
        X_parts.append(f[idx])
        y_parts.append(np.full(n, label))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    clf.fit(X_tr, y_tr)
    return {
        'separability': float(clf.score(X_te, y_te)),
        'n_per_domain': int(n),
        'chance': 1.0 / len(domains),
        'domains': domains,
    }
