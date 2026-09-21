import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def domain_separability(src_feats, tgt_feats, seed=SEED):
    src_feats = np.asarray(src_feats)
    tgt_feats = np.asarray(tgt_feats)

    # Equal counts: subsample the larger set (usually Sketch) down to the smaller,
    # deterministically with seed 6304.
    rng = np.random.RandomState(seed)
    n = min(len(src_feats), len(tgt_feats))
    src = src_feats[rng.choice(len(src_feats), n, replace=False)]
    tgt = tgt_feats[rng.choice(len(tgt_feats), n, replace=False)]

    X = np.concatenate([src, tgt])
    y = np.concatenate([np.zeros(n), np.ones(n)])   # 0 = source, 1 = target

    # 70/30 split, seed 6304, stratified so both classes stay balanced.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    clf.fit(X_tr, y_tr)
    return {'separability': float(clf.score(X_te, y_te)),  # chance = 0.5
            'n_per_domain': int(n)}
