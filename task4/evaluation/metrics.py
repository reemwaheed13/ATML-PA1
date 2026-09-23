import numpy as np
from sklearn.metrics import roc_auc_score


# known = in-distribution (label 0), unknown = novel (label 1), score = unknownness u
def auroc(u_known, u_unknown):
    y = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    s = np.concatenate([u_known, u_unknown])
    return float(roc_auc_score(y, s))
