import numpy as np


# u_MLS = -max_k z_k
def mls_score(logits):
    return -logits.max(axis=1)
