import numpy as np


# u_MSP = 1 - max_k softmax(z)_k
def msp_score(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return 1.0 - p.max(axis=1)
