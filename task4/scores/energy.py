import numpy as np


# u_Energy = -log sum_k exp(z_k)  (computed stably)
def energy_score(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))
    return -lse
