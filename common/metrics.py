import numpy as np
from sklearn.metrics import f1_score
import torch


def top1_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return (preds == labels).mean()


def macro_f1(logits, labels):
    preds = np.argmax(logits, axis=1)
    return f1_score(labels, preds, average='macro')


def mean_max_confidence(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.max(axis=1).mean()


def prediction_consistency(preds_clean, preds_transformed):
    return (preds_clean == preds_transformed).mean()
