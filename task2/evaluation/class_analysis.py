import numpy as np


def per_class_accuracy(preds, labels, num_classes):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    accs = []
    for c in range(num_classes):
        mask = labels == c
        accs.append(float((preds[mask] == c).mean()) if mask.any() else float('nan'))
    return accs


def confusion_matrix(preds, labels, num_classes):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def top_confusions_for_class(cm, classes, true_class, k=2):
    row = cm[true_class]
    order = np.argsort(row)[::-1]
    out = []
    for j in order:
        if j != true_class and row[j] > 0:
            out.append({'predicted': classes[j], 'count': int(row[j])})
        if len(out) == k:
            break
    return out


def largest_gain_and_drop(delta, classes):
    delta = np.asarray(delta, dtype=float)
    valid = ~np.isnan(delta)
    idx = np.where(valid)[0]
    gain = int(idx[np.argmax(delta[idx])])
    drop = int(idx[np.argmin(delta[idx])])
    return {
        'largest_gain': {'class': classes[gain], 'delta': float(delta[gain])},
        'largest_drop': {'class': classes[drop], 'delta': float(delta[drop])},
    }
