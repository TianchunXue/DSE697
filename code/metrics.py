# metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

def evaluate_binary(y_pred, y_true, y_logits=None):
    # y_pred, y_true: numpy arrays of 0/1
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_logits) if y_logits is not None else 0.0
    aupr = average_precision_score(y_true, y_logits) if y_logits is not None else 0.0
    return acc, pre, rec, f1, auc, aupr
