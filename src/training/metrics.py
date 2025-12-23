# src/training/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from typing import Dict


def compute_classification_metrics(y_true, y_pred, y_prob, num_classes: int) -> Dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Specificity per class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tn = []
    fp = []
    fn = []
    tp = []
    for i in range(num_classes):
        tp_i = cm[i, i]
        fn_i = cm[i, :].sum() - tp_i
        fp_i = cm[:, i].sum() - tp_i
        tn_i = cm.sum() - (tp_i + fn_i + fp_i)
        tp.append(tp_i)
        fn.append(fn_i)
        fp.append(fp_i)
        tn.append(tn_i)

    specificity = [tn[i] / (tn[i] + fp[i] + 1e-8) for i in range(num_classes)]
    metrics["specificity_macro"] = float(np.mean(specificity))

    # ROC-AUC (macro, if probabilities provided)
    if y_prob is not None and y_prob.shape[1] == num_classes:
        try:
            y_true_onehot = np.eye(num_classes)[y_true]
            metrics["roc_auc_macro"] = roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
        except Exception:
            metrics["roc_auc_macro"] = None
    else:
        metrics["roc_auc_macro"] = None

    metrics["confusion_matrix"] = cm
    return metrics
