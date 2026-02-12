import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_prob, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    if num_classes > 2:
        metrics["auroc"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    else:
        metrics["auroc"] = roc_auc_score(y_true, y_prob[:, 1])

    return metrics
