"""Metrics used by the experiment harness."""

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC from binary labels and raw model scores."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape:
        raise ValueError(f"label/score shape mismatch: {labels.shape} vs {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    return float(roc_auc_score((labels > 0).astype(int), scores))
