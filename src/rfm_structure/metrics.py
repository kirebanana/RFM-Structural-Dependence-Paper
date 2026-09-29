"""AUROC helper for binary prediction scores."""

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Score raw logits (or probabilities), treating labels > 0 as positive.

    Validates equal shapes and finite scores. AUROC depends on score order, so
    applying sigmoid to logits is unnecessary.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape:
        raise ValueError(f"label/score shape mismatch: {labels.shape} vs {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    return float(roc_auc_score((labels > 0).astype(int), scores))
