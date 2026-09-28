"""Small, validation-first metrics used by structural experiments."""

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUROC from labels and finite raw model scores.

    Labels greater than zero are treated as the positive class. ``scores`` may
    be logits because AUROC depends on ranking, so applying sigmoid would not
    change the result. Raises ``ValueError`` for mismatched shapes or non-finite
    scores; scikit-learn raises if the labels do not contain both classes.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape:
        raise ValueError(f"label/score shape mismatch: {labels.shape} vs {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    return float(roc_auc_score((labels > 0).astype(int), scores))
