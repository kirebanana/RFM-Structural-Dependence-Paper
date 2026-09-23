import numpy as np

from rfm_structure.metrics import binary_auroc


def test_binary_auroc_accepts_raw_logits():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([-2.0, 3.0, -1.0, 4.0])
    assert binary_auroc(labels, scores) == 1.0
