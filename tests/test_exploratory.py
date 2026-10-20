import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rfm_structure.exploratory import (
    clustered_auc_delta,
    grouped_sensitivity_prediction,
    rank_credit,
)


def test_pair_credit_and_bootstrap_point_match_paired_auroc_with_ties():
    labels = np.array([0, 1, 0, 1, 0, 1])
    clean = np.array([0, 1, 1, 1, 0, 2])
    arms = np.array([clean[::-1], clean + np.array([1, 0, 1, 0, 0, 0])])
    assert rank_credit(labels, clean).mean() == pytest.approx(roc_auc_score(labels, clean))
    result = clustered_auc_delta(labels, clean, arms, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], replicates=100)
    expected = np.mean([roc_auc_score(labels, arm) for arm in arms]) - roc_auc_score(labels, clean)
    assert result["point"] == pytest.approx(expected)
    assert result == clustered_auc_delta(labels, clean, arms, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], replicates=100)


def test_paired_identity_has_zero_intervals_under_every_resampling_scheme():
    labels = np.tile([0, 1], 10)
    scores = np.arange(20)
    result = clustered_auc_delta(labels, scores, scores[None, :], np.repeat(np.arange(10), 2),
                                 np.tile([0, 1], 10), replicates=100)
    for name in ("date", "driver", "crossed"):
        assert result[f"{name}_percentile_95"] == [0, 0]


def test_grouped_prediction_is_deterministic_and_generalizes_a_simple_signal():
    x = np.linspace(0, 1, 100)
    sensitivity = np.expm1(x)
    groups = np.repeat(np.arange(10), 10)
    result = grouped_sensitivity_prediction(x[:, None], sensitivity, groups)
    assert result == grouped_sensitivity_prediction(x[:, None], sensitivity, groups)
    assert result["out_of_fold_r2"] > 0.9
    assert result["n_clusters"] == 10
