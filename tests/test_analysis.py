import json
import shutil

import numpy as np
import pytest
from experiment_fixtures import (
    FakeModel,
    evaluator,
    runner_functions,
    write_metadata,
)

from rfm_structure.analysis import (
    analyze_matrix,
    assert_prediction_alignment,
    paired_statistics,
    realized_corruption,
    summary_markdown,
)


def test_paired_raw_statistics_have_known_values_and_undefined_correlation():
    result = paired_statistics(np.arange(4), np.arange(4) * 2)
    assert result["mean_absolute_raw_shift"] == 1.5
    assert result["median_absolute_raw_shift"] == 1.5
    assert result["pearson"] == pytest.approx(1)
    assert paired_statistics(np.ones(4), np.arange(4))["pearson_status"] == "undefined_zero_variance"
    with pytest.raises(ValueError, match="finite"):
        paired_statistics([0, 1], [0, np.nan])


@pytest.mark.parametrize("field", ["labels", "node_idxs"])
def test_misaligned_targets_or_labels_fail_before_pairing(field):
    base = {"labels": np.array([0, 1]), "node_idxs": np.array([5, 6]), "preds": np.array([0.1, 0.8])}
    arm = {key: array.copy() for key, array in base.items()}
    arm[field] = arm[field][::-1]
    with pytest.raises(ValueError, match="misaligned"):
        assert_prediction_alignment(base, arm, count=2)


def test_query_mean_global_fractions_and_zero_denominators_are_distinct():
    result = realized_corruption({
        "changed_edges": np.array([1, 0, 0]), "eligible_edges": np.array([1, 9, 0]),
        "max_changed_edges": np.array([1, 3, 0]),
    })
    assert result["mean_changed_eligible"] == 0.5
    assert result["global_changed_eligible"] == 0.1
    assert result["global_changed_max"] == 0.25
    assert result["zero_eligible_queries"] == result["zero_max_queries"] == 1
    result = realized_corruption({"changed_edges": np.array([0]), "eligible_edges": np.array([0]), "max_changed_edges": np.array([0])})
    assert result["global_changed_max"] is None
    assert result["mean_changed_eligible"] is None


@pytest.fixture(scope="module")
def saved_matrix(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("synthetic_analysis")
    write_metadata(temporary)
    runner = runner_functions()
    root = temporary / "matrix"
    runner["execute_matrix"](
        {48: evaluator(48), 128: evaluator(128)}, temporary,
        [101, 202, 303], [0, 0.5, 1], root, dry_only=False, model_factory=FakeModel,
    )
    return root


def test_completed_saved_arrays_produce_14_rows_six_summaries_without_rt(saved_matrix):
    with pytest.raises(ValueError, match="synthetic test"):
        analyze_matrix(saved_matrix)
    result = analyze_matrix(saved_matrix, allow_test_artifacts=True)
    assert len(result["rows"]) == 14
    assert len(result["summaries"]) == 6
    for row in result["rows"]:
        assert row["delta_clean"] == 0
        assert row["mean_absolute_raw_shift"] == 0
        assert row["global_changed_max"] is not None
        assert row["timestamp_unknowns"] == 0
    for summary in result["summaries"]:
        assert summary["auroc_min"] == summary["auroc_max"]
    assert "synthetic_test" in summary_markdown(result)
    assert "undefined" in summary_markdown(result)
    json.dumps(result, allow_nan=False)


def test_incomplete_matrix_is_rejected(tmp_path):
    (tmp_path / "matrix.json").write_text(json.dumps({"status": "failed", "complete": False}))
    with pytest.raises(ValueError, match="complete two-context"):
        analyze_matrix(tmp_path)


def test_saved_prediction_tampering_is_rejected(saved_matrix, tmp_path):
    root = tmp_path / "matrix"
    shutil.copytree(saved_matrix, root)
    path = root / "ctx48_lctx24/preds_a050_s101.npz"
    with np.load(path) as handle:
        arrays = {name: handle[name] for name in handle.files}
    arrays["preds"][0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        analyze_matrix(root, allow_test_artifacts=True)
