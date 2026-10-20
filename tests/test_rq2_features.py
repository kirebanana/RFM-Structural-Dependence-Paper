"""RQ2 input categories must not relabel assignment outcomes as context."""
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

ANALYZER = runpy.run_path(str(Path(__file__).parents[1] / "experiments/analyze_relation_controls.py"))


@pytest.fixture
def schema():
    return ANALYZER["rq2_feature_schema"](["database_all", "task_driver", "task_query_cycles"])


@pytest.mark.parametrize("name", ["plus_context_only_exposure", "context_only_exposure_and_support"])
def test_context_only_specs_exclude_assignment_budget_clean_score_and_diagnostics(schema, name):
    spec = schema["specifications"][name]
    excluded = {"targeted_maximum", "max_changed_edges", "changed", "feat_absolute_delta",
                "nbr_absolute_delta", "abs_clean_score"}
    assert not excluded.intersection(spec["feature_names"])
    assert "assignment_informed_budget" not in spec["feature_categories"]
    assert "post_intervention_diagnostics" not in spec["feature_categories"]
    assert "targeted_eligible" in spec["feature_names"]
    assert set(schema["feature_categories"]["context_only_exposure"]) <= set(spec["feature_names"])

    # Altering every assignment/response-derived column cannot alter this fit's inputs.
    features = np.arange(5 * len(schema["feature_names"]), dtype=float).reshape(5, -1)
    changed = features.copy()
    for feature in excluded.intersection(schema["feature_names"]):
        changed[:, schema["feature_names"].index(feature)] = -1000
    np.testing.assert_array_equal(features[:, spec["feature_indices"]], changed[:, spec["feature_indices"]])


def test_context_and_assignment_comparisons_differ_only_by_assignment_budget(schema):
    specs = schema["specifications"]
    for context_name, assignment_name in (
        ("plus_context_only_exposure", "plus_assignment_informed_exposure_composition"),
        ("context_only_exposure_and_support", "assignment_informed_exposure_and_support"),
    ):
        assignment = specs[assignment_name]["feature_names"]
        assert "targeted_maximum" in assignment
        assert [name for name in assignment if name != "targeted_maximum"] == specs[context_name]["feature_names"]


@pytest.mark.parametrize("group_names", [["task_driver"], ["a", "b", "c", "d", "e", "f", "g", "h"]])
def test_historical_mapping_preserves_every_original_column_and_its_order(group_names):
    schema = ANALYZER["rq2_feature_schema"](group_names)
    nbase, nexp, nsupport = 1 + len(group_names), 10, 2
    historical = {
        "group_context": list(range(nbase)),
        "plus_exposure_composition": list(range(nbase + nexp)),
        "plus_support_only": list(range(nbase)) + [nbase + nexp],
        "plus_clean_score_only": list(range(nbase)) + [nbase + nexp + 1],
        "exposure_and_support": list(range(nbase + nexp + 1)),
        "plus_support_clean_score": list(range(nbase)) + list(range(nbase + nexp, nbase + nexp + nsupport)),
        "all_pre_intervention": list(range(nbase + nexp + nsupport)),
        "plus_realized_fanout_diagnostics": list(range(nbase + nexp + nsupport + 3)),
    }
    assert set(schema["historical_specification_mapping"]) == set(historical)
    for old, indices in historical.items():
        current = schema["historical_specification_mapping"][old]
        spec = schema["specifications"][current]
        assert spec["feature_indices"] == indices
        assert spec["feature_names"] == [schema["feature_names"][i] for i in indices]
    assert "all_pre_intervention" not in schema["specifications"]
    assert "plus_exposure_composition" not in schema["specifications"]


def test_categories_partition_features_and_distinguish_clean_score_from_context(schema):
    categorized = [name for members in schema["feature_categories"].values() for name in members]
    assert len(categorized) == len(set(categorized))
    assert set(categorized) == set(schema["feature_names"])
    assert schema["feature_categories"]["assignment_informed_budget"] == ["targeted_maximum"]
    assert schema["feature_categories"]["pre_inference_clean_score"] == ["abs_clean_score"]
    assert schema["feature_categories"]["retained_context_support"] == ["retained_support"]
    note = schema["feature_category_notes"]["pre_inference_clean_score"]
    assert "mathematically coupled" in note
    assert "not independent mechanism" in note
    for spec in schema["specifications"].values():
        assert spec["feature_names"] == [schema["feature_names"][i] for i in spec["feature_indices"]]


def test_corrected_release_retains_every_historical_fit_under_explicit_mapping():
    root = Path(__file__).parents[1] / "results/final/2026-10-19/relation_analysis"
    old = json.loads((root / "pre_review/analysis.json").read_text())
    current = json.loads((root / "analysis.json").read_text())
    prior, revised = old["exposure_prediction"], current["exposure_prediction"]
    mapping = revised["historical_specification_mapping"]
    checked = 0
    for grouping in ("date", "driver"):
        for name, fit in prior[grouping].items():
            assert revised[grouping][mapping[name]] == fit
            checked += 1
    for group, grouped in prior["within_control"].items():
        for grouping, specifications in grouped.items():
            for name, fit in specifications.items():
                assert revised["within_control"][group][grouping][mapping[name]] == fit
                checked += 1
    assert checked == 96
    assert old["relation_effects"] == current["relation_effects"]
    assert old["interactions"] == current["interactions"]
