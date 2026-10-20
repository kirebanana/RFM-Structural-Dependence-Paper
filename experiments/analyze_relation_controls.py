"""Exploratory relation effects and RQ2 sensitivity models from saved evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from rfm_structure.analysis import assert_prediction_alignment, paired_statistics
from rfm_structure.exploratory import (
    clustered_auc_delta,
    grouped_sensitivity_prediction,
)
from rfm_structure.metrics import binary_auroc
from rfm_structure.provenance import file_identity
from rfm_structure.validation import reconcile_exposure


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def relation_count(exposure, relation_ids, field="eligible_edges"):
    counts = np.zeros(len(exposure["target_node_idxs"]), dtype=float)
    selected = np.isin(exposure["relation_ids"], relation_ids)
    np.add.at(counts, exposure["relation_query_indices"][selected], exposure[f"relation_{field}"][selected])
    return counts


def rq2_feature_schema(group_names):
    """Separate context measurements from assignment budgets and diagnostics.

    Keep the historical column order and mapped specifications unchanged so
    their numerical fits remain directly comparable with saved analyses.
    """
    base = ["context128"] + [f"group_{name}" for name in group_names]
    exposure = ["eligible", "unique_parents", "targeted_eligible", "targeted_maximum",
                "retained_tokens", "share_task", "share_db_driver", "share_db_constructor", "share_db_race", "share_circuit"]
    support, clean_score = ["retained_support"], ["abs_clean_score"]
    post = ["changed", "feat_absolute_delta", "nbr_absolute_delta"]
    names = base + exposure + support + clean_score + post
    context = [name for name in exposure if name != "targeted_maximum"]
    categories = {
        "context_and_control_design": base,
        "context_only_exposure": context,
        "retained_context_support": support,
        "pre_inference_clean_score": clean_score,
        "assignment_informed_budget": ["targeted_maximum"],
        "post_intervention_diagnostics": post,
    }
    historical_mapping = {
        "group_context": "group_context",
        "plus_exposure_composition": "plus_assignment_informed_exposure_composition",
        "plus_support_only": "plus_support_only",
        "plus_clean_score_only": "plus_clean_score_only",
        "exposure_and_support": "assignment_informed_exposure_and_support",
        "plus_support_clean_score": "plus_support_clean_score",
        "all_pre_intervention": "assignment_informed_exposure_support_clean_score",
        "plus_realized_fanout_diagnostics": "plus_realized_fanout_diagnostics",
    }
    selections = {
        "group_context": base,
        "plus_context_only_exposure": base + context,
        historical_mapping["plus_exposure_composition"]: base + exposure,
        "plus_support_only": base + support,
        "plus_clean_score_only": base + clean_score,
        "context_only_exposure_and_support": base + context + support,
        historical_mapping["exposure_and_support"]: base + exposure + support,
        "plus_support_clean_score": base + support + clean_score,
        historical_mapping["all_pre_intervention"]: base + exposure + support + clean_score,
        "plus_realized_fanout_diagnostics": names,
    }
    return {
        "feature_names": names,
        "feature_categories": categories,
        "feature_category_notes": {
            "context_and_control_design": "Context size and proposed control identity; no solved assignment outcomes.",
            "context_only_exposure": "Retained-context counts and relation shares, including eligibility for the proposed relation group; no assignment-derived maximum.",
            "retained_context_support": "Retained labeled-support count before intervention, not a measure of independent mechanism.",
            "pre_inference_clean_score": "Absolute clean score is available before corrupted inference, but is mathematically coupled to |corrupted - clean| through the shared clean score; predictive gain is not independent mechanism evidence.",
            "assignment_informed_budget": "targeted_maximum is log1p(max_changed_edges) from the solved, targeted assignment (and cycle restriction), using the first saved seed as historically. It is not context-only exposure; the raw max_changed_edges count equals changed_edges per seed at full strength.",
            "post_intervention_diagnostics": "Seed-mean realized changed edges and fanout deltas; diagnostic associations, not context-only prediction.",
        },
        "specifications": {
            name: {"feature_names": selected,
                   "feature_indices": [names.index(feature) for feature in selected],
                   "feature_categories": [category for category, members in categories.items()
                                          if any(feature in members for feature in selected)]}
            for name, selected in selections.items()
        },
        "historical_specification_mapping": historical_mapping,
        "historical_fit_preservation": "Mapped historical specifications retain the same columns, column order, affected-record selection, seed aggregation, and fixed ridge evaluation. Renaming changes interpretation, not their numerical fits.",
    }


def analyze(runs, reference, raw_dir, output, panel_path=None):
    results = {"evidence_kind": "exploratory_relation_and_exposure_analysis", "runs": [str(r) for r in runs],
               "relation_effects": [], "interactions": {}, "exposure_prediction": {},
               "limitations": ["Post-exploration controls on the same single-task panel, not confirmatory tests.",
                               "Groups differ in changed-edge count and exposure; effect per edge is not a causal estimate.",
                               "Query-cycle controls include other rows in those cycles, not isolated query-edge replacements.",
                               "Cluster intervals are sensitivity checks, conditional on three seeds and one checkpoint.",
                                "Crossed cluster resampling does not eliminate temporal or outcome-window dependence.",
                                "Held-out sensitivity models predict model response, not labels or causal structure effects.",
                                "Context-only describes predictor inputs, not cohort selection: RQ2 records are restricted to queries affected by the realized interventions.",
                                "Assignment-informed targeted_maximum is not context-only exposure; at full strength max_changed_edges equals changed_edges.",
                                "Clean-score magnitude is a pre-inference predictor mathematically coupled to |corrupted - clean|, not independent mechanism evidence.",
                               "Original partial/full strengths differ in relation composition; no clean dose-response claim.",
                               "Availability masks retained inputs after sampling; historical retrieval validity is not proven."]}
    group_data = {}
    canonical = npz(reference / "ctx48_lctx24/preds_base.npz")
    policy = json.loads((reference / "matrix.json").read_text())["availability_policy"]
    task_range = next(info for info in policy["tables"] if info["table"] == "driver-dnf:Test")
    if panel_path is not None:
        panel = npz(panel_path)
        if not np.array_equal(panel["node_idxs"], canonical["node_idxs"]):
            raise ValueError("panel metadata differs from prediction order")
        labels, dates, drivers = panel["labels"], panel["dates"], panel["entities"]
    else:
        import pandas as pd

        table = pd.read_parquet(raw_dir / "tasks/driver-dnf/test.parquet").iloc[
            canonical["node_idxs"] - task_range["start"]
        ].reset_index(drop=True)
        labels = table["did_not_finish"].to_numpy().astype(int)
        dates = table["date"].astype(str).to_numpy()
        drivers = table["driverId"].to_numpy()
    if not np.array_equal(labels, (canonical["labels"] > 0).astype(int)):
        raise ValueError("raw task labels differ from saved paired labels")
    results["panel"] = {"queries": len(labels), "dates": len(set(dates)), "drivers": len(set(drivers))}
    for run in runs:
        matrix = json.loads((run / "controls.json").read_text())
        if not matrix["complete"] or matrix["status"] != "complete":
            raise ValueError("relation-control run is incomplete")
        for ctx in (48, 128):
            directory = run / f"ctx{ctx}_lctx{ctx // 2}"
            manifest = json.loads((directory / "manifest.json").read_text())
            checked = json.loads((directory / "validation.json").read_text())
            ref_directory = reference / directory.name
            ref_manifest = json.loads((ref_directory / "manifest.json").read_text())
            clean = npz(ref_directory / "preds_base.npz")
            if checked["status"] != "passed" or not checked["complete"]:
                raise ValueError("control has no passed validation")
            if checked["batch_fingerprints"] != ref_manifest["validation"]["batch_fingerprints"]:
                raise ValueError("control sampled context differs from reference")
            if any(manifest["checkpoint"][key]["sha256"] != ref_manifest["checkpoint_provenance"][key]["sha256"]
                   for key in ("config", "weights")):
                raise ValueError("control checkpoint differs from reference")
            for group, relation_ids in matrix["groups"].items():
                if (ctx, group) in group_data:
                    raise ValueError("duplicate context/control across supplied runs")
                arm_scores, exposures, per_seed = [], [], []
                for seed in (101, 202, 303):
                    name = f"{group}_s{seed}"
                    arm = manifest["arms"][name]
                    pred_path, exposure_path = directory / arm["predictions"], directory / arm["exposure"]
                    if file_identity(pred_path)["sha256"] != arm["prediction_sha256"] or file_identity(exposure_path)["sha256"] != arm["exposure_sha256"]:
                        raise ValueError("control checksum mismatch")
                    prediction, exposure = npz(pred_path), npz(exposure_path)
                    assert_prediction_alignment(clean, prediction)
                    if not np.array_equal(exposure["target_node_idxs"], clean["node_idxs"]):
                        raise ValueError("control exposure differs from target order")
                    reconcile_exposure({**exposure, "is_prediction_query": np.ones(702, dtype=bool)}, arm["totals"])
                    if any(exposure[f"{prefix}temporal_{key}"].sum() for prefix in ("", "base_") for key in ("unknown", "known_violations")):
                        raise ValueError("control temporal status is not valid")
                    untouched = exposure["changed_edges"] == 0
                    if not np.array_equal(prediction["preds"][untouched], clean["preds"][untouched]):
                        raise ValueError("control changed predictions without changed links")
                    scores = prediction["preds"].astype(float)
                    if not np.isclose(binary_auroc(labels, scores), arm["auroc"], atol=1e-12):
                        raise ValueError("recomputed AUROC differs")
                    arm_scores.append(scores)
                    exposures.append(exposure)
                    per_seed.append({"seed": seed, "auroc": binary_auroc(labels, scores),
                                     "changed_edges": int(exposure["changed_edges"].sum()),
                                     "affected_queries": int((~untouched).sum()),
                                     **paired_statistics(clean["preds"], scores)})
                score_array = np.asarray(arm_scores)
                sensitivity = np.mean(np.abs(score_array - clean["preds"]), axis=0)
                changed = np.mean([exposure["changed_edges"] for exposure in exposures], axis=0)
                affected = changed > 0
                summary = {"context": ctx, "group": group, "relation_ids": relation_ids,
                           "clean_auc": binary_auroc(labels, clean["preds"]),
                           "auroc_mean": float(np.mean([r["auroc"] for r in per_seed])),
                           "auroc_range": [min(r["auroc"] for r in per_seed), max(r["auroc"] for r in per_seed)],
                           "affected_queries": int(affected.sum()), "changed_edges_mean": float(changed.sum()),
                           "mean_abs_shift": float(sensitivity.mean()), "median_query_mean_abs_shift": float(np.median(sensitivity)),
                           "per_seed": per_seed,
                           "uncertainty": clustered_auc_delta(labels, clean["preds"], score_array, dates, drivers)}
                summary["affected_subset_auc_delta"] = (
                    float(np.mean([binary_auroc(labels[affected], s[affected]) for s in score_array])
                          - binary_auroc(labels[affected], clean["preds"][affected]))
                    if affected.any() and len(np.unique(labels[affected])) == 2 else None
                )
                group_data[(ctx, group)] = {"clean": clean["preds"], "scores": score_array,
                                           "exposures": exposures, "sensitivity": sensitivity,
                                           "changed": changed, "relation_ids": relation_ids}
                results["relation_effects"].append(summary)
    required = {"task_driver", "database_all", "database_driver", "database_constructor",
                "database_race", "race_circuit", "task_query_cycles", "task_other_cycles"}
    if set(group_data) != {(ctx, group) for ctx in (48, 128) for group in required}:
        raise ValueError("analysis requires both the six relation groups and two task-cycle controls at both contexts")
    for ctx in (48, 128):
        clean = group_data[(ctx, "task_driver")]["clean"]
        full = np.array([npz(reference / f"ctx{ctx}_lctx{ctx // 2}" / f"preds_a100_s{seed}.npz")["preds"]
                         for seed in (101, 202, 303)])
        task = group_data[(ctx, "task_driver")]["scores"]
        database = group_data[(ctx, "database_all")]["scores"]
        residual = (full - clean) - ((task - clean) + (database - clean))
        query = group_data[(ctx, "task_query_cycles")]["scores"]
        other = group_data[(ctx, "task_other_cycles")]["scores"]
        task_cycle_residual = (task - clean) - ((query - clean) + (other - clean))
        results["interactions"][str(ctx)] = {"mean_absolute_nonadditive_score_residual": float(np.abs(residual).mean()),
                                           "fraction_scores_nonadditive": float((residual != 0).mean()),
                                           "task_cycle_mean_absolute_nonadditive_residual": float(np.abs(task_cycle_residual).mean()),
                                           "caution": "Nonadditivity is an exact score identity, not a causal interaction estimate."}
    records, responses, date_groups, driver_groups = [], [], [], []
    group_names = sorted({group for _, group in group_data})
    feature_schema = rq2_feature_schema(group_names)
    for (ctx, group), data in group_data.items():
        e = data["exposures"][0]
        registry = json.loads((reference / f"ctx{ctx}_lctx{ctx // 2}/manifest.json").read_text())["relation_index"]
        categories = {
            "task": [int(r) for r, m in registry.items() if m["child_table"] == "driver-dnf"],
            "db_driver": [int(r) for r, m in registry.items() if m["parent_table"] == "drivers" and m["child_table"] != "driver-dnf"],
            "db_constructor": [int(r) for r, m in registry.items() if m["parent_table"] == "constructors"],
            "db_race": [int(r) for r, m in registry.items() if m["parent_table"] == "races"],
            "circuit": [int(r) for r, m in registry.items() if m["parent_table"] == "circuits"],
        }
        denominators = np.maximum(e["eligible_edges"], 1)
        numeric = np.column_stack([
            np.log1p(e["eligible_edges"]), np.log1p(e["unique_parent_nodes"]),
            np.log1p(relation_count(e, data["relation_ids"])), np.log1p(e["max_changed_edges"]),
            e["context_token_counts"] / ctx,
            *[relation_count(e, ids) / denominators for ids in categories.values()],
            np.log1p(e["labeled_support_counts"]), np.abs(data["clean"]), np.log1p(data["changed"]),
            np.log1p(np.mean([x["fanout_feat_absolute_delta"] for x in data["exposures"]], axis=0)),
            np.log1p(np.mean([x["fanout_nbr_absolute_delta"] for x in data["exposures"]], axis=0)),
        ])
        for i in np.flatnonzero(data["changed"] > 0):
            records.append([float(ctx == 128), *[float(group == name) for name in group_names], *numeric[i]])
            responses.append(data["sensitivity"][i])
            date_groups.append(dates[i])
            driver_groups.append(drivers[i])
    features = np.asarray(records)
    specs = {name: spec["feature_indices"] for name, spec in feature_schema["specifications"].items()}
    results["exposure_prediction"].update(feature_schema)
    results["exposure_prediction"]["scope"] = "Affected query/control records only; all contexts/control repetitions of a held-out date or driver stay held out. Fixed linear ridge, no tuning."
    for grouping, groups in (("date", date_groups), ("driver", driver_groups)):
        results["exposure_prediction"][grouping] = {
            name: grouped_sensitivity_prediction(features[:, indices], responses, groups)
            for name, indices in specs.items()
        }
    results["exposure_prediction"]["within_control"] = {}
    for group in group_names:
        selected = features[:, 1 + group_names.index(group)] == 1
        if selected.sum() < 100:
            continue
        results["exposure_prediction"]["within_control"][group] = {}
        for grouping, clusters in (("date", np.asarray(date_groups)), ("driver", np.asarray(driver_groups))):
            results["exposure_prediction"]["within_control"][group][grouping] = {
                name: grouped_sensitivity_prediction(features[selected][:, indices], np.asarray(responses)[selected], clusters[selected])
                for name, indices in specs.items() if name in (
                    "group_context", "plus_context_only_exposure", "plus_assignment_informed_exposure_composition",
                    "plus_support_only", "plus_clean_score_only", "context_only_exposure_and_support",
                    "assignment_informed_exposure_support_clean_score"
                )
            }
    for ctx in (48, 128):
        for group in ("task_driver", "database_all", "task_query_cycles", "task_other_cycles"):
            data = group_data[(ctx, group)]
            support = data["exposures"][0]["labeled_support_counts"]
            associations = {}
            for field, vector in (("support", support), ("changed", data["changed"]),
                                  ("eligible", data["exposures"][0]["eligible_edges"])):
                associations[field] = float(spearmanr(vector, data["sensitivity"]).statistic) if np.ptp(vector) else None
            summary = next(row for row in results["relation_effects"] if row["context"] == ctx and row["group"] == group)
            summary["sensitivity_associations"] = associations
            summary["support_subgroups"] = []
            for name, mask in (("none", support == 0), ("one_to_three", (support > 0) & (support <= 3)), ("four_plus", support >= 4)):
                if not mask.any():
                    continue
                summary["support_subgroups"].append({"name": name, "n": int(mask.sum()),
                    "mean_abs_shift": float(data["sensitivity"][mask].mean()),
                    "auc_delta": float(np.mean([binary_auroc(labels[mask], s[mask]) for s in data["scores"]])
                                       - binary_auroc(labels[mask], data["clean"][mask])) if len(np.unique(labels[mask])) == 2 else None})
    output.mkdir(exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["# Exploratory relation controls", "", "Detailed contrasts and RQ2 models: [analysis.json](analysis.json).", "",
              "Same 702 sampled queries and context fingerprints; three matching seeds; no confirmatory claims.",
              "Affected queries are the union across seeds; changed edges are the seed mean. Budgets are not matched.", "",
              "| Context | Control | AUROC mean | Δ clean | Affected queries (union) | Changed edges (mean) | Mean abs score shift | Crossed-cluster interval for Δ |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in results["relation_effects"]:
        lo, hi = row["uncertainty"]["crossed_percentile_95"]
        lines.append(f"| {row['context']} | {row['group']} | {row['auroc_mean']:.4f} | "
                     f"{row['uncertainty']['point']:+.4f} | {row['affected_queries']} | "
                     f"{row['changed_edges_mean']:.0f} | {row['mean_abs_shift']:.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "## Held-out sensitivity prediction", "", "Response: log1p(seed-mean absolute score shift). Out-of-fold R² is not causal evidence.", "",
              "Context-only specifications exclude assignment-derived budgets and realized diagnostics. The affected-record cohort is still selected after intervention.",
              "Assignment-informed comparisons retain the historical numerical fits under corrected names; they are not context-only exposure models.",
              "Clean-score magnitude is available before corrupted inference but is mathematically coupled to |corrupted - clean|, not independent mechanism evidence.", "",
              "| Features | Date holdout R² | Driver holdout R² |", "|---|---:|---:|"]
    for name in specs:
        lines.append(f"| {name} | {results['exposure_prediction']['date'][name]['out_of_fold_r2']:.4f} | "
                      f"{results['exposure_prediction']['driver'][name]['out_of_fold_r2']:.4f} |")
    lines += ["", "### Feature categories", ""]
    for category, members in feature_schema["feature_categories"].items():
        lines.append(f"- **{category}**: {', '.join(members)}. {feature_schema['feature_category_notes'][category]}")
    lines += ["", "### Historical specification mapping", "", feature_schema["historical_fit_preservation"], "",
              "| Historical name | Current name (same numerical fit) |", "|---|---|"]
    for old, new in feature_schema["historical_specification_mapping"].items():
        lines.append(f"| {old} | {new} |")
    lines += ["", "## Limitations", *[f"- {text}" for text in results["limitations"]]]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(output / "SUMMARY.md")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("results/preliminary/2026-10-05_validated"))
    metadata = parser.add_mutually_exclusive_group(required=True)
    metadata.add_argument("--raw-dir", type=Path)
    metadata.add_argument("--panel", type=Path, help="curated panel.npz; avoids external raw data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs, args.reference, args.raw_dir, args.output, args.panel)


if __name__ == "__main__":
    main()
