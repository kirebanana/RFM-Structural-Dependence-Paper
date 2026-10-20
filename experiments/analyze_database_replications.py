"""Analyze frozen cross-database controls; no exploratory predictor fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from rfm_structure.analysis import assert_prediction_alignment, paired_statistics
from rfm_structure.exploratory import clustered_auc_delta
from rfm_structure.metrics import binary_auroc
from rfm_structure.provenance import file_identity
from rfm_structure.validation import reconcile_exposure


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def paired_mean_interval(differences, entities, replicates=2000):
    """Entity-cluster resampling, conditional on the observed query dates/seeds."""
    unique, codes = np.unique(entities, return_inverse=True)
    draws = np.random.default_rng(20261007).multinomial(len(unique), np.ones(len(unique)) / len(unique), size=replicates)
    weights = draws[:, codes]
    estimates = (weights @ differences) / weights.sum(axis=1)
    return {"point": float(np.mean(differences)), "entity_percentile_95": np.quantile(estimates, [.025, .975]).tolist(),
            "replicates": replicates, "scope": "conditional on test date and three matching seeds"}


def entity_auc_uncertainty(labels, reference, scores, dates, entities):
    """Use entity terminology for panels whose entities are not F1 drivers."""
    return {key.replace("driver", "entity"): value
            for key, value in clustered_auc_delta(labels, reference, scores, dates, entities).items()}


def analyze_run(root):
    state = json.loads((root / "replication.json").read_text())
    if not state["complete"] or state["status"] != "complete":
        raise ValueError("replication did not complete")
    database, task = state["database"], state["task"]
    if (root / "panel.npz").exists():
        panel = read_npz(root / "panel.npz")
        if not np.array_equal(panel["node_idxs"], state["query_ids"]):
            raise ValueError("panel metadata differs from prediction order")
        labels, dates, entities = panel["labels"], panel["dates"], panel["entities"]
    else:
        import pandas as pd

        source = Path(state["provenance"]["preprocessing"]["meta.json"]["path"]).parent
        raw = Path(json.loads((source / "meta.json").read_text())["source"])
        task_spec = yaml.safe_load((raw / "tasks" / task / "manifest.yaml").read_text())
        offset = next(info["start"] for info in state["availability_policy"]["tables"] if info["table"] == f"{task}:Test")
        rows = pd.read_parquet(raw / "tasks" / task / "test.parquet").iloc[np.asarray(state["query_ids"]) - offset]
        labels = rows[task_spec["target_col"]].to_numpy().astype(int)
        dates = rows[task_spec["time_col"]].astype(str).to_numpy()
        entities = rows[task_spec["entity_col"]].to_numpy()
    output = {"database": database, "task": task, "protocol": state["protocol"], "panel": state["panel"],
              "official_test_size": state["official_test_size"], "queries": state["n_queries"], "run": str(root),
              "contexts": {}, "interpretation": "fixed small replication; no full-test or temporal-generalization claim"}
    checkpoint = None
    sensitivities = {}
    reference_clean = None
    for ctx in (48, 128):
        directory = root / f"ctx{ctx}_lctx{ctx // 2}"
        manifest = json.loads((directory / "manifest.json").read_text())
        validation = json.loads((directory / "validation.json").read_text())
        if manifest["status"] != "complete" or not manifest["complete"] or validation["status"] != "passed":
            raise ValueError("context did not pass validation and complete inference")
        identity = (manifest["checkpoint"]["weights"]["sha256"], manifest["checkpoint"]["config"]["sha256"])
        if checkpoint is not None and identity != checkpoint:
            raise ValueError("checkpoint differs across contexts")
        checkpoint = identity
        loaded = {}
        exposure = {}
        for name, arm in manifest["arms"].items():
            pred_path, exposure_path = directory / arm["predictions"], directory / arm["exposure"]
            if file_identity(pred_path)["sha256"] != arm["prediction_sha256"] or file_identity(exposure_path)["sha256"] != arm["exposure_sha256"]:
                raise ValueError("artifact checksum mismatch")
            loaded[name], exposure[name] = read_npz(pred_path), read_npz(exposure_path)
            assert_prediction_alignment(loaded[name], loaded[name], count=len(labels))
            if not np.array_equal(loaded[name]["node_idxs"], state["query_ids"]) or not np.array_equal(loaded[name]["labels"], validation["labels"]):
                raise ValueError("prediction query/label order differs from validation")
            if not np.array_equal(loaded[name]["labels"] > 0, labels.astype(bool)):
                raise ValueError("stored normalized labels differ from raw task labels")
            reconcile_exposure({**exposure[name], "is_prediction_query": np.ones(len(labels), dtype=bool)}, arm["totals"])
            if not np.array_equal(exposure[name]["target_node_idxs"], state["query_ids"]):
                raise ValueError("exposure/prediction target order differs")
            if any(exposure[name][f"{prefix}temporal_{key}"].sum() for prefix in ("", "base_") for key in ("unknown", "known_violations")):
                raise ValueError("unresolved temporal status")
            if not np.isclose(binary_auroc(labels, loaded[name]["preds"]), arm["auroc"], atol=1e-12):
                raise ValueError("AUROC differs from manifest")
        clean = loaded["base"]
        assert_prediction_alignment(reference_clean if reference_clean is not None else clean, clean, count=len(labels))
        reference_clean = clean
        context = {"clean_auc": binary_auroc(labels, clean["preds"]), "groups": {},
                   "availability": validation["availability_totals"]}
        output["contexts"][str(ctx)] = context
        for group in ("task_entity", "database"):
            scores = np.array([loaded[f"{group}_s{seed}"]["preds"] for seed in (101, 202, 303)])
            es = [exposure[f"{group}_s{seed}"] for seed in (101, 202, 303)]
            for seed in (101, 202, 303):
                assert_prediction_alignment(clean, loaded[f"{group}_s{seed}"], count=len(labels))
            shift = np.abs(scores - clean["preds"])
            sensitivities[(ctx, group)] = shift.mean(axis=0)
            aucs = [binary_auroc(labels, score) for score in scores]
            context["groups"][group] = {
                "auroc_mean": float(np.mean(aucs)), "auroc_range": [min(aucs), max(aucs)],
                "delta_clean": float(np.mean(aucs) - context["clean_auc"]), "aurocs_by_seed": aucs,
                "mean_abs_shift": float(shift.mean()), "pooled_median_abs_shift": float(np.median(shift)),
                "paired_by_seed": [paired_statistics(clean["preds"], score) for score in scores],
                "changed_edges_per_seed": [int(e["changed_edges"].sum()) for e in es],
                "affected_queries_per_seed": [int((e["changed_edges"] > 0).sum()) for e in es],
                "auroc_uncertainty": entity_auc_uncertainty(labels, clean["preds"], scores, dates, entities),
            }
        task_scores = np.array([loaded[f"task_entity_s{s}"]["preds"] for s in (101, 202, 303)])
        db_scores = np.array([loaded[f"database_s{s}"]["preds"] for s in (101, 202, 303)])
        context["task_minus_database_abs_sensitivity"] = paired_mean_interval(
            sensitivities[(ctx, "task_entity")] - sensitivities[(ctx, "database")], entities
        )
        context["task_minus_database_auroc"] = entity_auc_uncertainty(labels, db_scores, task_scores, dates, entities)
    output["context_sensitivity_change"] = {
        group: paired_mean_interval(sensitivities[(128, group)] - sensitivities[(48, group)], entities)
        for group in ("task_entity", "database")
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyses = [analyze_run(root) for root in args.runs]
    args.output.mkdir(exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(analyses, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen cross-database replication", "", "Detailed paired contrasts: [analysis.json](analysis.json).", "",
              "512 uniformly selected official test queries per task; three matching seeds.",
             "Single-date panels: intervals are conditional entity-resampling sensitivity checks, not temporal replication.", "",
             "| Database/task | Context | Clean AUROC | Task-link AUROC mean | DB-link AUROC mean | Task abs shift | DB abs shift |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for result in analyses:
        for ctx, values in result["contexts"].items():
            task, db = values["groups"]["task_entity"], values["groups"]["database"]
            lines.append(f"| {result['database']}/{result['task']} | {ctx} | {values['clean_auc']:.4f} | "
                         f"{task['auroc_mean']:.4f} | {db['auroc_mean']:.4f} | {task['mean_abs_shift']:.4f} | {db['mean_abs_shift']:.4f} |")
    lines += ["", "Task/entity and DB groups use different changed-edge counts, not matched dose.",
              "Declared static covariates and post-sampling availability filtering retain the existing interpretation limits."]
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")


if __name__ == "__main__":
    main()
