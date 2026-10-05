"""Paired descriptive analysis of the saved preliminary matrix, without RT."""

import json
from pathlib import Path

import numpy as np

from .metrics import binary_auroc
from .provenance import file_identity
from .validation import (
    AVAILABILITY_FIELDS,
    FANOUT_FIELDS,
    TEMPORAL_FIELDS,
    experiment_arms,
    reconcile_exposure,
    reconcile_temporal,
)


def paired_statistics(clean, scores):
    clean = np.asarray(clean, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if clean.ndim != 1 or clean.shape != scores.shape or not len(clean):
        raise ValueError("paired scores must be equal nonempty vectors")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(scores)):
        raise ValueError("paired scores must be finite")
    shift = np.abs(scores - clean)
    correlation = None if np.ptp(clean) == 0 or np.ptp(scores) == 0 else float(np.corrcoef(clean, scores)[0, 1])
    return {"mean_absolute_raw_shift": float(shift.mean()), "median_absolute_raw_shift": float(np.median(shift)),
            "pearson": correlation, "pearson_status": "defined" if correlation is not None else "undefined_zero_variance"}


def assert_prediction_alignment(base, arm, count=702):
    for arrays in (base, arm):
        for field in ("labels", "preds", "node_idxs"):
            if arrays[field].shape != (count,) or not np.all(np.isfinite(arrays[field])):
                raise ValueError(f"invalid prediction array {field}")
        if len(np.unique(arrays["node_idxs"])) != count:
            raise ValueError("duplicate prediction target IDs")
    if not np.array_equal(base["node_idxs"], arm["node_idxs"]) or not np.array_equal(base["labels"], arm["labels"]):
        raise ValueError("paired target IDs or labels are misaligned")


def realized_corruption(exposure):
    changed = np.asarray(exposure["changed_edges"])
    eligible = np.asarray(exposure["eligible_edges"])
    maximum = np.asarray(exposure["max_changed_edges"])
    result = {}
    for name, denominator in (("eligible", eligible), ("max", maximum)):
        defined = denominator != 0
        result[f"mean_changed_{name}"] = float(np.mean(changed[defined] / denominator[defined])) if np.any(defined) else None
        result[f"global_changed_{name}"] = float(changed.sum() / denominator.sum()) if denominator.sum() else None
        result[f"zero_{name}_queries"] = int((~defined).sum())
    return result


def _read_npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _artifact_path(directory, name):
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("artifact must be a file in its context directory")
    return path


def verify_policy_exposure(exposure, validation, name):
    """Independently reconcile saved availability, temporal and fanout evidence."""
    for prefix, summary in (("", validation["per_arm"][name]["temporal"]),
                            ("base_", validation["base_temporal"])):
        queries = [
            {key: int(exposure[f"{prefix}temporal_{key}"][i]) for key in TEMPORAL_FIELDS}
            for i in range(len(exposure["target_node_idxs"]))
        ]
        reconcile_temporal({**summary, "queries": queries}, int(exposure["eligible_edges"].sum()))
    for key in AVAILABILITY_FIELDS:
        if int(exposure[f"availability_{key}"].sum()) != validation["availability_totals"][key]:
            raise ValueError(f"saved availability accounting differs for {key}")
    original = exposure["availability_original_token_counts"]
    removed = exposure["availability_removed_future_tokens"] + exposure["availability_removed_unmatured_label_tokens"]
    if not np.array_equal(original - removed, exposure["context_token_counts"]):
        raise ValueError("original/removed/retained token counts do not reconcile")
    for key in FANOUT_FIELDS:
        if int(exposure[f"fanout_{key}"].sum()) != validation["per_arm"][name]["fanout_totals"][key]:
            raise ValueError(f"saved fanout diagnostics differ for {key}")


def analyze_matrix(root, *, allow_test_artifacts=False):
    """Require the complete 14-arm matrix and independently check saved pairing."""
    root = Path(root)
    matrix = json.loads((root / "matrix.json").read_text())
    if matrix["status"] != "complete" or not matrix["complete"] or set(matrix["contexts"]) != {"48", "128"}:
        raise ValueError("analysis requires a complete two-context inference matrix")
    rows, summaries, shifts = [], [], {}
    first_clean = None
    first_checkpoint = None
    expected_arms = experiment_arms([101, 202, 303])
    for ctx in (48, 128):
        directory = root / f"ctx{ctx}_lctx{ctx // 2}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest["complete"] or manifest["status"] != "complete" or manifest["n_queries"] != 702:
            raise ValueError("context run is incomplete")
        if manifest["evidence_kind"] != "real" and not allow_test_artifacts:
            raise ValueError("synthetic test artifacts are not research results")
        if manifest["ctx"] != ctx or manifest["local_ctx"] != ctx // 2:
            raise ValueError("context metadata differs from directory")
        if manifest["db_name"] != "rel-f1" or manifest["task_table"] != "driver-dnf" or manifest["checkpoint"] != "stanford-star/rt-plurel/classification":
            raise ValueError("task/model differs from the preliminary research setup")
        if set(manifest["arms"]) != {name for name, _, _ in expected_arms}:
            raise ValueError("expected exactly seven agreed arms")
        validation = manifest["validation"]
        if validation["status"] != "passed" or not validation["complete"]:
            raise ValueError("inference lacks complete passed validation evidence")
        if manifest["evidence_kind"] == "real":
            policy = validation.get("availability_policy")
            if not policy or policy["version"] != "query_available_incidence_v1" or policy != matrix.get("availability_policy"):
                raise ValueError("real analysis requires the recorded common availability/incidence policy")
            checkpoint = manifest["checkpoint_provenance"]
            identity = (checkpoint["weights"]["sha256"], checkpoint["config"]["sha256"])
            if first_checkpoint is not None and identity != first_checkpoint:
                raise ValueError("checkpoint differs across contexts")
            first_checkpoint = identity
        base = _read_npz(_artifact_path(directory, manifest["arms"]["base"]["predictions"]))
        assert_prediction_alignment(first_clean if first_clean is not None else base, base)
        first_clean = base
        base_auc = binary_auroc(base["labels"], base["preds"])
        context_rows = []
        for name, alpha, seed in expected_arms:
            arm = manifest["arms"][name]
            if arm["requested_strength"] != alpha or arm["matching_seed"] != seed or arm["seed"] != (None if alpha == 0 else seed):
                raise ValueError("requested strength/seed metadata differs from planned arm")
            pred_path = _artifact_path(directory, arm["predictions"])
            exposure_path = _artifact_path(directory, arm["exposure"])
            if file_identity(pred_path)["sha256"] != arm["prediction_sha256"] or file_identity(exposure_path)["sha256"] != arm["exposure_sha256"]:
                raise ValueError("saved artifact checksum differs from completed manifest")
            pred, exposure = _read_npz(pred_path), _read_npz(exposure_path)
            assert_prediction_alignment(base, pred)
            if not np.array_equal(exposure["target_node_idxs"], pred["node_idxs"]):
                raise ValueError("exposure/prediction target IDs are misaligned")
            if not np.array_equal(pred["node_idxs"], validation["target_node_idxs"]) or not np.array_equal(pred["labels"], validation["labels"]):
                raise ValueError("saved predictions differ from validated targets/labels")
            reconcile_exposure({**exposure, "is_prediction_query": np.ones(702, dtype=bool)}, arm["totals"])
            if validation.get("availability_policy"):
                verify_policy_exposure(exposure, validation, name)
            if alpha == 0 and np.any(exposure["changed_edges"]):
                raise ValueError("clean arm has changed FK edges")
            if alpha == 1 and not np.array_equal(exposure["changed_edges"], exposure["max_changed_edges"]):
                raise ValueError("full arm does not realize its maximum matching")
            if not np.all(exposure["requested_strength"] == alpha) or not np.all(exposure["configured_context_sizes"] == ctx):
                raise ValueError("exposure strength/context differs from arm")
            if np.any(exposure["context_token_counts"] > ctx) or np.any(exposure["context_token_counts"] < 1):
                raise ValueError("actual context token counts are invalid")
            if any(exposure[f"{prefix}temporal_{field}"].sum() for prefix in ("", "base_") for field in ("unknown", "known_violations")):
                raise ValueError("saved exposure has unresolved temporal uncertainty/violations")
            if not set(exposure["relation_ids"].tolist()).issubset({int(key) for key in manifest["relation_index"]}):
                raise ValueError("exposure includes an unknown relation ID")
            auc = binary_auroc(pred["labels"], pred["preds"])
            if not np.isclose(auc, arm["auroc"], atol=1e-12, rtol=0):
                raise ValueError("saved AUROC differs from recomputed score")
            row = {"context": ctx, "local_context": ctx // 2, "arm": name, "requested_strength": alpha,
                   "seed": arm["seed"], "auroc": auc, "delta_clean": auc - base_auc,
                   **paired_statistics(base["preds"], pred["preds"]), **realized_corruption(exposure),
                    "timestamp_unknowns": validation["per_arm"][name]["temporal"]["unknown"]}
            row["fanout_diagnostics"] = validation["per_arm"][name].get("fanout_totals", {})
            row["availability_filter_totals"] = validation.get("availability_totals", {})
            rows.append(row)
            context_rows.append(row)
            shifts[name] = np.abs(pred["preds"] - base["preds"])
        for alpha in (0, 0.5, 1):
            selected = [row for row in context_rows if row["requested_strength"] == alpha]
            scores = [row["auroc"] for row in selected]
            summary = {"context": ctx, "local_context": ctx // 2, "requested_strength": alpha,
                       "seeds": [row["seed"] for row in selected], "auroc_mean": float(np.mean(scores)),
                       "auroc_min": min(scores), "auroc_max": max(scores),
                       "delta_clean_mean": float(np.mean([row["delta_clean"] for row in selected])),
                       "mean_absolute_raw_shift": float(np.mean([row["mean_absolute_raw_shift"] for row in selected])),
                       "pooled_median_absolute_raw_shift": float(np.median(np.concatenate([shifts[row["arm"]] for row in selected])))}
            for key in ("mean_changed_eligible", "mean_changed_max", "global_changed_eligible", "global_changed_max"):
                # Global ratios must be pooled from edge counts, not seed percentages.
                if key.startswith("global"):
                    denominator = "eligible_edges" if key.endswith("eligible") else "max_changed_edges"
                    totals = [manifest["arms"][row["arm"]]["totals"] for row in selected]
                    den = sum(total[denominator] for total in totals)
                    summary[key] = sum(total["changed_edges"] for total in totals) / den if den else None
                else:
                    zero_key = "zero_eligible_queries" if key.endswith("eligible") else "zero_max_queries"
                    defined = sum(702 - row[zero_key] for row in selected)
                    summary[key] = sum((row[key] or 0) * (702 - row[zero_key]) for row in selected) / defined if defined else None
            summaries.append(summary)
    return {"status": "complete", "result_set": str(root.resolve()),
             "evidence_kind": "synthetic_test" if allow_test_artifacts else "real",
             "availability_policy": matrix.get("availability_policy"),
            "rows": rows, "summaries": summaries,
            "limitations": ["Preliminary descriptive evidence; no significance or causal context-size claims.",
                             "This is a total FK-incidence effect: per-token fanout/normalization can change and is recorded, not held fixed.",
                            "Requested strength is relative to seeded maximum changed parent IDs, not all eligible edges.",
                            "Zero eligible/max denominators are undefined; query means omit these rows and report their counts.",
                             "Common filtering masks future rows and unclosed forecast labels without refilling the sampled context.",
                             "Timeless tables follow explicit benchmark schema metadata, not proof of historical attribute availability.",
                             "Retained temporal input uses query-time cutoffs and mature rolling support labels; this is not train-only context.",
                             "Sampling precedes availability masking; historically leakage-free retrieval selection has not been established.",
                             "The historical pilot is separate and its checkpoint revision was not pinned."]}


def summary_markdown(analysis):
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"

    lines = ["# Preliminary structural-reliance results", "",
             f"Result set: `{analysis['result_set']}`", f"Evidence: **{analysis['evidence_kind']}**", "",
             "## Context-by-strength summary", "",
             "| Context/local | Requested | Seeds | AUROC mean [range] | Δ clean | Global changed/eligible | Global changed/max |",
             "|---|---:|---|---|---:|---:|---:|"]
    for row in analysis["summaries"]:
        seeds = "clean" if row["requested_strength"] == 0 else ", ".join(str(seed) for seed in row["seeds"])
        lines.append(f"| {row['context']}/{row['local_context']} | {row['requested_strength']:.0%} | {seeds} | "
                     f"{row['auroc_mean']:.4f} [{row['auroc_min']:.4f}, {row['auroc_max']:.4f}] | "
                     f"{row['delta_clean_mean']:+.4f} | {fmt(row['global_changed_eligible'])} | {fmt(row['global_changed_max'])} |")
    lines += ["", "## Per-arm paired statistics", "",
              "| Context | Requested | Seed | AUROC | Δ clean | Changed/eligible | Changed/max | Mean abs raw shift | Median abs raw shift | Pearson | Unknown comparisons |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["rows"]:
        lines.append(f"| {row['context']} | {row['requested_strength']:.0%} | {row['seed'] if row['seed'] is not None else 'clean'} | "
                     f"{row['auroc']:.4f} | {row['delta_clean']:+.4f} | {fmt(row['global_changed_eligible'])} | {fmt(row['global_changed_max'])} | {row['mean_absolute_raw_shift']:.4f} | "
                     f"{row['median_absolute_raw_shift']:.4f} | {fmt(row['pearson'])} | {row['timestamp_unknowns']} |")
    if analysis.get("availability_policy"):
        lines += ["", "## Availability and incidence policy", "",
                  f"Policy: `{analysis['availability_policy']['version']}`. Future rows and unclosed forecast labels are masked before every arm; no refill.",
                  "Fanout changes are part of the measured incidence effect, not a held-fixed control.", "",
                  "| Context | Future cells masked | Unclosed label cells masked | Original cells |",
                  "|---|---:|---:|---:|"]
        for row in analysis["rows"]:
            if row["requested_strength"] == 0:
                counts = row["availability_filter_totals"]
                lines.append(f"| {row['context']} | {counts['removed_future_tokens']} | "
                             f"{counts['removed_unmatured_label_tokens']} | {counts['original_token_counts']} |")
    lines += ["", "## Limitations", "", *[f"- {text}" for text in analysis["limitations"]]]
    return "\n".join(lines) + "\n"
