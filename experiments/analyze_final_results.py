"""Verify the portable evidence release and independently recalculate its metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from rfm_structure.analysis import assert_prediction_alignment, verify_policy_exposure
from rfm_structure.exploratory import clustered_auc_delta
from rfm_structure.metrics import binary_auroc
from rfm_structure.relation_controls import relation_groups
from rfm_structure.replication import replication_groups
from rfm_structure.validation import (
    AVAILABILITY_FIELDS,
    FANOUT_FIELDS,
    TEMPORAL_FIELDS,
    reconcile_exposure,
    reconcile_temporal,
)

SEEDS = (101, 202, 303)
CONTEXTS = (48, 128)
RELATION_GROUPS = ("task_driver", "database_all", "database_driver", "database_constructor", "database_race", "race_circuit")
# This verifier is a release gate for the fixed published protocol, not a generic
# analyzer of arbitrary coherent manifests. Experimental identity is not inferred
# from whichever arms happen to be supplied.
EXPECTED = {
    "primary": ("rel-f1", "driver-dnf", 702, 495, 29, "matrix.json", ("a050", "a100")),
    "relations": ("rel-f1", "driver-dnf", 702, 495, 29, "controls.json", RELATION_GROUPS),
    "task_cycles": ("rel-f1", "driver-dnf", 702, 495, 29, "controls.json", ("task_query_cycles", "task_other_cycles")),
    "labeled_support": ("rel-f1", "driver-dnf", 702, 495, 29, "controls.json", ("labeled_support_only",)),
    "driver_top3": ("rel-f1", "driver-top3", 726, 128, 30, "matrix.json", ("a050", "a100")),
    "arxiv": ("rel-arxiv", "paper-citation", 512, 225, 1, "replication.json", ("task_entity", "database")),
    "trial": ("rel-trial", "study-outcome", 512, 292, 1, "replication.json", ("task_entity", "database")),
}
PINNED_CHECKPOINT = ("4a369f065d8f69e1da11ee4326ceb9d32cf78ee69255a438449e971f681b7f83",
                     "f93b00b5037b99d748cd1965e1ec710dea0e81773c6712af7e171659e3d2e728")
REQUIRED_RELEASE_FILES = {
    "catalog.json", "data_sources.json", "preparation_verification.json",
    "support_link_audit.json", "support_link_audit.npz", "source_at_finalization.patch", "source_identity.json",
    "recalculated/analysis.json", "relation_analysis/analysis.json", "cross_database_analysis/analysis.json",
    "driver_top3/analysis/analysis.json", "README.md", "SUMMARY.md", "AUDIT.md",
}


def read_json(path):
    return json.loads(path.read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_checksums(root):
    check((root / "checksums.json").is_file(), "release requires a complete checksums.json inventory")
    inventory = read_json(root / "checksums.json")
    check(bool(inventory), "empty release inventory")
    for name, identity in inventory.items():
        path = root / name
        check(not Path(name).is_absolute() and path.resolve().is_relative_to(root.resolve())
              and not path.is_symlink() and name != "checksums.json", "invalid inventory path")
        check(path.is_file(), f"missing artifact: {path}")
        content = path.read_bytes()
        check(len(content) == identity["bytes"] and hashlib.sha256(content).hexdigest() == identity["sha256"],
              f"checksum mismatch: {path}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != root / "checksums.json"}
    check(actual == set(inventory), "release contains unlisted artifacts")


def expected_arms(name):
    groups = EXPECTED[name][-1]
    arms = {f"{group}_s{seed}" for group in groups for seed in SEEDS}
    return arms | ({"base"} if name in ("primary", "driver_top3", "arxiv", "trial") else set())


def arm_order(name):
    groups = EXPECTED[name][-1]
    return (["base"] if "base" in expected_arms(name) else []) + [f"{group}_s{seed}" for group in groups for seed in SEEDS]


def verify_protocol(name, info, state, panel):
    database, task, n, positive, dates, state_file, groups = EXPECTED[name]
    check((info["database"], info["task"], info["state_file"]) == (database, task, state_file), "catalog protocol differs")
    check(set(state["contexts"]) == {"48", "128"}, "expected exactly two contexts")
    if name != "primary":
        check(state.get("seeds", state.get("matching_seeds")) == list(SEEDS), "matching seeds differ from protocol")
    check(set(panel) == {"node_idxs", "labels", "dates", "entities"}, "panel fields differ")
    check(all(array.shape == (n,) for array in panel.values()), "panel count/shape differs from protocol")
    check(set(np.unique(panel["labels"])) == {0, 1} and int(panel["labels"].sum()) == positive
          and len(np.unique(panel["dates"])) == dates, "panel labels/dates differ from protocol")
    if state_file == "matrix.json" and name != "primary":
        check(state["fractions"] == [0.0, 0.5, 1.0] and state["task"] == f"{database}/{task}"
              and state["expected_queries"] == n, "matrix protocol differs")
    elif state_file != "matrix.json":
        check(set(state["groups"]) == set(groups), "condition groups differ from protocol")
        if state_file == "replication.json":
            check((state["database"], state["task"]) == (database, task)
                  and state["selection_seed"] == 20261007 and state["n_queries"] == n
                  and state["query_ids"] == panel["node_idxs"].tolist(), "replication panel selection differs")
    if name == "labeled_support":
        check(state["source_selection"] == {"labeled_support_only": "visible_labeled_support"}, "support source selection differs")
    if name == "task_cycles":
        check(state["cycle_modes"] == {"task_query_cycles": "touching_query", "task_other_cycles": "excluding_query"},
              "task cycle selection differs")


def verify_registry(directory, state):
    path = directory / "relation_index.json"
    expected = state["provenance"]["preprocessing"]["relation_index.json"]
    content = path.read_bytes()
    check(len(content) == expected["bytes"] and hashlib.sha256(content).hexdigest() == expected["sha256"],
          "relation registry differs from authenticated preprocessing identity")
    registry = read_json(path)
    source = read_json(directory / "panel_source.json")
    tables = [item["table"].removesuffix(":Db") for item in state["availability_policy"]["tables"] if item["table"].endswith(":Db")]
    reconstructed = replication_groups(registry, state["task"], source["manifest"]["entity_table"], tables)
    check(reconstructed == state["groups"], "task/entity versus database grouping differs from relation registry")
    return registry


def verify_f1_groups(name, state, registry):
    reconstructed = relation_groups(registry)
    if name == "relations":
        expected = reconstructed
    else:
        expected = {group: reconstructed["task_driver"] for group in EXPECTED[name][-1]}
    check(state["groups"] == expected, "F1 group membership differs from saved relation semantics")


def verify_accounting(exposure, validation, arm_name, arm, clean_exposure):
    """Apply the same per-query identities to every saved validation format.

    Older control JSON omits temporal/availability aggregate fields: reconcile
    its complete NPZ fields and compare held-fixed counts to its verified clean
    reference. Compare every aggregate when that format actually saved one.
    """
    n = len(exposure["target_node_idxs"])
    validated = validation.get("per_arm", validation.get("arms", {}))[arm_name]
    check(validated["totals"] == arm["totals"], "validation/manifest edge totals differ")
    for prefix in ("", "base_"):
        queries = []
        for i in range(n):
            counts = {}
            for key in TEMPORAL_FIELDS:
                values = exposure[f"{prefix}temporal_{key}"]
                check(values.shape == (n,) and np.issubdtype(values.dtype, np.integer), "invalid temporal count vector")
                counts[key] = int(values[i])
            check(counts["comparisons"] == int(exposure["eligible_edges"][i]), "per-query temporal eligibility differs")
            queries.append(counts)
        totals = {key: sum(q[key] for q in queries) for key in TEMPORAL_FIELDS}
        reconcile_temporal({**totals, "queries": queries}, int(exposure["eligible_edges"].sum()))
        check(totals["unknown"] == totals["known_violations"] == 0, "unresolved temporal status")
        for key in TEMPORAL_FIELDS:
            check(np.array_equal(exposure[f"{prefix}temporal_{key}"], clean_exposure[f"temporal_{key}"]),
                  "temporal counts differ from fixed-context clean reference")
        aggregate = validation.get("base_temporal") if prefix else validated.get("temporal")
        if aggregate is not None:
            check(all(aggregate[key] == totals[key] for key in TEMPORAL_FIELDS), "saved temporal totals differ")
        if not prefix:
            for record in (validated, arm):
                for key in ("known_violations", "unknown", "declared_timeless"):
                    if key in record:
                        check(record[key] == totals[key], "compact temporal totals differ")
    for key in AVAILABILITY_FIELDS:
        field = f"availability_{key}"
        values = exposure[field]
        check(values.shape == (n,) and np.issubdtype(values.dtype, np.integer) and np.all(values >= 0),
              "invalid availability count vector")
        check(np.array_equal(values, clean_exposure[field]), "availability differs from fixed-context clean reference")
        if "availability_totals" in validation:
            check(int(values.sum()) == validation["availability_totals"][key], "saved availability totals differ")
    tokens = exposure["context_token_counts"]
    check(np.array_equal(tokens, clean_exposure["context_token_counts"]), "retained context differs from clean")
    check(np.array_equal(exposure["availability_original_token_counts"] - exposure["availability_removed_future_tokens"]
                         - exposure["availability_removed_unmatured_label_tokens"], tokens), "availability accounting differs")
    for field in ("eligible_edges", "unique_parent_nodes", "labeled_support_counts", "configured_context_sizes"):
        check(np.array_equal(exposure[field], clean_exposure[field]), f"held-fixed exposure differs: {field}")
    unchanged = exposure["changed_edges"] == 0
    for record in (validated, arm):
        if "affected_queries" in record:
            check(record["affected_queries"] == int((~unchanged).sum()), "affected-query totals differ")
        if "unchanged_queries" in record:
            check(record["unchanged_queries"] == int(unchanged.sum()), "unchanged-query totals differ")
    for key in FANOUT_FIELDS:
        values = exposure[f"fanout_{key}"]
        check(values.shape == (n,) and np.issubdtype(values.dtype, np.integer) and np.all(values >= 0),
              "invalid fanout count vector")
        check(not np.any(values[unchanged]), "fanout changed without changed edges")
        if "fanout_totals" in validated:
            check(int(values.sum()) == validated["fanout_totals"][key], "saved fanout totals differ")
    if "base_temporal" in validation:
        verify_policy_exposure(exposure, validation, arm_name)


def verify_support_audit(root):
    audit = read_json(root / "support_link_audit.json")
    check(set(audit["contexts"]) == {"48", "128"}, "support audit contexts incomplete")
    for groups in audit["contexts"].values():
        check(set(groups) == {"task_cycles", "labeled_support"}, "support audit groups incomplete")
        check(all(set(seeds) == {str(seed) for seed in SEEDS} for seeds in groups.values()), "support audit seeds incomplete")
    arrays = read_npz(root / audit["per_query_arrays"]["path"])
    for ctx, groups in audit["contexts"].items():
        for group, seeds in groups.items():
            context = root / group / f"ctx{ctx}_lctx{int(ctx) // 2}"
            manifest = read_json(context / "manifest.json")
            for seed, totals in seeds.items():
                prefix = f"ctx{ctx}_{group}_s{seed}_"
                arm_name = f"{'task_other_cycles' if group == 'task_cycles' else 'labeled_support_only'}_s{seed}"
                exposure = read_npz(context / manifest["arms"][arm_name]["exposure"])
                check(np.array_equal(arrays[prefix + "target_node"], exposure["target_node_idxs"]), "audit target order differs")
                changed = sum(arrays[prefix + "changed_" + kind] for kind in ("query", "visible_label", "hidden_label"))
                check(np.array_equal(changed, exposure["changed_edges"]), "audit exposure differs")
                for field in ("eligible", "changed"):
                    for kind in ("query", "visible_label", "hidden_label"):
                        check(int(arrays[prefix + field + "_" + kind].sum()) == totals[field][kind], "audit totals differ")
                check(totals["changed"]["query"] == 0, "audit query links changed")
                if group == "labeled_support":
                    check(totals["changed"]["hidden_label"] == 0, "audit hidden-label links changed")


def analyze(root, *, bootstrap_replicates=2000):
    root = Path(root)
    verify_checksums(root)
    check(all((root / name).is_file() for name in REQUIRED_RELEASE_FILES), "release missing required evidence/analysis files")
    verify_support_audit(root)
    catalog = read_json(root / "catalog.json")
    check(catalog["schema_version"] == 1 and set(catalog["runs"]) == set(EXPECTED) - {"primary"},
          "release catalog must contain exactly the six expected follow-up panels")
    check(catalog["baseline"] == "81e4fa4fdb98ea6ce5973ec4772f91dc4ca9a468", "release baseline differs")
    check(catalog["primary"] == "../../preliminary/2026-10-05_validated", "primary reference differs from fixed release")
    primary = (root / catalog["primary"]).resolve()
    verify_checksums(primary)
    runs = {"primary": {"path": primary, "state_file": "matrix.json", "database": "rel-f1", "task": "driver-dnf"}}
    runs.update({name: {**info, "path": root / name} for name, info in catalog["runs"].items()})
    rows, groups, checkpoints, support = [], [], set(), {}
    for name, info in runs.items():
        directory = info["path"]
        state = read_json(directory / info["state_file"])
        check(state["status"] == "complete" and state["complete"], "incomplete run")
        policy = state.get("availability_policy", state.get("policy"))
        check(policy["version"] == "query_available_incidence_v1", "availability policy differs")
        metadata_dir = root / "relations" if name == "primary" else directory
        panel = read_npz(metadata_dir / "panel.npz")
        verify_protocol(name, info, state, panel)
        n = len(panel["labels"])
        check(len(np.unique(panel["node_idxs"])) == n, "duplicate panel IDs")
        registry = verify_registry(directory, state) if name in ("arxiv", "trial") else None
        lineage = read_json(directory / "artifact_lineage.json") if name != "primary" else None
        lineage_paths = set()
        for ctx in (48, 128):
            context = f"ctx{ctx}_lctx{ctx // 2}"
            path = directory / context
            manifest = read_json(path / "manifest.json")
            validation = manifest.get("validation") or read_json(path / "validation.json")
            check(manifest["status"] == "complete" and manifest["complete"] and validation["status"] == "passed"
                  and validation["complete"], "incomplete or unvalidated context")
            expected = expected_arms(name)
            check(set(manifest["arms"]) == expected
                  and set(validation.get("per_arm", validation.get("arms", {}))) == expected, "expected arm/seed set differs")
            check(manifest.get("ctx", manifest.get("context")) == ctx, "context setting differs")
            if "n_queries" in manifest:
                check(manifest["n_queries"] == n, "manifest panel count differs")
            if name in ("primary", "driver_top3"):
                check(manifest["seeds"] == list(SEEDS) and manifest["fractions"] == [0.0, 0.5, 1.0]
                      and manifest["task_table"] == EXPECTED[name][1] and manifest["db_name"] == EXPECTED[name][0],
                      "matrix manifest task/seed/strength protocol differs")
            if "local_ctx" in manifest or "local_context" in manifest:
                check(manifest.get("local_ctx", manifest.get("local_context")) == ctx // 2, "local context differs")
            checkpoint = manifest.get("checkpoint_provenance", manifest["checkpoint"])
            identity = tuple(checkpoint[key]["sha256"] for key in ("weights", "config"))
            check(identity == PINNED_CHECKPOINT, "checkpoint differs from pinned protocol")
            checkpoints.add(identity)
            clean_path = path if "base" in manifest["arms"] else primary / context
            clean = read_npz(clean_path / "preds_base.npz")
            clean_manifest = read_json(clean_path / "manifest.json")
            clean_exposure = read_npz(clean_path / clean_manifest["arms"]["base"]["exposure"])
            context_registry = registry if registry is not None else manifest["relation_index"]
            if name in ("relations", "task_cycles", "labeled_support"):
                check(context_registry == clean_manifest["relation_index"], "control registry differs from primary reference")
                verify_f1_groups(name, state, context_registry)
            check(np.array_equal(clean["node_idxs"], panel["node_idxs"])
                  and np.array_equal(clean["labels"] > 0, panel["labels"]), "metadata/prediction alignment differs")
            if clean_path != path:
                reference = read_json(clean_path / "manifest.json")
                check(validation["batch_fingerprints"] == reference["validation"]["batch_fingerprints"],
                      "control sampled contexts differ from primary")
            loaded, scores_by_group, group_rows = {}, {}, {}
            for arm_name in arm_order(name):
                arm = manifest["arms"][arm_name]
                for field, sha in (("predictions", "prediction_sha256"), ("exposure", "exposure_sha256")):
                    filename = arm[field]
                    check(Path(filename).name == filename, "arm artifact must be in its context directory")
                    content = (path / filename).read_bytes()
                    check(hashlib.sha256(content).hexdigest() == arm[sha], "arm hash differs")
                    if lineage is not None:
                        key = f"{context}/{filename}"
                        lineage_paths.add(key)
                        check(key in lineage and lineage[key]["curated"]["sha256"] == arm[sha]
                              and lineage[key]["curated"]["bytes"] == len(content), "curation lineage differs")
                prediction, exposure = read_npz(path / arm["predictions"]), read_npz(path / arm["exposure"])
                assert_prediction_alignment(clean, prediction, count=n)
                check(np.array_equal(prediction["node_idxs"], validation["target_node_idxs"])
                      and np.array_equal(prediction["labels"], validation["labels"]), "validated prediction order differs")
                check(np.array_equal(exposure["target_node_idxs"], prediction["node_idxs"]), "exposure order differs")
                reconcile_exposure({**exposure, "is_prediction_query": np.ones(n, dtype=bool)}, arm["totals"])
                verify_accounting(exposure, validation, arm_name, arm, clean_exposure)
                check(np.all(exposure["configured_context_sizes"] == ctx)
                      and np.all((exposure["context_token_counts"] > 0) & (exposure["context_token_counts"] <= ctx)),
                      "context token budget differs")
                group = arm_name.rsplit("_s", 1)[0] if "_s" in arm_name else arm_name
                strength = 0 if group == "base" else 0.5 if group == "a050" else 1
                check(np.all(exposure["requested_strength"] == strength), "requested strength differs")
                if strength in (0, 1):
                    check(np.array_equal(exposure["changed_edges"], np.zeros(n, dtype=int) if strength == 0 else exposure["max_changed_edges"]),
                          "clean/full assignment counts differ")
                if name in ("primary", "driver_top3"):
                    matching_seed = SEEDS[0] if group == "base" else int(arm_name.rsplit("_s", 1)[1])
                    check(arm["requested_strength"] == strength and arm["matching_seed"] == matching_seed
                          and arm["seed"] == (None if group == "base" else matching_seed), "matrix arm strength/seed differs")
                check(set(exposure["relation_ids"].tolist()) <= {int(key) for key in context_registry}, "unknown relation ID")
                if "groups" in state and group != "base":
                    relations = state["groups"][group]
                    selected = np.isin(exposure["relation_ids"], relations)
                    check(not np.any(exposure["relation_changed_edges"][~selected])
                          and not np.any(exposure["relation_max_changed_edges"][~selected]), "non-selected relations changed")
                    if registry is not None:
                        check(arm["relation_ids"] == relations and arm["seed"] == int(arm_name.rsplit("_s", 1)[1]),
                              "replication arm relation/seed metadata differs")
                untouched = exposure["changed_edges"] == 0
                check(np.array_equal(prediction["preds"][untouched], clean["preds"][untouched]), "untouched scores differ")
                auc = float(roc_auc_score(panel["labels"], prediction["preds"]))
                check(abs(auc - binary_auroc(prediction["labels"], prediction["preds"])) < 1e-12, "independent AUROC differs")
                check(abs(auc - arm["auroc"]) < 1e-12, "manifest AUROC differs")
                row = {"run": name, "database": info["database"], "task": info["task"], "context": ctx,
                       "arm": arm_name, "group": group, "auroc": auc,
                       "clean_auroc": float(roc_auc_score(panel["labels"], clean["preds"])),
                       "changed_edges": int(exposure["changed_edges"].sum()),
                       "eligible_edges_all_relations": int(exposure["eligible_edges"].sum()),
                       "maximum_changed_edges_selected_pool": int(exposure["max_changed_edges"].sum()),
                       "affected_queries": int((~untouched).sum()),
                       "mean_abs_shift": float(np.abs(prediction["preds"].astype(float) - clean["preds"]).mean()),
                       "n_queries": n}
                rows.append(row)
                group_rows.setdefault(group, []).append(row)
                scores_by_group.setdefault(group, []).append(prediction["preds"])
                loaded[arm_name] = (prediction, exposure)
            for group, selected in group_rows.items():
                groups.append({"run": name, "database": info["database"], "task": info["task"], "context": ctx,
                               "group": group, "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
                               "clean_auroc": selected[0]["clean_auroc"],
                               "mean_abs_shift": float(np.mean([r["mean_abs_shift"] for r in selected])),
                               "changed_edges_by_seed": [r["changed_edges"] for r in selected],
                               "affected_queries_by_seed": [r["affected_queries"] for r in selected], "n_queries": n})
            if name == "labeled_support":
                scores = np.asarray(scores_by_group["labeled_support_only"])
                masks = [e["changed_edges"] > 0 for _, e in loaded.values()]
                check(all(np.array_equal(masks[0], mask) for mask in masks), "support coverage differs by seed")
                mask = masks[0]
                old_manifest = read_json(root / "task_cycles" / context / "manifest.json")
                old_scores = np.asarray([read_npz(root / "task_cycles" / context / old_manifest["arms"][f"task_other_cycles_s{s}"]["predictions"])["preds"]
                                         for s in (101, 202, 303)])
                def uncertainty(reference, panel=panel, scores=scores):
                    return clustered_auc_delta(panel["labels"], reference, scores, panel["dates"],
                                               panel["entities"], replicates=bootstrap_replicates)
                support[str(ctx)] = {"delta_clean_uncertainty": uncertainty(clean["preds"]),
                                     "delta_old_control_uncertainty": uncertainty(old_scores),
                                     "affected_subset": {"n": int(mask.sum()), "positive": int(panel["labels"][mask].sum()),
                                         "clean_auroc": float(roc_auc_score(panel["labels"][mask], clean["preds"][mask])),
                                         "support_auroc_mean": float(np.mean([roc_auc_score(panel["labels"][mask], s[mask]) for s in scores])),
                                         "old_auroc_mean": float(np.mean([roc_auc_score(panel["labels"][mask], s[mask]) for s in old_scores])),
                                         "support_mean_abs_shift": float(np.abs(scores[:, mask] - clean["preds"][mask]).mean()),
                                         "old_mean_abs_shift": float(np.abs(old_scores[:, mask] - clean["preds"][mask]).mean())}}
        if lineage is not None:
            check(set(lineage) == lineage_paths, "curation lineage missing or extra files")
    check(len(rows) == 110, "release does not contain the expected 110 prediction conditions")
    check(len(checkpoints) == 1, "checkpoint differs across panels")
    return {"status": "verified", "independent_auroc": "sklearn.metrics.roc_auc_score versus project rank-sum implementation",
            "checkpoint_hashes": list(next(iter(checkpoints))), "prediction_files": len(rows), "rows": rows,
            "groups": groups, "labeled_support": support}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Independently recalculated saved results", "", "Detailed metrics and support contrasts: [analysis.json](analysis.json).", "",
             f"Verified {result['prediction_files']} prediction files.", "",
             "AUROC averages per-seed AUROCs, not averaged predictions. Counts/coverage list seeds in order 101/202/303.",
             "Eligible counts in the arrays cover all relations; maximum counts apply to the selected pool.", "",
             "| Panel | Context | Group | Clean AUROC | AUROC mean | Mean abs shift | Changed edges | Affected queries |",
             "|---|---:|---|---:|---:|---:|---|---|"]
    for row in result["groups"]:
        lines.append(f"| {row['run']} | {row['context']} | {row['group']} | {row['clean_auroc']:.4f} | {row['auroc_mean']:.4f} | "
                     f"{row['mean_abs_shift']:.4f} | {row['changed_edges_by_seed']} | {row['affected_queries_by_seed']} |")
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")


if __name__ == "__main__":
    main()
