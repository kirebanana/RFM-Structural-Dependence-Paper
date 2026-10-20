"""Exploratory paired relation controls on the already validated rel-f1 panel.

Reuse clean/all-relations predictions only after exact sampled-input replay and
checkpoint checks. Relation/cycle controls restrict the existing assignment;
visible-support controls rematch within a selected source pool. Queries,
sampled cells and common availability preparation are unchanged.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_clean_rewire import (
    BaseWrap,
    RewireWrap,
    build_tasks,
    load_frozen_model,
    save_json,
)

from rfm_structure.analysis import analyze_matrix
from rfm_structure.data import (
    load_availability_policy,
    load_expected_target_ids,
    load_relation_index,
)
from rfm_structure.metrics import binary_auroc
from rfm_structure.provenance import (
    file_identity,
    preprocessing_identity,
    software_versions,
    source_identity,
)
from rfm_structure.relation_controls import relation_groups
from rfm_structure.validation import (
    EDGE_FIELDS,
    attach_availability,
    available_context,
    batch_fingerprint,
    capture_evaluator_masks,
    configure_complete_evaluation,
    evaluator_query_arrays,
    real_query_batch,
    rewire_and_validate_batch,
    visible_support_source_mask,
    write_exposure_npz,
)

SEEDS = (101, 202, 303)
CYCLE_MODES = {"task_query_cycles": "touching_query", "task_other_cycles": "excluding_query"}
SOURCE_SELECTIONS = {"labeled_support_only": "visible_labeled_support"}


def read_predictions(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


class CachedClean(BaseWrap):
    """Prepare the validated clean batch without repeating clean RT inference."""

    def __init__(self, model, holder, cache):
        super().__init__(model, holder, SEEDS[0], inference_batch_size=8)
        self.cache = dict(zip(cache["node_idxs"].tolist(), cache["preds"].tolist(), strict=True))

    def predict_checked(self, batch, expected, eval_ctx_sizes, device, task, bool_as_num):
        import torch

        ids, _, _ = evaluator_query_arrays(expected)
        predictions = torch.zeros(len(self.holder.batch_mask), dtype=torch.float32)
        predictions[self.holder.batch_mask] = torch.tensor([self.cache[int(node)] for node in ids])
        return {ctx: predictions for ctx in eval_ctx_sizes}


def validate_groups(ev, groups, policy, reference, directory):
    """Validate every control before weights load and persist compact exposure."""
    report = {"status": "in_progress", "complete": False, "arms": {},
              "target_node_idxs": [], "labels": [], "batch_fingerprints": []}
    evidence_batches = {f"{group}_s{seed}": [] for group in groups for seed in SEEDS}
    try:
        loader = configure_complete_evaluation(ev, 702)
        for raw in loader:
            real, _ = real_query_batch(raw)
            base, counts = available_context(real, policy)
            report["batch_fingerprints"].append(batch_fingerprint(base))
            ids, labels, _ = evaluator_query_arrays(base)
            report["target_node_idxs"].extend(ids.tolist())
            report["labels"].extend(labels.tolist())
            for seed in SEEDS:
                full = rewire_and_validate_batch(base, seed, 1.0, policy)
                if full["errors"] or full["temporal"]["unknown"]:
                    raise ValueError("all-relations reference no longer validates")
                for group, relations in groups.items():
                    name = f"{group}_s{seed}"
                    source_mask = visible_support_source_mask(base) if group in SOURCE_SELECTIONS else None
                    evidence = rewire_and_validate_batch(base, seed, 1.0, policy, relations, CYCLE_MODES.get(group), source_mask)
                    if evidence["errors"] or evidence["temporal"]["unknown"]:
                        raise ValueError(f"{name}: {evidence['errors']}")
                    selected = np.isin(base["f2p_rel_idxs"], relations)
                    changed = evidence["batch"]["f2p_nbr_idxs"] != base["f2p_nbr_idxs"]
                    compared = changed if group in CYCLE_MODES else selected
                    if group not in SOURCE_SELECTIONS and not np.array_equal(evidence["batch"]["f2p_nbr_idxs"][compared],
                                          full["batch"]["f2p_nbr_idxs"][compared]):
                        raise ValueError("targeted assignment differs from all-relations matching")
                    attach_availability(evidence["exposure"], counts)
                    evidence_batches[name].append(evidence["exposure"])
        for key in ("target_node_idxs", "labels", "batch_fingerprints"):
            if report[key] != reference["validation"][key]:
                raise ValueError(f"{key} differs from the original validated panel")
        for name, batches in evidence_batches.items():
            exposure = write_exposure_npz(directory / f"validation_exposure_{name}.npz", batches)
            report["arms"][name] = {"totals": {key: int(exposure[key].sum()) for key in EDGE_FIELDS},
                                   "unchanged_queries": int((exposure["changed_edges"] == 0).sum())}
        report.update(status="passed", complete=True)
    except Exception as error:
        report.update(status="failed", errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        save_json(directory / "validation.json", report)
    return report


def run_controls(evaluators, pre_dir, checkpoint_dir, reference_root, out, *, dry_run=False, group_names=None):
    """Pair new controls with existing clean/full evidence; never overwrite it."""
    analyze_matrix(reference_root)
    out.mkdir(exist_ok=False)
    registry = load_relation_index(pre_dir, "rel-f1")
    policy = load_availability_policy(pre_dir)
    groups = relation_groups(registry)
    groups.update({name: groups["task_driver"] for name in CYCLE_MODES})
    groups.update({name: groups["task_driver"] for name in SOURCE_SELECTIONS})
    if group_names is not None:
        if not group_names or len(set(group_names)) != len(group_names) or not set(group_names).issubset(groups):
            raise ValueError("groups must be unique known relation controls")
        groups = {name: groups[name] for name in group_names}
    status = {"status": "in_progress", "complete": False, "evidence_kind": "exploratory_relation_controls",
              "reference": str(reference_root.resolve()), "groups": groups, "seeds": list(SEEDS),
              "policy": policy, "cycle_modes": {name: CYCLE_MODES.get(name) for name in groups},
              "source_selection": {name: SOURCE_SELECTIONS.get(name) for name in groups},
              "matching_rule": "source-selected controls rematch within their selected pool; others restrict full assignment",
              "contexts": {},
              "provenance": {"source": source_identity(Path(__file__).resolve().parents[1]),
                             "software": software_versions(), "preprocessing": preprocessing_identity(pre_dir, "rel-f1")}}
    try:
        references, validated = {}, {}
        for ctx, ev in evaluators.items():
            directory = out / f"ctx{ctx}_lctx{ctx // 2}"
            directory.mkdir()
            reference = json.loads((reference_root / directory.name / "manifest.json").read_text())
            if reference["validation"]["availability_policy"] != json.loads((reference_root / "matrix.json").read_text())["availability_policy"]:
                raise ValueError("reference availability policies differ")
            # Curated policies omit local file paths; compare scientific ranges/rules.
            if any(policy[key] != reference["validation"]["availability_policy"][key]
                   for key in ("version", "tables", "cutoff_rule", "timeless_rule", "fanout_rule", "refill_removed_cells")):
                raise ValueError("new availability policy differs from reference")
            expected = load_expected_target_ids(pre_dir, ev.tasks[0], 702)
            if not np.array_equal(np.sort(reference["validation"]["target_node_idxs"]), expected):
                raise ValueError("reference target set differs from dataset")
            references[ctx] = reference
            validated[ctx] = validate_groups(ev, groups, policy, reference, directory)
            status["contexts"][str(ctx)] = {"status": "validated"}
        if dry_run:
            status.update(status="validated", complete=True)
            return status
        for ctx, ev in evaluators.items():
            directory = out / f"ctx{ctx}_lctx{ctx // 2}"
            reference_directory = reference_root / directory.name
            reference = references[ctx]
            started = time.time()
            model, identity = load_frozen_model(checkpoint_dir, "cuda")
            if any(identity[key]["sha256"] != reference["checkpoint_provenance"][key]["sha256"]
                   for key in ("weights", "config")):
                raise ValueError("checkpoint differs from cached clean/full reference")
            cache = read_predictions(reference_directory / "preds_base.npz")
            holder = SimpleNamespace(batch=None, batch_index=0, forward_calls=0,
                                     fingerprints=validated[ctx]["batch_fingerprints"], availability_policy=policy)
            capture_evaluator_masks(ev, holder)
            wrappers = [("base", CachedClean(model, holder, cache))]
            reused = {}
            for group, relations in groups.items():
                for seed in SEEDS:
                    name = f"{group}_s{seed}"
                    if validated[ctx]["arms"][name]["totals"]["changed_edges"] == 0:
                        reused[name] = "base"  # Sparse/no-change control needs no forward.
                    else:
                        wrappers.append((name, RewireWrap(model, seed, 1.0, holder, 8, relations, CYCLE_MODES.get(group), SOURCE_SELECTIONS.get(group))))
            import torch

            with torch.inference_mode():
                outputs = list(ev.evaluate_raw([(wrapper, name) for name, wrapper in wrappers], [ctx], with_node_idxs=True))
            if len(outputs) != 1:
                raise ValueError("expected exactly one context/task output")
            _, _, labels, scores, _, ids = outputs[0]
            if not np.array_equal(ids, cache["node_idxs"]) or not np.array_equal(labels, cache["labels"]):
                raise ValueError("new controls differ in cached target/label pairing")
            if holder.batch_index != len(holder.fingerprints):
                raise ValueError("inference omitted a validated sampled batch")
            manifest = {"status": "in_progress", "complete": False, "ctx": ctx,
                        "checkpoint": identity, "relation_index": registry, "arms": {},
                        "reference_clean": file_identity(reference_directory / "preds_base.npz")}
            for name, wrapper in wrappers:
                if name == "base":
                    continue
                arrays = write_exposure_npz(directory / f"exposure_{name}.npz", wrapper.exposure_batches)
                with np.load(directory / f"validation_exposure_{name}.npz") as checked:
                    if set(checked.files) != set(arrays) or any(not np.array_equal(checked[key], arrays[key], equal_nan=True) for key in checked.files):
                        raise ValueError("relation control exposure differs from validation")
            for group in groups:
                for seed in SEEDS:
                    name = f"{group}_s{seed}"
                    prediction = cache["preds"] if name in reused else np.asarray(scores[name], dtype=float)
                    if prediction.shape != (702,) or not np.isfinite(prediction).all():
                        raise ValueError("invalid relation-control scores")
                    path = directory / f"preds_{name}.npz"
                    with path.open("xb") as handle:
                        np.savez(handle, labels=labels, preds=prediction, node_idxs=ids)
                    manifest["arms"][name] = {"auroc": binary_auroc(labels, prediction),
                                              "predictions": path.name, "prediction_sha256": file_identity(path)["sha256"],
                                              "exposure": f"validation_exposure_{name}.npz",
                                              "exposure_sha256": file_identity(directory / f"validation_exposure_{name}.npz")["sha256"],
                                              "reused_clean_no_changes": name in reused,
                                              **validated[ctx]["arms"][name]}
            manifest.update(status="complete", complete=True, elapsed_sec=time.time() - started,
                            model_forward_attempts=holder.forward_calls)
            save_json(directory / "manifest.json", manifest)
            status["contexts"][str(ctx)] = {"status": "complete", "elapsed_sec": manifest["elapsed_sec"]}
            del model, wrappers
        status.update(status="complete", complete=True)
    except Exception as error:
        status.update(status="failed", complete=False, errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        save_json(out / "controls.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/preliminary/2026-10-05_validated"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--groups", default="task_driver,database_all,database_driver,database_constructor,database_race,race_circuit")
    args = parser.parse_args()
    from rt.eval_utils import build_evaluator

    tasks = build_tasks(str(args.pre_dir))
    evaluators = {ctx: build_evaluator(tasks, str(args.pre_dir), embedding_model="all-MiniLM-L12-v2", d_text=384,
                                      device="cpu" if args.dry_run else "cuda", ctx_size=ctx, local_ctx_size=ctx // 2,
                                      items_per_task=702, num_workers=0, context_seed=0, shuffle_seed=0)
                  for ctx in (48, 128)}
    out = Path("results/runs") / f"relation_controls_{time.time_ns()}"
    run_controls(evaluators, args.pre_dir, args.checkpoint_dir, args.reference, out,
                 dry_run=args.dry_run, group_names=args.groups.split(","))
    print(out)


if __name__ == "__main__":
    main()
