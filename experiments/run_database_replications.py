"""Frozen 512-query task/entity-versus-database replication, not a search.

Complete DB tables are retained. Explicit target nodes are selected uniformly
without labels; the same sampled contexts, availability policy and checkpoint
are used for clean and the two full-matching relation controls.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_clean_rewire import BaseWrap, RewireWrap, load_frozen_model, save_json

from rfm_structure.data import load_availability_policy, load_relation_index
from rfm_structure.metrics import binary_auroc
from rfm_structure.provenance import (
    file_identity,
    preprocessing_identity,
    software_versions,
    source_identity,
)
from rfm_structure.replication import (
    replication_evaluator,
    replication_groups,
    selected_query_ids,
    validate_embedding_scope,
)
from rfm_structure.validation import (
    EDGE_FIELDS,
    attach_availability,
    available_context,
    batch_fingerprint,
    capture_evaluator_masks,
    evaluator_query_arrays,
    numpy_batch,
    real_query_batch,
    rewire_and_validate_batch,
    write_exposure_npz,
)

TASKS = {"rel-arxiv": ("paper-citation", "papers", 182),
         "rel-trial": ("study-outcome", "studies", 365)}
SEEDS = (101, 202, 303)
CHECKPOINT_SHA = "4a369f065d8f69e1da11ee4326ceb9d32cf78ee69255a438449e971f681b7f83"


def arms(groups):
    return [("base", 0.0, 101, None)] + [
        (f"{group}_s{seed}", 1.0, seed, ids)
        for group, ids in groups.items() for seed in SEEDS
    ]


def validate_panel(ev, ids, groups, policy, directory, scope):
    report = {"status": "in_progress", "complete": False, "target_node_idxs": [], "labels": [],
              "batch_fingerprints": [], "availability_totals": {}, "per_arm": {}}
    batches = {name: [] for name, _, _, _ in arms(groups)}
    try:
        for raw in ev.eval_loaders[ev.tasks[0]]:
            arrays = numpy_batch(raw)
            if not all(np.isfinite(arrays[key]).all() for key in ("text_values", "col_name_values")):
                raise ValueError("frozen panel touched an unencoded/nonfinite embedding")
            if scope and batch_fingerprint({k: v for k, v in arrays.items() if k not in ("text_values", "col_name_values")}) != scope["context_fingerprints"][str(ev.ctx_sizes[0])]:
                raise ValueError("sampling differs from the scoped embedding coverage pass")
            real, _ = real_query_batch(raw)
            base, availability = available_context(real, policy)
            report["batch_fingerprints"].append(batch_fingerprint(base))
            target_ids, labels, _ = evaluator_query_arrays(base)
            report["target_node_idxs"].extend(target_ids.tolist())
            report["labels"].extend(labels.tolist())
            for key, value in availability.items():
                report["availability_totals"][key] = report["availability_totals"].get(key, 0) + int(value.sum())
            for name, alpha, seed, relations in arms(groups):
                evidence = rewire_and_validate_batch(base, seed, alpha, policy, relations)
                if evidence["errors"] or evidence["base_temporal"]["unknown"] or evidence["temporal"]["unknown"]:
                    raise ValueError(f"{name}: structural/temporal failure: {evidence['errors'][:20]}")
                attach_availability(evidence["exposure"], availability)
                batches[name].append(evidence["exposure"])
        if not np.array_equal(report["target_node_idxs"], ids):
            raise ValueError("not exactly the frozen target set/order")
        if len(set(np.asarray(report["labels"]) > 0)) != 2:
            raise ValueError("frozen selected panel has only one class")
        for name, parts in batches.items():
            exposure = write_exposure_npz(directory / f"validation_exposure_{name}.npz", parts)
            report["per_arm"][name] = {
                "totals": {key: int(exposure[key].sum()) for key in EDGE_FIELDS},
                "affected_queries": int((exposure["changed_edges"] > 0).sum()),
                "known_violations": int(exposure["temporal_known_violations"].sum()),
                "unknown": int(exposure["temporal_unknown"].sum()),
                "declared_timeless": int(exposure["temporal_declared_timeless"].sum()),
            }
        report.update(status="passed", complete=True)
    except Exception as error:
        report.update(status="failed", errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        save_json(directory / "validation.json", report)
    return report


def run_replication(pre_dir, database, checkpoint, output, *, dry_run=False):
    import pandas as pd
    from rt.tasks import tasks_from_preprocessed

    task_name, entity, horizon = TASKS[database]
    tasks = [t for t in tasks_from_preprocessed(str(pre_dir), splits=("test",), dbs=[database]) if t.table_name == task_name]
    if len(tasks) != 1 or tasks[0].task_type != "clf":
        raise ValueError("expected exactly the frozen classification task")
    task = tasks[0]
    policy = load_availability_policy(pre_dir, database)
    registry = load_relation_index(pre_dir, database)
    schema_tables = {info["table"].rsplit(":", 1)[0] for info in policy["tables"] if info["table"].endswith(":Db")}
    groups = replication_groups(registry, task_name, entity, schema_tables)
    ids = selected_query_ids(pre_dir, task)
    task_info = next(info for info in policy["tables"] if info["table"] == f"{task_name}:Test")
    if task_info["horizon_seconds"] != horizon * 86400:
        raise ValueError("task horizon differs from frozen protocol")
    meta = json.loads((pre_dir / database / "meta.json").read_text())
    raw = Path(meta["source"])
    task_rows = pd.read_parquet(raw / "tasks" / task_name / "test.parquet").iloc[ids - task_info["start"]]
    import yaml

    task_spec = yaml.safe_load((raw / "tasks" / task_name / "manifest.yaml").read_text())
    raw_labels = task_rows[task_spec["target_col"]].to_numpy().astype(int)
    scope_path = pre_dir / database / "embedding_scope.json"
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else None
    validate_embedding_scope(pre_dir, database, scope)
    output.mkdir(exist_ok=False)
    status = {"status": "in_progress", "complete": False, "database": database, "task": task_name,
              "protocol": "frozen_task_entity_vs_database_v1", "selection_seed": 20261007,
              "query_ids": ids.tolist(), "n_queries": len(ids), "official_test_size": task_info["stop"] - task_info["start"],
              "matching_seeds": list(SEEDS), "groups": groups, "horizon_days": horizon,
              "availability_policy": policy, "embedding_scope": scope, "contexts": {},
              "provenance": {"source": source_identity(Path(__file__).resolve().parents[1]),
                             "software": software_versions(), "preprocessing": preprocessing_identity(pre_dir, database)},
              "panel": {"positive": int(raw_labels.sum()), "dates": int(task_rows[task_spec["time_col"]].nunique()),
                        "entities": int(task_rows[task_spec["entity_col"]].nunique())}}
    save_json(output / "replication.json", status)
    try:
        evaluators, validated = {}, {}
        for ctx in (48, 128):
            directory = output / f"ctx{ctx}_lctx{ctx // 2}"
            directory.mkdir()
            ev = replication_evaluator(pre_dir, task, ids, ctx)
            report = validate_panel(ev, ids, groups, policy, directory, scope)
            if not np.array_equal(np.asarray(report["labels"]) > 0, raw_labels.astype(bool)):
                raise ValueError("raw task labels differ from normalized evaluator labels")
            evaluators[ctx], validated[ctx] = ev, report
            status["contexts"][str(ctx)] = "validated"
            save_json(output / "replication.json", status)
            print(database, ctx, "validated all 512 queries and seven arms", flush=True)
        if dry_run:
            status.update(status="validated", complete=True)
            return status
        for ctx, ev in evaluators.items():
            directory = output / f"ctx{ctx}_lctx{ctx // 2}"
            started = time.monotonic()
            model, identity = load_frozen_model(checkpoint, "cuda")
            if identity["weights"]["sha256"] != CHECKPOINT_SHA:
                raise ValueError("not the frozen RT-PluRel checkpoint")
            ev.device = "cuda"
            holder = SimpleNamespace(batch=None, batch_index=0, forward_calls=0,
                                     fingerprints=validated[ctx]["batch_fingerprints"], availability_policy=policy)
            capture_evaluator_masks(ev, holder)
            wrappers = [("base", BaseWrap(model, holder, 101, 8))]
            for name, _, seed, relations in arms(groups)[1:]:
                if validated[ctx]["per_arm"][name]["totals"]["changed_edges"]:
                    wrappers.append((name, RewireWrap(model, seed, 1.0, holder, 8, relations)))
            outputs = list(ev.evaluate_raw([(wrapper, name) for name, wrapper in wrappers], [ctx], with_node_idxs=True))
            if len(outputs) != 1:
                raise ValueError("expected one task/context output")
            _, _, labels, scores, _, returned_ids = outputs[0]
            if not np.array_equal(returned_ids, ids) or not np.array_equal(labels, validated[ctx]["labels"]):
                raise ValueError("paired inference differs from validated target/label order")
            if holder.batch_index != len(holder.fingerprints):
                raise ValueError("not every validated batch was replayed")
            manifest = {"status": "in_progress", "complete": False, "database": database, "task": task_name,
                        "context": ctx, "local_context": ctx // 2, "checkpoint": identity, "arms": {}}
            runtime_wrappers = dict(wrappers)
            for name, alpha, seed, relations in arms(groups):
                path = directory / f"preds_{name}.npz"
                predictions = np.asarray(scores[name] if name in scores else scores["base"], dtype=float)
                if predictions.shape != (len(ids),) or not np.isfinite(predictions).all():
                    raise ValueError("invalid replication scores")
                with path.open("xb") as handle:
                    np.savez(handle, labels=labels, preds=predictions, node_idxs=ids)
                reference_path = directory / f"validation_exposure_{name}.npz"
                with np.load(reference_path, allow_pickle=False) as expected:
                    if name in runtime_wrappers:
                        runtime = write_exposure_npz(directory / f"exposure_{name}.npz", runtime_wrappers[name].exposure_batches)
                        if set(runtime) != set(expected.files) or any(not np.array_equal(runtime[key], expected[key], equal_nan=True) for key in expected.files):
                            raise ValueError("runtime exposure differs from validation")
                    untouched = expected["changed_edges"] == 0
                    if not np.array_equal(predictions[untouched], np.asarray(scores["base"])[untouched]):
                        raise ValueError("predictions changed without changed FK assignments")
                manifest["arms"][name] = {"seed": None if alpha == 0 else seed, "relation_ids": relations,
                                          "predictions": path.name, "prediction_sha256": file_identity(path)["sha256"],
                                          "exposure": reference_path.name, "exposure_sha256": file_identity(reference_path)["sha256"],
                                          "auroc": binary_auroc(labels, predictions),
                                          **validated[ctx]["per_arm"][name]}
            manifest.update(status="complete", complete=True, seconds=time.monotonic() - started,
                            model_forward_attempts=holder.forward_calls)
            save_json(directory / "manifest.json", manifest)
            status["contexts"][str(ctx)] = "complete"
            del model, wrappers
        status.update(status="complete", complete=True)
    except Exception as error:
        status.update(status="failed", complete=False, errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        save_json(output / "replication.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=TASKS, required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = Path("results/runs") / f"replication_{args.database}_{time.time_ns()}"
    run_replication(args.pre_dir, args.database, args.checkpoint_dir, output, dry_run=args.dry_run)
    print(output)


if __name__ == "__main__":
    main()
