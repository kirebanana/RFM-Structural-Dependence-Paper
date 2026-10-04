"""Validate and run the paired RT-PluRel corruption matrix on rel-f1/driver-dnf.

No weights are loaded in --dry-run. Full runs validate every context/arm first,
then require adequate resources and a local checkpoint: no implicit downloads.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rfm_structure.data import load_expected_target_ids, load_relation_index
from rfm_structure.metrics import binary_auroc
from rfm_structure.provenance import (
    file_identity,
    preprocessing_identity,
    software_versions,
    source_identity,
)
from rfm_structure.validation import (
    EDGE_FIELDS,
    TEMPORAL_FIELDS,
    assert_held_fixed,
    batch_fingerprint,
    capture_evaluator_masks,
    configure_complete_evaluation,
    experiment_arms,
    numpy_batch,
    real_query_batch,
    rewire_and_validate_batch,
    validate_batches,
    write_exposure_npz,
)

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
CKPT = "stanford-star/rt-plurel/classification"
DB_NAME = "rel-f1"
TASK_TABLE = "driver-dnf"
ITEMS_PER_TASK = 702
STRENGTH_RULE = "whole cycles in shuffled-index order; strict distance improvement toward Python round-to-even(alpha * max changed IDs); ties unchanged"


def save_json(path, content):
    """Replace only this run's status file atomically; directories are exclusive."""
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)
    temporary.replace(path)


class BaseWrap:
    """Check replay identity and record clean exposure before predicting once."""

    def __init__(self, inner, holder, seed):
        self.inner = inner
        self.holder = holder
        self.seed = seed
        self.stats = {}
        self.exposure_batches = []

    def eval(self):
        self.inner.eval()

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        real, _ = real_query_batch({**batch, "batch_mask": self.holder.batch_mask})
        index = self.holder.batch_index
        if index >= len(self.holder.fingerprints) or batch_fingerprint(real) != self.holder.fingerprints[index]:
            raise ValueError("sampled input differs from the validated batch replay")
        self.holder.batch_index += 1
        self.holder.batch = real
        self.record(rewire_and_validate_batch(real, self.seed, 0.0))
        return self.predict_checked(batch, real, eval_ctx_sizes, device, task, bool_as_num)

    def predict_checked(self, batch, expected, eval_ctx_sizes, device, task, bool_as_num):
        """Count the forward and ensure prediction leaves sampled inputs intact."""
        self.holder.forward_calls += 1
        predictions = self.inner.predict(batch, eval_ctx_sizes, device, task, bool_as_num)
        after, _ = real_query_batch({**batch, "batch_mask": self.holder.batch_mask})
        if batch_fingerprint(after) != batch_fingerprint(expected):
            raise ValueError("model prediction mutated its sampled input")
        return predictions

    def record(self, evidence):
        if evidence["errors"] or evidence["base_temporal"]["unknown"] or evidence["temporal"]["unknown"]:
            raise ValueError(f"structural/temporal validation failed or uncertain: {evidence['errors'][:20]}")
        for field in EDGE_FIELDS:
            self.stats[field] = self.stats.get(field, 0) + evidence["totals"].get(field, 0)
        self.exposure_batches.append(evidence["exposure"])


class RewireWrap(BaseWrap):
    """Only FK parents change; preserve evaluator phantom prediction shapes."""

    def __init__(self, inner, seed, alpha, holder):
        super().__init__(inner, holder, seed)
        self.alpha = alpha

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        real, _ = real_query_batch({**batch, "batch_mask": self.holder.batch_mask})
        assert_held_fixed(self.holder.batch, real)
        if not np.array_equal(self.holder.batch["f2p_nbr_idxs"], real["f2p_nbr_idxs"]):
            raise ValueError("arm did not receive clean FK parents")
        evidence = rewire_and_validate_batch(real, self.seed, self.alpha)
        self.record(evidence)
        full = numpy_batch({"nbr": batch["f2p_nbr_idxs"]})["nbr"]
        full[self.holder.batch_mask] = evidence["batch"]["f2p_nbr_idxs"]
        parent_tensor = batch["f2p_nbr_idxs"]
        if hasattr(parent_tensor, "detach"):
            import torch

            full = torch.from_numpy(full).to(dtype=parent_tensor.dtype, device=parent_tensor.device)
        model_input = {**batch, "f2p_nbr_idxs": full}
        return self.predict_checked(model_input, evidence["batch"], eval_ctx_sizes, device, task, bool_as_num)


def build_tasks(pre_dir):
    from rt.tasks import tasks_from_preprocessed

    tasks = [task for task in tasks_from_preprocessed(pre_dir, splits=("test",), dbs=[DB_NAME]) if task.table_name == TASK_TABLE]
    if len(tasks) != 1:
        raise ValueError("expected exactly one driver-dnf test classification task")
    return tasks


def dry_run(ev, pre_dir, seeds, out_dir, local_ctx, fractions=None):
    """Persist full coverage/evidence; exits 0 passed, 1 failed, 2 uncertain."""
    report_path = Path(out_dir) / "validation.json"
    report = {"status": "in_progress", "complete": False, "inference_ready": False,
              "ctx": int(ev.ctx_sizes[0]), "local_ctx": int(local_ctx), "seeds": list(seeds)}
    with report_path.open("x") as handle:
        json.dump(report, handle)
    started = time.time()
    try:
        expected = load_expected_target_ids(pre_dir, ev.tasks[0], ITEMS_PER_TASK)
        relations = load_relation_index(pre_dir, DB_NAME)
        loader = configure_complete_evaluation(ev, len(expected))
        report, exposures = validate_batches(loader, expected, seeds, ev.ctx_sizes[0], fractions)
        report.update(db_name=DB_NAME, task_table=TASK_TABLE, pre_dir=str(pre_dir), local_ctx=int(local_ctx),
                      relation_index=relations, eval_bs=int(ev.eval_bs), scheduled_batches=len(loader.dataset),
                      evaluator_batch_cap=None, strength_rule=STRENGTH_RULE)
        if report["complete"]:
            for name, batches in exposures.items():
                suffix = f"rw{name}" if isinstance(name, int) else name
                arrays = write_exposure_npz(Path(out_dir) / f"validation_exposure_{suffix}.npz", batches)
                if not np.array_equal(arrays["target_node_idxs"], report["target_node_idxs"]):
                    raise ValueError(f"arm {name}: saved target order differs")
                if not set(arrays["relation_ids"].tolist()).issubset(relations):
                    raise ValueError(f"arm {name}: unknown relation IDs")
                for key in TEMPORAL_FIELDS:
                    if int(arrays[f"temporal_{key}"].sum()) != report["per_arm"][str(name)]["temporal"][key]:
                        raise ValueError(f"arm {name}: temporal exposure mismatch for {key}")
                    if int(arrays[f"base_temporal_{key}"].sum()) != report["base_temporal"][key]:
                        raise ValueError(f"arm {name}: base temporal exposure mismatch for {key}")
        report["inference_ready"] = report["status"] == "passed"
    except Exception as error:  # noqa: BLE001 -- preserve incomplete evidence on sampler/writer errors
        report.update(status="failed", complete=False, inference_ready=False)
        report.setdefault("errors", []).append(f"{type(error).__name__}: {error}")
    report["elapsed_sec"] = time.time() - started
    save_json(report_path, report)
    print(f"[validation] {report['status']}: {report_path}", flush=True)
    return {"passed": 0, "temporal_uncertainty": 2}.get(report["status"], 1)


def load_frozen_model(checkpoint_dir, device):
    """Load local weights only, after real validation and resource checks."""
    import torch
    from rt import RelationalTransformer

    checkpoint = Path(checkpoint_dir).expanduser()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    weights = checkpoint / config.get("checkpoint_file", "model.safetensors")
    if not weights.is_file():
        raise ValueError("compatible local classification checkpoint weights are required")
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("real inference requires a suitable CUDA runtime; CPU validation remains available")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise ValueError("insufficient CUDA capability for RT bfloat16 inference")
    # A preceding context can leave unoccupied allocator memory reserved.
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    required = max(8 * 2**30, 3 * weights.stat().st_size)
    if free < required:
        raise ValueError("insufficient CUDA capability/free memory for RT bfloat16 inference")
    available_ram = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))) * 1024
    if available_ram < max(2 * 2**30, 2 * weights.stat().st_size):
        raise ValueError("insufficient available host RAM for checkpoint loading and RT inference")
    model = RelationalTransformer.from_pretrained(str(checkpoint), device=device).to(torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # RT's evaluate_raw already runs predictions inside torch.inference_mode.
    return model, {"name": CKPT, "weights": file_identity(weights), "config": file_identity(config_path),
                      "local_revision": next((part for index, part in enumerate(checkpoint.parts)
                                              if index and checkpoint.parts[index - 1] == "snapshots"
                                              and len(part) == 40 and all(char in "0123456789abcdef" for char in part)), None),
                      "device": torch.cuda.get_device_name(), "capability": list(capability),
                      "free_vram_bytes_before_load": free, "total_vram_bytes": total}


def run(ev, pre_dir, seeds, out_dir, device, local_ctx, *, fractions=(0.0, 0.5, 1.0),
        validation_report=None, checkpoint_dir=None, model_factory=None, provenance=None):
    """Technically gated inference; factory injection is only for lightweight tests."""
    out_dir = Path(out_dir)
    if validation_report is None:
        rc = dry_run(ev, pre_dir, seeds, out_dir, local_ctx, fractions)
        if rc:
            raise ValueError("inference blocked: validation failed or timestamps remain uncertain")
        validation_report = json.loads((out_dir / "validation.json").read_text())
    manifest_path = out_dir / "manifest.json"
    manifest = {"status": "in_progress", "complete": False, "db_name": DB_NAME, "task_table": TASK_TABLE,
                "ctx": int(ev.ctx_sizes[0]), "local_ctx": int(local_ctx), "seeds": list(seeds),
                "fractions": list(fractions), "checkpoint": CKPT, "validation": validation_report,
                "relation_index": load_relation_index(pre_dir, DB_NAME), "strength_rule": STRENGTH_RULE,
                "provenance": provenance or {}, "evidence_kind": "synthetic_test" if model_factory else "real"}
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle)
    started = time.time()
    holder = None
    try:
        arms = experiment_arms(seeds, fractions)
        if arms[0][0] != "base":
            raise ValueError("inference requires a clean arm first")
        if validation_report["status"] != "passed" or not validation_report["complete"]:
            raise ValueError("complete technically passed validation is required")
        if set(validation_report["per_arm"]) != {name for name, _, _ in arms}:
            raise ValueError("validation arm set differs from inference")
        for name, alpha, seed in arms:
            checked = validation_report["per_arm"][name]
            if checked["requested_strength"] != alpha or checked["matching_seed"] != seed:
                raise ValueError("validation strength/seed differs from inference")
            if alpha > 0 and checked["totals"]["changed_edges"] == 0:
                raise ValueError(f"arm {name}: no realized corruption; inference would be uninformative")
        configure_complete_evaluation(ev, ITEMS_PER_TASK)
        if model_factory:
            model = model_factory()
            manifest["checkpoint_provenance"] = {"test_only": True}
        else:
            if checkpoint_dir is None:
                raise ValueError("a local checkpoint is required; weights are never implicitly downloaded")
            model, manifest["checkpoint_provenance"] = load_frozen_model(checkpoint_dir, device)
        holder = SimpleNamespace(batch=None, batch_index=0, forward_calls=0, fingerprints=validation_report["batch_fingerprints"])
        capture_evaluator_masks(ev, holder)
        wrappers = [(name, BaseWrap(model, holder, seed) if alpha == 0 else RewireWrap(model, seed, alpha, holder))
                    for name, alpha, seed in arms]
        outputs = list(ev.evaluate_raw([(wrapper, name) for name, wrapper in wrappers], ev.ctx_sizes, with_node_idxs=True))
        if len(outputs) != 1:
            raise ValueError("expected one task/context output")
        _, _, labels, preds, _, ids = outputs[0]
        expected = load_expected_target_ids(pre_dir, ev.tasks[0], ITEMS_PER_TASK)
        if len(ids) != ITEMS_PER_TASK or not np.array_equal(np.sort(ids), expected):
            raise ValueError("predictions do not cover exactly the 702 intended targets")
        if not np.array_equal(ids, validation_report["target_node_idxs"]) or not np.array_equal(labels, validation_report["labels"]):
            raise ValueError("prediction target/label order differs from validated data")
        if holder.batch_index != len(holder.fingerprints):
            raise ValueError("inference did not traverse every validated batch")
        manifest["arms"] = {}
        for (name, alpha, seed), (_, wrapper) in zip(arms, wrappers, strict=True):
            scores = np.asarray(preds[name], dtype=float)
            if scores.shape != (ITEMS_PER_TASK,) or not np.all(np.isfinite(scores)):
                raise ValueError(f"arm {name}: invalid raw scores")
            exposure_path = out_dir / f"exposure_{name}.npz"
            exposure = write_exposure_npz(exposure_path, wrapper.exposure_batches)
            if not np.array_equal(exposure["target_node_idxs"], ids):
                raise ValueError(f"arm {name}: exposure/prediction alignment failed")
            with np.load(out_dir / f"validation_exposure_{name}.npz", allow_pickle=False) as checked:
                if set(checked.files) != set(exposure) or any(
                    not np.array_equal(checked[key], exposure[key], equal_nan=True) for key in checked.files
                ):
                    raise ValueError(f"arm {name}: exposure differs from pre-inference validation")
            for key in EDGE_FIELDS:
                if int(exposure[key].sum()) != wrapper.stats.get(key, 0):
                    raise ValueError(f"arm {name}: aggregate exposure mismatch")
            prediction_path = out_dir / f"preds_{name}.npz"
            with prediction_path.open("xb") as handle:
                np.savez(handle, labels=labels, preds=scores, node_idxs=ids)
            manifest["arms"][name] = {"requested_strength": alpha, "seed": None if alpha == 0 else seed,
                                      "matching_seed": seed, "auroc": binary_auroc(labels, scores), "totals": wrapper.stats,
                                      "predictions": prediction_path.name, "exposure": exposure_path.name,
                                      "prediction_sha256": file_identity(prediction_path)["sha256"],
                                      "exposure_sha256": file_identity(exposure_path)["sha256"]}
        manifest.update(status="complete", complete=True, n_queries=len(ids))
    except Exception as error:
        manifest.update(status="failed", complete=False, errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        manifest["model_forward_attempts"] = holder.forward_calls if holder else 0
        if manifest["status"] == "in_progress":
            manifest.update(status="interrupted", complete=False)
        manifest["elapsed_sec"] = time.time() - started
        save_json(manifest_path, manifest)
    return manifest


def execute_matrix(evaluators, pre_dir, seeds, fractions, out_dir, *, dry_only=True,
                   checkpoint_dir=None, device="cuda", model_factory=None, provenance=None):
    """Validate every context before any weight load or model forward."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    status = {"status": "in_progress", "complete": False, "contexts": {}, "inference_ran": False,
              "task": f"{DB_NAME}/{TASK_TABLE}", "seeds": list(seeds), "fractions": list(fractions),
              "provenance": provenance or {}, "evidence_kind": "synthetic_test" if model_factory else "validation_only" if dry_only else "real"}
    save_json(root / "matrix.json", status)
    try:
        validated = {}
        for ctx, ev in evaluators.items():
            directory = root / f"ctx{ctx}_lctx{ctx // 2}"
            directory.mkdir()
            dry_run(ev, pre_dir, seeds, directory, ctx // 2, fractions)
            report = json.loads((directory / "validation.json").read_text())
            validated[ctx] = report
            status["contexts"][str(ctx)] = report["status"]
        if any(result != "passed" for result in status["contexts"].values()):
            status["status"] = "validation_failed" if "failed" in status["contexts"].values() else "temporal_uncertainty"
            return status
        reference = next(iter(validated.values()))
        if any(report["target_node_idxs"] != reference["target_node_idxs"] or report["labels"] != reference["labels"]
               for report in validated.values()):
            raise ValueError("target/label alignment differs across contexts")
        if dry_only:
            status.update(status="validated", complete=True)
            return status
        if set(evaluators) != {48, 128} or set(fractions) != {0.0, 0.5, 1.0} or seeds != [101, 202, 303]:
            raise ValueError("preliminary inference requires both contexts and all 14 agreed arms")
        for ctx, ev in evaluators.items():
            directory = root / f"ctx{ctx}_lctx{ctx // 2}"
            run(ev, pre_dir, seeds, directory, device, ctx // 2, fractions=fractions,
                validation_report=validated[ctx], checkpoint_dir=checkpoint_dir, model_factory=model_factory, provenance=provenance)
        status.update(status="complete", complete=True)
        return status
    except Exception as error:
        status.update(status="failed", complete=False, errors=[f"{type(error).__name__}: {error}"])
        raise
    finally:
        status["inference_ran"] = any(
            json.loads(path.read_text()).get("model_forward_attempts", 0) > 0
            for path in root.glob("ctx*/manifest.json")
        )
        if status["status"] == "in_progress":
            status.update(status="interrupted", complete=False)
        save_json(root / "matrix.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx", type=int, choices=(48, 128))
    parser.add_argument("--local-ctx", type=int)
    parser.add_argument("--contexts", default="48,128")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--fractions", default="0,0.5,1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--out-root", default="results/runs")
    args = parser.parse_args()
    try:
        contexts = [args.ctx] if args.ctx else [int(value) for value in args.contexts.split(",")]
        seeds = [int(value) for value in args.seeds.split(",")]
        fractions = [float(value) for value in args.fractions.split(",")]
        experiment_arms(seeds, fractions)
        if len(set(contexts)) != len(contexts) or not contexts or any(ctx not in (48, 128) for ctx in contexts):
            raise ValueError("contexts must be unique values from 48 and 128")
        if args.local_ctx is not None and (args.ctx is None or args.local_ctx != args.ctx // 2):
            raise ValueError("local context must be 24 for 48 or 64 for 128")
        pre_dir = Path(args.pre_dir).expanduser().resolve()
        if not pre_dir.is_dir():
            raise ValueError("pre-dir must be an existing local compatible artifact; no downloads")
        if not args.dry_run and (set(contexts) != {48, 128} or seeds != [101, 202, 303] or set(fractions) != {0, 0.5, 1}):
            raise ValueError("inference requires the complete preliminary matrix")
        if not args.dry_run and (args.checkpoint_dir is None or not Path(args.checkpoint_dir).expanduser().is_dir()):
            raise ValueError("inference requires an existing local checkpoint directory")
    except ValueError as error:
        parser.error(str(error))
    root = Path(args.out_root).expanduser() / f"preliminary_{time.time_ns()}"
    try:
        provenance = {"source": source_identity(Path(__file__).resolve().parents[1]),
                      "software": software_versions(), "preprocessing": preprocessing_identity(pre_dir, DB_NAME),
                      "effective_arguments": vars(args), "sampling": {"context_seed": 0, "shuffle_seed": 0,
                      "embedding_model": "all-MiniLM-L12-v2", "d_text": 384, "tokens_per_gpu": 2**18,
                      "bfs_width": 32, "num_walks": 10000, "walk_length": 20, "prefer_latest": True}}
        from rt.eval_utils import build_evaluator

        tasks = build_tasks(str(pre_dir))
        evaluators = {ctx: build_evaluator(tasks, str(pre_dir), embedding_model="all-MiniLM-L12-v2", d_text=384,
                       device="cpu" if args.dry_run else "cuda", ctx_size=ctx, local_ctx_size=ctx // 2,
                       items_per_task=ITEMS_PER_TASK, num_workers=0, shuffle_seed=0, context_seed=0) for ctx in contexts}
        status = execute_matrix(evaluators, pre_dir, seeds, fractions, root, dry_only=args.dry_run,
                                checkpoint_dir=args.checkpoint_dir, provenance=provenance)
        print(f"[{status['status']}] {root}")
        return 0 if status["complete"] else 2 if status["status"] == "temporal_uncertainty" else 1
    except Exception as error:  # noqa: BLE001 -- persist setup errors without loading weights
        root.mkdir(parents=True, exist_ok=True)
        path = root / "matrix.json"
        if not path.exists():
            save_json(path, {"status": "setup_failed", "complete": False, "errors": [f"{type(error).__name__}: {error}"]})
        print(f"[failed] {error}; evidence: {root}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
