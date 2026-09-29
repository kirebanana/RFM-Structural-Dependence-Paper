"""Compare clean and rewired FK links on RelBench ``rel-f1/driver-dnf``.

Each evaluator batch contains one sampled context per prediction query. The
clean arm and three seeded rewired arms share a single frozen RT-PluRel model;
only the FK parent IDs differ. The rewiring algorithm and its checks live in
``rfm_structure.rewiring``. Input comes from ``scripts/preprocess.sh`` and
generated outputs go under ``results/runs/`` by default.

``--dry-run`` validates only the first evaluator batch without model inference.
It does not establish that all prediction queries have been checked.

Usage:
  python experiments/run_clean_rewire.py --ctx 48 --local-ctx 24 --dry-run \
      --pre-dir artifacts/clean_rewire_preprocessed
  python experiments/run_clean_rewire.py --ctx 48 --local-ctx 24 --seeds 101,202,303 \
      --pre-dir artifacts/clean_rewire_preprocessed
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np
import torch
from rt import RelationalTransformer
from rt.eval_utils import build_evaluator
from rt.tasks import tasks_from_preprocessed

from rfm_structure.data import load_relation_index
from rfm_structure.rewiring import (
    assert_structural_invariants,
    attn_true_counts,
    rewire_f2p_nbr,
    summarize_query_exposure,
)

CKPT = "stanford-star/rt-plurel/classification"
DB_NAME = "rel-f1"
TASK_TABLE = "driver-dnf"
ITEMS_PER_TASK = 702


class _Captured:
    """Share the current clean batch between model wrappers."""


class BaseWrap(torch.nn.Module):
    """Run the clean arm and capture each batch for comparison with rewired arms."""

    def __init__(self, inner, holder):
        super().__init__()
        self.inner = inner
        self.holder = holder

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        """Capture the current control batch, then delegate to RT prediction."""
        self.holder.batch = {
            k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        return self.inner.predict(batch, eval_ctx_sizes, device, task, bool_as_num)


class RewireWrap(torch.nn.Module):
    """Check and run one seeded FK-parent rewiring arm on the shared RT model.

    ``stats`` accumulates eligible/changed edge instances across batches;
    ``exposure_batches`` retains the corresponding per-query relation counts.
    """

    def __init__(self, inner, seed, holder, stats):
        super().__init__()
        self.inner = inner
        self.seed = seed
        self.holder = holder
        self.stats = stats
        self.exposure_batches = []

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        """Verify held-fixed inputs, validate rewiring, then predict."""
        # The evaluator calls the clean arm first for this same sampled batch.
        base = self.holder.batch
        assert base is not None, "rewire arm ran before the control arm"
        assert set(batch) == set(base), "batch-equality gate found different keys"
        for k in batch:
            if k in ("f2p_nbr_idxs", "batch_mask"):
                # FK parents are the intervention; batch_mask is evaluator metadata.
                continue
            if isinstance(batch[k], torch.Tensor):
                assert torch.equal(
                    batch[k].cpu(), base[k].cpu()
                ), f"batch-equality gate failed on {k}"
            else:
                assert batch[k] == base[k], f"batch-equality gate failed on {k}"
        b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        nbr = b["f2p_nbr_idxs"].cpu().numpy().astype(np.int64)
        rel = b["f2p_rel_idxs"].cpu().numpy().astype(np.int64)
        nid = b["node_idxs"].cpu().numpy().astype(np.int64)
        pad = b["is_padding"].cpu().numpy()
        timestamps = b["timestamps"].cpu().numpy().astype(np.int64)
        is_targets = b["is_targets"].cpu().numpy().astype(bool)
        sequence_stats = []
        new_nbr = rewire_f2p_nbr(
            nbr,
            rel,
            nid,
            pad,
            self.seed,
            stats=self.stats,
            sequence_stats=sequence_stats,
        )
        errors = assert_structural_invariants(
            nbr,
            new_nbr,
            rel,
            nid,
            pad,
            timestamps=timestamps,
            is_targets=is_targets,
        )
        assert not errors, f"structural invariant gate failed: {errors}"
        self.exposure_batches.append(summarize_query_exposure(
            nid,
            rel,
            pad,
            is_targets,
            b["col_name_idxs"].cpu().numpy(),
            b["is_task_nodes"].cpu().numpy().astype(bool),
            sequence_stats,
        ))
        b["f2p_nbr_idxs"] = torch.from_numpy(new_nbr).to(b["f2p_nbr_idxs"].dtype)
        return self.inner.predict(b, eval_ctx_sizes, device, task, bool_as_num)


def build_tasks(pre_dir):
    """Find the driver-DNF test task in the local preprocessed RT data."""
    tasks = [
        t for t in tasks_from_preprocessed(pre_dir, splits=("test",), dbs=[DB_NAME])
        if t.table_name == TASK_TABLE
    ]
    assert tasks, f"no {TASK_TABLE} task found in {pre_dir}"
    return tasks


def dry_run(ev, pre_dir, seeds, out_dir):
    """Check the first sampled batch without loading the RT model.

    Saves that batch, per-seed query exposure, and a validation report. Returns
    nonzero if a check fails or fewer than 80% of non-singleton eligible edges
    change. This checks one batch, not every prediction query. ``pre_dir`` is
    already used by the evaluator and is not read directly here.
    """
    task = ev.tasks[0]
    loader = ev.eval_loaders[task]
    batch = next(iter(loader))
    report = {"seeds": [int(s) for s in seeds], "ctx": ev.ctx_sizes[0]}

    # Preserve the exact sampled batch inspected by this dry-run.
    cap = {k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    torch.save(cap, os.path.join(out_dir, "base_context.pt"))

    nbr = cap["f2p_nbr_idxs"].cpu().numpy().astype(np.int64)
    rel = cap["f2p_rel_idxs"].cpu().numpy().astype(np.int64)
    nid = cap["node_idxs"].cpu().numpy().astype(np.int64)
    pad = cap["is_padding"].cpu().numpy()

    fb, nb = attn_true_counts(nid, nbr, pad)
    report["base_feat_counts"] = fb
    report["base_nbr_counts"] = nb

    per_seed = {}
    for seed in seeds:
        stats = {}
        sequence_stats = []
        new_nbr = rewire_f2p_nbr(
            nbr,
            rel,
            nid,
            pad,
            int(seed),
            stats=stats,
            sequence_stats=sequence_stats,
        )
        errs = assert_structural_invariants(
            nbr,
            new_nbr,
            rel,
            nid,
            pad,
            timestamps=cap["timestamps"].cpu().numpy().astype(np.int64),
            is_targets=cap["is_targets"].cpu().numpy().astype(bool),
        )
        exposure = summarize_query_exposure(
            nid,
            rel,
            pad,
            cap["is_targets"].cpu().numpy().astype(bool),
            cap["col_name_idxs"].cpu().numpy(),
            cap["is_task_nodes"].cpu().numpy().astype(bool),
            sequence_stats,
        )
        write_exposure_npz(os.path.join(out_dir, f"exposure_rw{seed}.npz"), [exposure])
        eligible_edges = stats["eligible_edges"]
        changed_edges = stats["changed_edges"]
        singleton_edges = stats.get("singleton_edges", 0)
        raw_fraction = changed_edges / eligible_edges if eligible_edges else 0.0
        rewirable_edges = eligible_edges - singleton_edges
        rewirable_fraction = changed_edges / rewirable_edges if rewirable_edges else 0.0
        per_seed[int(seed)] = {
            "eligible_edges": eligible_edges,
            "changed_edges": changed_edges,
            "singleton_edges": singleton_edges,
            "changed_edge_fraction": raw_fraction,
            "rewirable_edge_fraction": rewirable_fraction,
            "unrewirable_strata": stats.get("unrewirable_strata", 0),
            "invariant_errors": errs,
        }
        if errs:
            print(f"[dry-run] seed {seed} INVARIANT FAILURES: {errs}", flush=True)
        else:
            print(
                f"[dry-run] seed {seed} OK eligible_edges={eligible_edges} "
                f"changed_edges={changed_edges} changed_edge_fraction={raw_fraction:.3f} "
                f"rewirable_edge_fraction={rewirable_fraction:.3f}",
                flush=True,
            )
    report["per_seed"] = per_seed
    with open(os.path.join(out_dir, "dry_run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[dry-run] wrote {out_dir}/dry_run_report.json", flush=True)
    failed = [s for s, v in per_seed.items() if v["invariant_errors"]]
    if failed:
        print(f"[dry-run] ABORT: invariants failed for seeds {failed}", flush=True)
        return 1
    if any(v["rewirable_edge_fraction"] < 0.8 for v in per_seed.values()):
        print("[dry-run] ABORT: rewirable edge fraction < 0.80 for some seed", flush=True)
        return 1
    print("[dry-run] all gates passed; ready for model forward.", flush=True)
    return 0


def write_exposure_npz(path, batches):
    """Save per-query FK counts and flattened per-relation counts as NPZ.

    Each batch uses local row indices. Exclude rows without one target, append
    the remaining queries in evaluation order, and remap relation rows to their
    index in the saved query arrays. ``changed_edge_fractions`` divides changed
    by eligible FK edge instances; it is zero when none are eligible.
    """
    query_fields = (
        "target_node_idxs",
        "context_token_counts",
        "eligible_edges",
        "changed_edges",
        "singleton_edges",
        "unrewirable_edges",
        "unrewirable_strata",
        "unique_parent_nodes",
        "labeled_support_counts",
    )
    query_values = {field: [] for field in query_fields}
    relation_values = {
        "relation_query_indices": [],
        "relation_ids": [],
        "relation_eligible_edges": [],
        "relation_changed_edges": [],
        "relation_singleton_edges": [],
        "relation_unrewirable_edges": [],
        "relation_unrewirable_strata": [],
    }
    query_offset = 0
    for batch in batches:
        query_mask = batch["is_prediction_query"]
        remap = np.full(len(query_mask), -1, dtype=np.int64)
        remap[query_mask] = np.arange(query_mask.sum(), dtype=np.int64) + query_offset
        for field in query_fields:
            query_values[field].append(batch[field][query_mask])
        relation_mask = query_mask[batch["relation_query_indices"]]
        relation_values["relation_query_indices"].append(
            remap[batch["relation_query_indices"][relation_mask]]
        )
        for field in tuple(relation_values)[1:]:
            relation_values[field].append(batch[field][relation_mask])
        query_offset += int(query_mask.sum())
    arrays = {
        field: np.concatenate(values) if values else np.array([], dtype=np.int64)
        for field, values in query_values.items()
    }
    arrays["changed_edge_fractions"] = np.divide(
        arrays["changed_edges"],
        arrays["eligible_edges"],
        out=np.zeros(len(arrays["eligible_edges"]), dtype=float),
        where=arrays["eligible_edges"] != 0,
    )
    arrays.update({
        field: np.concatenate(values) if values else np.array([], dtype=np.int64)
        for field, values in relation_values.items()
    })
    np.savez(path, **arrays)
    return arrays


def run(ev, pre_dir, seeds, out_dir, device, local_ctx):
    """Evaluate clean and rewired arms, then save aligned predictions and exposure.

    One evaluator pass uses the same sampled contexts and RT model for every
    arm. The saved NPZ files hold labels, raw scores, and target row IDs; the
    manifest records AUROC, aggregate edge counts, and FK relation metadata.
    """
    from sklearn.metrics import roc_auc_score

    holder = _Captured()
    holder.batch = None
    base = BaseWrap(None, holder)  # inner set after model load
    wrappers = [("base", base)]
    seed_stats = {}
    for s in seeds:
        st = {}
        seed_stats[int(s)] = st
        wrappers.append((f"rw{int(s)}", RewireWrap(None, int(s), holder, st)))

    # Sharing weights makes parent identity the only intended model input change.
    model = RelationalTransformer.from_pretrained(
        CKPT, device=device
    )
    model = model.to(torch.bfloat16)

    base.inner = model
    for _, w in wrappers[1:]:
        w.inner = model

    ctx = ev.ctx_sizes[0]
    labels_acc = {p: [] for p, _ in wrappers}
    preds_acc = {p: [] for p, _ in wrappers}
    nid_acc = []

    t0 = time.time()
    for out in ev.evaluate_raw(
        [(w, p) for p, w in wrappers], [ctx], with_node_idxs=True
    ):
        _task, _ctx_size, labels_np, preds_by_prefix, _num_labels, node_idxs_np = out
        nid_acc.append(node_idxs_np)
        for p, _ in wrappers:
            labels_acc[p].append(labels_np)
            preds_acc[p].append(preds_by_prefix[p])

    dt = time.time() - t0
    labels_all = {p: np.concatenate(v) for p, v in labels_acc.items()}
    preds_all = {p: np.concatenate(v) for p, v in preds_acc.items()}
    node_idxs_all = np.concatenate(nid_acc)

    for prefix, wrapper in wrappers[1:]:
        exposure_path = os.path.join(out_dir, f"exposure_{prefix}.npz")
        exposure = write_exposure_npz(exposure_path, wrapper.exposure_batches)
        assert np.array_equal(exposure["target_node_idxs"], node_idxs_all), (
            f"{prefix}: exposure rows do not align with evaluator predictions"
        )

    # Raw logits and sigmoid probabilities have the same AUROC ranking.
    n_unique = len(np.unique(node_idxs_all))
    assert n_unique == ITEMS_PER_TASK, f"expected 702 unique nodes, got {n_unique}"
    auroc = {}
    for p, labels in labels_all.items():
        lab = labels.astype(float)
        pr = preds_all[p].astype(float)
        assert np.all(np.isfinite(pr)), f"{p}: non-finite predictions"
        lab_bin = (lab > 0).astype(int)
        auroc[p] = float(roc_auc_score(lab_bin, pr))
        np.savez(
            os.path.join(out_dir, f"preds_{p}.npz"),
            labels=lab, preds=pr, node_idxs=node_idxs_all,
        )
        print(f"  {p}: preds min={pr.min():.4f} max={pr.max():.4f} AUROC={auroc[p]:.4f}",
              flush=True)

    manifest = {
        "db_name": DB_NAME,
        "task_table": TASK_TABLE,
        "ctx": int(ctx),
        "local_ctx": int(local_ctx),
        "seeds": [int(s) for s in seeds],
        "checkpoint": CKPT,
        "pre_dir": pre_dir,
        "items_per_task": ITEMS_PER_TASK,
        "n_unique_nodes": int(n_unique),
        "auroc": auroc,
        "seed_stats": {str(k): v for k, v in seed_stats.items()},
        "elapsed_sec": dt,
        # Relation IDs in exposure NPZs need this mapping to remain interpretable.
        "relation_index": load_relation_index(pre_dir, DB_NAME),
    }
    # Edge-instance counts; singleton strata cannot change.
    cov = {
        str(k): {
            "eligible_edges": v.get("eligible_edges", 0),
            "changed_edges": v.get("changed_edges", 0),
            "singleton_edges": v.get("singleton_edges", 0),
            "changed_edge_fraction": (
                v.get("changed_edges", 0) / v["eligible_edges"]
                if v.get("eligible_edges") else 0.0
            ),
            "rewirable_edge_fraction": (
                (v.get("changed_edges", 0) / (v["eligible_edges"] - v.get("singleton_edges", 0)))
                if (v.get("eligible_edges", 0) - v.get("singleton_edges", 0)) > 0 else 0.0
            ),
        }
        for k, v in seed_stats.items()
    }
    manifest["coverage"] = cov
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[run] ctx={ctx} done in {dt:.0f}s", flush=True)
    for p, score in auroc.items():
        print(f"  {p:>8} AUROC={score:.4f}", flush=True)
    print(f"[run] manifest -> {out_dir}/manifest.json", flush=True)
    return manifest


def main():
    """Select a context size, local RT artifact, seeds, and dry/full run mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--local-ctx", type=int, required=True)
    ap.add_argument("--seeds", type=str, default="101,202,303")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pre-dir", required=True)
    ap.add_argument("--out-root", default="results/runs")
    args = ap.parse_args()

    pre_dir = os.path.expanduser(args.pre_dir)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_root = os.path.expanduser(args.out_root)
    run_id = f"ctx{args.ctx}_lctx{args.local_ctx}"
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = build_tasks(pre_dir)

    ev = build_evaluator(
        tasks, pre_dir,
        embedding_model="all-MiniLM-L12-v2", d_text=384,
        device=device, ctx_size=args.ctx, local_ctx_size=args.local_ctx,
        items_per_task=ITEMS_PER_TASK, num_workers=0, shuffle_seed=0,
    )

    if args.dry_run:
        rc = dry_run(ev, pre_dir, seeds, out_dir)
        sys.exit(rc)

    manifest = run(ev, pre_dir, seeds, out_dir, device, args.local_ctx)
    # final table
    print("\n=== FINAL TABLE ===")
    print("| ctx | local_ctx | base AUROC | " + " | ".join(f"rw{s}" for s in seeds)
          + " | mean delta | seed range | changed/eligible edge instances | support/query | feat/nbr equal |")
    base_a = manifest["auroc"]["base"]
    rw = [manifest["auroc"][f"rw{s}"] for s in seeds]
    mean_delta = np.mean([a - base_a for a in rw])
    cov = manifest["coverage"]
    changed = sum(v["changed_edges"] for v in cov.values())
    eligible = sum(v["eligible_edges"] for v in cov.values())
    print(f"| {args.ctx} | {args.local_ctx} | {base_a:.4f} | "
          + " | ".join(f"{a:.4f}" for a in rw)
          + f" | {mean_delta:+.4f} | {min(seeds)}-{max(seeds)} | {changed}/{eligible} | "
          f"within-context | see manifest |")


if __name__ == "__main__":
    main()
