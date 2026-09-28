"""Run the RT-PluRel FK-parent rewiring experiment on ``rel-f1/driver-dnf``.

For one requested context configuration, this script evaluates four arms in a
single ``Evaluator.evaluate_raw`` call: the unmodified batch plus one rewired
batch for each seed. All arms share one frozen, loaded RT-PluRel model. The
rewiring implementation lives in :mod:`rfm_structure.rewiring`, not in this
directory.

In full mode, ``RewireWrap`` clones an evaluator batch and changes only
``f2p_nbr_idxs``. It checks that every other tensor matches the base batch before
calling RT. In ``--dry-run`` mode, the script instead rewires the first sampled
batch, runs ``assert_structural_invariants``, and writes a report without loading
the model or performing a forward pass.

The input is a locally preprocessed RT artifact made by ``scripts/preprocess.sh``.
Generated outputs are written below ``results/runs/`` by default and are ignored
by Git.

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

from rfm_structure.rewiring import (
    assert_structural_invariants,
    attn_true_counts,
    rewire_f2p_nbr,
)

CKPT = "stanford-star/rt-plurel/classification"
DB_NAME = "rel-f1"
TASK_TABLE = "driver-dnf"
ITEMS_PER_TASK = 702

BASE_TENSORS = (
    "node_idxs", "table_name_idxs", "col_name_idxs", "class_value_idxs",
    "sem_types", "number_values", "text_values", "datetime_values",
    "boolean_values", "col_name_values", "timestamps", "seed_node_idxs",
    "bfs_depths", "is_targets", "is_task_nodes", "is_padding", "f2p_rel_idxs",
)


class _Captured:
    """Mutable holder shared by the base and rewired model wrappers.

    ``BaseWrap`` stores a CPU clone of its first batch here. ``RewireWrap`` uses
    that clone as the reference for its full-mode equality gate.
    """


class BaseWrap(torch.nn.Module):
    """Evaluator-compatible wrapper for the unmodified control arm.

    The wrapper forwards batches to ``inner`` unchanged. On its first call it
    retains a CPU clone in ``holder.batch`` so rewired arms can assert that their
    non-structural inputs are unchanged.
    """

    def __init__(self, inner, holder):
        super().__init__()
        self.inner = inner
        self.holder = holder

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        """Capture the first control batch, then delegate to RT prediction."""
        if self.holder.batch is None:
            self.holder.batch = {
                k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        return self.inner.predict(batch, eval_ctx_sizes, device, task, bool_as_num)


class RewireWrap(torch.nn.Module):
    """Evaluator-compatible RT wrapper for one seeded structural intervention.

    ``seed`` selects one deterministic parent assignment. ``stats`` is mutated
    by :func:`rfm_structure.rewiring.rewire_f2p_nbr` and later becomes the
    aggregate edge-coverage section of the run manifest.
    """

    def __init__(self, inner, seed, holder, stats):
        super().__init__()
        self.inner = inner
        self.seed = seed
        self.holder = holder
        self.stats = stats

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        """Clone ``batch``, rewire only its FK parent IDs, and call RT.

        The equality gate excludes ``f2p_nbr_idxs`` because that is the intended
        intervention and ``batch_mask`` because it is evaluator bookkeeping.
        This method does not call the more expensive structural-invariant checker;
        that check is performed by :func:`dry_run`.
        """
        # Batch-equality gate: every tensor except f2p_nbr_idxs must equal base.
        base = self.holder.batch
        if base is not None:
            for k in batch:
                if k in ("f2p_nbr_idxs", "batch_mask"):
                    continue
                if isinstance(batch[k], torch.Tensor):
                    assert torch.equal(
                        batch[k].cpu(), base[k].cpu()
                    ), f"batch-equality gate failed on {k}"
        b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        nbr = b["f2p_nbr_idxs"].cpu().numpy().astype(np.int64)
        rel = b["f2p_rel_idxs"].cpu().numpy().astype(np.int64)
        nid = b["node_idxs"].cpu().numpy().astype(np.int64)
        pad = b["is_padding"].cpu().numpy()
        new_nbr = rewire_f2p_nbr(nbr, rel, nid, pad, self.seed, stats=self.stats)
        b["f2p_nbr_idxs"] = torch.from_numpy(new_nbr).to(b["f2p_nbr_idxs"].dtype)
        return self.inner.predict(b, eval_ctx_sizes, device, task, bool_as_num)


def build_tasks(pre_dir):
    """Load the configured RelBench test task from a preprocessed RT artifact.

    Args:
        pre_dir: Directory produced by the vendored RT preprocessing command.

    Returns:
        The nonempty list of ``driver-dnf`` test tasks for the fixed ``rel-f1``
        database.
    """
    tasks = [
        t for t in tasks_from_preprocessed(pre_dir, splits=("test",), dbs=[DB_NAME])
        if t.table_name == TASK_TABLE
    ]
    assert tasks, f"no {TASK_TABLE} task found in {pre_dir}"
    return tasks


def dry_run(ev, pre_dir, seeds, out_dir):
    """Validate rewiring on the evaluator's first batch without RT inference.

    The function saves the raw batch as ``base_context.pt``, computes aggregate
    feature/neighbor attention counts, applies every requested seeded rewire,
    and writes ``dry_run_report.json``. It returns ``0`` only when every
    invariant check passes and every seed changes at least 80% of its rewirable
    (non-singleton) edge instances.

    ``pre_dir`` is accepted for parity with :func:`run` but is not read directly;
    the already constructed evaluator owns the preprocessed input.
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
        new_nbr = rewire_f2p_nbr(nbr, rel, nid, pad, int(seed), stats=stats)
        errs = assert_structural_invariants(nbr, new_nbr, rel, nid, pad)
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
        # sample map: first 5 rows of (b=0) f2p_nbr before/after for relation slots
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


def run(ev, pre_dir, seeds, out_dir, device, local_ctx):
    """Evaluate base and seeded rewired arms with one shared RT-PluRel model.

    The evaluator yields predictions for the same test targets under a control
    wrapper and one rewired wrapper per seed. This function concatenates those
    predictions, verifies the expected 702 unique target node IDs, computes
    AUROC from raw logits, and writes one NPZ archive per arm plus a manifest.

    Args:
        ev: Vendored RT evaluator configured by :func:`main`.
        pre_dir: Recorded verbatim in the output manifest for provenance.
        seeds: Integer rewiring seeds, each defining one experimental arm.
        out_dir: Existing directory for generated NPZ and JSON files.
        device: ``"cuda"`` when available, otherwise ``"cpu"``.
        local_ctx: Local-context size recorded in the manifest.

    Returns:
        The manifest dictionary after it has been written to disk.
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

    # load model once, share across all wrappers
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

    # scoring gate
    n_unique = len(np.unique(node_idxs_all))
    assert n_unique == ITEMS_PER_TASK, f"expected 702 unique nodes, got {n_unique}"
    auroc = {}
    for p, labels in labels_all.items():
        lab = labels.astype(float)
        pr = preds_all[p].astype(float)
        # clf with bool_as_num emits raw logits; AUROC is rank-invariant to the
        # sigmoid, so score on raw scores (matches rt.eval_utils._score).
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
    """Parse CLI options, construct the evaluator, then run validation or RT.

    ``--ctx`` and ``--local-ctx`` select the sampled-context configuration.
    ``--pre-dir`` must point at locally generated RT artifacts. ``--dry-run``
    validates one sampled batch without loading the checkpoint; omitting it runs
    full base-plus-rewired inference and writes results below ``--out-root``.
    """
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
