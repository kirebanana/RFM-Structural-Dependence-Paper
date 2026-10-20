"""Replay frozen F1 contexts and classify changed task links without model inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_clean_rewire import build_tasks

from rfm_structure.data import load_availability_policy, load_relation_index
from rfm_structure.provenance import file_identity
from rfm_structure.relation_controls import relation_groups
from rfm_structure.rewiring import rewire_f2p_nbr
from rfm_structure.validation import (
    available_context,
    batch_fingerprint,
    configure_complete_evaluation,
    evaluator_query_arrays,
    real_query_batch,
    visible_support_source_mask,
)


def classify_links(base, rewired, relation):
    """Count source/FK slots once, classifying label visibility from retained cells."""
    output = []
    for b in range(len(base["node_idxs"])):
        kept = np.flatnonzero(~base["is_padding"][b])
        target = next(s for s in kept if base["is_targets"][b, s])
        query = int(base["node_idxs"][b, target])
        label_column = base["col_name_idxs"][b, target]
        present = set(base["node_idxs"][b, kept].tolist())
        visible = {int(base["node_idxs"][b, s]) for s in kept
                   if not base["is_targets"][b, s] and base["is_task_nodes"][b, s]
                   and base["col_name_idxs"][b, s] == label_column
                   and np.isfinite(base["number_values"][b, s]).all()
                   and int(base["node_idxs"][b, s]) != query}
        row = {"target_node": query, "eligible": dict.fromkeys(("query", "visible_label", "hidden_label"), 0),
               "changed": dict.fromkeys(("query", "visible_label", "hidden_label"), 0),
               "visible_pool_distinct_parents": 0}
        seen, parents = set(), set()
        for s in kept:
            node = int(base["node_idxs"][b, s])
            kind = "query" if node == query else "visible_label" if node in visible else "hidden_label"
            for slot in range(base["f2p_nbr_idxs"].shape[-1]):
                if base["f2p_rel_idxs"][b, s, slot] != relation:
                    continue
                parent = int(base["f2p_nbr_idxs"][b, s, slot])
                if (node, slot) in seen or parent not in present:
                    continue
                seen.add((node, slot))
                row["eligible"][kind] += 1
                if kind == "visible_label":
                    parents.add(parent)
                row["changed"][kind] += int(rewired[b, s, slot] != parent)
        row["visible_pool_distinct_parents"] = len(parents)
        output.append(row)
    return output


def audit(pre_dir, release):
    from rt.eval_utils import build_evaluator

    policy = load_availability_policy(pre_dir)
    relation, = relation_groups(load_relation_index(pre_dir, "rel-f1"))["task_driver"]
    tasks = build_tasks(str(pre_dir))
    report = {"method": "CPU context replay; independent retained-cell classification; source/FK slots counted once",
              "contexts": {}}
    for ctx in (48, 128):
        context = f"ctx{ctx}_lctx{ctx // 2}"
        ev = build_evaluator(tasks, str(pre_dir), embedding_model="all-MiniLM-L12-v2", d_text=384,
                             device="cpu", ctx_size=ctx, local_ctx_size=ctx // 2, items_per_task=702,
                             num_workers=0, context_seed=0, shuffle_seed=0)
        loader = configure_complete_evaluation(ev, 702)
        records = {name: {str(seed): [] for seed in (101, 202, 303)} for name in ("task_cycles", "labeled_support")}
        fingerprints, ids = [], []
        for raw in loader:
            real, _ = real_query_batch(raw)
            base, _ = available_context(real, policy)
            fingerprints.append(batch_fingerprint(base))
            query_ids, _, _ = evaluator_query_arrays(base)
            ids.extend(query_ids.tolist())
            for name in records:
                for seed in (101, 202, 303):
                    rewired = rewire_f2p_nbr(base["f2p_nbr_idxs"], base["f2p_rel_idxs"], base["node_idxs"],
                        base["is_padding"], seed, alpha=1.0, relation_ids=[relation],
                        cycle_mode="excluding_query" if name == "task_cycles" else None,
                        query_node_ids=np.asarray(query_ids),
                        source_mask=visible_support_source_mask(base) if name == "labeled_support" else None)
                    records[name][str(seed)].extend(classify_links(base, rewired, relation))
        summary = {}
        for name, seeds in records.items():
            directory = release / name / context
            validation = json.loads((directory / "validation.json").read_text())
            if fingerprints != validation["batch_fingerprints"] or ids != validation["target_node_idxs"]:
                raise ValueError("replayed contexts differ from saved run")
            manifest = json.loads((directory / "manifest.json").read_text())
            summary[name] = {}
            for seed, rows in seeds.items():
                group = "task_other_cycles" if name == "task_cycles" else "labeled_support_only"
                arm = manifest["arms"][f"{group}_s{seed}"]
                with np.load(directory / arm["exposure"], allow_pickle=False) as exposure:
                    changed = np.asarray([sum(row["changed"].values()) for row in rows])
                    if not np.array_equal(changed, exposure["changed_edges"]):
                        raise ValueError("visibility audit differs from saved edge exposure")
                totals = {field: {kind: sum(row[field][kind] for row in rows)
                                  for kind in ("query", "visible_label", "hidden_label")} for field in ("eligible", "changed")}
                if totals["changed"]["query"] or (name == "labeled_support" and totals["changed"]["hidden_label"]):
                    raise ValueError("excluded source links changed")
                summary[name][seed] = {**totals, "affected_queries": int((changed > 0).sum()),
                    "visible_pool_zero_queries": sum(row["eligible"]["visible_label"] == 0 for row in rows),
                    "visible_pool_singleton_queries": sum(row["eligible"]["visible_label"] == 1 for row in rows),
                    "per_query": rows}
        report["contexts"][str(ctx)] = summary
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays_path = args.output.with_suffix(".npz")
    if args.output.exists() or arrays_path.exists():
        raise FileExistsError("audit output already exists")
    result = audit(args.pre_dir, args.release)
    arrays = {}
    for ctx, groups in result["contexts"].items():
        for group, seeds in groups.items():
            for seed, summary in seeds.items():
                rows = summary.pop("per_query")
                prefix = f"ctx{ctx}_{group}_s{seed}_"
                for field in ("target_node", "visible_pool_distinct_parents"):
                    arrays[prefix + field] = np.asarray([row[field] for row in rows])
                for field in ("eligible", "changed"):
                    for kind in ("query", "visible_label", "hidden_label"):
                        arrays[prefix + field + "_" + kind] = np.asarray([row[field][kind] for row in rows])
    with arrays_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result["per_query_arrays"] = {**file_identity(arrays_path), "path": arrays_path.name}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
