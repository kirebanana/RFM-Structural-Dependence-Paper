"""No-inference validation at the sampled-batch NumPy seam.

Real rows are selected by the sampler's mask, never by target presence. A
complete expected-ID gate catches failed sampler items masked as phantoms.
Temporal uncertainty is quantified separately from known validity.
"""

import hashlib

import numpy as np

from .rewiring import (
    TEMPORAL_FIELDS,
    assert_structural_invariants,
    attention_fanout_counts,
    attn_true_counts,
    effective_parent_set_sizes,
    rewire_f2p_nbr,
    summarize_query_exposure,
    summarize_temporal_status,
)

__all__ = ["assert_structural_invariants", "attn_true_counts"]

EDGE_FIELDS = (
    "eligible_edges", "changed_edges", "singleton_edges",
    "unrewirable_edges", "unrewirable_strata", "max_changed_edges",
)
QUERY_FIELDS = (
    "target_node_idxs", "context_token_counts", *EDGE_FIELDS,
    "unique_parent_nodes", "labeled_support_counts",
    "requested_strength",
    "configured_context_sizes",
)
AVAILABILITY_FIELDS = (
    "original_token_counts", "removed_future_tokens", "removed_unmatured_label_tokens",
    "declared_timeless_tokens",
)
FANOUT_FIELDS = (
    "feat_changed_tokens", "nbr_changed_tokens", "feat_absolute_delta",
    "nbr_absolute_delta", "parent_set_changed_tokens",
)


def timeless_parent_ranges(policy):
    """Return only node ranges explicitly declared timeless by the schema."""
    return [(info["start"], info["stop"]) for info in policy["tables"]
            if info["time_col"] is None] if policy else []


def available_context(batch, policy):
    """Mask unavailable cells once, before clean/rewired arms; do not refill.

    This is common input preparation, not the structural intervention. Every
    retained temporal row must have a known time <= the query. Forecast labels
    need a fully closed outcome window. Explicitly timeless tables are allowed
    under the benchmark convention; genuinely missing required times fail.
    """
    result = dict(batch)
    padding = batch["is_padding"].copy()
    nodes = batch["node_idxs"]
    times = batch["timestamps"].astype(np.int64)
    present = ~padding
    targets = batch["is_targets"] & present
    if not np.all(targets.sum(axis=1) == 1):
        raise ValueError("availability requires one target per real prediction query")
    target_positions = targets.argmax(axis=1)
    cutoffs = times[np.arange(len(nodes)), target_positions][:, None]
    missing = np.iinfo(np.int32).min
    if np.any(cutoffs == missing):
        raise ValueError("prediction query timestamp is missing")
    classified = np.zeros_like(padding)
    future = np.zeros_like(padding)
    immature = np.zeros_like(padding)
    timeless = np.zeros_like(padding)
    for info in policy["tables"]:
        positions = present & (nodes >= info["start"]) & (nodes < info["stop"])
        classified |= positions
        if info["time_col"] is None:
            timeless |= positions
            continue
        if np.any(positions & (times == missing)):
            raise ValueError(f"missing timestamp in temporal table {info['table']}")
        future |= positions & (times > cutoffs)
        if info["label_column"] is not None:
            label_cells = positions & (batch["col_name_idxs"] == info["label_column"]) & ~targets
            immature |= label_cells & (times + info["horizon_seconds"] > cutoffs)
    if np.any(present & ~classified):
        raise ValueError("sampled node has no source-schema availability metadata")
    if np.any(targets & (future | immature)):
        raise ValueError("availability must not remove the evaluated target")
    padding |= future | immature
    result["is_padding"] = padding
    return result, {
        "original_token_counts": present.sum(axis=1, dtype=np.int64),
        "removed_future_tokens": future.sum(axis=1, dtype=np.int64),
        "removed_unmatured_label_tokens": (immature & ~future).sum(axis=1, dtype=np.int64),
        "declared_timeless_tokens": (timeless & ~padding).sum(axis=1, dtype=np.int64),
    }


def attach_availability(exposure, counts):
    """Persist common filtering counts alongside every arm's query exposure."""
    if counts is not None:
        exposure.update({f"availability_{key}": value.copy() for key, value in counts.items()})


def experiment_arms(seeds, fractions=(0.0, 0.5, 1.0)):
    """One clean arm, plus each requested non-clean strength/seed pair."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if not fractions or len(set(fractions)) != len(fractions) or any(
        fraction not in (0.0, 0.5, 1.0) for fraction in fractions
    ):
        raise ValueError("fractions must be unique values from 0, 0.5 and 1")
    return [
        ("base", 0.0, seeds[0]) if alpha == 0 else (f"a{round(alpha * 100):03d}_s{seed}", alpha, seed)
        for alpha in sorted(fractions)
        for seed in (seeds[:1] if alpha == 0 else seeds)
    ]


def batch_fingerprint(batch):
    """Bind later inference to the exact real sampled input validated earlier."""
    digest = hashlib.sha256()
    for key, values in sorted(batch.items()):
        array = np.ascontiguousarray(values)
        digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def numpy_batch(batch, row_mask=None):
    """Copy NumPy or CPU/GPU RT tensors without importing Torch or RT."""
    result = {}
    for key, value in batch.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if row_mask is not None:
                value = value[row_mask]
            if str(value.dtype) == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        elif row_mask is not None:
            value = np.asarray(value)[row_mask]
        result[key] = np.asarray(value).copy()
    return result


def real_query_batch(batch):
    """Filter phantoms and reject malformed real targets before accounting."""
    mask = numpy_batch({"mask": batch["batch_mask"]})["mask"]
    size = len(batch["node_idxs"])
    if mask.dtype != np.bool_ or mask.shape != (size,):
        raise ValueError("batch_mask must be a boolean vector with one entry per row")
    sampled = {key: value for key, value in batch.items() if key != "batch_mask"}
    if any(value.ndim == 0 or len(value) != size for value in sampled.values()):
        raise ValueError("all sampled tensors must have the same batch dimension")
    real = numpy_batch(sampled, row_mask=mask)
    targets = real["is_targets"] & ~real["is_padding"]
    if np.any(real["is_targets"] & real["is_padding"]):
        raise ValueError("real row has a target on padding")
    if not np.all(targets.sum(axis=1) == 1):
        raise ValueError("every real row must contain exactly one prediction target")
    return real, int((~mask).sum())


def evaluator_query_arrays(batch):
    """Mirror evaluate_raw's target-ID, label and same-column support rules."""
    targets = batch["is_targets"]
    nodes = batch["node_idxs"]
    target_ids = (nodes * targets).sum(axis=1, dtype=np.int64)
    target_cols = (batch["col_name_idxs"] * targets).sum(axis=1)[:, None]
    support = (
        batch["is_task_nodes"] & ~batch["is_padding"]
        & (batch["col_name_idxs"] == target_cols)
        & (nodes != target_ids[:, None])
    ).sum(axis=1, dtype=np.int64)
    labels = (batch["number_values"].squeeze(-1) * targets).sum(axis=1)
    if not np.all(np.isfinite(labels)):
        raise ValueError("non-finite target labels")
    return target_ids, labels, support


def assert_held_fixed(base, arm):
    """Parent IDs are the only permitted sampled-input change."""
    if set(base) != set(arm):
        raise ValueError("held-fixed inputs have different keys")
    for key in base:
        if key != "f2p_nbr_idxs" and (
            base[key].dtype != arm[key].dtype or not np.array_equal(base[key], arm[key], equal_nan=True)
        ):
            raise ValueError(f"held-fixed input changed: {key}")


def reconcile_temporal(status, eligible_edges=None):
    """Require exhaustive checked/unknown accounting at query and batch levels."""
    for counts in (*status["queries"], status):
        if any(counts[key] < 0 for key in TEMPORAL_FIELDS):
            raise ValueError("negative temporal accounting count")
        if (
            counts["comparisons"] != counts["checked"] + counts["unknown"] + counts["declared_timeless"]
            or counts["checked"] != counts["valid"] + counts["known_violations"]
            or counts["unknown"] != counts["missing_target"] + counts["missing_parent"] - counts["missing_both"]
            or counts["missing_both"] > min(counts["missing_target"], counts["missing_parent"])
            or max(counts["missing_target"], counts["missing_parent"]) > counts["unknown"]
        ):
            raise ValueError("temporal accounting checked/unknown/invalid counts do not reconcile")
    for key in TEMPORAL_FIELDS:
        if sum(query[key] for query in status["queries"]) != status[key]:
            raise ValueError(f"temporal accounting query/aggregate mismatch for {key}")
    if eligible_edges is not None and status["comparisons"] != eligible_edges:
        raise ValueError("temporal accounting comparisons differ from eligible FK edge instances")


def reconcile_exposure(exposure, totals=None):
    """Require relation-to-query and optional run-total equality, row by row."""
    n = len(exposure["target_node_idxs"])
    if not np.all(exposure["is_prediction_query"]):
        raise ValueError("exposure contains a malformed real prediction query")
    indices = exposure["relation_query_indices"]
    if len(exposure["relation_ids"]) != len(indices):
        raise ValueError("relation IDs must have one entry per relation exposure row")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("relation exposure query index is out of range")
    pairs = list(zip(indices.tolist(), exposure["relation_ids"].tolist()))
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate relation stratum in exposure")
    sums = {}
    for field in EDGE_FIELDS:
        query = exposure[field]
        relation = exposure[f"relation_{field}"]
        if len(query) != n or len(relation) != len(indices):
            raise ValueError(f"exposure length mismatch for {field}")
        if np.any(query < 0) or np.any(relation < 0):
            raise ValueError(f"negative exposure count for {field}")
        relation_sums = np.zeros(n, dtype=np.int64)
        np.add.at(relation_sums, indices, relation)
        if not np.array_equal(query, relation_sums):
            raise ValueError(f"relation/query exposure mismatch for {field}")
        sums[field] = int(query.sum())
        if totals is not None and sums[field] != totals.get(field, 0):
            raise ValueError(f"query/aggregate exposure mismatch for {field}")
    for prefix in ("", "relation_"):
        eligible = exposure[prefix + "eligible_edges"]
        unavailable = (
            exposure[prefix + "singleton_edges"] + exposure[prefix + "unrewirable_edges"]
        )
        if np.any(exposure[prefix + "changed_edges"] + unavailable > eligible):
            raise ValueError("changed/singleton/unrewirable counts exceed eligible edges")
        maximum = exposure[prefix + "max_changed_edges"]
        if np.any(exposure[prefix + "changed_edges"] > maximum) or np.any(maximum + unavailable > eligible):
            raise ValueError("realized/max/eligible edge counts do not reconcile")
    return sums


def write_exposure_npz(path, batches):
    """Save validated real rows and flatten relation indices across batches.

    Malformed rows are errors rather than silently omitted. Unknown fractions
    for zero eligible edges are NaN; JSON reports use null for this case.
    """
    temporal_fields = tuple(
        f"{prefix}temporal_{key}" for prefix in ("", "base_") for key in TEMPORAL_FIELDS
    )
    fields = QUERY_FIELDS + tuple(
        key for key in temporal_fields if batches and key in batches[0]
    )
    fields += tuple(
        f"{prefix}{key}" for prefix, keys in (("availability_", AVAILABILITY_FIELDS), ("fanout_", FANOUT_FIELDS))
        for key in keys if batches and f"{prefix}{key}" in batches[0]
    )
    query = {key: [] for key in fields}
    relation = {"relation_query_indices": [], "relation_ids": []}
    relation.update({f"relation_{key}": [] for key in EDGE_FIELDS})
    offset = 0
    for batch in batches:
        reconcile_exposure(batch)
        for key in fields:
            if len(batch[key]) != len(batch["target_node_idxs"]):
                raise ValueError(f"query exposure length mismatch for {key}")
            query[key].append(batch[key])
        for key, accumulator in relation.items():
            values = batch[key]
            accumulator.append(values + offset if key == "relation_query_indices" else values)
        offset += len(batch["target_node_idxs"])
    arrays = {
        key: np.concatenate(values) if values else np.array([], dtype=np.int64)
        for key, values in (query | relation).items()
    }
    arrays["changed_edge_fractions"] = np.divide(
        arrays["changed_edges"], arrays["eligible_edges"],
        out=np.full(offset, np.nan), where=arrays["eligible_edges"] != 0,
    )
    arrays["realized_max_fractions"] = np.divide(
        arrays["changed_edges"], arrays["max_changed_edges"],
        out=np.full(offset, np.nan), where=arrays["max_changed_edges"] != 0,
    )
    arrays["no_changeable_edges"] = arrays["max_changed_edges"] == 0
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    return arrays


def configure_complete_evaluation(ev, expected_count):
    """Avoid the redundant floor batch cap after the sampler item limit is set.

    Only this single-task/single-rank project runner is supported. Rustler was
    already constructed with its item limit; clearing the evaluator-only cap
    leaves its sampled items, context policy and batch size unchanged. Use this
    for both no-inference traversal and later evaluate_raw calls.
    """
    if len(ev.tasks) != 1 or ev.world_size != 1:
        raise ValueError("validation requires exactly one task and one evaluator rank")
    task = ev.tasks[0]
    loader = ev.eval_loaders[task]
    dataset = loader.dataset
    if dataset.rustler_dataset.num_items != expected_count:
        raise ValueError(f"sampler must contain exactly {expected_count} intended items")
    ev.items_per_task = None
    return loader


class MaskCaptureLoader:
    """Keep sampler metadata available when evaluate_raw pops batch_mask.

    The original batch, including all phantom slots, still reaches RT. Only
    project validation/exposure uses the captured mask to select real rows.
    """

    def __init__(self, loader, holder):
        self.loader = loader
        self.dataset = loader.dataset
        self.holder = holder

    def __iter__(self):
        for batch in self.loader:
            self.holder.batch_mask = numpy_batch({"mask": batch["batch_mask"]})["mask"]
            yield batch


def capture_evaluator_masks(ev, holder):
    """Install mask capture for subsequent full evaluator passes, not RT code."""
    for task, loader in ev.eval_loaders.items():
        captured = MaskCaptureLoader(loader, holder)
        ev.eval_loaders[task] = captured
        ev.eval_loader_iters[task] = iter(captured)


def rewire_and_validate_batch(base, seed, alpha=1.0, availability_policy=None):
    """Apply maximum rewiring and inspect a real-row batch at one shared seam.

    Return structural/temporal errors alongside exposure rather than losing
    offending-batch evidence. Malformed inputs or inconsistent accounting raise.
    No model invocation occurs here; callers must gate on errors/uncertainty.
    """
    cap = {key: value.copy() for key, value in base.items()}
    arm = {key: value.copy() for key, value in base.items()}
    totals = {}
    sequence_stats = []
    arm["f2p_nbr_idxs"] = rewire_f2p_nbr(
        base["f2p_nbr_idxs"], base["f2p_rel_idxs"], base["node_idxs"],
        base["is_padding"], seed, stats=totals, sequence_stats=sequence_stats, alpha=alpha,
    )
    assert_held_fixed(cap, base)
    if not np.array_equal(cap["f2p_nbr_idxs"], base["f2p_nbr_idxs"]):
        raise ValueError("rewiring mutated the clean parent input")
    assert_held_fixed(cap, arm)
    temporal = {}
    for name, batch in (("base", cap), ("rewired", arm)):
        temporal[name] = summarize_temporal_status(
            batch["f2p_nbr_idxs"], batch["node_idxs"], batch["timestamps"],
            batch["is_padding"], batch["is_targets"],
            timeless_ranges=timeless_parent_ranges(availability_policy),
        )
        reconcile_temporal(temporal[name], totals["eligible_edges"])
    structural_errors = assert_structural_invariants(
        cap["f2p_nbr_idxs"], arm["f2p_nbr_idxs"], arm["f2p_rel_idxs"],
        arm["node_idxs"], arm["is_padding"],
        enforce_fanout=availability_policy is None,
    )
    errors = structural_errors.copy()
    for name, counts in temporal.items():
        errors.extend(f"{name} temporal: {error}" for error in counts["errors"])
    exposure = summarize_query_exposure(
        arm["node_idxs"], arm["f2p_rel_idxs"], arm["is_padding"],
        arm["is_targets"], arm["col_name_idxs"], arm["is_task_nodes"], sequence_stats,
    )
    exposure["configured_context_sizes"] = np.full(len(cap["node_idxs"]), cap["node_idxs"].shape[1], dtype=np.int64)
    before = attention_fanout_counts(cap["node_idxs"], cap["f2p_nbr_idxs"], cap["is_padding"])
    after = attention_fanout_counts(arm["node_idxs"], arm["f2p_nbr_idxs"], arm["is_padding"])
    for key in ("feat", "nbr"):
        exposure[f"fanout_{key}_changed_tokens"] = (before[key] != after[key]).sum(axis=1, dtype=np.int64)
        exposure[f"fanout_{key}_absolute_delta"] = np.abs(after[key] - before[key]).sum(axis=1, dtype=np.int64)
    exposure["fanout_parent_set_changed_tokens"] = (
        effective_parent_set_sizes(cap["f2p_nbr_idxs"], cap["node_idxs"], cap["is_padding"])
        != effective_parent_set_sizes(arm["f2p_nbr_idxs"], arm["node_idxs"], arm["is_padding"])
    ).sum(axis=1, dtype=np.int64)
    ids, _, support = evaluator_query_arrays(cap)
    if not np.array_equal(exposure["target_node_idxs"], ids):
        raise ValueError("exposure target IDs differ from evaluator target IDs")
    if not np.array_equal(exposure["labeled_support_counts"], support):
        raise ValueError("exposure labeled support differs from evaluator")
    reconcile_exposure(exposure, totals)
    for name, counts in temporal.items():
        prefix = "base_" if name == "base" else ""
        for key in TEMPORAL_FIELDS:
            exposure[f"{prefix}temporal_{key}"] = np.array([
                query[key] for query in counts["queries"]
            ], dtype=np.int64)
    return {
        "batch": arm, "exposure": exposure, "totals": totals,
        "base_temporal": temporal["base"], "temporal": temporal["rewired"],
        "structural_errors": structural_errors, "errors": errors,
    }


def validate_batches(batches, expected_target_ids, seeds, ctx, fractions=None, availability_policy=None):
    """Traverse real batches and return per-arm validation evidence and exposure.

    Failures return an explicitly incomplete report, including already observed
    coverage/temporal counts. Unknown timestamps remain uncertainty rather than
    validity. No model is loaded by this function.
    """
    seeds = [int(seed) for seed in seeds]
    arms = [(seed, 1.0, seed) for seed in seeds] if fractions is None else experiment_arms(seeds, fractions)
    report = {
        "status": "failed", "complete": False, "inference_ready": False,
        "ctx": int(ctx), "seeds": seeds, "batches_processed": 0,
        "phantom_rows": 0, "target_node_idxs": [], "labels": [], "errors": [],
        "context_token_counts": [], "base_temporal_batches": 0,
        "batch_fingerprints": [], "fractions": list(fractions) if fractions is not None else [1.0],
        "fanout_check": ("record per-token fanout as an incidence effect, not a rejection gate"
                         if availability_policy else "strict per-token key-count preservation"),
        "base_temporal": {key: 0 for key in TEMPORAL_FIELDS},
        "per_arm": {
            str(name): {"totals": {key: 0 for key in EDGE_FIELDS},
                        "requested_strength": alpha, "matching_seed": seed,
                        "temporal": {key: 0 for key in TEMPORAL_FIELDS},
                        "accounted_batches": 0, "validated_batches": 0,
                        "feature_fanout_preserved": None, "neighbor_fanout_preserved": None,
                        "structural_error_count": 0, "structural_errors": []}
            for name, alpha, seed in arms
        },
        "availability_policy": availability_policy,
        "availability_totals": {key: 0 for key in AVAILABILITY_FIELDS},
    }
    exposures = {name: [] for name, _, _ in arms}
    try:
        expected = np.asarray(expected_target_ids, dtype=np.int64)
        if expected.ndim != 1 or not len(expected) or len(np.unique(expected)) != len(expected):
            raise ValueError("expected target IDs must be a nonempty unique vector")
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be nonempty, unique and nonnegative")
        report["expected_queries"] = len(expected)
        for raw in batches:
            report["batches_processed"] += 1
            base, phantom_count = real_query_batch(raw)
            availability_counts = None
            if availability_policy and len(base["node_idxs"]):
                base, availability_counts = available_context(base, availability_policy)
                for key in AVAILABILITY_FIELDS:
                    report["availability_totals"][key] += int(availability_counts[key].sum())
            report["batch_fingerprints"].append(batch_fingerprint(base))
            report["phantom_rows"] += phantom_count
            if base["node_idxs"].shape[1] != ctx:
                raise ValueError("sampled context size differs from configured context")
            ids, labels, _support = evaluator_query_arrays(base)
            report["target_node_idxs"].extend(ids.tolist())
            report["labels"].extend(labels.tolist())
            report["context_token_counts"].extend(
                (~base["is_padding"]).sum(axis=1).tolist()
            )
            if not len(ids):
                continue
            if np.any(base["node_idxs"][~base["is_padding"]] < 0):
                raise ValueError("non-padding cell has an invalid source node ID")
            if np.any(
                (base["f2p_nbr_idxs"] >= 0) & (base["f2p_rel_idxs"] < 0)
                & ~base["is_padding"][..., None]
            ):
                raise ValueError("present FK parent has no relation identity")
            base_time = summarize_temporal_status(
                base["f2p_nbr_idxs"], base["node_idxs"], base["timestamps"],
                base["is_padding"], base["is_targets"],
                timeless_ranges=timeless_parent_ranges(availability_policy),
            )
            reconcile_temporal(base_time)
            for key in TEMPORAL_FIELDS:
                report["base_temporal"][key] += base_time[key]
            report["base_temporal_batches"] += 1
            if base_time["errors"]:
                report["base_temporal_error_count"] = len(base_time["errors"])
                raise ValueError(f"base temporal (first 20 errors): {base_time['errors'][:20]}")
            for name, alpha, seed in arms:
                evidence = rewire_and_validate_batch(base, seed, alpha, availability_policy)
                attach_availability(evidence["exposure"], availability_counts)
                summary = report["per_arm"][str(name)]
                for key in TEMPORAL_FIELDS:
                    summary["temporal"][key] += evidence["temporal"][key]
                for key in EDGE_FIELDS:
                    summary["totals"][key] += evidence["totals"].get(key, 0)
                summary["accounted_batches"] += 1
                summary["structural_error_count"] += len(evidence["structural_errors"])
                summary["structural_errors"].extend(evidence["structural_errors"][:20])
                for label, code in (("feature", "feat"), ("neighbor", "nbr")):
                    key = f"{label}_fanout_preserved"
                    changed = int(evidence["exposure"][f"fanout_{code}_changed_tokens"].sum())
                    summary[key] = summary[key] is not False and changed == 0
                summary.setdefault("fanout_totals", {key: 0 for key in FANOUT_FIELDS})
                for key in FANOUT_FIELDS:
                    summary["fanout_totals"][key] += int(evidence["exposure"][f"fanout_{key}"].sum())
                exposures[name].append(evidence["exposure"])
                if evidence["errors"]:
                    raise ValueError(f"arm {name} (first 20 errors): {evidence['errors'][:20]}")
                summary["validated_batches"] += 1
        observed = np.asarray(report["target_node_idxs"], dtype=np.int64)
        if len(observed) != len(expected) or not np.array_equal(np.sort(observed), np.sort(expected)):
            raise ValueError("real target coverage differs from the exact intended ID set")
        for name, alpha, seed in arms:
            arm_ids = np.concatenate([batch["target_node_idxs"] for batch in exposures[name]])
            if not np.array_equal(arm_ids, observed):
                raise ValueError(f"seed {seed}: exposure is not in evaluator target order")
            totals = report["per_arm"][str(name)]["totals"]
            for key in EDGE_FIELDS:
                if sum(int(batch[key].sum()) for batch in exposures[name]) != totals[key]:
                    raise ValueError(f"seed {seed}: aggregate exposure mismatch for {key}")
            eligible = totals["eligible_edges"]
            denominator = eligible - totals["singleton_edges"]
            report["per_arm"][str(name)]["changed_edge_fraction"] = (
                totals["changed_edges"] / eligible if eligible else None
            )
            report["per_arm"][str(name)]["rewirable_edge_fraction"] = (
                totals["changed_edges"] / denominator if denominator else None
            )
            report["per_arm"][str(name)]["realized_max_fraction"] = (
                totals["changed_edges"] / totals["max_changed_edges"] if totals["max_changed_edges"] else None
            )
            if alpha == 1 and denominator and totals["changed_edges"] / denominator < 0.8:
                raise ValueError(f"seed {seed}: maximum-rewiring coverage below existing 80% gate")
        report["complete"] = True
        unknown = report["base_temporal"]["unknown"] or any(
            summary["temporal"]["unknown"] for summary in report["per_arm"].values()
        )
        report["status"] = "temporal_uncertainty" if unknown else "passed"
        report["inference_ready"] = not unknown
    except Exception as error:  # noqa: BLE001 -- even sampler failures need incomplete evidence
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["observed_queries"] = len(report["target_node_idxs"])
    report["unique_targets"] = len(set(report["target_node_idxs"]))
    if "expected_queries" in report:
        report["missing_target_ids"] = np.setdiff1d(expected, report["target_node_idxs"]).tolist()
        report["unexpected_target_ids"] = np.setdiff1d(report["target_node_idxs"], expected).tolist()
    return report, exposures
