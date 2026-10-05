"""Reassign FK parents within a prediction query's sampled RT context.

``f2p_nbr_idxs[b, s, k]`` holds the parent row ID for FK slot ``k`` of cell
``s`` in sampled context ``b``; ``-1`` means no in-context link. The parallel
``f2p_rel_idxs`` identifies the FK relation. Several cells may represent the
same source row, so rewiring changes all their matching slots together.

Only parent IDs change. The original parent instances are reassigned within
each (sampled context, relation) group, preserving the parent multiset. The
checks below verify slot presence, self-links, the multiset, and per-cell and
per-token fanout counts. Timestamp checks detect known future parents, but
missing timestamps and database-wide FK validity remain unverified.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_F2P_NBRS = 5
TEMPORAL_FIELDS = (
    "comparisons", "checked", "valid", "known_violations", "unknown",
    "missing_target", "missing_parent", "missing_both",
    "declared_timeless",
)


def _present_nodes(node_idxs_b, pad_b):
    """Return node IDs at non-padding positions of one sampled sequence."""
    return {int(x) for x in node_idxs_b[~pad_b].tolist()}


def _cells_by_source(node_idxs_b, pad_b, s_max):
    """Map source node id -> list of cell positions (s) in this sequence."""
    out = defaultdict(list)
    for s in range(s_max):
        if pad_b[s]:
            continue
        out[int(node_idxs_b[s])].append(s)
    return out


def rewire_f2p_nbr(
    nbr,
    rel,
    node_idxs,
    is_padding,
    seed,
    stats=None,
    registry=None,
    sequence_stats=None,
    alpha=1.0,
):
    """Reassign eligible FK parent IDs without changing the input array.

    One eligible FK edge instance is a (sampled context, relation, source row,
    FK slot) with a parent in the same context. A seeded matching permutes the
    parent instances within each relation stratum. Self-links are forbidden;
    keeping the original parent is allowed but discouraged. A one-edge stratum
    cannot be permuted and stays unchanged.

    Args:
        nbr, rel: Parallel parent-ID and relation-ID arrays, shaped (B, S, F).
        node_idxs, is_padding: Source row IDs and padding mask, shaped (B, S).
        seed: Makes the matching reproducible for fixed inputs.
        stats: Optional mutable totals across all sampled contexts in this call.
        registry: Reserved for relation metadata; currently unused.
        sequence_stats: Optional list filled with one summary per sampled
            context, including counts by relation ID.

    Returns a new parent-ID array. ``changed_edges`` counts eligible edge
    instances whose assigned parent ID actually differs, not prediction queries.
    ``alpha`` targets the fraction of changed IDs in this seed's maximum
    matching. Intermediate strengths select whole cycles in shuffled-index
    order, accepting only strict improvements toward Python's round-to-even
    target. Cycle selection consumes no RNG, preserving the legacy 1.0 stream.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite fraction between 0 and 1")
    nbr = nbr.copy()
    B, S, F = nbr.shape
    if stats is None:
        stats = {}
    rng = np.random.default_rng(np.random.PCG64(seed))
    total_eligible = 0
    total_changed = 0
    total_max_changed = 0
    unrewirable = 0
    unrewirable_edges = 0

    for b in range(B):
        pad = is_padding[b]
        seq_stats = {
            "eligible_edges": 0,
            "changed_edges": 0,
            "max_changed_edges": 0,
            "requested_strength": float(alpha),
            "singleton_edges": 0,
            "unrewirable_edges": 0,
            "unrewirable_strata": 0,
            "unique_parent_nodes": 0,
            "relations": {},
        }
        if pad.all():
            if sequence_stats is not None:
                sequence_stats.append(seq_stats)
            continue
        present = _present_nodes(node_idxs[b], pad)
        if not present:
            if sequence_stats is not None:
                sequence_stats.append(seq_stats)
            continue
        cells_by_u = _cells_by_source(node_idxs[b], pad, S)
        parent_nodes = set()

        # One source row can occupy several cells; all copies must agree on FK slots.
        for u, cells in cells_by_u.items():
            rep = cells[0]
            for k in range(F):
                rv = int(rel[b, rep, k])
                nv = int(nbr[b, rep, k])
                for s in cells[1:]:
                    assert int(rel[b, s, k]) == rv, (
                        f"inconsistent rel idx across cells of source {u} slot {k}"
                    )
                    assert int(nbr[b, s, k]) == nv, (
                        f"inconsistent nbr idx across cells of source {u} slot {k}"
                    )

        # A relation stratum contains only links of one FK type in this context.
        rel_ids = set()
        for u, cells in cells_by_u.items():
            rep = cells[0]
            for k in range(F):
                rv = int(rel[b, rep, k])
                if rv >= 0:
                    rel_ids.add(rv)

        for r in sorted(rel_ids):
            # Collect eligible edges: (source u, slot k, original parent p)
            edges = []
            for u, cells in cells_by_u.items():
                rep = cells[0]
                for k in range(F):
                    if int(rel[b, rep, k]) != r:
                        continue
                    p = int(nbr[b, rep, k])
                    if p < 0:
                        continue
                    if p not in present:
                        continue
                    edges.append((u, k, p))
            if not edges:
                continue

            M = len(edges)
            relation_stats = seq_stats["relations"].setdefault(
                int(r),
                {
                    "eligible_edges": 0,
                    "changed_edges": 0,
                    "max_changed_edges": 0,
                    "singleton_edges": 0,
                    "unrewirable_edges": 0,
                    "unrewirable_strata": 0,
                },
            )
            parent_nodes.update(p for _, _, p in edges)
            total_eligible += M
            seq_stats["eligible_edges"] += M
            relation_stats["eligible_edges"] += M
            if M == 1:
                # Count singletons as eligible even though they cannot change.
                stats["singleton_edges"] = stats.get("singleton_edges", 0) + 1
                seq_stats["singleton_edges"] += 1
                relation_stats["singleton_edges"] += 1
                continue
            p_arr = np.array([e[2] for e in edges], dtype=np.int64)
            u_arr = np.array([e[0] for e in edges], dtype=np.int64)

            # Shuffling breaks ties between valid minimum-cost assignments.
            perm = rng.permutation(M)
            edges = [edges[i] for i in perm]
            p_arr = p_arr[perm]
            u_arr = u_arr[perm]

            # Each column is one original parent instance: assigning every
            # column once preserves multiplicities, including duplicate IDs.
            # Self-links are forbidden; unchanged parent IDs cost more.
            INF = 10 ** 9
            eq_self = p_arr[None, :] == u_arr[:, None]
            eq_id = p_arr[None, :] == p_arr[:, None]
            cost = np.where(eq_self, INF, np.where(eq_id, 1, 0)).astype(np.int64)

            _, col_ind = linear_sum_assignment(cost)
            feasible = all(cost[i, col_ind[i]] < INF for i in range(M))
            if not feasible:
                unrewirable += 1
                unrewirable_edges += M
                seq_stats["unrewirable_strata"] += 1
                seq_stats["unrewirable_edges"] += M
                relation_stats["unrewirable_strata"] += 1
                relation_stats["unrewirable_edges"] += M
                continue

            full_changed = int(np.sum(p_arr[col_ind] != p_arr))
            total_max_changed += full_changed
            seq_stats["max_changed_edges"] += full_changed
            relation_stats["max_changed_edges"] += full_changed
            selected = col_ind
            if alpha < 1:
                selected = np.arange(M)
                visited = set()
                changed = 0
                target = round(alpha * full_changed)
                for start in range(M):
                    if start in visited:
                        continue
                    cycle = []
                    i = start
                    while i not in visited:
                        visited.add(i)
                        cycle.append(i)
                        i = int(col_ind[i])
                    weight = int(np.sum(p_arr[col_ind[cycle]] != p_arr[cycle]))
                    if weight and abs(changed + weight - target) < abs(changed - target):
                        selected[cycle] = col_ind[cycle]
                        changed += weight
            for i in range(M):
                u, k, p = edges[i]
                new_p = int(p_arr[selected[i]])
                if new_p != p:
                    total_changed += 1
                    seq_stats["changed_edges"] += 1
                    relation_stats["changed_edges"] += 1
                for s in cells_by_u[u]:
                    nbr[b, s, k] = new_p

        seq_stats["unique_parent_nodes"] = len(parent_nodes)
        if sequence_stats is not None:
            sequence_stats.append(seq_stats)

    stats["eligible_edges"] = stats.get("eligible_edges", 0) + total_eligible
    stats["changed_edges"] = stats.get("changed_edges", 0) + total_changed
    stats["max_changed_edges"] = stats.get("max_changed_edges", 0) + total_max_changed
    stats["unrewirable_strata"] = stats.get("unrewirable_strata", 0) + unrewirable
    stats["unrewirable_edges"] = stats.get("unrewirable_edges", 0) + unrewirable_edges
    return nbr


def effective_parent_set_sizes(nbr, node_idxs, is_padding):
    """Count distinct in-context parents per cell, excluding its own row."""
    B, S, F = nbr.shape
    out = np.zeros((B, S), dtype=np.int64)
    for b in range(B):
        pad = is_padding[b]
        present = _present_nodes(node_idxs[b], pad)
        for s in range(S):
            if pad[s]:
                continue
            u = int(node_idxs[b, s])
            parents = set()
            for k in range(F):
                p = int(nbr[b, s, k])
                if p >= 0 and p in present and p != u:
                    parents.add(p)
            out[b, s] = len(parents)
    return out


def attn_true_counts(node_idxs, nbr, is_padding):
    """Sum feature and neighbor attention key counts over non-padding tokens."""
    fanout = attention_fanout_counts(node_idxs, nbr, is_padding)
    return int(fanout["feat"].sum()), int(fanout["nbr"].sum())


def attention_fanout_counts(node_idxs, nbr, is_padding):
    """Count RT feature and neighbor keys for each cell/token.

    Feature keys share the token's source row or belong to one of its FK parents;
    neighbor keys belong to source rows pointing to that token's row. Repeated
    cells count as separate keys. Returns ``feat`` and ``nbr`` arrays of shape
    (B, S), matching RT's key-count rules, not full attention-mask equality.
    """
    B, S, F = nbr.shape
    feat = np.zeros((B, S), dtype=np.int64)
    nbr_count = np.zeros((B, S), dtype=np.int64)
    for b in range(B):
        pad = is_padding[b]
        present_positions = np.flatnonzero(~pad)
        node_positions = defaultdict(list)
        parent_sources = defaultdict(set)
        for s in present_positions:
            node_positions[int(node_idxs[b, s])].append(s)
            for k in range(F):
                parent = int(nbr[b, s, k])
                if parent >= 0:
                    parent_sources[parent].add(s)
        for s in present_positions:
            candidates = {int(node_idxs[b, s])}
            candidates.update(int(p) for p in nbr[b, s] if p >= 0)
            feat[b, s] = sum(len(node_positions[parent]) for parent in candidates)
            nbr_count[b, s] = len(parent_sources[int(node_idxs[b, s])])
    return {"feat": feat, "nbr": nbr_count}


def summarize_temporal_status(nbr, node_idxs, timestamps, is_padding, is_targets, timeless_ranges=()):
    """Count in-context parent comparisons once per source-node/FK-slot.

    Callers must filter evaluator phantoms first. Unknown timestamps are
    ``i32::MIN``. Missing-target and missing-parent counts overlap; subtract
    missing-both to obtain unknown. Checked includes known violations:
    comparisons = checked + unknown + declared_timeless. Declared timeless
    parent ranges must come from source schema metadata, not missing values.
    Repeated cells must agree on timestamps and parent slots. No timestamp
    is inferred from context membership or another row.
    """
    missing_timestamp = np.iinfo(np.int32).min
    fields = TEMPORAL_FIELDS
    result = {key: 0 for key in fields}
    result["queries"] = []
    errors = []
    B, _, F = nbr.shape
    for b in range(B):
        counts = {key: 0 for key in fields}
        result["queries"].append(counts)
        target_positions = np.flatnonzero(is_targets[b] & ~is_padding[b])
        if len(target_positions) != 1:
            errors.append(f"b={b}: expected one prediction target, found {len(target_positions)}")
            continue
        target_s = int(target_positions[0])
        target_ts = int(timestamps[b, target_s])
        node_timestamps = {}
        cells_by_source = _cells_by_source(node_idxs[b], is_padding[b], nbr.shape[1])
        for node, cells in cells_by_source.items():
            values = {int(timestamps[b, s]) for s in cells}
            if len(values) != 1:
                errors.append(f"b={b}: node {node} has inconsistent timestamps")
            node_timestamps[node] = (
                next(iter(values)) if len(values) == 1 else missing_timestamp
            )
        present = set(cells_by_source)
        for node, cells in cells_by_source.items():
            s = cells[0]
            for k in range(F):
                parent = int(nbr[b, s, k])
                if any(int(nbr[b, other, k]) != parent for other in cells[1:]):
                    errors.append(f"b={b}: node {node} has inconsistent parent slot {k}")
                if parent < 0 or parent not in present:
                    continue
                parent_ts = node_timestamps[parent]
                counts["comparisons"] += 1
                target_missing = target_ts == missing_timestamp
                timeless = any(start <= parent < stop for start, stop in timeless_ranges)
                if timeless and not target_missing:
                    counts["declared_timeless"] += 1
                    continue
                parent_missing = parent_ts == missing_timestamp and not timeless
                counts["missing_target"] += int(target_missing)
                counts["missing_parent"] += int(parent_missing)
                counts["missing_both"] += int(target_missing and parent_missing)
                if target_missing or parent_missing:
                    counts["unknown"] += 1
                    continue
                counts["checked"] += 1
                if parent_ts > target_ts:
                    counts["known_violations"] += 1
                    errors.append(
                        f"b={b} s={s} k={k}: parent {parent} timestamp "
                        f"{parent_ts} exceeds target timestamp {target_ts}"
                    )
                else:
                    counts["valid"] += 1
        for key in fields:
            result[key] += counts[key]
    result["errors"] = errors
    return result


def temporal_parent_violations(nbr, node_idxs, timestamps, is_padding, is_targets):
    """Return known temporal/malformed-row errors, deduplicated by source slot.

    An empty list is not proof of complete validity: use
    ``summarize_temporal_status`` to quantify missing timestamp comparisons.
    """
    return summarize_temporal_status(
        nbr, node_idxs, timestamps, is_padding, is_targets
    )["errors"]


def assert_structural_invariants(
    base_nbr,
    new_nbr,
    rel,
    node_idxs,
    is_padding,
    registry=None,
    timestamps=None,
    is_targets=None,
    enforce_fanout=True,
):
    """Return errors when rewiring breaks a checked property.

    Checks FK-slot presence, self-links and parent multisets per (sampled context,
    relation). Strict mode additionally requires unchanged token key counts and
    distinct parents. Incidence experiments set enforce_fanout=False and record
    those changes as diagnostics instead. With
    timestamps and target flags, also checks known future-parent links in both
    clean and rewired inputs. The reserved ``registry`` is not used; this does
    not check database-wide FK validity or unknown timestamps.
    """
    errs = []
    B = base_nbr.shape[0]

    # Missing/out-of-context FK slots must stay missing.
    if not np.array_equal(base_nbr >= 0, new_nbr >= 0):
        errs.append("present/absent slot pattern changed")

    for b in range(B):
        present = _present_nodes(node_idxs[b], is_padding[b])
        cells_by_source = _cells_by_source(node_idxs[b], is_padding[b], base_nbr.shape[1])
        for source, cells in cells_by_source.items():
            rep = cells[0]
            for slot in range(base_nbr.shape[2]):
                if any(new_nbr[b, cell, slot] != new_nbr[b, rep, slot] for cell in cells[1:]):
                    errs.append(f"b={b} source={source} slot={slot}: inconsistent rewired FK copies")
                for cell in cells:
                    parent = int(base_nbr[b, cell, slot])
                    if (parent < 0 or parent not in present or rel[b, cell, slot] < 0) and new_nbr[b, cell, slot] != parent:
                        errs.append(f"b={b} s={cell} k={slot}: ineligible FK slot changed")

    # A source row must not become its own FK parent.
    for b in range(B):
        pm = ~is_padding[b]
        for s in range(base_nbr.shape[1]):
            if not pm[s]:
                continue
            u = int(node_idxs[b, s])
            for k in range(base_nbr.shape[2]):
                p = int(new_nbr[b, s, k])
                if p >= 0 and p == u:
                    errs.append(f"self-link created at b={b} s={s} k={k}")
                    break

    # A source row may occupy several cells, but each FK slot is one edge
    # instance. Counting every cell would incorrectly weight the parent multiset
    # by the source row's sampled column count.
    base_ms = defaultdict(lambda: defaultdict(int))
    new_ms = defaultdict(lambda: defaultdict(int))
    for b in range(B):
        pad = is_padding[b]
        present = _present_nodes(node_idxs[b], pad)
        cells_by_source = _cells_by_source(node_idxs[b], pad, base_nbr.shape[1])
        for cells in cells_by_source.values():
            s = cells[0]
            for k in range(base_nbr.shape[2]):
                r = int(rel[b, s, k])
                if r < 0:
                    continue
                bp = int(base_nbr[b, s, k])
                np_ = int(new_nbr[b, s, k])
                if bp >= 0 and bp in present:
                    base_ms[(b, r)][bp] += 1
                if np_ >= 0 and np_ in present:
                    new_ms[(b, r)][np_] += 1
    for key in set(base_ms) | set(new_ms):
        if dict(base_ms[key]) != dict(new_ms[key]):
            b, r = key
            errs.append(f"parent multiset changed for b={b} relation={r}")

    # Duplicated FK targets count once toward a cell's distinct parent set.
    e_base = effective_parent_set_sizes(base_nbr, node_idxs, is_padding)
    e_new = effective_parent_set_sizes(new_nbr, node_idxs, is_padding)
    if enforce_fanout and not np.array_equal(e_base, e_new):
        errs.append("effective parent-set size per cell changed")

    # Match RT's number of keys per token, not the identities of those keys.
    base_fanout = attention_fanout_counts(node_idxs, base_nbr, is_padding)
    new_fanout = attention_fanout_counts(node_idxs, new_nbr, is_padding)
    for name in ("feat", "nbr"):
        if enforce_fanout and not np.array_equal(base_fanout[name], new_fanout[name]):
            errs.append(f"{name} attention fanout changed per token")

    # Check both arms: a sampled parent may already be dated after the query.
    if (timestamps is None) != (is_targets is None):
        errs.append("timestamps and is_targets must be supplied together")
    elif timestamps is not None:
        errs.extend(f"base temporal: {error}" for error in temporal_parent_violations(
            base_nbr, node_idxs, timestamps, is_padding, is_targets
        ))
        errs.extend(f"rewired temporal: {error}" for error in temporal_parent_violations(
            new_nbr, node_idxs, timestamps, is_padding, is_targets
        ))

    return errs


def summarize_query_exposure(
    node_idxs,
    rel,
    is_padding,
    is_targets,
    col_name_idxs,
    is_task_nodes,
    sequence_stats,
):
    """Summarize FK exposure for each prediction query in a batch.

    Each entry of ``sequence_stats`` describes one sampled context. Eligible
    edges include singletons and unrewirable strata; changed edges have a new
    parent ID. ``unique_parent_nodes`` counts distinct in-context FK targets.
    ``labeled_support_counts`` counts other task-row cells in the target column.

    Query arrays have one entry per batch row. Relation arrays have one entry
    per nonempty relation stratum and use ``relation_query_indices`` to point
    back to that row. Rows without exactly one target are marked as non-queries.
    """
    B, _ = node_idxs.shape
    if len(sequence_stats) != B:
        raise ValueError("sequence_stats must have one entry per batch sequence")
    result = {
        "is_prediction_query": np.zeros(B, dtype=bool),
        "target_node_idxs": np.full(B, -1, dtype=np.int64),
        "context_token_counts": (~is_padding).sum(axis=1, dtype=np.int64),
        "eligible_edges": np.zeros(B, dtype=np.int64),
        "changed_edges": np.zeros(B, dtype=np.int64),
        "max_changed_edges": np.zeros(B, dtype=np.int64),
        "requested_strength": np.zeros(B, dtype=float),
        "singleton_edges": np.zeros(B, dtype=np.int64),
        "unrewirable_edges": np.zeros(B, dtype=np.int64),
        "unrewirable_strata": np.zeros(B, dtype=np.int64),
        "unique_parent_nodes": np.zeros(B, dtype=np.int64),
        "labeled_support_counts": np.full(B, -1, dtype=np.int64),
        "relation_query_indices": [],
        "relation_ids": [],
        "relation_eligible_edges": [],
        "relation_changed_edges": [],
        "relation_max_changed_edges": [],
        "relation_singleton_edges": [],
        "relation_unrewirable_edges": [],
        "relation_unrewirable_strata": [],
    }
    for b, stats in enumerate(sequence_stats):
        for key in (
            "eligible_edges",
            "changed_edges",
            "max_changed_edges",
            "requested_strength",
            "singleton_edges",
            "unrewirable_edges",
            "unrewirable_strata",
            "unique_parent_nodes",
        ):
            result[key][b] = stats[key]
        target_positions = np.flatnonzero(is_targets[b] & ~is_padding[b])
        if len(target_positions) != 1:
            continue
        target_s = int(target_positions[0])
        target_node = int(node_idxs[b, target_s])
        target_col = col_name_idxs[b, target_s]
        result["is_prediction_query"][b] = True
        result["target_node_idxs"][b] = target_node
        result["labeled_support_counts"][b] = int(np.sum(
            is_task_nodes[b]
            & ~is_padding[b]
            & (col_name_idxs[b] == target_col)
            & (node_idxs[b] != target_node)
        ))
        for relation_id, relation_stats in stats["relations"].items():
            result["relation_query_indices"].append(b)
            result["relation_ids"].append(relation_id)
            for key in (
                "eligible_edges",
                "changed_edges",
                "max_changed_edges",
                "singleton_edges",
                "unrewirable_edges",
                "unrewirable_strata",
            ):
                result[f"relation_{key}"].append(relation_stats[key])
    for key in (
        "relation_query_indices",
        "relation_ids",
        "relation_eligible_edges",
        "relation_changed_edges",
        "relation_max_changed_edges",
        "relation_singleton_edges",
        "relation_unrewirable_edges",
        "relation_unrewirable_strata",
    ):
        result[key] = np.asarray(result[key], dtype=np.int64)
    return result


def _self_test():
    """Check rewiring on two unpadded contexts with separate parent pools."""
    B, S, F = 2, 6, 3
    # child rows 0..2 have FK; rows 3..5 are pure parent rows (pool B)
    node_idxs = np.array(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=np.int64
    )
    rel = np.full((B, S, F), -1, dtype=np.int64)
    rel[:, :3, 0] = 100  # relation 100 (pool A: 10..12 / 20..22)
    rel[:, :3, 1] = 200  # relation 200 (pool B: 13..15 / 23..25)
    is_padding = np.zeros((B, S), dtype=bool)

    nbr = np.full((B, S, F), -1, dtype=np.int64)
    for b in range(B):
        c0, p0 = (10, 13) if b == 0 else (20, 23)
        # relation 100: cycle over pool A {c0, c0+1, c0+2}
        nbr[b, 0, 0] = c0 + 1
        nbr[b, 1, 0] = c0 + 2
        nbr[b, 2, 0] = c0
        # relation 200: child i -> pool B node p0+i
        nbr[b, 0, 1] = p0
        nbr[b, 1, 1] = p0 + 1
        nbr[b, 2, 1] = p0 + 2

    stats = {}
    new_nbr = rewire_f2p_nbr(nbr, rel, node_idxs, is_padding, seed=101, stats=stats)
    errs = assert_structural_invariants(nbr, new_nbr, rel, node_idxs, is_padding)
    assert not errs, f"invariants failed: {errs}"
    assert stats["eligible_edges"] == B * 3 * 2, stats
    assert stats["changed_edges"] > 0, "nothing changed"
    from collections import Counter
    # relation 100 parent multiset per sequence must stay {c0,c0+1,c0+2}
    for b in range(B):
        c0 = 10 if b == 0 else 20
        vals = [int(new_nbr[b, i, 0]) for i in range(3)]
        assert Counter(vals) == Counter([c0, c0 + 1, c0 + 2]), (b, vals)
    print(
        f"[self-test] OK eligible_edges={stats['eligible_edges']} "
        f"changed_edges={stats['changed_edges']}"
    )


if __name__ == "__main__":
    _self_test()
