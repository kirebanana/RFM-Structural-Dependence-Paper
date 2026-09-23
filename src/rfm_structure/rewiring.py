"""Clean structure-preserving FK rewiring for RT-PluRel rel-f1 eval.

Pure functions only (numpy + scipy). No torch, no rt import, so this module
is unit-testable without GPU. The rewiring permutes the *parent* node indices
of foreign-key links *within* each (query context b, relation id r) stratum:

  * slot count per (source row, relation) is preserved  -> source degree kept
  * the multiset of present parent node ids per relation is preserved
    -> every node's in-degree (how many rows point at it) is preserved
    -> RT's feat/nbr attention true-counts are preserved
  * no self-link is ever created
  * absent (-1 / out-of-context) slots are copied byte-for-byte

`f2p_rel_idxs` is NOT modified; only `f2p_nbr_idxs` values are permuted.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_F2P_NBRS = 5


def _present_nodes(node_idxs_b, pad_b):
    return {int(x) for x in node_idxs_b[~pad_b].tolist()}


def _cells_by_source(node_idxs_b, pad_b, s_max):
    """Map source node id -> list of cell positions (s) in this sequence."""
    out = defaultdict(list)
    for s in range(s_max):
        if pad_b[s]:
            continue
        out[int(node_idxs_b[s])].append(s)
    return out


def rewire_f2p_nbr(nbr, rel, node_idxs, is_padding, seed, stats=None, registry=None):
    """Return a copy of ``nbr`` with eligible FK parents rewired.

    nbr, rel  : int64 arrays (B, S, F)
    node_idxs : int64 array (B, S)
    is_padding: bool array (B, S)
    seed      : int rewire seed
    stats     : optional dict to accumulate edge-instance counts and failures
    registry  : optional relation_index dict (relation_id str -> metadata); only
                used for the parent-table sanity assertion in tests.
    """
    nbr = nbr.copy()
    B, S, F = nbr.shape
    if stats is None:
        stats = {}
    rng = np.random.default_rng(np.random.PCG64(seed))
    total_eligible = 0
    total_changed = 0
    unrewirable = 0

    for b in range(B):
        pad = is_padding[b]
        if pad.all():
            continue
        present = _present_nodes(node_idxs[b], pad)
        if not present:
            continue
        cells_by_u = _cells_by_source(node_idxs[b], pad, S)

        # Assert all cells of a source row agree on (rel, nbr) per slot.
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

        # Relation ids present in this sequence (>=0).
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
            if M == 1:
                # A single eligible edge has no alternative to permute to; it is
                # a "singleton stratum" that must stay unchanged (the plan keeps
                # these in the denominator but they cannot be rewired).
                total_eligible += 1
                stats["singleton_edges"] = stats.get("singleton_edges", 0) + 1
                continue
            p_arr = np.array([e[2] for e in edges], dtype=np.int64)
            u_arr = np.array([e[0] for e in edges], dtype=np.int64)

            # Deterministic-but-seed-dependent derangement: shuffle the edge
            # order with the seeded RNG so that, among the (often many) optimal
            # min-cost assignments, a different one is selected per seed. The
            # cost matrix is just the original one with rows/cols permuted, so
            # every selected assignment still respects the self-link / identity
            # constraints and preserves the parent multiset exactly.
            perm = rng.permutation(M)
            edges = [edges[i] for i in perm]
            p_arr = p_arr[perm]
            u_arr = u_arr[perm]

            # Cost matrix for constrained derangement over candidate instances.
            #   self-link (p_j == u_i)        -> INF   (hard forbidden)
            #   identity  (p_j == p_i)         -> 1     (soft: prefer change)
            #   otherwise                      -> 0     (preferred)
            INF = 10 ** 9
            eq_self = p_arr[None, :] == u_arr[:, None]
            eq_id = p_arr[None, :] == p_arr[:, None]
            cost = np.where(eq_self, INF, np.where(eq_id, 1, 0)).astype(np.int64)

            _, col_ind = linear_sum_assignment(cost)
            feasible = all(cost[i, col_ind[i]] < INF for i in range(M))
            if not feasible:
                unrewirable += 1
                continue

            for i in range(M):
                u, k, p = edges[i]
                new_p = int(p_arr[col_ind[i]])
                total_eligible += 1
                if new_p != p:
                    total_changed += 1
                for s in cells_by_u[u]:
                    nbr[b, s, k] = new_p

    stats["eligible_edges"] = stats.get("eligible_edges", 0) + total_eligible
    stats["changed_edges"] = stats.get("changed_edges", 0) + total_changed
    stats["unrewirable_strata"] = stats.get("unrewirable_strata", 0) + unrewirable
    return nbr


def effective_parent_set_sizes(nbr, node_idxs, is_padding):
    """Per-cell count of DISTINCT present parents (excluding the cell's own node)."""
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
    """Total feat / nbr attention true-counts, replicating rt.model mask logic.

    feat query i includes kv j when same_node(i,j) or node_idxs[j] in f2p_nbr_idxs[i].
    nbr  query i includes kv j when f2p_nbr_idxs[j] contains node_idxs[i].
    """
    B = nbr.shape[0]
    feat_total = 0
    nbr_total = 0
    for b in range(B):
        pm = ~is_padding[b]
        if not pm.any():
            continue
        nids = node_idxs[b]
        fpr = nbr[b]  # (S, F)
        # eq[p, k, i] = (fpr[p, k] == nids[i])
        eq = fpr[:, :, None] == nids[None, None, :]  # (S, F, S)
        ref = eq.any(1)  # (S, S): ref[p, i] -> p references i
        ref = ref & pm[:, None] & pm[None, :]
        nbr_cnt = ref.sum(0)  # per i: how many present rows reference i
        feat_ref = eq.any(1)  # (S, S): feat_ref[i, j] -> i references j
        feat_ref = feat_ref & pm[:, None] & pm[None, :]
        feat_cnt = feat_ref.sum(1) + 1  # +1 for same_node self
        feat_cnt[~pm] = 0
        nbr_cnt[~pm] = 0
        feat_total += int(feat_cnt.sum())
        nbr_total += int(nbr_cnt.sum())
    return feat_total, nbr_total


def assert_structural_invariants(base_nbr, new_nbr, rel, node_idxs, is_padding, registry=None):
    """Return a list of error strings (empty == all invariants hold)."""
    errs = []
    B = base_nbr.shape[0]

    # 1. present/absent slots identical
    if not np.array_equal(base_nbr >= 0, new_nbr >= 0):
        errs.append("present/absent slot pattern changed")

    # 2. no self-link in rewired result
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

    # 3. parent multiset per relation preserved
    base_ms = defaultdict(lambda: defaultdict(int))
    new_ms = defaultdict(lambda: defaultdict(int))
    for b in range(B):
        pm = ~is_padding[b]
        present = _present_nodes(node_idxs[b], pm)
        for s in range(base_nbr.shape[1]):
            if not pm[s]:
                continue
            for k in range(base_nbr.shape[2]):
                r = int(rel[b, s, k])
                if r < 0:
                    continue
                bp = int(base_nbr[b, s, k])
                np_ = int(new_nbr[b, s, k])
                if bp >= 0 and bp in present:
                    base_ms[r][bp] += 1
                if np_ >= 0 and np_ in present:
                    new_ms[r][np_] += 1
    for r in set(base_ms) | set(new_ms):
        if dict(base_ms[r]) != dict(new_ms[r]):
            errs.append(f"parent multiset changed for relation {r}")

    # 4. effective parent-set size per cell identical
    e_base = effective_parent_set_sizes(base_nbr, node_idxs, is_padding)
    e_new = effective_parent_set_sizes(new_nbr, node_idxs, is_padding)
    if not np.array_equal(e_base, e_new):
        errs.append("effective parent-set size per cell changed")

    # 5. feat / nbr attention true-counts identical
    fb, nb = attn_true_counts(node_idxs, base_nbr, is_padding)
    fn, nn = attn_true_counts(node_idxs, new_nbr, is_padding)
    if fb != fn:
        errs.append(f"feat true-count changed {fb} -> {fn}")
    if nb != nn:
        errs.append(f"nbr true-count changed {nb} -> {nn}")

    return errs


def _self_test():
    """Synthetic check that rewiring preserves every invariant.

    Uses two relations with DISJOINT parent pools (mirroring real data where
    different FK columns point to different tables / disjoint id ranges), so a
    node never receives the same parent id from two relations.
    """
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
