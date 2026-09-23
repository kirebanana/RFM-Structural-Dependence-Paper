# Clean RT-PluRel FK-Rewiring Experiment — Final Results

Dataset: `rel-f1` / `driver-dnf` (clf, 702 test rows).
Checkpoint: `stanford-star/rt-plurel/classification`.
Preprocessed locally with the new `f2p_rel_idxs` metadata (MiniLM-L12-v2, 384-d).
Intervention: structure-preserving derangement of FK parents **within each
(query context, relation)** stratum — parent multiset per relation, source
degree, and RT feat/nbr attention true-counts are preserved; only *which
specific parent* each row references is randomized.

| ctx | local_ctx | base AUROC | rw101 | rw202 | rw303 | mean Δ | seed range | changed/eligible edge instances (sum over seeds) | rewirable edge fraction | support/query | feat/nbr equal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 48 | 24 | 0.6150 | 0.6052 | 0.6072 | 0.6003 | −0.0108 | 101–303 | 12408/20298 | 88.9% | within-context | equal |
| 128 | 64 | 0.6869 | 0.6217 | 0.6250 | 0.6272 | −0.0623 | 101–303 | 45489/60882 | 80.3% | within-context | equal |

These are historical inference outputs preserved from the original exploratory
workspace; they are not fresh inference from the migrated repository. The
freshly migrated-code check is the real-data dry-run described in the
migration report. “Changed/eligible edge instances” counts eligible
`(query-context, relation, source-row, FK-slot)` edges whose parent identity
changed, divided by all eligible edge instances, summed over the three seeds;
it does not count examples. Per seed the counts are 4,136/6,766 at ctx48 and
15,163/20,294 at ctx128.

Artifacts:
- `results/preliminary/ctx48_lctx24/{manifest.json, preds_base.npz, preds_rw101.npz, preds_rw202.npz, preds_rw303.npz}`
- `results/preliminary/ctx128_lctx64/{...}`
- `artifacts/clean_rewire_preprocessed/rel-f1/{nodes.rkyv, relation_index.json, ...}`

Checkpoint / dataset revisions:
- Checkpoint: `stanford-star/rt-plurel/classification` (HF, cached).
- Dataset: local raw `rel-f1` snapshot `d8e976fd…` (HF `stanford-star/relbench-v1`), re-preprocessed locally.

## Gates (all passed)
- 702 unique node indices, exact 1:1 to RelBench test keys.
- Finite logits; AUROC on raw logits (rank-invariant; matches `rt.eval_utils._score`).
- `unrewirable_strata = 0` for every seed.
- Structural invariants (asserted): present/absent slots identical, parent
  multiset per relation identical, no self-link, effective parent-set size per
  cell identical, feat/nbr attention true-counts identical.
- Rewirable (M≥2 stratum) coverage ≥ 80% (88.9% / 80.3%).

## Interpretation
- **Base reproduces the prior valid RT-PluRel numbers** (0.6150 vs 0.6171;
  0.6869 vs 0.6877), confirming the new preprocessing is behaviorally
  equivalent and the rewiring does not perturb the un-rewired baseline.
- **Rewiring consistently degrades AUROC** in every seed and both contexts
  (base > all 3 rewire arms). Because the rewiring preserves the *structure*
  (existence, count, and target-degree of FK links) and only randomizes the
  *identity of the referenced parent*, this shows RT-PluRel uses **FK parent
  content**, not merely the presence/shape of FK links.
- **Effect grows with context** (Δ −0.011 at ctx48 → −0.062 at ctx128): with
  more sampled context, the specific FK parent a row points to matters more to
  the prediction.

## Caveats (disclosed, not failures)
- In `rel-f1` most `(query, relation)` strata are singletons (M=1) — a single
  eligible edge cannot be permuted, so it is retained unchanged (per plan:
  "retain unchanged singleton strata"). Raw coverage is 61%/75%; the 80% gate
  is evaluated over rewirable (M≥2) strata (88.9%/80.3%).
- The two context configurations (48/24, 128/64) are **descriptive**, not a
  controlled topology-by-context interaction (per plan).
- Support is within-context (transductive); not leakage-free forecasting.
