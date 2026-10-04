# Preliminary experiment methodology

## What is held fixed?

RT receives a sampled sequence of **cells/tokens**, not the entire database.
A source row can occupy several cells. A prediction query is one target task
row; its sampled context includes typed cell values, schema information,
source IDs, FK parent IDs, target/task flags and timestamps.

Every arm uses the same query and sampled cells in the same order, including
their values, labels, timestamps and source IDs. The frozen model is shared
across arms within a context. The intervention changes only `f2p_nbr_idxs`,
which determines concrete FK parent identities and therefore relational
attention connections. `f2p_rel_idxs` identifies the FK relation for matching
and accounting; it is not an additional learned relation input.

An **eligible FK edge instance** is a `(sampled context, relation, source node,
FK slot)` with a parent present in that context. Repeated cells representing
the same source slot count once. A **changed** instance receives a different
parent node ID. Neither count is a count of prediction queries.

## Corruption strengths

For each context/relation group, seeded minimum-cost matching reassigns the
original parent instances while prohibiting self-links and minimizing
unchanged parent IDs. Duplicate parent IDs remain separate instances in the
assignment. All copies of a source slot are updated together.

The resulting assignment is a permutation, decomposed into disjoint cycles:

- **0%:** apply no cycles; parent identities remain original.
- **100%:** apply the full assignment, preserving the original full-rewiring
  behavior and RNG stream for fixed input/seed.
- **50%:** visit cycles in seeded shuffled-index order. Select a whole cycle
  only if its changed-ID count strictly improves the distance to
  `round(0.5 × full-matching changed IDs)` for that group. Ties stay unchanged;
  cycles changing no IDs are ignored. Python rounding uses ties-to-even.

Selecting complete cycles preserves the parent-instance multiset. It does
not by itself guarantee all token-level invariants, which are checked
separately. Discrete cycle sizes can make the achieved intermediate strength
approximate or zero in small groups.

Requested strength is relative to the **current seeded maximum matching**,
not all eligible edges. Both achieved ratios are reported:

```text
changed / eligible
changed / full-matching maximum changed
```

Zero denominators are undefined, represented as NaN in NPZ and null in JSON.
Singleton and infeasible groups remain eligible but unchanged, with separate
accounting. The inherited 80%-of-non-singleton coverage check applies only to
100% arms. Inference also rejects non-clean arms with no realized changes.

## Validation before inference

Validation traverses every evaluator batch without loading weights. It uses
`batch_mask` to exclude phantom rows, then requires exactly the 702 distinct
metadata-defined target IDs. Failed/timed-out intended sampler items can also
be masked as phantoms; the exact-ID check detects their loss.

At the default token budget, batch sizes are 5,461 for context 48 and 2,048 for
context 128, so both configurations fit the 702 intended queries in one batch.
The runner avoids RT's redundant floor batch cap after sampler construction;
it does not change the sampled item limit or batch size.

Every arm checks:

- held-fixed sampled inputs, source-slot copy consistency and unchanged
  absent/ineligible slots;
- no self-links, original parent multisets and distinct-parent counts;
- feature/neighbor attention **key counts per token**;
- target-ID/label alignment and evaluator-equivalent labeled support;
- relation → query → aggregate exposure reconciliation.

Key-count preservation does **not** mean identical attention adjacency:
which rows attend to one another is intentionally changed. Source-edge
permutations can fail token-level checks when repeated-row cell multiplicities
or duplicate parents interact. Such failures are rejected rather than waived.

Temporal checks compare each in-context parent against the query timestamp,
once per source-node/FK-slot, in both clean and rewired contexts. Rustler's
`i32::MIN` means missing timestamp. Reports distinguish checked, valid,
known-future and unknown comparisons, including missing target, parent or both.
Missing-target and missing-parent counts overlap; subtract missing-both to
obtain unknown. Inconsistent timestamps across copies of a node are errors.

Missing timestamps are not inferred from context membership. The current
runner stops on unresolved temporal uncertainty; interpretation or a defensible
methodological restriction must be established before repeating validation.
These checks concern sampled in-context links, not database-wide FK validity.

Both contexts and all planned arms are validated before any forward call.
During inference, a SHA-256 fingerprint binds each real batch to the sampled
input validated earlier. Input equality and post-prediction mutation checks
protect pairing throughout the run.

## Execution and artifacts

The experiment is fixed to `rel-f1/driver-dnf` and the RT-PluRel classification
checkpoint, with 48/24 and 128/64 contexts. Explicit CLI flags configure
contexts, seeds, fractions and local paths; `configs/*.yaml` are reference
records, not executable configuration.

```bash
# Maximum-only validation; repeat with --ctx 128 --local-ctx 64.
uv run python experiments/run_clean_rewire.py --ctx 48 --local-ctx 24 \
  --fractions 1 --dry-run --pre-dir artifacts/clean_rewire_preprocessed

# Complete two-context matrix validation.
uv run python experiments/run_clean_rewire.py --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed

# Validation followed by conditional inference, using existing local weights.
uv run python experiments/run_clean_rewire.py \
  --pre-dir artifacts/clean_rewire_preprocessed \
  --checkpoint-dir /path/to/local/rt-plurel/classification
```

The full matrix contains seven arms per context: `base`, `a050_s101/202/303`
and `a100_s101/202/303`. Clean inference runs once. The local checkpoint folder
must actually contain the intended RT-PluRel weights; recorded hashes identify
the files used but do not establish their original model-family provenance.

Current conservative deployment checks require CUDA compute capability >=8,
free VRAM >=8 GiB or 3× checkpoint bytes, and available host RAM >=2 GiB or 2×
checkpoint bytes, whichever is larger. These are operational checks, not
measured minimum model requirements. Actual runtime/OOM failures still leave
incomplete reports. The runner downloads neither data nor weights.

Each invocation creates a fresh directory under ignored `results/runs/`:

| Artifact | Contents |
|---|---|
| `matrix.json` | Overall validation/execution status and provenance |
| `<context>/validation.json` | Coverage, structural/temporal checks, counts and replay fingerprints |
| `<context>/validation_exposure_<arm>.npz` | Pre-inference per-query/relation exposure |
| `<context>/preds_<arm>.npz` | Aligned `labels`, raw `preds`, `node_idxs`, each length 702 |
| `<context>/exposure_<arm>.npz` | Query/relation eligible, changed, maximum, singleton/unrewirable counts; token/support/parent and timestamp summaries |
| `<context>/manifest.json` | Arm AUROCs, counters, relation metadata, effective settings, checkpoint/file hashes and completion status |

Source revision plus worktree hash, software versions and preprocessing/
embedding identities are recorded. Checkpoint path/hash and a local snapshot
revision when identifiable are recorded independently of historical weights.
Validation and inference artifacts are distinct and existing files are not
silently replaced. Failed/interrupted matrices are explicitly incomplete.
Do not interpret partial outputs as the completed preliminary comparison.

Exit 0 means completed technical checks/execution; exit 1 means failure;
exit 2 means `temporal_uncertainty`. Validation-only runs save neither model
predictions nor large sampled contexts. Software tests use marked synthetic
fixtures and do not establish validity on actual RelBench batches.

## Analysis and interpretation

Analysis loads saved arrays without RT, checks checksums and target/label
alignment across arms/contexts, and reconciles exposure with recorded totals.
It produces six context-by-strength summaries and all 14 arm rows with:

- AUROC per arm, mean/range over rewiring seeds and signed delta
  **arm minus clean**;
- mean/median absolute paired **raw-score** change and clean-versus-arm Pearson
  correlation, undefined when either vector has zero variance;
- query-mean and global achieved changed/eligible and changed/maximum ratios.

Global ratios use summed edge counts, not averages of query percentages.
Query means omit zero-denominator rows and report their counts. Summary medians
pool absolute shifts across query/seed pairs. Synthetic test artifacts are
rejected by the analysis CLI.

These are preliminary descriptive comparisons: no significance tests or causal
context-size conclusion. Context construction and structural exposure differ
between 48 and 128. Historical four-arm results remain separate, and their
unproven checkpoint revision prevents claiming identical historical weights.
