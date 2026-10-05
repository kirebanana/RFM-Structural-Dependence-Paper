# Methodology

## What the model sees

A prediction query is a `driver-dnf` task row: driver identity, query time and
a binary label for failure to finish in the next 30 days. RT receives a sampled
sequence of cells/tokens. A database row can contribute several cells; token
count is not row count. A context also contains labeled task rows as support.

For each context setting, one frozen RT-PluRel model evaluates a clean condition
and six corruption conditions (50%/100% requested strengths × seeds 101/202/303).
All arms use exactly the same 702 target IDs and labels in paired order.

## Which information is available to a query?

Before comparing clean and rewired links, we apply the same availability rules
to their shared context. The policy is named **`query_available_incidence_v1`**
in the saved files. It uses the raw dataset/task manifests to determine which
tables have timestamps and when support labels become available:

1. Keep temporal row cells only when `row time <= query time`.
2. Keep non-target forecast-label cells only when
   `label-row time + task horizon <= query time`. Driver-DNF's SQL aggregates
   outcomes in `(query time, query time + 30 days]`, so a past row date alone
   does not establish label availability.
3. Allow tables explicitly declaring `time_col: null`: drivers, constructors
   and circuits. Count these separately from unknown timestamps.
4. Reject missing query times, missing times in temporal tables, or unclassified
   node IDs. RT's missing-time sentinel is not enough to declare a table timeless.

The filter changes the padding mask for unavailable cells; it does not refill
or reorder the retained context. The 2026-10-05 run masked 55 future race cells
and 1,379 unclosed support-label cells in each context configuration. Both arms
then use the same filtered cells, values, labels and timestamps.

Past qualifying rows can point to a later race. No schedule-availability
timestamp is supplied, so we exclude the future race cells rather than inventing
a same-day exception. Mature support labels can come from earlier train,
validation or test query windows; this is a **rolling-label context**, not a
train-only support protocol.

This validates retained inputs under benchmark conventions. It does not prove
that static attributes were historically immutable, and the original sampler
can use information later masked from the input when selecting context rows.
We therefore do not claim a wholly leakage-free retrieval/forecasting system.

## How FK links are changed

`f2p_nbr_idxs` stores parent row IDs reached through FK slots. The parallel
`f2p_rel_idxs` identifies the FK column/relation and is used for matching, not
as an extra learned model input.

An **eligible FK edge instance** is a (sampled context, relation, source row,
FK slot) with an in-context parent. Repeated cells of a source row count once
for that slot. A **changed edge instance** receives a different parent ID.

For each sampled context, we group eligible links by FK relation. This group is
called a **relation stratum**. A minimum-cost matching reassigns the original
parent instances, including duplicates, within that group. The algorithm forbids
a row from becoming its own parent and prefers changing the parent ID. A group
with just one link cannot be permuted; it stays eligible but unchanged. The same
is true if no allowed matching exists.

For fixed inputs and seed, the full assignment is deterministic. To create a
partial change, we apply complete cycles of that assignment rather than selecting
individual links. For example, applying a two-row swap changes both links or
neither. This preserves the original parent counts, also called the parent
multiset.

Cycles are visited in seeded shuffled order and accepted only when they bring
the number of changed IDs closer to `round(alpha × maximum changed IDs)`.
Ties stay unchanged; rounding uses Python's ties-to-even rule.

- **0%:** no cycles applied.
- **100%:** the seeded full assignment.
- **50%:** approximately half the maximum changed IDs, subject to discrete cycles.

Report both `changed / eligible` and `changed / maximum changed`. Undefined
denominators are NaN in NPZ and null in JSON. Requested strength is not the
fraction of all eligible links, and discrete selection need not realize 50%.

## What is controlled

| Fixed between paired arms | Intervention | Measured consequences |
|---|---|---|
| Query, retained cells/order, values, labels, timestamps, source IDs, relation IDs, FK slots, parent-instance multiset, model weights | FK parent identity within strata | Attention incidence, per-token key counts/fanout, normalization, scores |

Hard checks verify held-fixed inputs, source-slot copy consistency, unchanged
absent/ineligible slots, no self-links, and parent multisets **once per source
row/slot**, not once per cell. Query/relation/run-level exposure counts must
reconcile; labels, target IDs and prediction/exposure rows must align.

Fanout and distinct-parent-count changes are diagnostics, not rejection gates
for this incidence experiment. Strict fanout checking remains available in the
low-level validator. This experiment measures the **total effect of incidence
corruption**, not parent content independently of degree or attention scaling.
It also does not construct a globally valid counterfactual database.

## What is checked and saved

All arms and contexts are validated before weights load. Only real sampler
rows count, but the exact 702-target coverage gate detects lost intended rows.
The run binds inference to validated sampled inputs with fingerprints and checks
for post-prediction mutation. Model microbatching occurs after matching and does
not alter contexts or the matching RNG.

Per-query exposure stores token/support/unique-parent counts; eligible, changed,
maximum, singleton and unrewirable edge counts; requested/realized strengths;
availability removals; fanout diagnostics; and temporal status. Flattened
relation records identify their query row and relation ID. Relation metadata is
saved in the manifest.

Analysis checks artifact hashes, target/label alignment, checkpoint identity
across contexts, and saved exposure/temporal/availability accounting. It reports
AUROC, arm-minus-clean differences, seed means/ranges, absolute paired raw-score
shifts and Pearson correlations. Zero-variance correlations are undefined.
Global ratios use summed counts; query means omit undefined denominators.

## Reading the comparisons

This is one model/task with three rewiring seeds. Seed ranges are variability
across interventions, not confidence intervals. Corruption response can be
non-monotonic. Context/local-context settings jointly change sampling and
exposure; their comparison does not identify a causal context-size effect.
The common availability policy changes historical inputs, so the old pilot is
not a directly equivalent baseline.

See [reproduction](REPRODUCIBILITY.md) for execution details and
[current results](../results/preliminary/2026-10-05_validated/README.md) for the
completed evidence set.
