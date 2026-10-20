# Methodology

## What the model sees

A prediction query is an entity/time task row with a binary outcome. In the
primary `driver-dnf` task it asks about failure to finish in the next 30 days.
The query's outcome is masked as the prediction target, not supplied as support.
RT receives a sampled
sequence of cells/tokens. A database row can contribute several cells; token
count is not row count. A context also contains labeled task rows as support.

All experiments use the same frozen classification checkpoint, eager BF16
inference, context/local-context settings 48/24 and 128/64, and matching seeds
101/202/303. Within a panel every arm uses identical target IDs and labels in
paired order. Context microbatches of eight do not change matching or sampling.

| Database/task | Queries | Positive outcomes | Query dates | Forecast horizon | Conditions |
|---|---:|---:|---:|---:|---|
| rel-f1/driver-dnf | 702 | 495 | 29 | 30 days | Clean, partial/full all-relations; relation/cycle/support controls |
| rel-f1/driver-top3 | 726 | 128 | 30 | 30 days | Clean, partial/full all-relations |
| rel-arxiv/paper-citation | 512 | 225 | 1 | 182 days | Clean, task-entity, database |
| rel-trial/study-outcome | 512 | 292 | 1 | 365 days | Clean, task-entity, database |

F1 uses complete official test panels. ArXiv and Trial use uniformly sampled
official test rows without replacement, selection seed 20261007. Database tables
remain intact; those two preprocessing runs include the selected task's splits
but omit unrelated task-label tables. They are not full-test evaluations.

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
3. Allow tables explicitly declaring `time_col: null` (for example, F1 drivers,
   constructors and circuits). Count these separately from unknown timestamps.
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

### Relation, cycle and visible-support controls

Relation-restricted controls retain the same seeded assignment as all-relations
rewiring for the selected relations, leaving the rest unchanged. Task-cycle
controls partition complete matching cycles into those touching the prediction
query and those excluding it. A query-touching cycle may move other task rows;
an excluding-query cycle may move both visible- and hidden-label rows. They
do not isolate query edges or visible labeled support individually.

The separate `labeled_support_only` control selects non-query source rows with
a retained, finite, non-target label cell in the prediction column after the
common availability mask. All retained cells of a selected source share its
changed grounding. Matching is recomputed within this source pool, independently
of label value; query, hidden-label and other-relation slots cannot exchange
parents with it. A zero/one-row pool or a pool of identical parents cannot change
identities. Full parent multisets and excluded slots remain fixed.

This control measures visible support-**row grounding**, not label-value use
alone: dates and other features on the row share the changed FK. It is a fresh
matching, not a subset of the old excluding-query assignment; effects cannot be
subtracted to isolate a hidden-label contribution. The replay audit separately
records selected-pool sizes, singleton pools and distinct parents. General
`eligible_edges`/`singleton_edges` counters still describe the unrestricted
relation strata; `max_changed_edges` applies to the selected matching pool.

### Held-fixed inputs and incidence consequences

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
rows count, and exact panel-specific target coverage detects lost intended rows.
The run binds inference to validated sampled inputs with fingerprints and checks
for post-prediction mutation. Model microbatching occurs after matching and does
not alter contexts or the matching RNG.

Per-query exposure stores token/support/unique-parent counts; eligible, changed,
maximum, singleton and unrewirable edge counts; requested/realized strengths;
availability removals; fanout diagnostics; and temporal status. Flattened
relation records identify their query row and relation ID. F1 relation metadata
is saved in manifests; the ArXiv/Trial release also includes exact-byte
`relation_index.json` registries authenticated against the original preprocessing
hashes. Task/entity and ordinary-database groups can therefore be reconstructed
from portable metadata rather than trusted as unexplained numeric IDs.

Analysis checks artifact hashes, target/label alignment, checkpoint identity
across contexts, and saved exposure/temporal/availability accounting. It reports
AUROC, arm-minus-clean differences, seed means/ranges, absolute paired raw-score
shifts and Pearson correlations. Zero-variance correlations are undefined.
Global ratios use summed counts; query means omit undefined denominators.

## Reading the comparisons

This is one model/checkpoint, four tasks and three rewiring seeds. Seed ranges are variability
across interventions, not confidence intervals. Corruption response can be
non-monotonic. Context/local-context settings jointly change sampling and
exposure; their comparison does not identify a causal context-size effect.
The common availability policy changes historical inputs, so the old pilot is
not a directly equivalent baseline.

### Uncertainty and exploratory RQ2 models

Primary statistics are per-seed AUROC, arm-minus-clean AUROC and absolute raw-score
shift; seed means average statistics, not scores before AUROC. Raw-score movement
is not calibrated probability movement and is not directly comparable across
tasks. Larger movement need not imply greater ranking damage.

Paired percentile sensitivity intervals use 2,000 bootstrap replicates, shared
weights across paired arms and fixed matching seeds. F1 is resampled by date,
driver, and products of independent date/driver multiplicities. Single-date
ArXiv/Trial panels cannot estimate date uncertainty; their entity intervals are
conditional on the observed date and panel. These exploratory intervals do not
account for model/task selection, multiplicity, arbitrary temporal dependence or
overlapping outcome windows, and are not confirmatory significance tests.

RQ2 models predict `log1p(seed-mean absolute raw-score shift)` on affected
query/control records from the original eight F1 relation/cycle controls. Repeated
records across controls and contexts are not independent queries. Five-fold
GroupKFold holds out dates or drivers, keeping all repetitions of the held-out
group together. Standardization fits only inside training folds; ridge alpha is
fixed at 10 without tuning. Features distinguish:

- **Context-only exposure:** retained tokens, eligible links, unique parents,
  proposed-group eligibility, relation shares and labeled-support counts. These
  are available without solving the matching assignment.
- **Assignment-informed intervention budgets:** `targeted_maximum` is derived
  from the solved matching and cycle restriction (the first saved seed, as in the
  historical analysis). At full strength its raw maximum equals realized changed
  edges. It is not context-only exposure, even though known before model inference.
- **Post-intervention diagnostics:** seed-mean realized changes and fanout deltas.
- **Pre-inference clean-score magnitude:** available before corrupted inference,
  but mathematically coupled to the response `|corrupted - clean|` through their
  shared clean score. Its predictive gain is not independent mechanism evidence.

The corrected analysis includes context-only specifications that exclude the
assignment-derived maximum. Historical assignment-informed fits are preserved
under explicitly mapped names; their values are unchanged. “Context-only” refers
to predictor inputs, not the cohort: all RQ2 specifications still use records
selected for nonzero realized intervention exposure, so selection is not ex ante.
R²/error describe predictive association within this panel, not causal mediation,
outcome prediction, a universal exposure law or external-task generalization.

### Native temporal preprocessing and scoped text embeddings

Rustler previously assigned temporal row timestamps along outgoing-FK processing,
leaving temporal tables without outgoing FKs (notably Trial `studies`) with
missing timestamps. The local fix assigns declared timestamps on the feature
path too. A compiled-native regression covers temporal entities without outgoing
FKs. Replication
artifacts were rebuilt before the completed inference runs; the pre-fix data
were not used for reported Trial results. The fix does not change model weights.

Cross-database runs use a frozen-context embedding optimization: pinned
MiniLM-L12-v2 encodes the union of all text/schema lookups in both sampled
contexts in FP32, stored as 384-dimensional BF16 vectors. Unused rows remain
NaN. Coverage is collected before availability masking, including placeholders;
label outcomes do not determine which strings are encoded. Scope manifests bind
query IDs, encoder identity, covered IDs, non-embedding context fingerprints and
embedding file hashes. Finite-coverage checks prevent NaNs reaching inference;
hash reconciliation prevents a different finite file inheriting pinned-encoder
provenance. These files cannot be reused for arbitrary queries/contexts.

See [reproduction](REPRODUCIBILITY.md) for execution details and
[current results](../results/final/2026-10-19/SUMMARY.md) for the
completed evidence set.
