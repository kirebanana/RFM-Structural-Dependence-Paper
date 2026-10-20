# Measuring Structural Reliance in Relational Foundation Models

This project studies the sensitivity of a frozen **RT-PluRel** Relational
Transformer to foreign-key (FK) identity corruption in fixed sampled contexts.
Experiments cover four RelBench tasks across three databases: `rel-f1/driver-dnf`,
`rel-f1/driver-top3`, `rel-arxiv/paper-citation`, and `rel-trial/study-outcome`.
There is no model training or attack optimization.

A model can encounter useful values without associating them with the right
entities. We ask whether correct relational grounding matters beyond simply
having those values in the input:

- **RQ1:** How sensitive are predictions to changed FK identities when sampled
  information is kept fixed?
- **RQ2:** How does sensitivity vary with context size and exposed relational
  structure, and how well do those characteristics predict the model's response?

**[Final results](results/final/2026-10-19/SUMMARY.md)** ·
**[Methodology](docs/METHODOLOGY.md)** ·
**[Reproduction](docs/REPRODUCIBILITY.md)**

## How the experiment works

A **prediction query** is one entity at a particular date. Its **sampled context**
contains database cells and labeled task rows used as support. RT processes cells
as tokens; one source row can contribute several tokens.
RT-PluRel is a pretrained Relational Transformer: its attention uses row and FK
connectivity to combine information across a relational database. The query's
outcome is hidden; retained outcomes on other task rows provide labeled support.

For each query, we compare the original foreign-key (FK) links with rewired
links in the same sampled context:

```text
                 original links      rewired links
                 row A → parent X     row A → parent Y
                 row B → parent Y     row B → parent X

same query, sampled information, values, labels, timestamps, and model
```

Only parent row IDs are reassigned. We keep each FK relation separate and reuse
the same parent instances, rather than replacing them with arbitrary rows.
Source rows, FK slots and relation identities remain fixed. Matching permutes
parent instances within each context/relation, preserving their multiplicities
and forbidding self-links; it does not rebuild a globally consistent database.
The F1 matrices compare clean, partial and full rewiring with three matching
seeds. Follow-up controls restrict relations, query-touching cycles, or visible
labeled-support source rows. Cross-database panels compare task-entity and
database-link corruption. Actual changed-link counts and affected-query coverage
are reported: these are not matched-dose interventions.

Before this comparison, a shared availability filter removes future-dated cells
and support labels whose outcome window has not closed. Both conditions receive
the same filtered input. Validation checks the held-fixed inputs, permitted
rewiring, prediction alignment, and accounting. Attention connections and the
number of connected tokens can change as a consequence of rewiring.

See [methodology](docs/METHODOLOGY.md) for the matching algorithm, availability
rules, and definitions of the measurements.

## How the code fits together

The code has three layers. Stanford's implementation prepares the database,
samples the cells shown to the model, and produces predictions. Our Python
package changes selected relationships and checks that the comparison remains
controlled. The experiment scripts choose the task and comparison, run these
steps, and save the evidence used in the results.

```text
RelBench database tables + task splits/manifests
    |
    v
Rustler preprocessing + text/schema embeddings
    |  Native row data, timestamps and FK relation metadata
    v
Frozen query panel -> RT context sampler (48/24 or 128/64)
    |
    v
Common query-time availability mask (no refill/reordering)
    |
    +--> Clean: original FK parent IDs ----------+
    |                                            |
    +--> Corrupted: seeded parent reassignment --+
                                                 |
                                                 v
                         Pre-inference validation of every arm
                         Invariants, temporal rules, query coverage
                                                 |
                                                 v
                         Same frozen RT-PluRel classification model
                         Small forward batches; no context resampling
                                                 |
                                                 v
                         Replay/alignment checks + saved evidence
                         Predictions, exposure, manifests, hashes
                                                 |
                                                 v
                         Paired analysis + independent verification
                         AUROC, score shifts, uncertainty, RQ2 models
```

Every condition receives the same prepared cells and model weights. We hash the
sampled inputs to check that inference uses the contexts that passed validation.
The detailed F1 controls reuse saved clean predictions only after checking the
same contexts and checkpoint; if no links change, another model evaluation is
unnecessary. For ArXiv and Trial, text embeddings are prepared only for the
preselected queries' contexts. Checks ensure every required vector is present
and the embedding file matches its recorded identity.

### Shared implementation

| Component | What it does and why it exists |
|---|---|
| `vendor/relational-transformer/` | Stanford's model and Rustler data engine. Rustler converts tables into model-readable data and samples contexts. Local changes expose FK relation identities and preserve timestamps needed for our checks; the model architecture is unchanged. |
| `src/rfm_structure/rewiring.py` | Reassigns parent identities within each relationship type while preserving the original parent counts. It also supports selecting complete permutation cycles or restricting which source rows may change. This is the intervention, rather than a new sampling procedure. |
| `src/rfm_structure/validation.py` | Applies the common time-availability mask and checks that only permitted links change. It records how many cells, support labels and links each query actually exposes, so an unchanged prediction is not mistaken for robustness when no links could change. |
| Other `src/rfm_structure/` modules | Read task/schema metadata, define relation groups and query selection, record file/model identities, and calculate paired metrics and exploratory statistics. These shared functions keep the runners and analyses consistent. |
| `scripts/` and `configs/` | Set up the environment and preprocess F1 data; configuration files provide readable parameter records. The runners take their execution settings from command-line arguments, not these reference files. |
| `tests/` | Check the matching algorithm, temporal handling, runner behavior, statistics and saved-artifact integrity. Small synthetic cases test failure modes without requiring new research experiments. |
| `results/` and `docs/` | Hold the selected predictions, exposure records and provenance, together with methodology and reproduction instructions. Large local runs and temporary outputs are not part of the curated release. |

### Experiment runners and saved-result analysis

The runners perform inference; the analysis scripts read saved evidence without
loading RT or model weights. They answer different parts of the research question:

| Script in `experiments/` | Purpose |
|---|---|
| `run_clean_rewire.py` | Runs the main clean, partial and full rewiring comparisons on **either F1 task**, driver-DNF or driver-top3. It validates the complete requested comparison before loading weights and saves paired predictions and exposure for each context and seed. |
| `run_relation_controls.py` | Runs the detailed controls on **F1 driver-DNF**: selected relationship types, complete matching cycles that touch or exclude the query, and corruption restricted to rows with visible support labels. These distinguish intervention regimes that the all-relations experiment combines. |
| `run_database_replications.py` | Tests whether the findings extend beyond F1, on **ArXiv paper-citation and Trial study-outcome**. For each task it fixes a uniform sample of 512 official test queries before inference, then compares clean inputs, task-row/entity-link corruption and ordinary database-link corruption. |
| `analyze_preliminary.py` | Recalculates AUROC and paired score changes for the complete clean/partial/full matrices on either F1 task. It checks saved predictions against their manifests and produces per-condition and seed-summary tables. |
| `analyze_relation_controls.py` | Compares the detailed F1 relation and query-cycle controls. It calculates paired uncertainty checks and fits the exploratory RQ2 models, holding out dates or drivers to assess prediction of score sensitivity. |
| `analyze_database_replications.py` | Compares task-link and database-link effects on ArXiv and Trial, including score changes, AUROC and entity-resampling intervals. Their single query dates do not support estimation of temporal uncertainty. |
| `analyze_final_results.py` | Checks the complete curated release: expected conditions, checksums, query/label alignment, relation-group meaning and exposure/temporal accounting. It independently recalculates all 110 saved prediction conditions and the visible-support-only contrasts. |

Additional scripts prepare the ArXiv/Trial text embeddings, replay the F1
support-link assignments without model inference, and assemble the compact
result release with file hashes and source records. Their commands and required
inputs are in the [reproduction guide](docs/REPRODUCIBILITY.md).

## Experiments and key findings

**[Results and limitations across all tasks](results/final/2026-10-19/SUMMARY.md)**

All panels use contexts/local contexts **48/24 and 128/64**, one classification
checkpoint and three matching seeds (101/202/303). A context size counts tokens,
not database rows; the local-context setting controls the sampler's local budget.

| Database/task | Prediction queries | Comparisons |
|---|---:|---|
| F1 / driver-DNF | 702, complete test panel | Clean; partial/full all-relations; six relation groups; query-touching/excluding cycles; visible labeled-support only. |
| F1 / driver-top3 | 726, complete test panel | Clean and partial/full all-relations; same-database task replication. |
| ArXiv / paper-citation | 512, uniform test subset | Clean, task-entity links and ordinary database links. |
| Trial / study-outcome | 512, uniform test subset | Clean, task-entity links and ordinary database links. |

For each task, we compare predictions using the original relationships with
predictions after changing selected relationships. The model, sampled
information and correct answers remain the same. We investigate:

- **Partial versus full rewiring:** in the F1 matrices, we target roughly half
  the maximum link changes allowed by our matching algorithm, or the full
  maximum. Changes must preserve complete permutation cycles, so 50% is an
  approximate target, not half of every relationship type.
- **Different relationship types:** we separately change links connecting task
  examples to their entities and links connecting ordinary database tables.
- **Query versus supporting examples:** we compare matching cycles that touch
  or exclude the query. Query-touching cycles can move other examples too, and
  excluding-query cycles can move rows without visible labels. A separate
  control changes only rows whose support labels are available to the model.

For ArXiv and Trial, we randomly selected 512 official test queries per database
before inference, using fixed selection seed 20261007 for reproducibility.

### What did we find?

**1. Correct relationships matter, even when the current query remains correctly connected.**

Our most direct support control changes only links belonging to other examples
whose labels are available to the model. On F1 driver-DNF, with a context size
of 128 tokens:

- Mean AUROC decreases from **0.6427 to 0.5964** across three matching seeds.
- **390 of 702** queries have at least one changed relationship per seed.
- **1,552 relationships** are changed per seed; query links and links belonging
  to rows without visible labels stay fixed.

The model is sensitive to how supporting examples are connected to entities,
not only to the current query's connections. This does not establish reliance
on label values alone: dates and other information on those rows share the
changed relationship and could also contribute.

**2. Different relationships affect predictions differently.**

Across the studied F1 driver-DNF, ArXiv and Trial controls, task-example/entity
links produce larger average score changes than ordinary database links. These
interventions have unequal numbers of changed links and affected queries, so
this is not a comparison of importance per relationship.

Larger score changes do not necessarily mean worse predictive performance.
For Trial at context 128, ordinary database-link corruption reduces AUROC more
despite moving scores less. Partial rewiring can also hurt performance more
than full rewiring. Its selected relationship composition differs, and effects
of separate interventions do not simply add together: more corruption cannot
be assumed to mean more damage.

**3. Context counts add little predictive information about sensitivity (RQ2).**

We tested whether characteristics such as the number of available relationships
or supporting examples predict how strongly the model reacts. In the exploratory
date-held-out analysis, R² increases only from **0.0995 to 0.1011** when
context-only structural-exposure features are added to a group/context baseline.
All these models use examples affected by the interventions, not an unselected
population of queries.

The original prediction's magnitude is more informative, although it shares a
mathematical term with the measured change in prediction. These associations
do not identify a causal mechanism. The experiments establish sensitivity to
relational grounding, but do not fully explain why it varies between queries.

**Limits.** Budgets and affected-query coverage differ between controls and
contexts. Fanout/attention normalization can change with incidence. Availability
masking follows sampling, so historically leakage-free retrieval is not proven;
mature rolling support labels are not a train-only protocol. ArXiv and Trial
each have one query date. Uncertainty checks are exploratory and conditional on
one checkpoint and three seeds, not universal claims about relational models.

The [artifact guide](results/final/2026-10-19/README.md) describes compact saved
predictions, exposure, provenance and independent recalculation. The
[primary F1 matrix](results/preliminary/2026-10-05_validated/SUMMARY.md) remains
unchanged; historical pilot outputs are separate and not directly comparable.

## Setup and running

**[Setup, data preparation, validation, inference, and analysis instructions](docs/REPRODUCIBILITY.md)**

The guide includes pinned data/checkpoint downloads and commands for both new
experiments and reanalysis of the versioned results. Saved-result analysis does
not require downloading a model or running inference.

For a first inspection, from the repository root:

```bash
uv sync --extra dev
uv run python experiments/analyze_final_results.py results/final/2026-10-19 \
  --output results/runs/readme_reanalysis
```

Choose a fresh output directory. Full inference additionally needs the RT/native
environment, pinned datasets, embeddings and checkpoint described in the guide.

## Sources

- [RT-PluRel checkpoint](https://huggingface.co/stanford-star/rt-plurel) and
  [Stanford Relational Transformer](https://github.com/stanford-star/relational-transformer).
- [RelBench](https://relbench.stanford.edu/) and its
  [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25cd345233c65fac1fec0ce61d0f7836-Abstract-Datasets_and_Benchmarks_Track.html).
- Gany, Cautis and Maniu, [Structural Adversarial Attacks on Relational Deep
  Learning under Integrity Constraints](https://arxiv.org/abs/2607.07089), 2026:
  FK rewiring is prior art; our focus is controlled structural reliance in a
  pretrained Relational Transformer.

Vendored-source provenance and licensing are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See the
[redistribution assessment](docs/REDISTRIBUTION.md) before publishing code or
dataset-derived artifacts; unresolved permissions are not grants to redistribute.
