# Measuring Structural Reliance in Relational Foundation Models

This project explores how much a pretrained model relies on the
**correct connections between database rows**.
If the model sees the same information but different relationships, do its
predictions change?

We use **RT-PluRel**, a pretrained Relational Transformer, on **RelBench**, a
benchmark of prediction tasks over relational databases. Our current task is
`rel-f1 / driver-dnf`: predict whether a driver will fail to finish a race in
the next 30 days. The model is frozen—we run inference, not training.

## How the experiment works

A **prediction query** is one driver at a particular date. Its **sampled
context** is the collection of database cells and available labeled support
rows selected for that prediction. RT processes cells as tokens; one database
row can contribute several tokens.

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
There is one clean condition and six rewired conditions: requested strengths
50% and 100%, each with three seeds. The requested strength is relative to the
maximum matching, so the actual proportion of changed links is recorded too.

Before this comparison, a shared availability filter removes future-dated cells
and support labels whose outcome window has not closed. Both conditions receive
the same filtered input. Validation checks the held-fixed inputs, permitted
rewiring, prediction alignment, and accounting. Attention connections and the
number of connected tokens can change as a consequence of rewiring.

The questions are:
- **RQ1:** How sensitive are RT-PluRel predictions to changed FK identities when
  sampled information is kept fixed?
- **RQ2:** How does that sensitivity vary with context size and the amount of
  relational structure exposed to the model?

See [methodology](docs/METHODOLOGY.md) for the matching algorithm, availability
rules, and definitions of the measurements.

## How the code fits together

The repository has three layers: Stanford's RT implementation, our experiment
logic, and the executable runners that connect them.

```text
Raw RelBench tables
    │ scripts/preprocess.sh
    ▼
RT/Rustler data files and FK relation metadata
    │ Stanford RT sampler
    ▼
One sampled context per prediction query
    │ our common availability filter
    ├── clean FK links ─────────────────────┐
    └── seeded FK-parent rewiring ──────────┤
                                          ▼
                              same frozen RT-PluRel model
                                          │
                              predictions + exposure records
                                          │ experiments/analyze_preliminary.py
                                          ▼
                              paired statistics + result tables
```

### Experiment entry points

- **`experiments/run_clean_rewire.py`** builds the evaluator, validates every
  condition, loads the checkpoint once per context setting, and runs clean and
  rewired inputs through it. `BaseWrap` and `RewireWrap` let the evaluator use
  one model with different input conditions. Predictions run in small batches
  after matching; this does not resample the contexts.
- **`experiments/analyze_preliminary.py`** reads the saved arrays and generates
  AUROC, paired score changes, correlations, and summary tables without RT.

### Our Python package: `src/rfm_structure/`

| Module | Responsibility |
|---|---|
| `rewiring.py` | Match FK parents within each context/relation, select corruption cycles, and check structural properties. |
| `validation.py` | Filter unavailable cells, validate complete query coverage, collect per-query/per-relation exposure, and write exposure arrays. |
| `data.py` | Read node ranges, FK relation metadata, and source manifests used to establish availability. |
| `metrics.py` | Compute binary AUROC from raw prediction scores. |
| `analysis.py` | Verify saved pairing/accounting and compute descriptive comparisons with the clean condition. |
| `provenance.py` | Record hashes and identities of code, preprocessing files, software, and checkpoints. |

Two input arrays connect the sampler to our intervention:
`f2p_nbr_idxs` stores the parent row IDs for each sampled row's FK slots;
`f2p_rel_idxs` says which FK relation each slot belongs to. Rewiring changes
the first array, while the second keeps different relationships from being mixed.

### Supporting directories

```text
scripts/             environment setup and native RT preprocessing
tests/               algorithm, validation, runner, and saved-result checks
configs/             readable parameter records; execution uses CLI arguments
vendor/              pinned Stanford RT model and Rustler sampler/preprocessor
patches/             upstream revision used for the vendored code
results/preliminary/ versioned predictions, exposure, and result summaries
docs/                methodology and reproduction instructions
```

The small local Stanford patches carry FK relation IDs from Rust preprocessing
through the sampler into Python. Model architecture remains upstream. Setup
installs `rfm_structure` and the vendored RT package in the same `uv` environment;
Rustler is the native extension that reads and samples the prepared database.

## Results

The completed preliminary matrix covers 702 queries at context/local-context
settings 48/24 and 128/64, with clean, partial, and full rewiring conditions.

**[Read the current results summary](results/preliminary/2026-10-05_validated/SUMMARY.md)**

The [result artifact guide](results/preliminary/2026-10-05_validated/README.md)
explains the saved predictions, exposure, checksums, and run provenance. Older
historical pilot outputs are retained separately.

## Setup and running

**[Setup, data preparation, validation, inference, and analysis instructions](docs/REPRODUCIBILITY.md)**

The guide includes pinned data/checkpoint downloads and commands for both new
experiments and reanalysis of the versioned results. Saved-result analysis does
not require downloading a model or running inference.

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
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
