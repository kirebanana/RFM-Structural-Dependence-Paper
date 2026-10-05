# Measuring Structural Reliance in Relational Foundation Models

A two-student university research project at FINKI studying whether a pretrained
Relational Transformer uses the **correct connections between database rows**,
rather than just the information those rows contain.

We use frozen **RT-PluRel** on **RelBench `rel-f1 / driver-dnf`**. Each prediction
query asks whether a driver will fail to finish a race in the next 30 days.
The model receives a sampled context of database cells and available labeled
support rows, not the entire database.

## The experiment

```text
same driver query, available cells, values, labels, timestamps, and model
             correct FK parents  vs  reassigned FK parents

example:   row A → parent X        row A → parent Y
           row B → parent Y        row B → parent X
```

We permute parent instances within each sampled context and FK relation. This
retains source slots and the parent multiset; it does not rewrite the database
or retrieve a new context. Partial corruption selects whole permutation cycles.

| Held fixed across conditions | Changed | Allowed consequences |
|---|---|---|
| Query, retained cell order/content, labels, timestamps, source IDs, FK types/slots, model weights | Concrete FK parent identities | Attention connections, token-level fanout and normalization, predictions |

**RQ1:** How sensitive are predictions to FK-incidence corruption with sampled
information held fixed? **RQ2:** How does this sensitivity vary with context and
structural exposure? The current milestone provides descriptive context/strength
comparisons and exposure records; a controlled RQ2 analysis remains future work.

## Current preliminary results — 5 October 2026

All 14 conditions use the same 702 targets and three rewiring seeds
(101, 202, 303). Context/local-context settings are 48/24 and 128/64: the first
number is the token budget; the second controls local graph expansion. Actual
retained token counts may be smaller after availability filtering.

| Context/local | Requested strength | AUROC mean [seed range] | Δ clean | Changed / eligible FK edges |
|---|---:|---:|---:|---:|
| 48/24 | Clean | 0.5613 | — | 0% |
| 48/24 | 50% | 0.5610 [0.5516–0.5682] | −0.0003 | 25.74% |
| 48/24 | 100% | 0.5604 [0.5533–0.5661] | −0.0010 | 61.22% |
| 128/64 | Clean | 0.6427 | — | 0% |
| 128/64 | 50% | 0.5893 [0.5862–0.5912] | −0.0534 | 28.03% |
| 128/64 | 100% | 0.6143 [0.6071–0.6246] | −0.0285 | 74.76% |

Requested strength is relative to the seed's maximum matching, **not all FK
edges**. Whole cycles make the intermediate strength approximate. Δ is arm minus
clean; seed ranges describe rewiring variability, not confidence intervals.
Context-48 effects vary in sign; context-128 corruption lowers AUROC in every
seed, but **50% hurts more than 100%**. A monotonic response is not established.

[Curated results](results/preliminary/2026-10-05_validated/README.md) include all
predictions, exposure arrays, provenance, validation and paired score statistics.
The older pilot remains in `results/preliminary/ctx*/`; its unfiltered contexts
and unpinned checkpoint revision make it a separate historical experiment.

## Validation and availability

Before creating conditions, we apply one shared query-time filter: temporal
rows must be dated at or before the query; forecast support labels must have
closed their outcome window. Removed cells are not refilled. The completed run
masked 55 future race cells and 1,379 unclosed support-label cells per context.
Schema-declared timeless tables follow RelBench's static-covariate convention;
genuinely missing required timestamps stop execution.

Every arm passed source-slot, parent-multiset, self-link, input-equality,
target-pairing and exposure-accounting checks. Retained parent comparisons had
no known violations or unresolved times. These are validations under an explicit
policy, **not a claim of globally valid database rewiring or complete historical
feature availability**. See [methodology](docs/METHODOLOGY.md).

## Setup and reproduction

Requires Linux/WSL, `uv`, Rust and internet access for initial dependency/data
downloads. Run from the repository root:

```bash
bash scripts/setup.sh
uv run pytest -q
uv run ruff check .
```

The [reproduction guide](docs/REPRODUCIBILITY.md) gives pinned dataset/checkpoint
downloads, preprocessing and required files. After preparing them:

```bash
# Check every context and condition without loading model weights.
uv run python experiments/run_clean_rewire.py --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed

# Revalidate, then run inference only if all gates pass.
uv run python experiments/run_clean_rewire.py \
  --pre-dir artifacts/clean_rewire_preprocessed \
  --checkpoint-dir artifacts/rt-plurel/classification \
  --device auto --inference-batch-size 8

# Replace RUN_ID with the fresh directory printed by the runner.
uv run python experiments/analyze_preliminary.py results/runs/preliminary_RUN_ID \
  --output results/runs/preliminary_RUN_ID/analysis
```

The matrix ran on a **GTX 1650, 4 GB**, with eager BF16 emulation: approximately
160 and 206 seconds of inference per context. Runtime varies by environment.
Saved-array analysis of the curated results requires no RT or external data:

```bash
uv run python experiments/analyze_preliminary.py \
  results/preliminary/2026-10-05_validated --output results/runs/curated_analysis
```

## Repository

```text
src/rfm_structure/  matching, validation, availability, provenance, analysis
experiments/        inference and saved-array analysis entry points
tests/              lightweight synthetic and curated-artifact checks
scripts/            environment setup and RT preprocessing
configs/            reference parameter records (runner uses CLI)
vendor/             pinned Stanford RT with relation-metadata patches
patches/            upstream revision
results/preliminary/ versioned current results and separate historical pilot
```

## Limits and next steps

Evidence covers one task/model and three seeds, with no significance testing.
Context construction and exposure both differ between settings, so RQ2 remains
descriptive. The measured effect includes fanout changes. Support uses mature
rolling labels, not a train-only protocol; sampling occurs before the common
availability mask, so retrieval selection is not proven historically leakage-free.
Fresh native preprocessing has not been verified to reproduce every artifact
byte-for-byte. Next steps are controlled exposure analysis and uncertainty
estimation, then broader replication.

## Sources

- [RT-PluRel checkpoint](https://huggingface.co/stanford-star/rt-plurel) and
  [Stanford Relational Transformer](https://github.com/stanford-star/relational-transformer).
- [RelBench](https://relbench.stanford.edu/) and its
  [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25cd345233c65fac1fec0ce61d0f7836-Abstract-Datasets_and_Benchmarks_Track.html).
- Gany, Cautis and Maniu, [Structural Adversarial Attacks on Relational Deep
  Learning under Integrity Constraints](https://arxiv.org/abs/2607.07089), 2026:
  FK rewiring itself is prior art; this study characterizes non-adversarial
  structural reliance in a pretrained RT.

Vendored-source provenance and unresolved upstream redistribution licensing
are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
