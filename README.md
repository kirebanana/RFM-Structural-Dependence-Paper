# Measuring Structural Dependence in Relational Foundation Models

This Topic 2 project studies whether a pretrained Relational Foundation Model
uses the actual relational structure of a database for prediction when the
labeled examples available in its context are controlled.

## Research questions

**Primary:** How much does a pretrained Relational Foundation Model rely on the
actual relational structure of a database for prediction, when the labeled
examples available in its context are controlled?

**Secondary:** How does the importance of relational structure change as the
model receives more context?

The primary model is RT-PluRel. The first verified task is RelBench
`rel-f1/driver-dnf`. This repository contains the corrected relation-preserving
rewiring harness and preliminary evidence; it is not yet the final project
study.

## Preliminary result

| Context | Base AUROC | Rewire 101 | Rewire 202 | Rewire 303 | Mean delta |
|---:|---:|---:|---:|---:|---:|
| 48 (local 24) | 0.6150 | 0.6052 | 0.6072 | 0.6003 | -0.0108 |
| 128 (local 64) | 0.6869 | 0.6217 | 0.6250 | 0.6272 | -0.0623 |

These are preliminary single-task results over 702 test queries. The full
interpretation and limitations are in `docs/preliminary_findings.md` and
`docs/limitations.md`.

## Layout

- `src/rfm_structure/`: tested rewiring, validation, data, and metric helpers.
- `experiments/`: RT-PluRel multi-arm experiment runner.
- `vendor/relational-transformer/`: pinned RT source with the required native
  `f2p_rel_idxs`/`relation_index.json` changes.
- `configs/`: verified experiment parameters.
- `results/preliminary/`: small manifests, prediction arrays, and reports.
- `docs/`: methodology, setup, findings, and limitations.

## Setup

Requirements: `uv`, Rust, and access to the RT checkpoint and RelBench data.
The lockfile constrains PyTorch to the 2.13 line; the original run used
PyTorch 2.13.0+cu130 and Rust 1.98.0.

```bash
cd ~/iis/rfm-structural-dependence
bash scripts/setup.sh
uv run pytest
```

Prepare a fresh local RT artifact. The modified Rustler format is not compatible
with the old hosted `relbench-preprocessed` node files:

```bash
bash scripts/preprocess.sh /path/to/rel-f1 artifacts/clean_rewire_preprocessed
```

Run the real-data validation without model forward first:

```bash
uv run python experiments/run_clean_rewire.py \
  --ctx 48 --local-ctx 24 --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed
```

Then run one context at a time:

```bash
uv run python experiments/run_clean_rewire.py \
  --ctx 48 --local-ctx 24 --seeds 101,202,303 \
  --pre-dir artifacts/clean_rewire_preprocessed
uv run python experiments/run_clean_rewire.py \
  --ctx 128 --local-ctx 64 --seeds 101,202,303 \
  --pre-dir artifacts/clean_rewire_preprocessed
```

Generated local artifacts go under `artifacts/` and `results/runs/`, both
ignored by Git. Small verified preliminary outputs are already retained under
`results/preliminary/`.

## Status

**READY_WITH_LIMITATIONS.** The corrected pipeline, source changes, tests, and
preliminary evidence are organized and reproducible in principle. Before public
release, resolve upstream licensing for the vendored RT source, verify the
environment on a clean machine, audit temporal label visibility, and add the
planned second dataset and statistical analysis.

AI coding agents assisted the implementation and workspace migration. The
scientific claims remain limited to the verified artifacts and documented
scope.
