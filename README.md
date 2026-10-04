# Measuring Structural Reliance in Relational Foundation Models

A two-student FINKI research project studying how much a pretrained Relational
Transformer relies on correct instance-level foreign-key connectivity.

For each prediction query, we keep the sampled cells, values, labels,
timestamps and frozen model fixed, and compare **original FK parent links**
with **reassigned parent links**. Only parent identities in the sampled input
change; the database and retrieved context are not rewritten or resampled
between paired conditions.

## Research questions and experiment

- **RQ1:** How sensitive are predictions to connectivity corruption when
  sampled information is held fixed?
- **RQ2:** How does this sensitivity vary with context and structural exposure?

The preliminary experiment uses frozen **RT-PluRel**
(`stanford-star/rt-plurel/classification`) on RelBench **rel-f1 / driver-dnf**:
702 test queries, context/local-context sizes **48/24** and **128/64**.
Each context has one clean condition and 50%/100% requested corruption with
seeds 101, 202 and 303. Requested strength is relative to the seeded maximum
matching; achieved corruption is measured separately.

```text
Local data → RT preprocessing → sampled prediction contexts
                                  │
                      original / reassigned FK parents
                                  │
                    structural + temporal validation
                                  │
                    paired predictions and exposure
                                  │
                       saved-array analysis
```

**Software is implemented and tested; new real-data validation and inference
results have not yet been generated.** The [methodology](docs/METHODOLOGY.md)
explains the intervention, checks, artifact format and interpretation limits.

## Historical preliminary results

The earlier pilot evaluated 702 aligned queries with three full-rewiring seeds:

| Context/local | Clean AUROC | Mean rewired AUROC | Clean minus rewired |
|---|---:|---:|---:|
| 48/24 | 0.6150 | ~0.6042 | ~0.0108 |
| 128/64 | 0.6869 | ~0.6246 | ~0.0623 |

Predictions and manifests are preserved in `results/preliminary/`. These are
historical results, not evidence from the new validation/strength pipeline.
Context construction and exposure differ, so the context comparison is
descriptive rather than a causal context-size result. The historical checkpoint
revision was not pinned.

## Running the experiment

Use a compatible **local** preprocessed artifact. `scripts/setup.sh` installs
the locked Python/RT environment and builds Rustler; it requires uv and a Rust
toolchain. `scripts/preprocess.sh` converts an existing raw rel-f1 dataset.
Data and checkpoints are not downloaded by the experiment runner.

Validate both contexts and all seven conditions without loading model weights:

```bash
uv run python experiments/run_clean_rewire.py --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed
```

For inference, use the same command without `--dry-run` and provide
`--checkpoint-dir /path/to/local/rt-plurel/classification`. All conditions must
pass real validation first; local weights and suitable CUDA/RAM resources are
also required. Each invocation writes a new directory under `results/runs/`.
See [execution and artifacts](docs/METHODOLOGY.md#execution-and-artifacts)
for options, statuses and resource checks.

Analyze a completed real matrix without RT:

```bash
uv run python experiments/analyze_preliminary.py results/runs/preliminary_RUN_ID \
  --output results/runs/preliminary_RUN_ID/analysis
```

This produces `analysis.json` and a professor-readable `SUMMARY.md` with
AUROC, paired score shifts, correlation and achieved corruption.

## Code and tests

| Location | Responsibility |
|---|---|
| `src/rfm_structure/rewiring.py` | Parent matching, corruption strengths and structural/temporal checks |
| `src/rfm_structure/validation.py` | Query coverage, accounting and exposure artifacts |
| `src/rfm_structure/analysis.py` | Paired descriptive statistics and result tables |
| `experiments/` | Experiment and saved-array analysis entry points |
| `vendor/` | Pinned Stanford RT with relation-metadata patches |

The lightweight suite uses synthetic fixtures, not RT weights:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src experiments tests
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the vendored RT source
and its unresolved upstream licensing status.
