# Reproducibility

## Environment

Use a working Rust toolchain and `uv`; `uv` owns the Python environment,
dependency resolution, and editable local packages. The original
successful environment used Python 3.12.3, PyTorch 2.13.0+cu130, Rust 1.98.0,
maturin 1.15.0, RelBench 3.0.1, NumPy 2.5.2, SciPy 1.18.1, and
scikit-learn 1.9.0. CUDA is optional for the pure tests; RT inference is
substantially slower without it.

```bash
cd ~/iis/rfm-structural-dependence
bash scripts/setup.sh
```

The setup builds the vendored RT/Rustler extension. It does not download model
weights or datasets automatically. The pretrained checkpoint is
`stanford-star/rt-plurel/classification`. The raw dataset used for the verified
run was the local HF snapshot of `stanford-star/relbench-v1/rel-f1` at raw
snapshot `d8e976fd…`.

## Preprocessing

```bash
bash scripts/preprocess.sh /path/to/rel-f1 artifacts/clean_rewire_preprocessed
```

The output must contain `relation_index.json`, `nodes.rkyv`, and the RT text
embedding file. Do not use the old hosted preprocessed artifact with the new
Rustler layout.

Do not activate a virtual environment manually. Run project commands through
`uv run`, which uses the locked project environment.

## Tests and smoke run

```bash
uv run pytest
uv run python experiments/run_clean_rewire.py \
  --ctx 48 --local-ctx 24 --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed
uv run python experiments/run_clean_rewire.py \
  --ctx 128 --local-ctx 64 --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed
```

The dry-run reports eligible and changed FK **edge instances**, not test
examples. Singleton `(query-context, relation)` strata contribute one eligible
edge instance but cannot be rewired; the rewirable-edge fraction excludes these
singletons. The preserved AUROC table is from historical inference runs; a
dry-run validates the migrated rewiring implementation but does not reproduce
those model predictions.

Full inference commands are documented in `README.md`; they require the
checkpoint, a GPU or a long CPU runtime, and a prepared local artifact.
