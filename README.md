# Measuring Structural Reliance in Relational Foundation Models

## Project

This project studies how strongly a pretrained Relational Foundation Model
depends on truthful **instance-level PK/FK connectivity**. The comparison holds
the sampled RT input as fixed as the current implementation permits:

```text
same sampled information, model, target, values, labels, and timestamps
different concrete PK/FK parent identities
```

We compare `f(X, M_correct)` with `f(X, M_rewired)`, where `X` is the sampled
content and `M` is the concrete relational connectivity available to RT. The
intervention operates on the sampled RT context; it is not a claim to construct
a globally valid counterfactual database.

## Model and Benchmark

- Frozen RT-PluRel inference on RelBench.
- Current pilot: `rel-f1 / driver-dnf`.
- Historical checkpoint: `stanford-star/rt-plurel/classification`, the newer
  approximately 85.6M-parameter classifier generation. The historical Hub
  revision was not recorded.
- Controlled, non-adversarial structural perturbation of sampled FK parents.

## Research Questions

**RQ1.** When sampled input content and context membership are held fixed, how
sensitive are RT-PluRel's predictions to corruption of concrete PK/FK identities?

**RQ2.** How does this structural reliance change as relational context and
structural exposure increase?

## Preliminary Evidence

The historical pilot evaluated 702 aligned `rel-f1 / driver-dnf` test queries
with three rewiring seeds:

| Context | Correct AUROC | Mean rewired AUROC | Gap |
|---|---:|---:|---:|
| 48 (local 24) | 0.6150 | ~0.6042 | ~0.0108 |
| 128 (local 64) | 0.6869 | ~0.6246 | ~0.0623 |

These are preliminary, single-task findings. The 48-versus-128 comparison does
not establish that larger context causally increases structural reliance:
context construction and structural exposure also differ. The retained
prediction arrays and manifests are under `results/preliminary/`.

## Prior Work

Integrity-constrained PK/FK rewiring was already studied for task-trained
relational GNNs by Gany, Cautis, and Maniu, *Structural Adversarial Attacks on
Relational Deep Learning under Integrity Constraints* (2026). This project does
not claim valid FK rewiring itself is novel. Its focus is controlled,
non-adversarial characterization of structural reliance in a pretrained
Relational Transformer, including context and exposure analysis.

## Repository Layout

```text
src/          structural intervention and analysis code
experiments/  experiment runners
configs/      experiment configurations
tests/        invariant and unit tests
results/      preserved preliminary results
vendor/       pinned relational-transformer source
scripts/      setup and preprocessing helpers
```

## Status

This is an ongoing university research project. Current results are preliminary.
