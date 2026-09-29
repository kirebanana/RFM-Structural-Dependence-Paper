# Measuring Structural Reliance in Relational Foundation Models

This project asks how much a pretrained Relational Transformer (RT-PluRel)
relies on the correct parent rows behind foreign-key links. For each prediction
query, it compares two versions of the same sampled context:

```text
same query, sampled cells, values, labels, timestamps, and frozen model
                 clean FK parent links
                          vs
                rewired FK parent links
```

Only the concrete FK parent identities in the sampled RT input are reassigned;
this is not a database-wide rewrite. The current pilot uses the frozen
`stanford-star/rt-plurel/classification` checkpoint on the RelBench
`rel-f1 / driver-dnf` task. Rewiring is controlled and non-adversarial.

## Research Questions

**RQ1.** When sampled input content and context membership are held fixed, how
sensitive are RT-PluRel's predictions to corruption of concrete PK/FK identities?

**RQ2.** How does this structural reliance change as relational context and
structural exposure increase?

## Preliminary results

The historical pilot evaluated 702 aligned `rel-f1 / driver-dnf` test queries
with three rewiring seeds:

| Context | Correct AUROC | Mean rewired AUROC | Gap |
|---|---:|---:|---:|
| 48 (local 24) | 0.6150 | ~0.6042 | ~0.0108 |
| 128 (local 64) | 0.6869 | ~0.6246 | ~0.0623 |

These are **historical, preliminary** results, not new inference with the
current local validation code. The context comparison is descriptive: context
construction and exposure differ too. Saved predictions and manifests are in
`results/preliminary/`.

## Repository layout

```text
src/rfm_structure/  rewiring, validation, and exposure helpers
experiments/        experiment runner
tests/              first-party tests
scripts/            setup and preprocessing
vendor/             pinned Stanford RT with local relation-metadata patches
results/            saved preliminary outputs
```

This is an ongoing university research project.
