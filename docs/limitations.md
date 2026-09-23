# Limitations

- Evidence currently covers one RelBench task: `rel-f1/driver-dnf`.
- Three rewiring seeds are pilot robustness, not cross-dataset generalization.
- Context 48 and context 128 differ in context construction and are descriptive;
  they do not establish a causal context-size interaction.
- The evaluated support/context is transductive within the sampled context.
  Temporal visibility and label availability must be audited before describing
  the experiment as strictly leakage-free forecasting.
- The preliminary result is zero-shot RT-PluRel evidence, not a claim about all
  relational foundation models.
- The vendored upstream checkout has no license file. Licensing must be resolved
  with upstream before public redistribution of the vendored source.
