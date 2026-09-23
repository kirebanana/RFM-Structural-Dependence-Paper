# Methodology

The primary intervention samples the same RT-PluRel context and then rewires
foreign-key parent identities within each `(query context, relation)` stratum.
It preserves source-slot counts, eligible parent multisets, relation IDs,
present/absent slots, and RT feat/nbr attention true-counts. Singleton strata
cannot be changed and remain unchanged.

The base and three seeded rewire arms are evaluated through one evaluator pass
over the same 702 `rel-f1/driver-dnf` test queries. AUROC is computed from the
raw classification scores; sigmoid is monotonic and therefore does not change
AUROC.

The native metadata path is necessary: `f2p_nbr_idxs` alone does not identify
which FK relation produced a parent slot. The modified Rust preprocessor emits
the parallel `f2p_rel_idxs` tensor and `relation_index.json`.
