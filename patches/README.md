# RT Patch Surface

The vendored RT source is pinned to upstream commit
`455df27c1458e093eac00133d5bbf41a8263a2e3`. The research-specific changes are
preprocessor and sampler must agree on the serialized `rkyv` layout.

The four modified files are:

- `rustler/src/common.rs`: adds `Node.f2p_rel_idxs`.
- `rustler/src/pre.rs`: emits relation IDs and `relation_index.json`.
- `rustler/src/fly.rs`: carries relation IDs through sampler output.
- `src/rt/data.py`: reshapes relation IDs for Python.

The original upstream checkout and its dirty diff are preserved in
