# Reproducible module layout

The submission is self-contained in `algorithms/physical_belief_rocket_landing/`. Code, configurations, checkpoints, the frozen benchmark snapshot, reports, and recorded results stay together. The repository's environment modules are unchanged. Run the commands in [the technical report](../README.md) from this directory.

The frozen snapshot preserves the environment used by the recorded experiment, including its original `src/common` import namespace. At launch, `scripts/snapshot.py` extracts the snapshot into a fresh run directory, and the model modules from `core/` are copied into that private runtime namespace. This allows later repository reorganizations to coexist with the recorded checkpoint and evaluation protocol.

`provenance.json` records the original source and checkpoint hashes and the packaging changes. `FILE_MANIFEST.json` lists every delivered file's size and SHA-256, excluding the manifest itself. Original experiment identifiers remain in historical metadata for traceability; public entry points and reports use descriptive names.

`results/packaging_check.json` records the earlier source/hash, model/optimizer loading, two-sample CPU forward/backward, and archived-episode checks. `results/integration_check.json` records checks after relocation into the repository. Packaging does not include a new GPU training run or full evaluation; the published rates come from the included completed experiment records.
