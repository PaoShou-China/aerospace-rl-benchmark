# Full-scale landing source integration

The recovered local commits are f9ab87397534a16ac218761bce7597d1b749dc58,
704801ff020204be6ceb19a8aa3642ef2cbb3d1b and
4e1c729cc94e1863c4d2fdfcf8fbc887e1590af0, based on
16fb1bd39453ae10dd3c7ca9a1bc56d48c0587ac. Zero-context transfer patches
were first reconstructed on that exact base, then three-way ported onto
civilian-sync 2346cae8cede779f62d1055b81b37cd86fe1d98e.

## Conflict decisions

- Preserve modern measured-wind reset/sampling before warmup. Apply the
  recovered initial-wind path only for other full-scale landing wind modes.
- Keep modern engine behavior, auxiliary potential shaping, precomputed
  full-trajectory BC returns and informative-subset collection.
- Add first-complete-episode BC as a separate protocol for full-scale landing.
  It retains all first-episode pairs, including terminal actions, excludes
  auto-reset samples, and disables informative-subset selection for this
  protocol. Explicit `collection="fixed_steps"` keeps the modern selector.
- Keep replica validation splitting for either protocol, recipe cache matching,
  failure retries and the bounded one-hour BC subprocess timeout. Include the
  collection protocol in cache fingerprints; keep revised rocket BC output
  directories distinct from historical results.
- Full-scale v1 primary scoring requires the final two seconds settled.
  Ever-settled remains a separate metric, not a substitute. Historical reports
  lacking valid terminal counts cannot be treated as current results.

## Validation boundary

No weights, rollout results or training improvements are introduced by this
source port. Dependency-light coverage tests validate first-episode masking,
precomputed-target compatibility, cache fingerprints and historical scoring.
The opt-in Genesis terminal simulation still requires `ROCKET_INTEGRATION=1`
and a working Genesis/headless renderer installation. Full training and
multi-seed evaluation must be reported separately from software tests.
