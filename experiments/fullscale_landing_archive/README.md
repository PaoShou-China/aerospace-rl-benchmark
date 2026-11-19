# Recovered full-scale landing study: aggregate evidence

These are curated records of a civilian full-scale rocket-landing study dated
2026-11-15 to 2026-11-16, **not a new run or a verified reproduction on main**.
The recorded source pin is `4e1c729cc94e1863c4d2fdfcf8fbc887e1590af0`;
this package does not establish its full dependency closure. Original file hashes
and curated-file hashes are in `manifest.json`. Scheduling deadlines and device
selection metadata were removed; scientific counts, seeds, checkpoint hashes,
selection rules and recorded timestamps were retained. Process status files and
machine-specific operational tools are not included.

## Results versus plans

| Recorded evaluation | Successes | Denominator | Budget actually scored |
| --- | ---: | ---: | --- |
| Frozen BC reference | 11 | 640 | 0 PPO updates |
| PPO seed 11 | 0 | 640 | 130 updates |
| PPO seed 22 | 0 | 640 | 160 updates |
| PPO seed 33 | 0 | 640 | 159 updates |
| PPO seed 44 | 0 | 640 | 140 updates |

There are 25 aggregate panels: five model variants, each evaluated at five seeds
with 128 episodes. BC has 629 non-successes; each PPO policy has 640. These are
**non-success counts**, not classified collision/timeout counts: episode-level
failure reasons were not transferred. The planned maximum was 2000 PPO updates
per training seed; `complete: true` does not mean that budget was reached.
Unequal final update counts do not support an equal-budget algorithm comparison.

The evaluator source checked `terminal_stable_2s` and complete first-episode
accounting before emitting the aggregate. Its raw per-episode reports are absent,
so that metric assertion cannot be independently rerun from this package.
This differs from historical ever-success rocket records.

### Data/width scaling experiment

Small/large data comprised 4096/12288 training episodes; the widths were 128/256.
A separate 512-episode demonstration validation set supported BC fitting. Each
of four BC variants then had a 512-episode rollout selection panel. Successes
were respectively 1, 2, 2, 7 for small-128, small-256, large-128, large-256.
The selected large-256 variant is **selection-set evidence, not a test result**.

For subsequent paired PPO, all eight policies had frozen checkpoints at updates
64 and 128. Each milestone's validation denominator was 2048 across eight models,
two seeds and 128 episodes. Both totals were zero; the documented tie rule chose
update 64. Selected hashes agree with the frozen hash manifest.

On the final eight test seeds, every policy recorded **0/1024**, or 0/4096 per
arm and 0/8192 across the two arms. The recorded paired difference and bootstrap
interval are zero. With all observed outcomes zero, that degenerate interval is
not evidence of equivalence or a population success rate of exactly zero.
The aggregate totals match the plan; the raw test panels needed to independently
verify pairing and recompute the bootstrap are missing.

## What can be verified here

Run `python experiments/fullscale_landing_archive/audit.py` from the repository
root. This new, standalone standard-library utility performs read-only hash,
coverage, denominator, selection and checkpoint-manifest consistency checks.
It does not load weights, start training, contact services or manage storage.

Missing: checkpoint bytes, demonstration shards, episode-level evaluations,
per-seed scaling test panels, complete source snapshot and environment lockfile.
The original instrumentation depends on absent `source`, `models` and `expert`
directories and intertwined imports. It also includes process supervisors and
storage mutation utilities. Those scripts were deliberately not copied as an
apparently runnable training package. Their useful evidence-accounting checks
are represented by the standalone audit, not by a claimed training-code port.

## Patch reconciliation boundary

The separately ported patches 0001–0003 cover the full-scale environment,
final-two-second scoring and complete-first-descent collection/fitting. This
package does not duplicate or reapply them.

The additional civilian working-tree diff contains overlapping environment and
evaluation files plus unported options for leg-end contact geometry, full-domain
ground, ground-triggered shutdown, pad-load feedback, civilian scenario profiles,
navigation-outage handling and archived-policy loading. Some reference extra
assets/modules outside that diff. No broad port was attempted here. These
variants change evaluation conditions and must not be silently combined with
this study's aggregates or described as tested by them.
