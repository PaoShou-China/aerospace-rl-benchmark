# Civilian experiment evidence reconciliation

Inventory date: 2026-11-19. This document separates historical archive claims
from recoverable repository evidence. It reports **no new training or evaluation
on current main**. Counts from different protocols must not be pooled.

## Evidence tiers

1. **Repository artifacts:** pinned source, checkpoint and/or evaluation records
   can be inspected. Their presence is not an independent reproduction.
2. **Archive-reported aggregates:** contemporaneous summaries describe runs, but
   this reconciliation has not recovered their matching source, weights and raw
   episode records. These are not independently verified results.
3. **Plans or incomplete work:** proposed changes and missing outputs establish
   neither an implemented method nor a result.

## Historical cloud experiment aggregates (tier 2)

The archive describes frozen-candidate validation using three batches of 256
new episodes. It distinguishes development tuning from subsequent independent
seed evaluation. Exact seed IDs, sampling distributions and matching artifact
hashes have not been recovered, so “independent” below is the archive's label,
not a verified experimental guarantee or evidence of out-of-distribution testing.

| Civilian task | Last reported count | Archive classification |
| --- | --- | --- |
| QuadX trajectory tracking | 768/768 | Independent-seed validation |
| Low-thrust orbital transfer | 768/768 | Independent-seed validation |
| Spacecraft rendezvous | 768/768 | Independent-seed validation |
| QuadPlane four-waypoint navigation | 768/768 | Independent-seed validation |
| QuadX waypoint navigation | 767/768 | Independent-seed validation; one collision |
| Fixed-wing waypoint navigation | 509/512 | Best development candidate; no final independent validation recovered |

The later QuadX development result was 2047/2048, with a subsequent final
checkpoint scoring 511/512 on the original development set. Neither replaces
the 767/768 independent-seed record. Failed validation cases were reportedly
reused for development before another validation with new seeds. Sample counts
alone do not establish generalization or perfect population success.

Missing for these six-task claims: recoverable frozen source revisions,
checkpoint bytes and hashes, exact seeds and reset conditions, full episode
records, training configuration/budgets and a source-to-result manifest.
Do not reconstruct missing implementation from these summaries and label it
recovered code. No new-main performance claim follows from this table.

## Recoverable repository lineage (tier 1)

### Repository audit snapshot

The later audit inspected a clean checkout at
[`7dbf037869b08459197a05ee9bc979816a8f08e7`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/7dbf037869b08459197a05ee9bc979816a8f08e7).
That snapshot is already an ancestor of the pre-consolidation target
[`73a7ec278d6d2f59b5f2cc5cc77877dd0a0e2da9`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/73a7ec278d6d2f59b5f2cc5cc77877dd0a0e2da9).
It is not a separate new implementation to import. Its
[controller-prior pilot report](https://github.com/PaoShou-China/aerospace-rl-benchmark/blob/7dbf037869b08459197a05ee9bc979816a8f08e7/experiments/controller_priors/RESULTS.md)
is a different experiment from the six-task archive above. The audit did not
recover the full five-seed campaign or those six-task checkpoints.

### Existing MoE rocket-landing study

[`f6af06e9f76af9415994eac62bf53af356847419`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/f6af06e9f76af9415994eac62bf53af356847419)
already packages the phase-routed policy, frozen weights, feedback-repair
training and evidence under `algorithms/moe_rocket_landing/`.
The [pinned report](https://github.com/PaoShou-China/aerospace-rl-benchmark/blob/f6af06e9f76af9415994eac62bf53af356847419/algorithms/moe_rocket_landing/README.md)
records 444/512 CPU development **ever-success**, but 0/512 final strict settling.
This is exploratory development evidence, not robust terminal landing success.
Its historical environment is pinned to
`39e6bb6fd5d0f5432f7405a76946938a469492ce`.
The package is not a complete retraining archive for every earlier expert.
Historical outcomes must retain that environment/protocol label after adaptation.

### Measured wind

[`edb5d0b941f1a0c1ef26025b05a40e14b956a9de`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/edb5d0b941f1a0c1ef26025b05a40e14b956a9de)
adds measured-wind replay for full-scale rocket landings; the target already
contains it through `73a7ec2`. See the
[pinned civilian environment guide](https://github.com/PaoShou-China/aerospace-rl-benchmark/blob/edb5d0b941f1a0c1ef26025b05a40e14b956a9de/environments/rocket/README.md).
Availability of wind replay does not establish that any historical policy was
trained or independently evaluated under those wind conditions.

### Performance branch lineage

The inspected `performance-speed-up` tip
[`0825e9253316aa7028d740c924f96196c49ea8c9`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/0825e9253316aa7028d740c924f96196c49ea8c9)
is already an ancestor of `73a7ec2`; it needs no duplicate import. The tip itself
is a README revision. Neither its branch name nor its inclusion proves a
measured speedup on the consolidated implementation.

## Reconciliation boundary

The inspected main pin is
[`b54883dead5e79d5f2997aee8565c9634c3ee0c8`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/b54883dead5e79d5f2997aee8565c9634c3ee0c8).
Compatibility work against that pin must be reported separately from historical
results. Source import, unit tests, model loading and an independently evaluated
policy are different milestones. Preserve original evidence unchanged; record
new source/configuration hashes and evaluation conditions before publishing any
new performance comparison. This inventory is intentionally limited to the
verified civilian entries above; it is not a claim that all historical work or
all checkpoints have been recovered.

## Integrated source and how to use it

- [Full-scale landing and first-descent BC](../method/fullscale-integration.md)
  ports three local commits against the current civilian interfaces. It keeps
  measured wind and the current BC data selector available. The new strict
  terminal score is not the historical ever-success metric.
- [Civilian scenario evaluation](../civil_scenarios.md) adds an opt-in recent
  local scenario suite. It does not silently enable the recovered interventions
  under the existing benchmark defaults.
- [Full-scale landing archive](../../experiments/fullscale_landing_archive/)
  retains negative results and distinguishes actual PPO updates from plans.
  Run its standard-library audit to validate the saved tables, not to replay
  the experiment.
- [Physical-belief landing](../../algorithms/physical_belief_rocket_landing/),
  [observable-survival](../../algorithms/observable_survival_rocket/), and
  [offline Phy-AI](../../experiments/phyai_offline_touchdown/) retain their own
  historical protocols. Their code coexists with the current tree; the
  archived actors were not recalibrated or benchmarked on its dynamics.

The civilian sync applies the relevant main changes without rewriting
`Yuxuan-Gong`'s history. This branch is not a byte-for-byte mirror of main.
