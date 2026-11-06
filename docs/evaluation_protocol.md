# Frozen Evaluation Protocol (2026-10-22)

**English** · [简体中文](evaluation_protocol.zh-CN.md)

Criterion, budget and reporting rules frozen before the formal experiments. After
freezing, revisions are allowed only as appended entries, and each must state
the revision date and reason in the body; already-published result entries may
not be retroactively edited.

## 1. Success definition and termination priority (frozen into code)

- Termination priority: **violation > success**. If the final step satisfies
  both the goal and any failure condition, it is recorded as a failure
  (`terminated`), not counted in `completed_successes`, and receives no terminal
  reward.
- This priority is implemented in `_CWHBase.step` in
  `satellite/rendezvous_docking/env.py` (`success = goal & ~violation`). Fix
  rationale: rendezvous's keep-out sphere (5 m) lies inside the goal sphere
  (10 m); the old behavior allowed "low-speed entry into keep-out" to be
  recorded simultaneously as success + terminated, making the success rate
  unintelligible.
- Docking has no keep-out, so a semantic change can only appear in the corner
  case "fuel runs out exactly on the capture step" (the baseline's maximum fuel
  is 0.912 kg < the 2 kg budget, so it is unaffected).
- Re-test after the semantic fix: the official rendezvous baseline remains
  1024/1024 = 100% (the 9 m hold point is outside keep-out). **RL training
  success rates from before the semantic fix (e.g., rendezvous 1.5%) are not
  directly comparable to the new criterion; formal experiments were re-run
  under the new criterion.**

## 2. Classical-controller (BC data source) audit

The BC warm-start data source = each task's `conventional_controller` (the
collector constructs and drives its own controller instance directly; the env
`info` does not contain controller actions). Every controller was tuned and
frozen on training/validation initial conditions before BC data collection
began. Freeze status:

| Task | Controller | Train success | Dev/test | Status | BC decision |
|---|---|---|---|---|---|
| quadx | SE(3) geometric (Lee et al. 2010) | 100% | 100% all scenarios | frozen | enabled |
| docking | CWH-LQR + shield | 100% (d=0/.5/1 × 3 seeds) | 63/64 on the 64-scenario grid | frozen | enabled |
| rendezvous | two-impulse CWH guidance | 100% (4096 eps) | 100% over the 150–1900 m grid | frozen | enabled |
| transfer | one-step greedy Gauss gating | 100% (d=0/.5/1) | optimality probe done | frozen | enabled |
| satellite | quaternion PD + CMG damped pseudo-inverse | 100% (1024 eps) | 100% random reset; singularity/180° records not shipped | frozen | enabled |
| fixedwing | L1 + TECS | 74.6% | dev 74.4% / test 73.1% | frozen (finalized after 21.3%→74.9% iteration; residual = structural OOB limit cycle) | enabled, but reports must state the ~75% expert ceiling inherited by BC |
| rocket | ZEM/PN + pulse-servo PDG | 84.1% (v1 instantaneous ever-success, 2000-step window) | 700-step suite A 69% / B 34% | frozen (v3 final; B_corners is a structural authority-saturation limit) | enabled, but reports must state the expert ceiling (v1 84.1%, v2 `ever_stable_2s` 100%) |

Conclusion: no controller is wildly out of control; the fixedwing/rocket
ceilings are structural controller limitations (causes documented). BC can only
inherit, not amplify, those ceilings, and reports must present them alongside
the expert success rate.

> Note: the table above is an audit snapshot frozen on 2026-10-22; its numbers
> are not current release artifacts. Current verifiable values are in
> [baselines.md](baselines.md):
> `results/fixedwing/baseline.json` is 97.0% (128 replicas / seed 13 / 1500
> steps, `complete=false`), and `results/rocket/baseline.json` is 100% (v2
> `ever_stable_2s`). The nonexistent `pendulum` task has been removed from the
> table — this repository registers only 10 tasks (see `ENVS` in
> `scripts/baseline_eval.py`).

Satellite steering-law ablation (does not change the frozen baseline): with the
same quaternion PD and gains, replace the damped pseudo-inverse with an ordinary
Moore–Penrose pseudo-inverse; registry name `quat_pd_mp_pinv_cmg`, evaluation
entry `scripts/satellite_steering_compare.py`. Configuration and near-singular /
exact-singular scenarios are in
[baseline-satellite-mp-pinv.md](method/baseline-satellite-mp-pinv.md).

## 3. Frozen evaluation protocol

For every task (uniform across all 10):

- **Training budget**: frozen as 1024 parallel envs × 1000 PPO updates; stop at
  budget, no early stopping, no checkpoint selection from training metrics
  (evaluate the final checkpoint only). The budgets actually hard-coded in each
  `train_<task>.py` are below (deviations from the frozen rule are logged in the
  2026-11-02 entry of §5):

  | Task | updates | steps/env | transitions |
  |---|---:|---:|---:|
  | quadx | 1000 | 100 | 102.4 M |
  | quadx_track | 200 | 100 | 20.5 M |
  | fixedwing | 1000 | 64 | 65.5 M |
  | quadplane | 1000 | 64 | 65.5 M |
  | rocket | 2000 | 64 | 131.1 M |
  | missile | 1000 | 64 | 65.5 M |
  | satellite | 1000 | 150 | 153.6 M |
  | transfer | 1000 | 32 | 32.8 M |
  | docking | 1000 | 32 | 32.8 M |
  | rendezvous | 1500 | 250 | 384.0 M |

  All entries use 1024 parallel replicas and seed 7.
- **Convergence thresholds (absolute success, training distribution and
  protocol)**:

  | Task | Threshold | Basis |
  |---|---|---|
  | quadx | ≥ 60% | current best 84.9% |
  | docking | ≥ 60% | MLP arm 77.9% |
  | transfer | ≥ 90% | current 98.9% |
  | rendezvous | ≥ 60% | current 82.2% (re-measured under the new success criterion) |
  | fixedwing | ≥ 60% | current 76.2% |
  | satellite | ≥ 10% | current 17.4% (baseline 100%; RL clearly unsolved, threshold is only a floor) |
  | rocket | ≥ 30% | 2026-10-22 snapshot: v1 baseline 84.1% (instantaneous ever-success; current v2 `ever_stable_2s` baseline is 100%) |

  Reaching the threshold is recorded as **converged**; not reaching it is
  recorded as **not converged** and reported as is. Narrative substitutes such
  as "still improving" are not allowed. Threshold revisions must be logged in
  §5. The "current" numbers above are 2026-10-22 snapshot values, not current
  release artifacts; the nonexistent `pendulum` row has also been removed (this
  repository does not have that task).
- **Evaluation interval**: during training, `model_*.pt` is saved every 200
  updates (rsl-rl `save_interval`); after training, exactly one deterministic
  evaluation is run on the final checkpoint (`eval.py`, restoring `env_kwargs`
  from the frozen `config.json`). No validation-set model selection during
  training.
- **Seeds and intervals**: formal experiments train each task with **3 seeds
  {7, 13, 21}**; report mean and [min, max] interval, not single-seed results.
- **Held-out test set**: training/validation use only the seed 7/13/21 IC
  streams; final generalization evaluation uses `eval.py --seed 101` (an RNG
  stream never seen in training), run once per task and reported unchanged.
  Baselines (`baseline_eval.py`) are also evaluated on seed 101 for
  same-distribution comparability.
- **Reporting rules**: success rates must state (1) whether they are
  training-protocol or deterministic-evaluation numbers, (2) the step window
  (rocket's 700-step and 2000-step windows differ enormously; default is 2000
  steps / official reset distribution), and (3) whether wind disturbance is
  enabled. Training may use wind; `eval.py` restores the training `env_kwargs`
  from `results/<task>/config.json` when present, so evaluation uses the
  training wind configuration rather than always calm air. Cross-method
  comparisons must explicitly fix the same wind configuration.

## 4. Cost and parallel-performance reporting (learning cost ≠ parallel performance)

RL main results report two comparisons, never mixed:

- **Equal interaction steps**: fixed transitions (`1024 envs × updates ×
  steps/env`; frozen rule 1024 × 1000 × 32 = 32.8 M; actual per-entry values in
  the §3 budget table), comparing final success/return — measures sample
  efficiency.
- **Equal wall-clock time**: report `train_wall_seconds` and
  `train_throughput_steps_per_s` (now recorded in `run.json`), comparing
  achievable success under the same wall-clock budget — measures engineering
  throughput. Report throughput, full training wall clock and peak VRAM
  (`peak_vram_gib`) separately (`summarize.py` now includes k steps/s and vram
  GiB columns).

BC reports its own cost, not charged to the RL training budget: demonstration
collection wall clock (`bc_warmstart.collect_wall_seconds`), pretraining wall
clock (`bc_warmstart.wall_seconds`), final actor/critic MSE, and collected
transitions (config.json `bc_warmstart` section). Arms using BC must note in
equal-interaction-step comparisons that PPO interaction steps do not include BC
collection steps.

## 5. Protocol revision log

Concise changelog of protocol revisions. Full detail of older entries is
available in git history.

- **2026-10-22 — initial freeze.** Success definition (violation > success),
  controller audit, per-task training budgets/convergence thresholds, seeds
  {7,13,21} + held-out {101}, and cost-accounting rules (§3–§4) are frozen.
- **2026-10-24 — env RNG stream fix.** `TensorRandom` seed assignment changed
  from `arange(n)+seed` to `mix32(seed ^ (i·0x9e3779b1))`, making the streams of
  seeds {7,13,21,101} pairwise disjoint. All env-RNG-dependent numbers from
  2026-10-22/23 are no longer bit-reproducible; baselines were re-run.
- **2026-10-24 — task-difficulty revision.** For the 5 tasks with 100% baseline
  success (quadx, satellite, transfer, docking, rendezvous) only initial/task
  parameter distributions were widened; success criterion, reward, horizon and
  failure bounds were untouched. Baselines re-run.
- **2026-10-26 — Rocket benchmark v2.** Six spherical-pad contact model +
  2-second continuous-settle primary metric (`ever_stable_2s`). Historical v1
  instantaneous scores are not comparable. See
  [env-rocket-landing.md](method/env-rocket-landing.md) §14.
- **2026-10-27 — Rocket policy rate 60→40 Hz.** Rocket policy/reward move to
  40 Hz (30 s horizon unchanged, 1800→1200 steps). Physics-step rates are now
  read from `PHYSICS_HZ` instead of being hard-coded. Pre-40-Hz rocket numbers
  are historical only.
- **2026-11-02 — documentation/code consistency (no result changes).** Task
  count confirmed at 10; §3 budget table lists actual hard-coded `train_<task>.py`
  values; evaluation wind protocol corrected (`eval.py` restores training
  `env_kwargs` from `config.json`); rocket reward is v5; rocket 84.1% is v1
  instantaneous, not v2; baseline sample sizes are 128 (protocol) / 256
  (variants), not 1024; sigma-pin mechanism retired (default Beta policy).
