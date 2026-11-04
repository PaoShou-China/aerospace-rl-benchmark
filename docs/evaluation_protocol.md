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
> [baseline_test_report.md](baseline_test_report.md):
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

- 2026-10-22 initial freeze (success definition, controller audit,
  thresholds/budget/seeds/test set, cost accounting).
- 2026-10-24 fix of env RNG stream seed assignment
  (`environments/common/plant.py` `TensorRandom`): changed from
  `arange(n)+seed` to `mix32(seed ^ (i·0x9e3779b1))`. Under the old scheme the
  i-th replica of base seed s+1 reused the same stream as the (i+1)-th replica
  of seed s; with 1024 envs adjacent seeds overlapped 1023/1024, and protocol
  seeds {7,13,21} overlapped 1018/1024 pairwise, so multiple seeds effectively
  shared one set of initial conditions (aggregate metrics could match bit for
  bit); after the fix the stream sets of {7,13,21,101} are pairwise disjoint.
  Consequently all env-RNG-related numbers produced on 2026-10-22/23 (including
  `results/<task>/baseline.json`, the §2 controller-audit success rates, and
  per-task harness numbers) are no longer bit-reproducible. The 8-task classical
  baselines were re-run under the new scheme (1024 envs / official reset
  distribution / seed 7, `scripts/baseline_eval.py`) and overwrote
  `results/<task>/baseline.json`; old values are historical only and must not be
  mixed or marged with new results.
- 2026-10-24 task-difficulty revision (user instruction: "slightly increase the
  difficulty of the tasks with 100% success"): for the 5 tasks with 100%
  baseline success (quadx / satellite / transfer / docking / rendezvous), only
  initial/task-parameter distributions were changed; **success criterion,
  reward, horizon and failure bounds were untouched**:
  - quadx: waypoint shell [1.0, 4.5] m → [1.5, 4.8] m (no near easy points;
    far points approach the 5 m OOB bound).
  - satellite: `reset_noise_scale` 0.001 → 0.01 (initial |ω| can reach ~0.017
    rad/s, close to the 0.02 rad/s success rate tolerance; momentum injection
    ≲2 N·m·s is well within CMG capacity).
  - transfer: `ecc0` 0.1 → 0.15 (rp0 9000→8500 km, ra0 11000→11500 km;
    ra_ref/rp_ref/vel_ref recomputed automatically).
  - docking: `init_range` (50, 100) m → (60, 120) m (top value is 40 m from the
    0.8·range_fail shield corridor; range_fail=200 m unchanged).
  - rendezvous: `init_range` (800, 1200) m → (1000, 1500) m (range_fail=2000 m
    unchanged, 500 m margin).
  After the revision the 5 classical baselines were re-run (same protocol 1024
  envs / seed 7) and the corresponding `results/<task>/baseline.json` files
  were overwritten; pre-revision 100% numbers are historical only. The "100%"
  rows in the §2 audit table are pre-revision.

### 2026-10-26: Rocket benchmark v2

`RocketLandingEnv` switched to the v2 six-spherical-pad contact model and the
2-second continuous settled primary metric; the original collision-bar assets
remain as comparison assets, and historical v1 scores are historical only. The
current main's 30 s horizon, initial distribution, observations/actions and
reward are retained. A 2 s continuous settle requires 81 observation points at
40 Hz policy steps (121 points at 60 Hz).

### 2026-10-27: Rocket policy rate 60→40 Hz; de-hardcoding physics step

- Rocket policy and reward move uniformly to 40 Hz (the same rate as the frozen
  research snapshot; the requested 50 Hz does not divide the 120 Hz control loop
  evenly, and upstream also rejects non-integer rates). The 30 s horizon is
  unchanged; policy steps go 1800→1200. The v4 reward coefficients are retained
  (at 40 Hz the safe bound for negative net payoff in a pure dive is
  u = 0.3/(0.1·0.025) = 120 m/s, still above the 100 m/s initial descent).
- Repository-wide hard-coded rates `1/240`, `/120`, `range(2)` now read the
  actual constructed values: `PHYSICS_HZ` in
  `environments/common/original_backend.py` is the single definition;
  `TensorPlant` reads back the actual scene step
  (`physics_dt`/`physics_hz`/`control_hz`/`physics_per_control`), and each env's
  inner physics substeps and PID gain scaling come from the plant. Rates are
  unchanged (240/120 Hz); apart from the rocket policy rate, dynamics are
  bit-identical.
- Rocket numbers recorded earlier under the 60 Hz protocol (including
  `results/rocket/baseline.json`) are historical only, not comparable to the new
  40 Hz results, and must be re-run before comparison.

The formal Rocket comparison uses the full 4×128 first episodes; other tasks
keep their existing sample sizes. The generic evaluation scripts fixed
double-counting of early-failed replicas; all tasks now take only each replica's
first episode. Reports with insufficient completeness are marked partial. New
training saves the environment configuration before starting; continuation
locks the disturbance profile and policy rate, and cross-configuration
experiments use separate directories.

Physics disturbance ranges, version boundaries, replacement rationale and
per-environment diagnostic evidence are in
[rocket_benchmark_v2.md](rocket_benchmark_v2.md). The above are new protocol
revisions and do not retroactively rewrite earlier frozen experiments.

### 2026-11-02: Documentation/code consistency revision (no experimental result changes)

This round only fixes descriptions and counts; it does not rewrite the frozen
2026-10-22/24/26/27 entries or change any published number:

- **Task count**: currently **10** tasks are registered (quadx / quadx_track /
  fixedwing / quadplane / rocket / missile / satellite / transfer / docking /
  rendezvous, see `ENVS` in `scripts/baseline_eval.py`). The nonexistent
  `pendulum` row in the §2 audit table has been removed; the rest of that table
  remains a 2026-10-22 audit snapshot, with current release artifacts noted
  below it (fixedwing 97.0%, rocket v2 100%).
- **Training budget**: §3 now lists the actual hard-coded values in each
  `train_<task>.py`. Deviations from the frozen rule "1024 envs × 1000 updates"
  are: rocket `UPDATES=2000`, rendezvous `UPDATES=1500`, quadx_track
  `UPDATES = 20_000 // 100 = 200`; the other entries use 1000 updates, but
  `steps/env` varies between 32 and 150.
- **Evaluation wind protocol**: the original statement "evaluation is always the
  calm-air baseline" is incorrect. `scripts/eval.py` restores the training
  `env_kwargs` (including `dryden`/`steady_wind`) from
  `results/<task>/config.json` when present; only when the config file is
  missing does it fall back to the environment's default calm air. §3 has been
  corrected accordingly.
- **Reward version**: the 2026-10-27 entry's "v4 reward coefficients retained"
  describes the state at that time; the current `environments/rocket/env.py`
  reward is **v5** (descent-speed corridor introduced 2026-10-28), see
  `docs/method/env-rocket-landing.md`.
- **Rocket metric protocol**: labeling 84.1% as "v2 reward" in the §2 audit
  table and §3 threshold table is a mislabel; that number comes from **v1
  instantaneous ever-success** (2000-step window) and is not comparable to the
  current v2 `ever_stable_2s` (2 s continuous settle, 40 Hz). The current
  `results/rocket/baseline.json` v2 metric is 100%.
- **Baseline sample size**: there are no 1024-replica baseline artifacts in the
  repository. Protocol dataset A is 128 replicas, four seeds; dataset B is 256
  replicas, single seed 7. Per-task metadata, Wilson intervals and failure modes
  are in [baseline_test_report.md](baseline_test_report.md).
- **`baseline.json` metadata**: the 2026-10-24 entry's statement "re-run at 1024
  envs / seed 7 and overwritten" describes that day's action; the current files
  are 128-replica protocol records (`quadx` exception: 32 replicas / seed 7;
  `quadx_track` 128 replicas / seed 21; the rest 128 replicas / seed 13). Always
  cite each file's own `num_envs`, `seed`, `env_steps`; do not reuse the old
  1024/seed 7 description.
- **sigma-pin mechanism retired (2026-11-02)**: `train_rendezvous.py` previously
  passed Gaussian-era knobs `sigma_pin=0.2` / `lr_sigma_coupling=1.5e-3`, which
  are incompatible with the default `BetaDistribution` (reading
  `log_std_param` raises `AttributeError`). By user decision the mechanism was
  **retired** rather than reverting to Gaussian: the entry now uses the default
  Beta policy like the other 9 entries and no longer passes those knobs;
  historical sigma tuning records are Gaussian-era references only.
