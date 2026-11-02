# PROGRESS — controller-prior study (BC warm start vs residual PPO)

Last updated: 2026-11-01 (dataset size budget added; full rerun in progress).

## 0. Repository state at start

* Repo: `github.com/PaoShou-China/aerospace-rl-benchmark` (local worktree
  `pyflyt-genesis`).
* `AGENTS.md`: **absent** in this checkout (nothing was read; no project-agent
  file exists at the root).
* Base commit: `b0e14cbe922228f3d9078477a262b839efa32500`
  ("Complete the quadplane chain, refresh rocket/track docs, add air variants").
* Pre-existing uncommitted work (BEFORE this task): 22 modified files and 8
  untracked files, including `algorithms/baselines/variants_*.py`,
  `environments/common/residual.py` (an env-internal residual mixin, unused by
  this study), `docs/baseline_test_report.md`, `scripts/baseline_suite.py`.
  All of it was preserved; this study only adds files and the opt-in hooks
  listed below.
* New uncommitted work from this task is listed in §9.

## 1. Scope and task selection

* First-round tasks: **docking**, **satellite**, **rendezvous**.
* Baseline re-verification with the *current source* (first-episode protocol,
  1024 envs, seed 7, official benchmark spawn):
  * docking: 100.0% (cwl_lqr), satellite: 100.0% (quat_pd_damped_cmg),
    rendezvous: 100.0% (cwh_two_impulse).
  * Re-run after the edits of this task at 256 envs (`_baseline_check_v2/`):
    100% on all three, confirming the opt-in design/plant hooks did not
    change default behaviour.
* Third task rationale: rendezvous was chosen (not rocket/transfer/quadplane)
  because its current controller was re-verified at 100%, its dynamics are
  pure-tensor, it has a genuinely stateful controller (online transfer-time
  ratchet) that stress-tests the residual state audit, and the 1000-step
  horizon makes demonstration coverage meaningful. Rocket was not used: its
  benchmark-v2 contract was not part of this study and its BC is known to fail
  on a near-bang-bang expert (documented in `docs/method/algo-bc-warmstart.md`).
  Quadplane is explicitly not validated as a conventional controller. Missile
  was not touched.
* Neither docking/satellite nor rendezvous is success-limited for the
  controller (all 100%); the study therefore reports time, propellant,
  actuator usage, terminal precision and learning cost alongside success.
* Documentation/source inconsistencies found and NOT refactored (source is
  authoritative):
  * `AGENTS.md` does not exist in this checkout.
  * `docs/method/algo-bc-warmstart.md` §5 says the per-task entries ship with
    `BC_WARMSTART=None`; the current `train_docking.py` / `train_satellite.py`
    / `train_rendezvous.py` set a non-None dict (2026-10-30 change). The
    experiment does not depend on those historical entries.
  * `docs/method/README.md` still points the quadplane controller doc at a
    "not yet validated" baseline; this study does not use quadplane.
  * `docs/baseline_test_report.md` and the extra `variants_*.py` modules are
    untracked work present before this task; the variants used here
    (`quat_pd_mp_pinv_cmg`, etc.) were checked in the current worktree.
  * README statements about old rocket "ever-success" numbers are explicitly
    marked non-comparable to benchmark v2; rocket was not used.

## 2. What was implemented

1. `algorithms/residual/residual_env.py`
   * `ResidualVecEnv` between the five-tuple env and `RslVecEnv`: owns the
     controller, computes nominal action once per step on the policy
     observation, composes `clamp(nominal + limit·tanh(u))`, records
     policy/nominal/residual/composed/applied, and calls
     `observe_applied_action` before `controller.reset_idx(term|trunc)`.
   * Per-channel authority `rho_i = limit_i/(upper−lower)`, physical-unit
     conversion per task, composition saturation / cancellation / clipping
     diagnostics.
   * `zero_init_policy_output` for residual PPO (deterministic zero residual).
   * `PolicyBaseController` for the optional frozen-BC residual arm.
2. `algorithms/baselines/baselines.py`
   * `Baseline.design` / `_design(name, default)`: explicit nominal design
     model, so a mismatch cannot be leaked by reading the shifted env.
   * `Baseline.observe_applied_action` (default no-op); docking override
     corrects `fuel_est` to the action actually delivered.
   * `build_baseline(..., **controller_kwargs)`.
   * Docking LQR / rendezvous two-impulse / satellite CMG PD read design
     constants (mass, thrust, Isp, inertia, rotor momentum, rate limit,
     gains) through the design model.
3. `algorithms/warmstart/expert_dataset.py`
   * Batched episode-level dataset (obs, expert action, episode/replica ID,
     in-episode time, terminated/truncated/success/done, terminal observation
     before reset, initial condition, controller config, seed, wall time).
   * Complete-episode collection, cut-tail marking, episode-disjoint 70/15/15
     split, flattening without cross-episode leakage.
4. `algorithms/warmstart/bc_warmstart.py`
   * Backward-compatible extensions: `normalizer_mode='fit_once'`,
     `freeze_normalizer`, validation metrics that never update normalizers,
     boundary-aware `returns_to_go` (success=terminal, timeout/cut excluded
     unless bootstrapped), `fit_critic=False` isolation.
5. `algorithms/train_utils.py`
   * `env_wrapper` hook, `runner_init_hook` (zero residual output),
     BC from an episode dataset or from an existing BC checkpoint, BC
     checkpoint save, `updates=0` BC-only runs, stage checkpoints
     (`stage_0`/`stage_N`), experiment metadata in `run.json`/`config.json`,
     residual diagnostics in the report.
   * Default `clean=True` still applies only to the *unique* experiment
     `out_dir`; official `results/<task>/` is untouched.
6. `environments/...` opt-in kwargs only (defaults unchanged):
   * docking/rendezvous: `init_range`, `init_vmax`, `plant_kwargs`
     (`mass`, `f_max`, `isp`, `g0`);
   * satellite: `reset_noise_scale`, `plant_kwargs`
     (`inertia`, `rotor_h_dyn`, `rotor_speed`).
7. Experiment package (`experiments/controller_priors/`): configs, conditions
   (all factor scans), dataset collection, arm training, first-episode
   evaluation with independent physical metrics, orchestration, aggregation,
   figures, README, tests.
8. Improvement ablations (one condition = one mechanism; official reward /
   success / termination unchanged):
   * `algorithms/ppo/kl_anchor.py`: `KLAnchorPPO` adds a linearly-decayed
     `KL(pi_theta || pi_BC)` term to the PPO loss (frozen anchor set from the
     loaded BC checkpoint).
   * `residual_env.py`: per-channel `rho_channels`, observation-only phase
     gate, and a success-gated authority curriculum
     (`segment_hook`, start/end/threshold/step/min_episodes).
   * `algorithms/residual/action_space.py`: `DeltaActionVecEnv` (bounded
     increments, reset per episode) and `ActionRatePenaltyVecEnv`
     (training-only reward penalty).
   * BC: `phase_weights` (terminal-phase-weighted actor MSE) and
     `delta_action_targets` (increment targets under the delta
     parameterization); both are applied only to the actor loss.
   * `expert_dataset.py`: `merge_datasets` for DAgger aggregation (episode
     table and per-step `episode_id` renumbered consistently).
   * `collect_dagger.py`: beta-mixture student rollouts labelled by the
     controller, 3 iterations, distinct `bc_dagger_dataset` directory.
   * Dataset size budget (<= 50 MiB per file, hard cap): the per-step
     `terminal_obs` copy is dropped (each episode's terminal observation
     stays in the episode table), states/actions are stored as float16,
     rewards as float32 and the id columns in the narrowest integer width;
     `load_expert_dataset` restores float32/float64/int64 for training.
     `save_expert_dataset(..., max_mb=50)` additionally drops whole env
     replicas (= whole complete episodes) if a file still exceeds the cap,
     recorded in `meta['storage']`. DAgger collects 1 episode per replica
     per iteration (`tasks.json: dagger.episodes_per_replica`).
     Measured (`improve_dagger_v2`, seed 7, million-transition scale):
     docking 25.7 / 23.8 MiB, satellite 38.6 / 49.8 MiB,
     rendezvous 18.8 / 26.5 MiB (`dataset.pt` / `dagger_dataset.pt`).
   * `train_arm.py` / `run_experiments.py`: **DAgger is the default BC path**.
     An internal `bc_bootstrap` arm fits the iteration-0 pure-expert data only
     to provide the DAgger student; the reported `bc_only` (and therefore
     `bc_ppo`) is always fitted on the merged DAgger dataset. No reported arm
     uses pure-expert data alone.
   * `select_checkpoint.py`: ranks `stage_*.pt` + `checkpoint.pt` by
     deterministic validation-seed success (then terminal error, return) and
     writes `selection/best_checkpoint.pt`; `evaluate.py --stage best` uses it.
   * `train_utils.py`: lr warmup + cosine, sigma annealing, actor freeze for
     the first N updates, update-segment hooks, env curriculum from scratch.
   * `train_arm.py` writes the final residual authority into
     `eval_config.json`; `evaluate.py` rebuilds the identical delta/rate/
     residual wrapper and composes the residual at the trained fraction.
   * `run_experiments.py --phase improvements`: 5 seeds
     `{7, 13, 21, 33, 45}`, independent condition per mechanism, optional
     validation-selection evaluation.
   * `analysis`: paired bootstrap CI helper for the 5-seed comparison.

## 3. Verification actually run

* `pytest experiments/controller_priors/tests/test_controller_priors.py -q`:
  **30 passed** (the original arm/contract tests plus phase weights, dataset
  merge offsets, delta wrapper, rate penalty, per-channel rho + phase gate,
  authority gating, DAgger labels, checkpoint ranking, bootstrap CI,
  KL-anchor decay, delta BC targets, satellite scalar-gain actions,
  aggregation of `*_stage_best` selection evaluations, and compact-encoding
  round-trip + size-cap replica trimming).
* Smoke phase (`run_experiments.py --phase smoke`, 2 tasks, 64 envs, 3 updates,
  all five arms): all commands returned 0; deterministic/stochastic
  first-episode evaluation complete for every arm.
* Improvement smoke with the **DAgger-default pipeline** (docking, 64 envs,
  3 updates, seed 7, tag `dagger_smoke`, `/tmp`): all 9 conditions returned 0.
  Verified: every reported `bc_only` is fitted on
  `bc_dagger_dataset/.../dagger_dataset.pt`; `bc_bootstrap` (pure-expert
  iteration-0 only) is an internal run with no evaluation; every `bc_ppo` run
  loads `bc_only/bc_checkpoint.pt`; `collect_dagger` recorded three iterations
  (beta 1.0/0.5/0.25); `improve_selection` ranked `stage_*.pt` on seed 1007
  and its `best_checkpoint.pt` scored 100% deterministic/stochastic.
  `KLAnchorPPO` is confirmed in `config.json` (`kl_anchor_coef=0.05`,
  decay 100); terminal weights, sigma anneal/actor freeze, scratch curriculum,
  phase gate, authority curriculum (`authority_fraction=0.01`) are visible in
  `run.json`/`selection.json`.
* Satellite delta smoke (`improve_delta_action`, 64 envs, seed 7, tag
  `delta_smoke`): the bootstrap student carries no `delta_action_scale`
  (it must emit absolute actions for DAgger mixing), `bc_only` fits the DAgger
  merge with `delta_action_scale=0.25`, `bc_ppo` loads that checkpoint, and
  three DAgger iterations were recorded.
* Shared-baseline fix: the pre-existing batched-gain edit in
  `algorithms/baselines/baselines.py` indexed scalar docking shield/guard
  thresholds and satellite PD gains with `[:, None]`. They are now broadcast
  to per-env tensors (scalars and `set_params` tensors both work); the docking
  expert then collected 212/212 successful complete episodes, and all tests
  still pass.
* Aggregation of validation-selected arms: `improve_selection` writes only
  `eval/det_stage_best.json` / `stoch_stage_best.json`; `analysis/aggregate.py`
  now falls back to these files (recording `stage='best'`) instead of labelling
  the arm `not_evaluated`. The paired-initial-state check includes `best`
  rows, so the selection arm is verified against the other arms' digest.
* Authority-curriculum evaluation (fresh directory): `eval_config.json`
  carries `authority_fraction=0.01`; the evaluation residual RMS is ~6.4e-5
  (vs a full-authority run), confirming the trained fraction is applied.
* Dry-run of `--phase improvements` for satellite and rendezvous prints the
  complete `collect_bc` → `collect_dagger` → `train_arm` → `evaluate`
  sequence; no missing conditions.
* Baseline re-verification after edits: docking/satellite/rendezvous 100%.
* Docking stage-save/evaluation, BC checkpoint reuse, residual zero-init,
  stochastic perturbation measurement, paired init digests verified.

## 4. Pilot protocol (actual)

* Evaluation seeds: training seeds `{7,13,21}`; all arms evaluated with the
  same protocol seed `--eval-seed 7`, 1024 envs (docking) / 512 envs
  (satellite/rendezvous), same benchmark env kwargs and paired initial
  digest.
* Budgets (pilot-scale, not the final paper budget):
  * docking: 1024 envs, 250 PPO updates (8.2M transitions per learned arm) —
    3 seeds.
  * satellite: 512 envs, 100 updates (7.7M transitions) — 3 seeds.
  * rendezvous: 512 envs, 200 updates (25.6M transitions) — 3 seeds.
* Stage checkpoints at 0/25/50/… and deterministic first-episode evaluation;
  stochastic evaluation at the final checkpoint.
* BC data: docking 3 episodes/replica (≈0.46M train pairs), satellite 3
  episodes/replica (≈0.27M train pairs), rendezvous 2 episodes/replica.
  `fit_critic=False`, `normalizer_mode='fit_once'`.

## 5. Observed pilot results

> **Archive note**: the pilot table below predates the DAgger-default BC
> pipeline (it uses pure-expert demonstrations). It is kept for provenance
> only; the reported configuration is the DAgger-default `improve_dagger_v1`
> rerun, which is not mixed into this table.

Full tables: [`RESULTS.md`](RESULTS.md); raw: `results/controller_priors/_analysis/`
and `results/controller_priors_scans/_analysis/` (**not shipped**).

Final deterministic first-episode success (3 seeds, mean ± sd):

| task | traditional | bc_only | ppo_scratch | bc_ppo | residual_ppo |
|---|---|---|---|---|---|
| docking | 1.000 | 1.000 | 0.071 ± 0.122 | 0.000 | 1.000 |
| satellite | 1.000 | 0.108 ± 0.087 | 0.993 ± 0.007 | 0.992 ± 0.003 | 1.000 |
| rendezvous | 1.000 | 0.976 ± 0.025 | 0.000 | 0.195 ± 0.075 | 1.000 |

Cost at rho=0.05 (all completed first episodes):
* docking: residual 0.696 kg / 188.6 s / saturation 0.436 vs controller
  0.713 kg / 189.6 s / 0.758; BC-only 0.710 kg / 0.406.
* rendezvous: residual 1.436 kg / 511.1 s / saturation 0.092 vs controller
  1.434 kg / 510.7 s / 0.173; BC-only 1.430 kg / 613.2 s / 0.078.
* satellite: residual 6.27 s / gimbal∫ 59.6 / saturation 0.251 vs controller
  6.11 s / 63.7 / 0.348; from-scratch 6.69 s / **24.6** / 0.128; BC→PPO
  5.90 s / 54.1 / 0.329.
of the three methods, direct PPO is the actuator-efficiency winner on satellite;
residual preserves 100% success but inherits controller actuator usage.

Headline findings:

1. The mechanism ranking is task-dependent. On docking and rendezvous,
   residual keeps the 100% controller guarantee and reduces actuator
   chattering (and, at moderate authority, fuel); BC-only also keeps success,
   while BC→PPO collapses under the shared recipe. On satellite, scratch PPO
   is stable and much more actuator-efficient than the controller, and small
   residual authority cannot inherit that efficiency; BC→PPO recovers to
   ~99.2%.
2. The BC→PPO collapse is partly an optimizer problem: with a pinned
   sigma 0.05 + lr-sigma coupling, docking BC→PPO holds 100% and improves
   fuel to 0.654 kg / 181.1 s. Rendezvous BC→PPO still collapses with both
   sigma 0.05 and 0.2 pins at 200 updates, so the rendezvous failure is a
   genuine task-level failure at this spawn/budget.
3. Model mismatch: with the controller's design model pinned to nominal and
   the true plant shifted (550 kg, 9 N/axis, Isp 250), traditional / BC-only /
   residual keep 100% success while scratch PPO reaches 0%; BC→PPO remains
   fragile. This is the clearest case for keeping the controller in the loop.
4. Residual authority: docking improves monotonically with rho (fuel
   0.713→0.451 kg at rho 0.4, success 1.0), but rho 0.4 is no longer a small
   correction. Satellite degrades (success 0.939 at rho 0.2, 0.912 at rho 0.4,
   settling time 8.4/10.1 s) even though its actuator integral remains near
   the controller's; the safe authority is task-dependent.
5. Demonstrations: BC-only success is insensitive in the tested range; BC→PPO
   improves with more samples (0.129→0.275 from 1x to 6x) and degrades with
   wider demonstration coverage (0.330 narrow → 0.113 wide).
6. Independent validation on held-out docking seed 101 reproduced the ranking
   exactly (traditional/BC/residual 100%, scratch/BC→PPO 0%; residual fuel
   0.692 vs 0.713 kg).

## 6. Known failures / not yet verified

* Satellite BC-only (~10.8% mean, 0.8–16.2% across seeds) is covariate shift /
  precision distillation of plain MSE BC, not a controller failure. That
  number is from the archived pure-expert run; DAgger is now the default BC
  data pipeline and the `improve_dagger_v1` rerun supersedes it.
* Rendezvous BC→PPO collapses within 25 updates in all 3 seeds and remains
  below 2% under the two sigma-pin ablations at this 200-update/benchmark-spawn
  budget.
* `fit_critic=True` (critic pretraining) is not part of the main pilot.
* `frozen_bc_residual_ppo` (optional arm F) has only the code scaffolding
  (`PolicyBaseController`); it was not run.
* Final full scans for satellite/rendezvous (quality/coverage/samples) were
  not budgeted; docking covers all factors, satellite only residual authority.
* No warm start beyond BC exists; this study is PPO-only by design.
* No optimal-control reference: only absolute physical cost and relative
  changes are reported, never a global-optimality gap.
* Scalar global metrics keep their run-wide meaning (`FirstEpisodes.report`);
  they are not per-first-episode per-trajectory.
* All runs in this round shared one RTX 3090 concurrently; `run.json` wall
  times/throughput include contention. Transitions and evaluations are
  unaffected.

## 7. Scans actually run

Docking (seed 7, 512 envs, 100 updates), `results/controller_priors_scans`
(not shipped):
controller quality ×3, demonstration coverage ×2, sample count ×2, residual
authority ×6 (including rho 0.05 and 0.4), model mismatch ×3 (plant, design,
both), BC→PPO pinned-sigma ablation ×2 outputs (bc_only + bc_ppo).
Satellite (seed 7, 512 envs, 100 updates): residual authority rho 0.2 and 0.4.
All tables in `RESULTS.md`.

## 8. Resource estimate for the formal study

Measured pilot throughput (from each `run.json`; CONCURRENT runs on one
RTX 3090): docking 110–121k steps/s, satellite 23.5–32k steps/s, rendezvous
82–150k steps/s. Peak VRAM 0.05 / 0.09 / 0.42 GiB. The shipped official
recipes use 1000–1500 updates at 1024 envs; per task and seed that is
3.3e7–3.8e8 transitions.

Recommended formal budget (5 arms × 3+ seeds × 3 tasks), using the measured
contended throughput as the conservative case:
* docking: 1000 updates × 1024 envs (3.3e7 transitions) ≈ 5 min/arm/seed →
  ~1.5 h for 15 runs + BC + ~10 stage evals.
* rendezvous: 1500 updates × 1024 envs (3.8e8 transitions) ≈ 64 min/arm/seed →
  ~16 h for 15 runs.
* satellite: 1000 updates × 1024 envs (1.5e8 transitions) ≈ 80 min/arm/seed →
  ~20 h for 15 runs.
* Single-factor scans (1 seed, 5 factors, 3–6 conditions, 3–5 arms): the
  executed docking scans + satellite authority (84 runs at 512 envs / 100
  updates) took ~1.5 h under contention; a full 3-task scan is ≈ 6–10 h.
* Independent validation (held-out seed/parameter region): 5 arms × 3 tasks ≈
  3 h.
Total ≈ 45–50 GPU-hours (conservative, concurrent-throughput basis), dominated
by satellite's 40 Hz RK4 + 10 substeps and the 1500-update rendezvous recipe.
Use 3 seeds for uncertainty; add seeds only after the pilot variance is known.

## 9. Files added/modified by this task

Added:
`algorithms/residual/{__init__,residual_env,action_space}.py`,
`algorithms/ppo/kl_anchor.py`,
`algorithms/warmstart/expert_dataset.py`,
`experiments/controller_priors/{README.md,PROGRESS.md,
RESULTS.md,configs/tasks.json,
scripts/{common,conditions,collect_bc,collect_dagger,select_checkpoint,
train_arm,evaluate,run_experiments}.py,
analysis/{task_metrics,aggregate,plot}.py,
tests/test_controller_priors.py}`.

Results (not source; **not shipped** in this checkout):
`results/controller_priors/{docking,satellite,rendezvous}/...` (pilot_v2 for
docking, pilot_v1 for satellite/rendezvous, holdout_v1 for docking seed 101,
stability ablations),
`results/controller_priors/_analysis/`,
`results/controller_priors/_analysis_holdout/`,
`results/controller_priors_scans/` (+ `_analysis`),
`results/controller_priors/_baseline_check_v2/` (post-edit baseline recheck).

Modified (opt-in paths, defaults preserved):
`algorithms/baselines/baselines.py`,
`algorithms/warmstart/bc_warmstart.py`,
`algorithms/train_utils.py`,
`environments/satellite/rendezvous_docking/env.py`,
`environments/satellite/attitude_control/env.py`.

## 10. Reproduce

```bash
PY=/home/ubuntu/.conda/envs/genesislab/bin/python
$PY -m pytest experiments/controller_priors/tests/test_controller_priors.py -q
$PY experiments/controller_priors/scripts/run_experiments.py --phase smoke
$PY experiments/controller_priors/scripts/run_experiments.py --phase pilot
$PY experiments/controller_priors/scripts/run_experiments.py --phase improvements
$PY experiments/controller_priors/analysis/aggregate.py
$PY experiments/controller_priors/analysis/plot.py
```

`--phase improvements` writes to `results/controller_priors_improvements/`
(current DAgger-default tag `improve_dagger_v1`, 5 seeds) and never overwrites
an existing run directory. The earlier pure-expert `improve_v1` archive is
kept separate and is not mixed into the reported tables.
