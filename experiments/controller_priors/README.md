# Controller Priors: imitation warm start vs residual RL

Reproducible comparison of two ways to exploit a conventional controller in
PPO: **behavior-cloning warm start** (BC → PPO) and **residual RL** (frozen
controller + learned action correction). The study is deliberately
non-committal: it measures where each mechanism helps, where it fails, and
which experimental factor (controller quality, demonstration coverage,
residual authority, model mismatch) predicts the better mechanism.

This directory is self-contained; the official task entries, rewards, success
criteria and `results/<task>/` directories are not modified.

## 1. Experimental arms

| Arm | Code | Initialization | Controller used at run time |
|---|---|---|---|
| A `traditional` | `build_baseline(task, env, variant, design=...)` | — | full conventional controller |
| B `bc_only` | DAgger-aggregated dataset → `fit_behavior_cloning` (`fit_critic=False`) | actor = BC mean | learned policy only |
| C `ppo_scratch` | random `MLPModel` init | random | learned policy only |
| D `bc_ppo` | loads the **same** `bc_only/bc_checkpoint.pt` | DAgger BC actor | learned policy only, PPO fine-tuned |
| E `residual_ppo` | zero-init actor output + frozen controller | residual = 0 | `clamp(nominal + limit·tanh(u))` |

The optional `frozen_bc_residual_ppo` arm (F: frozen BC policy as the residual
base) is intentionally **not** part of the first-round pilot. It requires a
different teacher/base contract and is documented as future work rather than
mixed into E.

All learned arms use the **same PPO configuration within a task** (the task's
current `build_cfg` recipe); no arm gets an extra lr schedule, sigma pin or
observation-normalizer freeze. `fit_critic=False` is explicit in the main
comparison so the BC arm isolates the actor warm start; critic pretraining is
a separate ablation (a custom arm dict with `fit_critic=True`).

The optimizer-dependence of BC→PPO is tested as an explicitly separate
ablation (`stability_pin_bc_ppo`, `stability_pin02_bc_ppo`: sigma pin + lr–sigma
coupling). It is never folded into the main mechanism comparison.

**Results of the first-round pilot:** see [`RESULTS.md`](RESULTS.md) and
[`PROGRESS.md`](PROGRESS.md).

## 2. Code map

```
algorithms/residual/residual_env.py     ResidualVecEnv (per-channel rho, phase gate,
                                        success-gated authority curriculum)
algorithms/residual/action_space.py     DeltaActionVecEnv, ActionRatePenaltyVecEnv
algorithms/ppo/kl_anchor.py             KLAnchorPPO (PPO + KL(pi||pi_BC) anchor)
algorithms/warmstart/expert_dataset.py  episode-level expert data, splits, DAgger merge
algorithms/warmstart/bc_warmstart.py    BC fit (+ boundary-aware returns, fit_once,
                                        phase weights, delta targets)
algorithms/baselines/baselines.py       design-model (`design=`) + observe_applied_action
algorithms/train_utils.py               run_training hooks: env wrapper, BC dataset/ckpt,
                                        stage checkpoints, update=0, metadata,
                                        lr warmup/cosine, sigma anneal, actor freeze,
                                        env curriculum, segment hooks
experiments/controller_priors/
    configs/tasks.json                  per-task PPO/BC/eval protocol
    scripts/conditions.py               all factor scans (incl. model mismatch)
                                        + improvement ablations
    scripts/collect_bc.py               iteration-0 pure-expert bootstrap data
    scripts/collect_dagger.py           DAgger aggregation (default BC data)
    scripts/train_arm.py                one arm (incl. internal bc_bootstrap)
    scripts/select_checkpoint.py        validation-success checkpoint selection
    scripts/evaluate.py                 first-episode det/stoch evaluation + physical metrics
    scripts/run_experiments.py          smoke / pilot / scans / improvements / plot
    analysis/task_metrics.py            independent physical cost metrics
    analysis/aggregate.py               tables (only existing data)
    analysis/plot.py                    figures
    tests/test_controller_priors.py     32 focused tests
```

## 3. Contracts preserved

* Five-tuple env protocol: `obs, reward, terminated, truncated, extras`.
* `RslVecEnv` converts to PPO's four-tuple; `extras['time_outs']` and
  `extras['terminal_observations']` are untouched.
* float64 physics, float32 observations/actions, device-resident batched
  tensors, batched reset, no per-env Python physics loop.
* Controller ownership stays **outside** the env (`ResidualVecEnv`), the task's
  official reward / success / termination code is never changed.
* PPO's ratio is computed on the sampled variable `u`; the residual map
  `r = limit·tanh(u)` and the final clamp are deterministic action maps. No
  log-probability is taken on the clipped environment action.
* Zero residual reproduces the controller trajectory bit-for-bit (tested).

### Controller state audit

`controller.actions(obs)` is called exactly once per control step, on the
observation the policy acted on. After `env.step`:

1. `observe_applied_action(applied)` advances/corrects state that depends on
   the **actually delivered** action (docking `fuel_est`; its internal
   `steps_left` advanced once by `actions`);
2. `controller.reset_idx(term | trunc)` starts the new episode for done
   replicas;
3. the next observation is cached for the next nominal action.

Observation-driven state (satellite CMG steering, rendezvous receding-horizon
schedule) is a no-op in `observe_applied_action`. No env-private state is read
to patch the controller. Every controller reads only the `policy` observation.

## 4. Demonstration data

`collect_expert_dataset` rolls the **training env** under the controller and
records: raw observations, expert actions, episode/replica ID, in-episode
time, `terminated`/`truncated`/`success`/`done`, terminal observation captured
from `extras` **before** auto-reset, initial condition, task/controller
configuration, seed and wall time. Collection is batched; only the one-shot
episode-table construction touches the host.

* Collection stops after `episodes_per_replica` complete episodes (default 3
  docking/satellite, 2 rendezvous). A fixed-length tail is marked `cut`, and
  is excluded from the supervised splits (never treated as a true termination).
* **DAgger is the default BC data pipeline.** An internal `bc_bootstrap`
  student is fitted on pure-expert (iteration-0) rollouts; three DAgger
  iterations then execute a `beta**i` mixture of that student and the
  controller (`beta=0.5`), label every visited state with the controller, and
  merge all iterations. The reported `bc_only` checkpoint is fitted on the
  merged dataset, and `bc_ppo` loads exactly that checkpoint. The bootstrap
  arm is an implementation detail and is never reported as a comparison arm;
  no reported arm is trained on pure-expert data alone.
* **Representative selection, not bulk near-duplicates.** After collection the
  episodes are scored on the z-scored initial policy observation and kept by
  farthest-point sampling (`data_selection.method = "farthest_point"`), so the
  supervised set covers the initial-condition space instead of repeating
  almost-identical rollouts. The counts live in `tasks.json`:
  `data_selection.episodes` (pure-expert pool for the bootstrap student) and
  `data_selection.dagger_episodes` (kept per DAgger iteration; three iterations
  are merged). Whole episodes are kept, so the episode-disjoint split stays
  valid; `selection` in `dataset.json`/`summary.json` records the requested,
  candidate and kept counts.
* **One dataset per (task, seed).** Improvement ablations only change the PPO
  side, so they all reuse the `default` condition's demonstrations and DAgger
  aggregation and only refit their own `bc_only` checkpoint (the terminal-weight
  ablation changes the BC objective, not the data).
* **Dataset files are capped at 50 MiB** (`--max-dataset-mb`, default 50 on
  `collect_bc.py` / `collect_dagger.py`). Storage is compact (float16
  states/actions, float32 rewards, int16/int32 ids, no per-step duplicate of
  the terminal observation) and `load_expert_dataset` restores the canonical
  float32/float64/int64 training dtypes. DAgger collects one episode per
  replica per iteration; if a file still exceeds the cap, whole env replicas
  (whole complete episodes) are dropped so the episode-disjoint split stays
  valid, recorded in `meta['storage']`.
* Split is **episode-disjoint** 70/15/15; adjacent time steps never cross
  splits. Validation never updates normalizer statistics.
* Main BC uses `normalizer_mode='fit_once'`: the empirical input normalizer is
  fitted once on the training split (not on every mini-batch of every epoch),
  then frozen during the fit. This removes the repeated-epoch drift of the
  legacy per-batch path and makes BC/BC→PPO statistics comparable.
* `returns_to_go` is boundary-aware: success is a true terminal event (zero
  terminal value); timeout or collection cut tails are **excluded** from the
  critic target unless a bootstrap value is supplied. The target construction
  and excluded-pair count are recorded in `config.json`.

## 5. Residual authority and channels

`rho_i = residual_limit_i / (action_upper_i − action_lower_i)`; `residual_limit_i`
is also recorded. `rho` is the fraction of the full normalized channel range
(the pilot default is 0.05). Physical conversion is task-specific and logged:

| task | native action | residual authority conversion |
|---|---|---|
| docking / rendezvous | normalized per-axis thrust | `limit · f_max` N (10 N/axis, 500 kg) |
| satellite | normalized CMG gimbal rate | `limit · action_scale` rad/s (3 rad/s) |

The pilot measures the actual environment-action perturbation
(`exploration_env_action_rms`) for the stochastic policy; residual scaling is
never assumed to make exploration comparable to direct PPO.

Run-wide residual diagnostics are recorded: mean/RMS residual, saturation of
residual and composed command, residual-opposes-nominal fraction, composed
command clipping, and active residual fraction. Per-channel values are saved.

## 6. Evaluation protocol

* `scripts/first_episode.py::FirstEpisodes`: every replica contributes exactly
  its **first complete episode**, run until all replicas finish (safety cap,
  default 2000–3000 steps). Incomplete cohorts are saved as **partial** and
  explicitly marked; no fast-finishing-only window is used.
* Terminal statistics are taken from `extras['terminal_observations']` before
  auto-reset, or from `completed_*` counters.
* Deterministic (mean) and stochastic (sampled policy) evaluations are separate
  reports; training rollout statistics are a third source.
* All arms are evaluated with the same protocol seed (`--eval-seed 7`), same
  num_envs and same env configuration. The initial observation digest is saved
  and `analysis/aggregate.py` verifies paired initialization across arms.
* Physical cost metrics are independent of the official reward: propellant,
  maneuver time, terminal range/speed (docking/rendezvous); gimbal-rate-squared
  integral (an **actuator-usage proxy, not electrical energy**), maneuver time,
  terminal attitude/rate error (satellite). Success-conditioned cost is always
  reported next to the success fraction.

## 7. Factors (single-factor scans)

`scripts/conditions.py`:

* **controller quality** (`quality_sluggish/aggressive/model_bias`): design
  gains or a nominal design-model bias; the **true plant is unchanged**.
* **demonstration coverage** (`coverage_narrow/wide`): same sample count,
  different initial-state coverage; evaluation stays on the official
  benchmark distribution (distribution shift).
* **sample count** (`samples_1x/…`): same coverage, different number of
  complete episodes.
* **residual authority** (`authority_0.01/0.025/0.1/0.2`).
* **model mismatch** (`mismatch_plant`, `mismatch_design`, `mismatch_both`):
  the controller receives an explicit nominal `design=` model (mandatory);
  it cannot read the shifted plant constants from the env. True-plant shifts
  are opt-in env kwargs (`plant_kwargs`) that leave benchmark defaults intact.

### Improvement ablations (`--phase improvements`)

Each condition changes exactly one mechanism; the shared PPO/BC recipe is
otherwise identical, so every entry is an independent ablation.

| condition | arm(s) | mechanism |
|---|---|---|
| `improve_kl_anchor` | bc_ppo | PPO objective + linearly-decayed `KL(pi_theta || pi_BC)` |
| `improve_terminal_weight` | bc_only, bc_ppo | terminal-phase-weighted BC MSE (obs-only phase signal) |
| `improve_optimizer` | ppo_scratch, bc_ppo | lr warmup + cosine, sigma anneal, actor freeze (the sigma term needs `distribution='gaussian'`; the default Beta policy ignores `init_std` and has no `log_std_param`) |
| `improve_grad_clip` | bc_ppo | tighter grad-norm clip + per-mini-batch advantage norm |
| `improve_rate_penalty` | bc_ppo | training-only `-coef ||a_t - a_{t-1}||^2` auxiliary reward |
| `improve_delta_action` | ppo_scratch, bc_ppo (satellite) | bounded increment action parameterization |
| `improve_phase_gate` | residual_ppo | per-channel residual authority gated by terminal phase |
| `improve_authority_curriculum` | residual_ppo | success-gated residual-authority curriculum |
| `improve_scratch_curriculum` | ppo_scratch | initial-condition curriculum from scratch |
| `improve_selection` | bc_ppo, residual_ppo | checkpoint chosen by validation-seed success |

The official reward / success / termination code is never changed: the rate
penalty is a training-only wrapper, evaluation uses the raw task reward.
The authority curriculum's final fraction is written to `eval_config.json`
and evaluation composes the residual at exactly that fraction. Under
`improve_delta_action` the `bc_only`/`bc_ppo` arms fit the increment target
`d_t = (a_t - a_{t-1}) / scale` (clipped), matching the wrapper's map; for
residual arms delta increments are rejected (they already provide bounded
increments).

The improvement phase uses 5 seeds `{7, 13, 21, 33, 45}` so the analysis can
report a paired bootstrap CI. `improve_selection` trains with
`--select-checkpoint`, which keeps `stage_*.pt`, ranks them on an independent
validation seed (`seed + 1000`), and evaluates `selection/best_checkpoint.pt`.

## 8. Commands

```bash
PY=/home/ubuntu/.conda/envs/genesislab/bin/python   # env with torch + rsl-rl

# 0) tests
$PY -m pytest experiments/controller_priors/tests/test_controller_priors.py -q

# 1) smoke (2 tasks, 3 updates, 64 envs, all five arms)
$PY experiments/controller_priors/scripts/run_experiments.py --phase smoke

# 2) pilot (5 seeds; docking 250 updates/1024 envs, satellite 100/512,
#    rendezvous 200/512)
$PY experiments/controller_priors/scripts/run_experiments.py --phase pilot \
    --seeds 7,13,21,33,45 --out-root results/controller_priors_dagger \
    --tag improve_dagger_v3

# 3) single-factor scans (docking by default)
$PY experiments/controller_priors/scripts/run_experiments.py --phase scans --scan-task docking

# 4) improvement ablations (independent conditions, 5 seeds)
$PY experiments/controller_priors/scripts/run_experiments.py --phase improvements \
    --out-root results/controller_priors_dagger --tag improve_dagger_v3
# subset example:
$PY experiments/controller_priors/scripts/run_experiments.py --phase improvements \
    --tasks satellite --improve-seeds 7 --improve-num-envs 64 --improve-updates 3 \
    --out-root /tmp/cp_improve_smoke --tag dagger_smoke
# (dry-run prints the full command sequence: add --dry-run)

# 5) tables + figures
$PY experiments/controller_priors/analysis/aggregate.py
$PY experiments/controller_priors/analysis/plot.py
# or: ... run_experiments.py --phase plot
```

Each run lives at
`results/controller_priors/<task>/<method>/<condition>/seed_<seed>__<tag>/`
(no run outputs are shipped in this checkout) and contains `checkpoint.pt`,
optional `stage_*.pt`, `config.json`, `run.json`,
`arm.json`, `eval_config.json`, `eval/det.json`, `eval/stoch.json`,
`eval/det_stage_*.json`, `eval/trajectory.npz`. The internal
`bc_bootstrap/` and `bc_dagger_dataset/` directories are separate; the
reported BC/BC→PPO arms only use the latter.

## 9. Known limitations (first-round pilot)

* `fit_critic=False` in the main BC arm; critic pretraining is not measured yet.
* Satellite BC plain-MSE distillation of the first-round pilot reached only
  ~16% closed-loop success from a 100%-success expert — a covariate-shift/
  precision effect, not a controller failure. DAgger is now part of the
  default BC pipeline (not a separate ablation); BC/BC→PPO results are
  therefore DAgger-based.
* Docking BC→PPO collapses under the task's shared PPO recipe in the pilot;
  this is reported as a mechanism × stability observation, not as a proof that
  BC warm start is useless. The optimizer ablation (`improve_optimizer`) now
  exposes sigma anneal / lr warmup / actor freeze as separate knobs.
* The improvement mechanisms are implemented and smoke-tested (docking full
  chain; satellite delta actions; authority-curriculum evaluation). A first
  5-seed docking pass used pure-expert BC and is archived separately; the
  DAgger-default rerun of all three tasks is the reported configuration
  (tag `improve_dagger_v3`, results written under a distinct run tag and
  never overwriting the `improve_v1` archive).
* Rendezvous uses the shared fixed-lr protocol (no sigma pin/lr coupling) so
  all mechanisms face the same optimizer; the task's current best recipe
  includes a task-specific sigma pin that is intentionally treated as a
  separate optimizer ablation, not part of the mechanism comparison. (That
  sigma-pin recipe is Gaussian-era: the default Beta policy exposes no
  `log_std_param`, so the ablation is only available under
  `distribution='gaussian'`.)
* No trustworthy optimal-control reference solution exists, so only absolute
  physical cost and changes relative to the baseline are reported — no
  global-optimality gap is claimed.
* The pilot budgets are small relative to the official 1000–1500-update
  entries; resource estimates for the full study are in `PROGRESS.md`.
