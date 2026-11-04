# Full Classical (Non-Learned) Baseline Test Report

**English** · [简体中文](baseline_test_report.zh-CN.md)

## 1. Data provenance and conventions

This report uses only JSON artifacts that **actually exist in the repository**;
no undelivered legacy numbers are carried forward. The two artifact families
correspond to two protocols:

| Dataset | Path | Parallel replicas | Seeds | Accounting |
|---|---|---|---|---|
| A: protocol multi-seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101 (held-out) | first episode per replica only (`FirstEpisodes`); 128 first episodes per seed |
| B: controller variants | `results/<task>/baselines/<controller>_seed7.json` | 256 | 7 | first episode per replica, single seed |

Common conventions:

- Success rate = `successes / episodes_finished` (i.e. each replica's first-episode
  `completed_successes / completed_episodes`).
- All dataset A/B JSON files have `env_kwargs = {}`, meaning calm air and default
  task difficulty. The generating script's convention differs: the current
  `scripts/baseline_suite.py` (dataset B) also writes `domain_randomize=False`,
  so freshly regenerated B artifacts carry that key; the old artifacts shipped
  in the repository still have `{}`. Any citation must include its `env_kwargs`.
- Step windows come from each JSON's `env_steps`, not a single hard-coded value.
- The success denominator is 128 (A) or 256 (B) first episodes;
  `episodes_completed_total` can be larger because replicas keep stepping after
  their first episode until the safety cap.

> **Difference from the old report**: the old report claimed "1024 parallel envs,
> 3 seeds, 3072 episodes". No 1024-replica baseline artifact exists in the current
> repository: dataset A is all 128 replicas, dataset B is 256 replicas (single
> seed). The 1024/3072 sample-size claim has been removed; report the actual JSON
> metadata instead.

## 2. Default controllers: protocol seeds {7, 13, 21} + held-out seed 101

Each task uses the default baseline registered in `BUILDERS`. The mean and
standard deviation use protocol seeds 7/13/21 only; the Wilson 95% interval pools
the three seeds' episodes (384 episodes).

| Task | Default controller | seed 7 | seed 13 | seed 21 | seed 101 | mean±sd (7/13/21) | [min,max] | Wilson 95% CI (pooled 384) | Mean return | Episodes | Step window |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| quadx | `se3_geometric` | 100.0 | 100.0 | 99.2 | 100.0 | 99.7±0.5 | [99.2, 100.0] | [98.5, 100.0] | 72.5 | 384 | 648 |
| quadx_track | `cv_kf_pid` | 89.1 | 85.9 | 82.8 | 82.0 | 85.9±3.1 | [82.8, 89.1] | [82.1, 89.1] | 86.5 | 384 | 1200 |
| fixedwing | `l1_tecs` | 89.8 | 87.5 | 83.6 | 90.6 | 87.0±3.2 | [83.6, 89.8] | [83.2, 90.0] | 208.8 | 384 | 3600 |
| rocket | `zem_zev_pdg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 899.5 | 384 | 1200 |
| missile | `apn_rate` | 75.0 | 71.1 | 85.9 | 70.3 | 77.3±7.7 | [71.1, 85.9] | [72.9, 81.2] | 103.3 | 384 | 607 |
| quadplane | `se3_pusher` | 85.9 | 89.1 | 92.2 | 85.2 | 89.1±3.1 | [85.9, 92.2] | [85.5, 91.8] | 268.6 | 384 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 65.2 | 384 | 320 |
| transfer | `gauss_tangential` | 100.0 | 100.0 | 97.7 | 100.0 | 99.2±1.4 | [97.7, 100.0] | [97.7, 99.7] | 0.0 | 384 | 166 |
| docking | `cwh_lqr` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 189.7 | 384 | 267 |
| rendezvous | `cwh_two_impulse` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 184.8 | 384 | 601 |

Notes:

- `rocket` success uses the v2 primary metric `ever_stable_2s` (2 s continuous
  settled), not the old instantaneous `ever_success`.
- `quadx_track` success means visibility fraction ≥ 0.85 with no collision; the
  held-out seed 101 value 82.0% is an independent confirmation outside the three
  protocol seeds.
- Every row satisfies `complete = true` (each replica's first episode finishes
  inside the window).

## 3. All 30 controller variants (seed 7, 256 replicas)

Dataset B is a single-seed, single-run evaluation and **provides no cross-seed
interval**; it ranks controllers within a task and must not be mixed with the
3-seed means in §2. The success denominator is fixed at 256 (first episode per
replica); the final "total episodes" column is `episodes_completed_total`, only
indicating how many extra episodes rolled forward inside the window.

| Task | Controller | Success | Mean return | Total episodes (`episodes_completed_total`) | Step window |
|---|---|---|---:|---:|---:|
| quadx | `cascaded_pid` | 99.2% | 72.74 | 257 | 800 |
| quadx | `se3_geometric` | 100.0% | 72.86 | 278 | 648 |
| quadx | `smc` | 100.0% | 72.86 | 267 | 557 |
| quadx_track | `cv_kf_pid` | 85.2% | 85.68 | 256 | 1200 |
| quadx_track | `pid_ema` | 75.8% | 75.98 | 256 | 1200 |
| quadx_track | `pursuit` | 75.8% | 71.85 | 256 | 1200 |
| fixedwing | `l1_tecs` | 98.6% | 230.09 | 280 | 1500 |
| fixedwing | `nlgl_tecs` | 94.5% | 222.94 | 292 | 1500 |
| fixedwing | `pid_cascade` | 91.6% | 219.07 | 188 | 1500 |
| rocket | `eguidance` | 34.4% | −3332.01 | 267 | 1200 |
| rocket | `pd_landing` | 100.0% | 22.34 | 256 | 1200 |
| rocket | `zem_zev_pdg` | 100.0% | 907.31 | 256 | 1200 |
| missile | `apn_rate` | 76.2% | 101.90 | 480 | 637 |
| missile | `ppn` | 2.0% | 27.21 | 704 | 900 |
| missile | `tpn` | 76.2% | 102.13 | 488 | 580 |
| quadplane | `hybrid_cascade` | 26.6% | 173.94 | 257 | 1200 |
| quadplane | `l1_wing` | 49.6% | 226.20 | 256 | 1200 |
| quadplane | `se3_pusher` | 87.5% | 271.97 | 258 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0% | 64.85 | 258 | 330 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0% | 64.54 | 260 | 366 |
| satellite | `quat_pd_wpinv_cmg` | 100.0% | 65.03 | 259 | 337 |
| transfer | `edelbaum` | 100.0% | 0.22 | 256 | 166 |
| transfer | `gauss_tangential` | 100.0% | 0.01 | 256 | 166 |
| transfer | `q_law_petropoulos` | 100.0% | 1.15 | 256 | 166 |
| docking | `cwh_lqr` | 100.0% | 188.62 | 261 | 274 |
| docking | `cwh_pole_placement` | 100.0% | 189.30 | 257 | 236 |
| docking | `zem_zev` | 100.0% | 190.06 | 256 | 313 |
| rendezvous | `cwh_compensated_pd` | 100.0% | 179.44 | 256 | 875 |
| rendezvous | `cwh_two_impulse` | 100.0% | 184.91 | 256 | 594 |
| rendezvous | `glideslope` | 100.0% | 180.96 | 256 | 805 |

Note: `fixedwing` uses `env_steps = 1500`, shorter than the 3600-step task
horizon, so only 216 / 237 / 179 of the 256 replicas finish their first episode
inside the window (`episodes_finished`); the success denominator is those
finished first episodes. The larger "total episodes" comes from finished
replicas continuing to step. `missile` and `rocket` total-episode counts are set
by their own termination conditions (e.g. many `ppn` timeouts roll more episodes
inside the window).

## 4. Failure modes (`task_metrics` of dataset B, seed 7)

| Task | Controller | crash | out-of-bounds | timeout |
|---|---|---:|---:|---:|
| quadx | `cascaded_pid` | 0.0% | 0.0% | 0.8% |
| quadx | `se3_geometric` | 0.0% | 0.0% | 0.0% |
| quadx | `smc` | 0.0% | 0.0% | 0.0% |
| quadx_track | `cv_kf_pid` | — | 0.0% | 14.8% |
| quadx_track | `pid_ema` | — | 0.0% | 24.2% |
| quadx_track | `pursuit` | — | 0.0% | 24.2% |
| fixedwing | `l1_tecs` | 0.0% | 2.5% | 0.0% |
| fixedwing | `nlgl_tecs` | 0.0% | 6.8% | 0.0% |
| fixedwing | `pid_cascade` | 0.0% | 10.1% | 0.0% |
| quadplane | `hybrid_cascade` | 0.0% | 4.7% | 68.5% |
| quadplane | `l1_wing` | 0.0% | 0.0% | 50.4% |
| quadplane | `se3_pusher` | 0.4% | 0.4% | 12.0% |
| missile | `apn_rate` | — | — | 0.0% |
| missile | `ppn` | — | — | 3.1% |
| missile | `tpn` | — | — | 0.0% |

- `quadx_track` failures are all timeouts (visibility fraction not reached within
  30 s), not collisions or out-of-bounds.
- `quadplane`'s `hybrid_cascade` / `l1_wing` failures are mostly timeouts
  (68.5% / 50.4%), showing the mode-switching autopilot cannot stably finish 4
  waypoints within 30 s; `se3_pusher` mostly times out near waypoints.
- `rocket` / `satellite` / `transfer` / `docking` / `rendezvous` JSON files do
  not provide per-failure counts, so they are omitted.

## 5. Method family tree (dataset B labels; `--baseline` CLI key in parentheses)

| Task | Controller (label / CLI variant key) | Method |
|---|---|---|
| quadx | `se3_geometric` | Lee SE(3) geometric control (default, `BUILDERS`) |
| quadx | `cascaded_pid` | position→attitude cascaded PID (Euler) |
| quadx | `smc` | sliding-mode position outer loop (exponential reaching law + boundary layer) + Euler attitude cascade |
| quadx_track | `cv_kf_pid` | CV Kalman filter target estimate + SE(3) tracking (default) |
| quadx_track | `pid_ema` | first-order EMA target filter + SE(3) tracking |
| quadx_track | `pursuit` | filter-free pure pursuit (raw noisy relative measurement) |
| fixedwing | `l1_tecs` | L1 lateral guidance + TECS longitudinal (default) |
| fixedwing | `nlgl_tecs` | Park nonlinear guidance law (NLGL) + TECS |
| fixedwing | `pid_cascade` | conventional cascaded PID attitude/altitude/airspeed |
| rocket | `zem_zev_pdg` | ZEM/ZEV midcourse + virtual-platform PDG + velocity-band pulse servo (default) |
| rocket | `pd_landing` | pure PD landing (no guidance layer) |
| rocket | `eguidance` | Apollo E-guidance quadratic acceleration + terminal pulse servo |
| missile | `apn_rate` | augmented proportional navigation APN (N=15) + half-N gravity compensation (default) |
| missile | `tpn` | true proportional navigation TPN (no (N/2)g augmentation) |
| missile | `ppn` | pure pursuit guidance (no lead) |
| quadplane | `se3_pusher` | quadrotor SE(3) throughout + forward-pusher allocation (default) |
| quadplane | `l1_wing` | wing-mode L1 lateral + quadrotor low-speed assist |
| quadplane | `hybrid_cascade` | mode-switching hybrid autopilot (low-speed rotor / high-speed wing) with wing-lift relief |
| satellite | `quat_pd_damped_cmg` | quaternion PD + Schaub–Junkins damped pseudo-inverse CMG (default) |
| satellite | `quat_pd_mp_pinv_cmg` | quaternion PD + ordinary Moore–Penrose pseudo-inverse CMG |
| satellite | `quat_pd_wpinv_cmg` | quaternion PD + Oh–Fleming weighted damped least-norm + Vadali null-space singularity avoidance |
| transfer | `gauss_tangential` | Gauss planetary-equation tangential-thrust one-step greedy (default) |
| transfer | `edelbaum` | Edelbaum analytic Δv-budget open-loop tangential thrust |
| transfer | `q_law_petropoulos` (CLI `q_law`) | Petropoulos Q-law Lyapunov guidance |
| docking | `cwh_lqr` | CWH full-state LQR + corridor / capture net / fuel guards (default) |
| docking | `cwh_pole_placement` (CLI `pole_placement`) | CWH pole placement (no extra constraint layer) |
| docking | `zem_zev` | CWH ZEM/ZEV single-impulse receding-horizon guidance + fuel/range guards |
| rendezvous | `cwh_two_impulse` | CWH receding-horizon two-impulse terminal guidance (default) |
| rendezvous | `cwh_compensated_pd` (CLI `cwh_pd`) | CWH compensated PD straight to origin |
| rendezvous | `glideslope` | range–rate braking corridor (glideslope) V-bar approach |

> Dataset B filenames and the `controller` field come from the controller class
> property `name`, while `baseline_eval.py:82-85`'s `--baseline` accepts only the
> `BASELINE_VARIANTS` dictionary keys. The two are not always identical:
> `q_law_petropoulos` ↔ `q_law`, `cwh_compensated_pd` ↔ `cwh_pd`,
> `cwh_pole_placement` ↔ `pole_placement`. The satellite's three arms share the
> same label and variant key; the default controllers of the remaining tasks
> (`se3_geometric`, `cv_kf_pid`, `l1_tecs`, `zem_zev_pdg`, `apn_rate`,
> `se3_pusher`, `gauss_tangential`, `cwh_lqr`, `cwh_two_impulse`) come from
> `BUILDERS` and have no `--baseline` key, so omit that argument when reproducing
> them.

## 6. Best classical method per task (dataset B, seed 7)

| Task | Best | Runner-up | Weakest | Spread |
|---|---|---|---|---|
| quadx | `se3_geometric` / `smc` (100.0%) | `cascaded_pid` (99.2%) | `cascaded_pid` | 0.8 pp |
| quadx_track | `cv_kf_pid` (85.2%) | `pid_ema` / `pursuit` (75.8%) | `pid_ema` / `pursuit` | 9.4 pp |
| fixedwing | `l1_tecs` (98.6%) | `nlgl_tecs` (94.5%) | `pid_cascade` (91.6%) | 7.0 pp |
| rocket | `zem_zev_pdg` / `pd_landing` (100.0%) | `eguidance` (34.4%) | `eguidance` | 65.6 pp |
| missile | `apn_rate` / `tpn` (76.2%) | `ppn` (2.0%) | `ppn` | 74.2 pp |
| quadplane | `se3_pusher` (87.5%) | `l1_wing` (49.6%) | `hybrid_cascade` (26.6%) | 60.9 pp |
| satellite | all three CMG arms 100.0% | — | — | 0.0 pp |
| transfer | all three arms 100.0% | — | — | 0.0 pp |
| docking | all three arms 100.0% | — | — | 0.0 pp |
| rendezvous | all three arms 100.0% | — | — | 0.0 pp |

## 7. Scope boundaries and known gaps

1. **There are no 1024-replica baseline artifacts.** Each
   `results/<task>/baseline.json` has different metadata: `quadx` is 32
   replicas / seed 7, most other tasks are 128 replicas / seed 13, and
   `quadx_track` is seed 21. Any citation must include `num_envs`, `seed` and
   `env_steps`; never normalize them to "1024 envs, seed 7".
2. **Dataset B has seed 7 only.** It cannot provide cross-seed variance; the
   3-seed table in §2 is the variance source for default controllers.
3. **`results/quadx/baseline_cascaded_pid.json` is a legacy artifact**: 512
   replicas, seed 7, 0.119 success rate, wildly different from the same
   controller's `results/quadx/baselines/cascaded_pid_seed7.json` (256 replicas,
   99.2%). The former predates the cascaded-PID fix and must not be cited.
4. **Variant JSON windows are not uniform**: `fixedwing` is 1500 steps (task
   horizon 3600), `missile` is 580–900 steps, `rocket` is 1200 steps. Cross-variant
   success comparisons must state the window.
5. **Specialized harness reports that were never delivered.** The following
   paths do not exist in the current repository; method documents that cite them
   must mark them "not shipped", and they cannot be treated as reproducible
   results:
   - `results/rocket/pdg_test_report.json`, `results/rocket_pd_tune/tune_report.json`
   - `results/fixedwing/autopilot_report.json`, `results/fixedwing_pid/comparison.json`, `results/fixedwing_pid/scenarios_pid.json`, `results/fixedwing_ppo/checkpoint.pt`
   - `results/quadplane/cascade_report.json`
   - `results/quadx/cascade_pid_tests.json`, `results/quadx/config.json`, `results/quadx/run.json`
   - `results/quadx_track/kf_tuning.json`, `results/quadx_track/ppo/config.json`, `results/quadx_track/ppo/eval.json`
   - `results/satellite/steering_compare.json`, `results/satellite/steering_compare.md`, `results/satellite_ppo/checkpoint.pt`
   - `results/transfer/comparison.json`, `results/transfer/baseline_q_law_petropoulos.json`
   - `results/missile/scenarios.json`

   The reproducible replacements for these paths are the dataset A/B JSON files
   (§2 and §3) and the artifacts marked as shipped under `docs/method/`.
6. **The old 3072-episode / 3×1024 claim is retired.** Restoring that sample size
   requires re-running `scripts/baseline_eval.py --num-envs 1024` and writing one
   artifact per seed.

## 8. Reproduction commands

```bash
# Dataset A: default controller, one directory per seed; baseline_eval writes baseline.json under --out
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --out results/baseline_protocol/seed101/<task>

# Dataset B: default controller + all variants, written to results/<task>/baselines/<controller>_seed7.json
python scripts/baseline_suite.py run --tasks <task> --num-envs 256 --seeds 7
```

Note that `baseline_eval.py --baseline <variant>` writes its report to
`results/<task>/baseline_<name>.json` (where `<name>` is the controller class
property), not dataset B's `baselines/<name>_seed7.json`; use
`baseline_suite.py` to reproduce dataset B's directory structure. Available
`<variant>` CLI keys are in §5, or run:

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```
