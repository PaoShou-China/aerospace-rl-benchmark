# Classical Baseline Results

**English** · [简体中文](baselines.zh-CN.md)

Every task ships a non-learned GNC baseline
([algorithms/baselines/](../algorithms/baselines/baselines.py)) on the same
environment, reward and success bar as the policy. Controllers read only the
`'policy'` observation group, take env constants at construction, and never
touch internal state buffers.

All numbers below were re-measured on 2026-11-12 (128 replicas per cell,
seeds 7/13/21 + held-out 101, calm air, nominal plant). The full per-controller
table with Wilson intervals and failure modes is
[baseline_test_report.md](baseline_test_report.md); the gains and bug fixes
applied in that pass are listed in §A.7.

| Task | Baseline | Reference success |
|---|---|---|
| QuadX-Waypoints | SE(3) geometric control + integral trim | 100% (128 replicas, seed 13, shipped `baseline.json`); 99.7±0.5% over the protocol seeds 7/13/21, held-out 101 = 100% |
| QuadX-Waypoints | position PID → attitude cascade | 100% (128 replicas, seeds 7/13/21/101; 0 crash / 0 out-of-bounds after the damping retune) |
| QuadX-Track | CV Kalman filter + fixed-offset SE(3) | 75.8% (128 replicas, seed 13); 79.2±3.5% over the protocol seeds. The task now has SOLID cylinder occluders, so the gap to the pre-change 85.9% is the new collision/visibility constraint |
| Fixedwing-Waypoints | L1 lateral + TECS longitudinal autopilot | 87.5% (128 replicas, seed 13, 3616-step window); 87.0±3.2% over the protocol seeds |
| Quadplane-VTOL | SE(3) + forward pusher | 98.4% (128 replicas, seed 13); 98.2±0.5% over the protocol seeds |
| Rocket-Landing | PDG: ZEM/ZEV braking + virtual-pad manifold + pulse servo | 100% under the v2 criterion (40 Hz, settled contact) |
| Satellite (CMG) | quaternion PD + Schaub–Junkins damped pseudo-inverse | 100% (128 replicas, all four seeds) |
| Transfer | Gauss one-step greedy tangential guidance | 100% (128 replicas, all four seeds) |
| Docking | CWH discrete LQR + shield/fuel derating | 100% (128 replicas, all four seeds) |
| Rendezvous | two-impulse CWH ZEM/PN, recursive re-solve | 100% (128 replicas, all four seeds) |
| Missile-Intercept | augmented PN (N = 15) → body-rate commands | 71.1% (128 replicas, seed 13); 77.6±8.1% over the protocol seeds |

Numbers are the shipped `results/<task>/baseline.json` artifacts (seed 13,
128 replicas, nominal plant, calm air). Task-internal variants live in
`results/<task>/baselines/`.

Two readings worth keeping:

- These are references, not strawmen. Fixed-wing `pid_cascade` keeps a
  documented out-of-bounds/timeout residue, the missile guidance law is
  authority-saturated in the high-crossing geometry, and `eguidance` /
  `ppn` / the quadplane mode-switching arms are deliberate lower-bound
  ablations of their method families.
- The rocket v1 figures in older notes (82.8% / 84.1%) are **instantaneous**
  success rates and are not comparable to the v2 settled criterion.
  Background and reproduction commands:
  [env-rocket-landing.md](method/env-rocket-landing.md).

Success rates are reported per task and per controller. They do not define a
cross-task ranking: task difficulty, observation budgets and success criteria
differ.

---

## Appendix: multi-seed protocol and full variant data

### A.1 Data provenance

Two artifact families, one convention:

| Dataset | Path | Parallel replicas | Seeds | Accounting |
|---|---|---|---|---|
| A: protocol multi-seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101 (held-out) | first episode per replica only (`FirstEpisodes`); 128 first episodes per seed |
| B: controller variants | `results/<task>/baselines/<controller>_seed<seed>.json` | 128 | 7 / 13 / 21 / 101 | first episode per replica |

- Success rate = `successes / episodes_finished` (each replica's first-episode
  `completed_successes / completed_episodes`).
- Both datasets evaluate the **nominal plant**: calm air, default difficulty,
  `domain_randomize = False` (the frozen benchmark contract). Matched-condition
  runs (policy vs baseline under domain randomization) opt in with
  `scripts/baseline_eval.py --domain-randomize on`.
- Step windows come from each JSON's `env_steps`.
- The success denominator is 128 first episodes per seed.

### A.2 Default controllers: protocol seeds {7, 13, 21} + held-out seed 101

Mean and standard deviation use protocol seeds 7/13/21 only; the Wilson 95%
interval pools the three seeds' episodes (384 episodes).

| Task | Default controller | seed 7 | seed 13 | seed 21 | seed 101 | mean±sd (7/13/21) | [min,max] | Wilson 95% CI (pooled 384) | Mean return | Episodes | Step window |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| quadx | `se3_geometric` | 100.0 | 100.0 | 99.2 | 100.0 | 99.7±0.5 | [99.2, 100.0] | [98.5, 100.0] | 72.5 | 384 | 672 |
| quadx_track | `cv_kf_pid` | 82.8 | 75.8 | 78.9 | 75.0 | 79.2±3.5 | [75.8, 82.8] | [74.8, 82.9] | 84.7 | 384 | 1216 |
| fixedwing | `l1_tecs` | 89.8 | 87.5 | 83.6 | 90.6 | 87.0±3.2 | [83.6, 89.8] | [83.2, 90.0] | 208.8 | 384 | 3616 |
| rocket | `zem_zev_pdg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 900.2 | 384 | 1216 |
| missile | `apn_rate` | 75.0 | 71.1 | 86.7 | 71.9 | 77.6±8.1 | [71.1, 86.7] | [73.2, 81.5] | 103.5 | 384 | 672 |
| quadplane | `se3_pusher` | 97.7 | 98.4 | 98.4 | 96.1 | 98.2±0.5 | [97.7, 98.4] | [96.3, 99.1] | 274.4 | 384 | 1216 |
| satellite | `quat_pd_damped_cmg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 65.2 | 384 | 352 |
| transfer | `gauss_tangential` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 0.8 | 384 | 192 |
| docking | `cwh_lqr` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 190.6 | 384 | 288 |
| rendezvous | `cwh_two_impulse` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 184.8 | 384 | 640 |

Notes:
- `rocket` uses the v2 primary metric `ever_stable_2s` (settled contact).
- `quadx_track` success = visibility fraction ≥ 0.85 with no collision.
- Every row satisfies `complete = true`.

### A.3 All 30 controller variants (128 replicas, seeds 7/13/21 + 101)

Each row is the mean over protocol seeds 7/13/21; the held-out column is seed
101. The table ranks controllers within a task only.

| Task | Controller | Success mean±sd | [min, max] | Held-out 101 | Mean return | Step window |
|---|---|---|---:|---:|---:|---:|
| quadx | `cascaded_pid` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 72.59 | 544 |
| quadx | `se3_geometric` | 99.7 ± 0.5 | [99.2, 100.0] | 100.0 | 72.49 | 672 |
| quadx | `smc` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 72.59 | 576 |
| quadx_track | `cv_kf_pid` | 79.2 ± 3.5 | [75.8, 82.8] | 75.0 | 84.72 | 1216 |
| quadx_track | `pid_ema` | 74.7 ± 4.0 | [70.3, 78.1] | 78.1 | 78.06 | 1216 |
| quadx_track | `pursuit` | 74.2 ± 4.3 | [69.5, 78.1] | 78.1 | 72.80 | 1216 |
| fixedwing | `l1_tecs` | 87.0 ± 3.2 | [83.6, 89.8] | 90.6 | 208.84 | 3616 |
| fixedwing | `nlgl_tecs` | 94.3 ± 1.2 | [93.0, 95.3] | 90.6 | 217.62 | 3616 |
| fixedwing | `pid_cascade` | 76.8 ± 4.6 | [73.4, 82.0] | 76.6 | 193.25 | 3616 |
| rocket | `eguidance` | 34.4 ± 5.1 | [28.9, 39.1] | 33.6 | −3304.80 | 1216 |
| rocket | `pd_landing` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 7.27 | 1216 |
| rocket | `zem_zev_pdg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 900.15 | 1216 |
| missile | `apn_rate` | 77.6 ± 8.1 | [71.1, 86.7] | 71.9 | 103.52 | 672 |
| missile | `ppn` | 2.3 ± 0.8 | [1.6, 3.1] | 2.3 | 27.58 | 928 |
| missile | `tpn` | 77.6 ± 7.4 | [71.9, 85.9] | 71.9 | 103.70 | 576 |
| quadplane | `hybrid_cascade` | 26.8 ± 2.4 | [24.2, 28.9] | 22.7 | 167.48 | 1216 |
| quadplane | `l1_wing` | 49.2 ± 5.6 | [43.0, 53.9] | 39.1 | 217.27 | 1216 |
| quadplane | `se3_pusher` | 98.2 ± 0.5 | [97.7, 98.4] | 96.1 | 274.35 | 1216 |
| satellite | `quat_pd_damped_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 65.22 | 352 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 64.92 | 384 |
| satellite | `quat_pd_wpinv_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 65.41 | 352 |
| transfer | `edelbaum` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 0.20 | 192 |
| transfer | `gauss_tangential` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 0.77 | 192 |
| transfer | `q_law_petropoulos` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 1.01 | 192 |
| docking | `cwh_lqr` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 190.59 | 288 |
| docking | `cwh_pole_placement` | 89.6 ± 3.0 | [87.5, 93.0] | 86.7 | 169.30 | 224 |
| docking | `zem_zev` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 191.21 | 320 |
| rendezvous | `cwh_compensated_pd` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 179.29 | 928 |
| rendezvous | `cwh_two_impulse` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 184.79 | 640 |
| rendezvous | `glideslope` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 180.82 | 864 |

### A.4 Failure modes (worst protocol seed per controller)

| Task | Controller | crash | out-of-bounds | timeout |
|---|---|---:|---:|---:|
| quadx | `cascaded_pid` | 0.0% | 0.0% | 0.0% |
| quadx | `se3_geometric` | 0.0% | 0.7% | 0.0% |
| quadx | `smc` | 0.0% | 0.0% | 0.0% |
| quadx_track | `cv_kf_pid` | — | 0.0% | 8.5% |
| quadx_track | `pid_ema` | — | 0.0% | 23.4% |
| quadx_track | `pursuit` | — | 0.0% | 24.2% |
| fixedwing | `l1_tecs` | 0.0% | 2.2% | 5.2% |
| fixedwing | `nlgl_tecs` | 0.0% | 7.0% | 1.0% |
| fixedwing | `pid_cascade` | 0.0% | 12.9% | 6.8% |
| quadplane | `hybrid_cascade` | 0.0% | 4.7% | 71.1% |
| quadplane | `l1_wing` | 0.0% | 0.0% | 57.0% |
| quadplane | `se3_pusher` | 2.0% | 0.0% | 0.7% |
| missile | `apn_rate` | — | — | 0.0% |
| missile | `ppn` | — | — | 3.9% |
| missile | `tpn` | — | — | 0.0% |

- `quadx_track` failures are all timeouts (visibility below the 0.85 bar while
  avoiding the now-solid cylinders).
- `quadplane` weak arms are timeout-dominated (the mode-switching autopilot
  cannot finish 4 waypoints in 30 s); `se3_pusher` keeps a small crash residue
  on the highest-speed approaches.
- `rocket`, `satellite`, `transfer`, `docking` and `rendezvous` JSON files do
  not provide per-failure counts.

### A.5 Scope boundaries

1. There are no 1024-replica baseline artifacts. `results/<task>/baseline.json`
   is 128 replicas / seed 13 for every task; the protocol tree uses 128
   replicas at seeds 7/13/21/101. Always cite `num_envs`, `seed`, `env_steps`.
2. Dataset B now covers all four seeds, so cross-seed variance is available for
   every controller, not only the defaults.
3. Step windows differ per task (`fixedwing` 3616, `missile` 576–928,
   `rocket` 1216, space tasks 192–928). Cross-controller comparisons must state
   the window.

### A.6 Reproduction commands

```bash
# Dataset A: default controller, one directory per seed (nominal plant)
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --max-steps 5000 --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --max-steps 5000 --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --max-steps 5000 --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --max-steps 5000 --out results/baseline_protocol/seed101/<task>

# Dataset B: default + all variants at the four seeds
python scripts/baseline_suite.py run --tasks <task> --num-envs 128 --seeds 7,13,21,101 --max-steps 5000

# Aggregate report (tables above)
python scripts/baseline_suite.py report --out docs/baseline_test_report.md

# Gain sweeps (tuning only; writes results/tuning/<task>/<controller>/)
python scripts/baseline_tune.py <task> --grid '{"KX": [3, 4, 5], "KV": [2, 3]}' --seeds 7
python scripts/baseline_tune.py docking --grid '{"w_vel": [3, 8, 15]}' --seeds 7,13
python scripts/baseline_tune.py fixedwing --config-json --grid '{"lateral.l1_distance": [40, 50]}'
python scripts/baseline_tune.py transfer --design --grid '{"rp0": [8500000, 8800000]}'
```

Available `<variant>` CLI keys:

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```

Matched-condition evaluation (domain randomization ON, e.g. for policy-vs-baseline
tables) adds `--domain-randomize on` to `baseline_eval.py`.

### A.7 2026-11-12 retune and regression fixes

| Item | Symptom | Cause | Change |
|---|---|---|---|
| rocket: all three arms | 0% (was 100%) | the 2026-11-05 contact-solver disable applied to every vehicle, so the booster sank through the pad instead of settling | the rigid contact solver stays ON for the rocket in legacy mode (`plant.py`) |
| docking: `cwh_lqr` | 0.4% | the new solid station hull makes a centre-to-origin approach push the nose through the port plane; `w_vel = 3` also braked too late for the 0.02 m/s² authority | hold the nose on the port plane (`docking_standoff`) and retune `w_vel` 3 → 15 → 100% on all seeds |
| quadx: `cascaded_pid` | 64.1% | the 2026-11-02 CPU-only sweep reduced `KD_POS` to 2.5, which is underdamped on the GPU plant (42.6% out-of-bounds) | restore `KD_POS = 4.0` → 100% on all seeds |
| quadplane: `se3_pusher` | 88.1% | `A_MAX = 6.5` could not finish the high/long waypoint legs inside 30 s | `A_MAX` 6.5 → 12, `V_MAX` 17 → 20 → 98.2% |
| transfer: `gauss_tangential` | 74.0% | the greedy burn under-delivers the perigee raise; the terminal objective was marginally negative | aim at a 300 km higher perigee (`rp0` 8.5e6 → 8.8e6) → 100% |
| `baseline_eval.py` | protocol ambiguity | the CLI inherited `domain_randomize = None` (ON) while the documented reference protocol is the nominal plant | new `--domain-randomize {off,on,default}`, default `off`; `run_5seed.py` opts into `on` for matched-condition comparisons |

No change was found to help `quadx_track` (static standoff offsets all
measured below the shipped 1.5 m offset), `fixedwing pid_cascade`,
`missile` APN/TPN (the commands are already saturated for every N ≥ 10),
`rocket eguidance`, or the quadplane mode-switching ablations; those arms are
left at their measured operating points.
