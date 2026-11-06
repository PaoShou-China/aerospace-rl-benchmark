# Classical Baseline Results

**English** · [简体中文](baselines.zh-CN.md)

Every task ships a non-learned GNC baseline
([algorithms/baselines/](../algorithms/baselines/baselines.py)) on the same
environment, reward and success bar as the policy. Controllers read only the
`'policy'` observation group, take env constants at construction, and never
touch internal state buffers.

| Task | Baseline | Reference success |
|---|---|---|
| QuadX-Waypoints | SE(3) geometric control + integral trim | 90.6% (32 replicas, seed 7); 99.7% mean over the multi-seed protocol (128 replicas × seeds 7/13/21; held-out seed 101 = 100%) |
| QuadX-Waypoints | position PID → attitude cascade | 99.2% (256 replicas, seed 7, 0 crash / OOB) |
| QuadX-Track | CV Kalman filter + fixed-offset SE(3) | 82.8% (128 replicas, seed 21); 90.2% visible time, 0.258 m standoff |
| Fixedwing-Waypoints | L1 lateral + TECS longitudinal autopilot | 97.0% (128 replicas, seed 13; the window ends before the 120 s limit) |
| Quadplane-VTOL | SE(3) + forward pusher | 89.1% (128 replicas, seed 13) |
| Rocket-Landing | PDG: ZEM/PN braking + virtual-pad manifold + pulse servo | 100% under the v2 criterion (40 Hz, continuously settled for 2 s) |
| Satellite (CMG) | quaternion PD + Schaub–Junkins damped pseudo-inverse | 100% (128 replicas, seed 13) |
| Transfer | Gauss one-step greedy tangential guidance | 100% (128 replicas, seed 13) |
| Docking | CWH discrete LQR + shield/fuel derating | 100% (128 replicas, seed 13) |
| Rendezvous | two-impulse CWH ZEM/PN, recursive re-solve | 100% (128 replicas, seed 13) |
| Missile-Intercept | augmented PN (N = 15) → body-rate commands | 71.1% (128 replicas, seed 13) |

Numbers are the shipped `results/<task>/baseline.json` artifacts; each file
records its own replica count, seed and step window. Within-task ablations
(256 replicas, seed 7) sit in `results/<task>/baselines/`.

Two readings worth keeping:

- These are references, not strawmen. The fixed-wing and rocket baselines keep
  documented residual failures (endgame limit cycles, authority-saturated
  braking) that a learned policy is expected to attack.
- The rocket v1 figures in older notes (82.8% / 84.1%) are **instantaneous**
  success rates and are not comparable to the v2 settled criterion. Background
  and reproduction commands:
  [env-rocket-landing.md](method/env-rocket-landing.md).

Success rates are reported per task and per controller. They do not define a
cross-task ranking: task difficulty, observation budgets and success criteria
differ, and the numbers come from different replica counts and seeds.

---

## Appendix: multi-seed protocol and full variant data

### A.1 Data provenance

Two artifact families, two conventions:

| Dataset | Path | Parallel replicas | Seeds | Accounting |
|---|---|---|---|---|
| A: protocol multi-seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101 (held-out) | first episode per replica only (`FirstEpisodes`); 128 first episodes per seed |
| B: controller variants | `results/<task>/baselines/<controller>_seed7.json` | 256 | 7 | first episode per replica, single seed |

- Success rate = `successes / episodes_finished` (each replica's first-episode
  `completed_successes / completed_episodes`).
- Dataset A/B JSON files have `env_kwargs = {}` (calm air, default difficulty).
- Step windows come from each JSON's `env_steps`.
- The success denominator is 128 (A) or 256 (B) first episodes.

### A.2 Default controllers: protocol seeds {7, 13, 21} + held-out seed 101

Mean and standard deviation use protocol seeds 7/13/21 only; the Wilson 95%
interval pools the three seeds' episodes (384 episodes).

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
- `rocket` uses the v2 primary metric `ever_stable_2s` (2 s continuous settle).
- `quadx_track` success = visibility fraction ≥ 0.85 with no collision.
- Every row satisfies `complete = true`.

### A.3 All 30 controller variants (seed 7, 256 replicas)

Dataset B is single-seed and provides no cross-seed interval; it ranks
controllers within a task only.

| Task | Controller | Success | Mean return | Step window |
|---|---|---|---:|---:|
| quadx | `cascaded_pid` | 99.2% | 72.74 | 800 |
| quadx | `se3_geometric` | 100.0% | 72.86 | 648 |
| quadx | `smc` | 100.0% | 72.86 | 557 |
| quadx_track | `cv_kf_pid` | 85.2% | 85.68 | 1200 |
| quadx_track | `pid_ema` | 75.8% | 75.98 | 1200 |
| quadx_track | `pursuit` | 75.8% | 71.85 | 1200 |
| fixedwing | `l1_tecs` | 98.6% | 230.09 | 1500 |
| fixedwing | `nlgl_tecs` | 94.5% | 222.94 | 1500 |
| fixedwing | `pid_cascade` | 91.6% | 219.07 | 1500 |
| rocket | `eguidance` | 34.4% | −3332.01 | 1200 |
| rocket | `pd_landing` | 100.0% | 22.34 | 1200 |
| rocket | `zem_zev_pdg` | 100.0% | 907.31 | 1200 |
| missile | `apn_rate` | 76.2% | 101.90 | 637 |
| missile | `ppn` | 2.0% | 27.21 | 900 |
| missile | `tpn` | 76.2% | 102.13 | 580 |
| quadplane | `hybrid_cascade` | 26.6% | 173.94 | 1200 |
| quadplane | `l1_wing` | 49.6% | 226.20 | 1200 |
| quadplane | `se3_pusher` | 87.5% | 271.97 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0% | 64.85 | 330 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0% | 64.54 | 366 |
| satellite | `quat_pd_wpinv_cmg` | 100.0% | 65.03 | 337 |
| transfer | `edelbaum` | 100.0% | 0.22 | 166 |
| transfer | `gauss_tangential` | 100.0% | 0.01 | 166 |
| transfer | `q_law_petropoulos` | 100.0% | 1.15 | 166 |
| docking | `cwh_lqr` | 100.0% | 188.62 | 274 |
| docking | `cwh_pole_placement` | 100.0% | 189.30 | 236 |
| docking | `zem_zev` | 100.0% | 190.06 | 313 |
| rendezvous | `cwh_compensated_pd` | 100.0% | 179.44 | 875 |
| rendezvous | `cwh_two_impulse` | 100.0% | 184.91 | 594 |
| rendezvous | `glideslope` | 100.0% | 180.96 | 805 |

### A.4 Failure modes (dataset B, seed 7)

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

- `quadx_track` failures are all timeouts.
- `quadplane` weak arms mostly time out (mode-switching autopilot cannot finish
  4 waypoints in 30 s).
- `rocket` / `satellite` / `transfer` / `docking` / `rendezvous` JSON files do
  not provide per-failure counts.

### A.5 Scope boundaries

1. There are no 1024-replica baseline artifacts. `results/<task>/baseline.json`
   metadata varies (`quadx` 32 replicas / seed 7, most tasks 128 / seed 13,
   `quadx_track` seed 21). Always cite `num_envs`, `seed`, `env_steps`.
2. Dataset B has seed 7 only; §A.2 is the cross-seed variance source.
3. Dataset B windows are not uniform (`fixedwing` 1500 steps, `missile`
   580–900, `rocket` 1200). Cross-variant comparisons must state the window.

### A.6 Reproduction commands

```bash
# Dataset A: default controller, one directory per seed
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --out results/baseline_protocol/seed101/<task>

# Dataset B: default + all variants to results/<task>/baselines/<controller>_seed7.json
python scripts/baseline_suite.py run --tasks <task> --num-envs 256 --seeds 7
```

Available `<variant>` CLI keys:

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```
