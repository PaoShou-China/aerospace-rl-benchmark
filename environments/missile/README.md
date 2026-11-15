# Missile-Intercept

6-DOF tail-controlled missile vs a kinematic 3-DOF maneuvering target
(project-authored task; `task_defaults: project`, informed by BVR Gym
arXiv:2403.17533, LAG arXiv:2506.14164 and the PPO-guidance literature's
overload/LOS-rate convention). The full normative spec lives in the package
docstring ([`__init__.py`](__init__.py)) and [`env.py`](env.py).

## Task

- **Plant**: 150 kg missile (body +x nose, four cross tail fins) on the shared
  Genesis `TensorPlant` at 240 Hz physics / 120 Hz rate-PID / 60 Hz policy,
  with a LiftingSurfaceBatch fin model, body drag and an autonomous
  boost-coast engine (20 kN for the first 4 s, then unpowered coast).
- **Target**: kinematic 3-DOF aircraft (250 m/s around 4 km altitude) advanced
  analytically every physics substep; maneuver type drawn per episode:
  0 straight line / 1 sinusoidal weave / 2 horizontal barrel roll.
- **Engagement**: the missile spawns 2500 m from the target with fire-control
  lead (straight-line shot solution at `t_go = 0.8 R / (V_M0 + V_T)`); the
  target heading points back at the missile inside a fixed +-37.5 deg cone.
- **Success**: miss distance < `KILL_RADIUS` = 15 m (proximity fuse) at ANY
  physics substep (fixed, never scaled). Failure: the
  closest-approach point passes, `|p|` > 15 km, `z` < 0; truncation at 900
  policy steps (15 s).

## Action / observation

- **Action (3,)**: body-axis rate commands in [-1, 1], affinely mapped to
  roll +-6 rad/s (`OMEGA_MAX`) and pitch/yaw +-1.0 rad/s (`OMEGA_LAT`, transient
  terminal-guidance authority above the steady coordinated-turn rate a/V)
  through the plant's rate PID and the skid-to-turn fin mixer. No throttle dim.
- **Observation (34,) float32**, `'policy'` TensorDict group: body omega (3),
  quat xyzw (4), body v (3), position (3), last action (3), fin actuations
  (4), ignited (1), burning (1), burn-time fraction (1), target relative
  position/velocity in body frame (3+3), LOS rate in body frame (3), closing
  speed Vc (1), target speed (1).

## Reward

Per inner step, active replicas only:
`+0.01 (d_prev - d)` closing progress, `+0.01 (v_hat . r_hat)` velocity/LOS
alignment, `+100` kill (once), and `-10 min(d_min / 500 m, 1)` on unrecoverable
miss (once); components reported as `Episode/rew_{progress,align,kill,miss}`.
The alignment term is the dense,
PN-consistent guidance signal: potential-like and locally computable, and
unlike a penalty on the LOS rate it cannot be gamed by flying away from the
target.

## Operating point

The engagement is a single fixed medium operating point. The static
`difficulty` constructor kwarg was removed on 2026-10-24, so there is no
difficulty sweep and no scaling of any geometry or of the kill gate:

- initial range 2500 m (`R_SPAWN`)
- target-heading cone half-width +-37.5 deg (`CONE`)
- target lateral acceleration 7.5 g (`A_LAT`)
- weave frequency 0.75 Hz (`F_WEAVE`)
- fire-control lead `t_go = TGO_SCALE * R / (V_M0 + V_T)` with
  `TGO_SCALE = 0.8` (4.0 s)
- kill radius 15 m, fixed (`KILL_RADIUS`)

Reset performs no warmup: the boost engine fires autonomously on the first
active physics step, so a zero-command warmup would burn thrust before the
first policy step, and an airborne missile has no contacts to settle.

## Classical baseline

`MissileAPNBaseline` ([algorithms/baselines/baselines.py](../../algorithms/baselines/baselines.py),
`name='apn_rate'`): augmented PN with navigation constant `N = 15` (retuned
from the N=4/N=10 lineage; half-N gravity compensation), mapped to the
body-rate interface through the skid-to-turn kinematic inverse
`omega = cross(b_x, a_lat) / V`, with a first-order LOS-rate filter.

Measured numbers:

| Protocol | Result |
|---|---|
| official `scripts/baseline_eval.py missile` (`results/missile/baseline.json`), 128 envs / seed 13 / 656 env steps / first episode per replica | SR 0.7109 (91/128), mean episode return 95.62, `complete = true` |
| `results/missile/baselines/apn_rate_seed7.json`, 256 envs / seed 7 / 637 env steps | SR 0.7617, mean episode return 101.90 |
| `test_missile_apn.py` T3, 256 envs, fixed operating point | pass gates only (mixed SR >= 0.45, straight SR >= 0.85); no artifact stored |

The APN law assumes a non-maneuvering straight-line shot solution, so its
residual failures concentrate on the weave/barrel target types — the gap RL
is expected to attack.

## Scenario harness

[`eval_missile_scenarios.py`](eval_missile_scenarios.py) measures one cell at
the fixed operating point (256 envs, ~256 completed episodes) and breaks it
down by the env's own maneuver type (`t_type` 0 straight / 1 weave / 2
barrel), measuring true min distance through a spy on `env.reset_idx` (the
`completed_metrics['miss_distance']` slots are stale at termination because
the env auto-resets internally). It writes `results/missile/scenarios.json`;
**that file is not shipped in the repository**, so no scenario breakdown is
quoted here.

Scope note: the spawn geometry (range, heading cone, maneuver severity) is
generated internally from fixed env constants and exposes no injection points
for aspect angle or range, and 5-8 km initial ranges are outside the 2500 m
spawn support — so the planned aspect-angle x range grid is replaced by this
per-maneuver-type breakdown (documented in the script docstring).

## Usage

```bash
python environments/missile/train_missile.py        # PPO + BC warm start (constants in-file)
python environments/missile/test_missile_apn.py     # T1-T3 test suite (default cpu; pass `gpu` to switch)
python environments/missile/eval_missile_scenarios.py   # fixed operating point, per-maneuver breakdown
python scripts/baseline_eval.py missile             # official baseline protocol
python scripts/train.py missile                     # legacy small-config aggregator
```

Results land in `results/missile/` (`run.json` / `config.json` from training,
`baseline.json` from the official baseline protocol, `scenarios.json` from
the scenario harness). The repository ships only `baseline.json` and
`results/missile/baselines/`; no training `run.json` / `eval.json` and no
`scenarios.json` are published, so no full-RL or scenario-breakdown number is
quoted.

## Honesty note

Only the official baseline and the dataset-B controller files listed above
are current measurements. No current full-RL result and no scenario
breakdown exist in this repository. Nothing here should be read as a
benchmark claim beyond those measurements.
