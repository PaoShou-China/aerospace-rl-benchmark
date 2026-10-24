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
- **Engagement**: the missile spawns 1.5-3.5 km away with fire-control lead
  (straight-line shot solution at `t_go = R / (V_m0 + V_t)`); the target
  heading points back at the missile inside a difficulty-widened cone.
- **Success**: miss distance < `KILL_RADIUS` = 15 m (proximity fuse) at ANY
  physics substep (fixed, never difficulty-scaled). Failure: the
  closest-approach point passes, `|p|` > 15 km, `z` < 0; truncation at 900
  policy steps (15 s).

## Action / observation

- **Action (3,)**: body-axis rate commands in [-1, 1], affinely mapped to
  roll +-6 rad/s (`OMEGA_MAX`) and pitch/yaw +-0.5 rad/s (`OMEGA_LAT`, the
  coordinated-turn limit a/V) through the plant's rate PID and the
  skid-to-turn fin mixer. No throttle dim.
- **Observation (34,) float32**, `'policy'` TensorDict group: body omega (3),
  quat xyzw (4), body v (3), position (3), last action (3), fin actuations
  (4), ignited (1), burning (1), burn-time fraction (1), target relative
  position/velocity in body frame (3+3), LOS rate in body frame (3), closing
  speed Vc (1), target speed (1).

## Reward

Per inner step, active replicas only:
`+0.01 (d_prev - d)` closing progress, `-0.5 |LOS rate|`, `-0.01 |action|^2`,
`+50` kill (once), `-10` unrecoverable miss (once); components reported as
`Episode/rew_{progress,los,ctrl,kill,miss}`.

## Difficulty

Static constructor kwarg, `None` == benchmark == 1.0. Scales initial range
1500 -> 3500 m, target-heading cone half-width +-15 -> +-60 deg and target
lateral acceleration 3 -> 12 g via endpoint-exact interpolation; the RNG draw
count never depends on d, so `seed(None)` and `seed(1.0)` are bit-identical
(tested, T4).

## Classical baseline

`MissileAPNBaseline` ([algorithms/baselines/baselines.py](../../algorithms/baselines/baselines.py),
`name='apn_rate'`): augmented PN (N=4) with half-N gravity compensation,
mapped to the body-rate interface through the skid-to-turn kinematic inverse
`omega = cross(b_x, a_lat) / V`, with a first-order LOS-rate filter.

Measured numbers (GPU, seed 7; success = proximity-fuse kill):

| Protocol | Result |
|---|---|
| `test_missile_apn.py` T3, d=0, 256 envs | SR 0.551 mixed / 1.000 straight, median miss (kills) 4.7 m |
| `test_missile_apn.py` T5, 128 envs/cell | d=0: SR 0.562 (straight 1.000, r0 1504 m) · d=0.5: 0.454 (0.775, r0 2503 m) · d=1.0: 0.062 (0.135, r0 3502 m) |
| official `scripts/baseline_eval.py missile`, benchmark d=1.0, 1024 envs | SR 0.0576 (59/1025 episodes), mean episode return -75.0 |

The APN law assumes a non-maneuvering straight-line shot solution, so its
residual failures concentrate on the weave/barrel target types at high
difficulty — the gap RL is expected to attack.

Scenario harness (256 envs / ~256 episodes per cell, `scenarios.json`):
overall SR 0.551 / 0.492 / 0.066 and median miss 7.4 / 8.1 / 84.7 m at
d = 0 / 0.5 / 1.0; per type at d=0: straight 1.000, weave 0.342, barrel
0.394; at d=1.0: straight 0.084, weave 0.115, barrel 0.000 (median miss
1036 m — the tight high-severity barrel circle defeats the straight-line
shot solution entirely).

## Scenario harness

[`eval_missile_scenarios.py`](eval_missile_scenarios.py) sweeps difficulty
d in {0, 0.5, 1.0} x maneuver type, 256 envs and ~256 episodes per cell,
measuring true min distance through a spy on `env.reset_idx` (the
`completed_metrics['miss_distance']` slots are stale at termination because
the env auto-resets internally). It writes `results/missile/scenarios.json`.
Scope note: the env's spawn geometry (range, heading cone, maneuver severity)
is generated internally from the single difficulty scalar and exposes no
injection points for aspect angle or range, and 5-8 km initial ranges are
outside the 1.5-3.5 km spawn support — so the planned aspect-angle x range
grid is replaced by this difficulty sweep (documented in the script
docstring).

## Usage

```bash
python environments/missile/train_missile.py        # PPO + BC warm start (constants in-file)
python environments/missile/test_missile_apn.py gpu # T1-T5 test suite (default gpu, cpu also supported)
python environments/missile/eval_missile_scenarios.py   # difficulty x maneuver-type grid
python scripts/baseline_eval.py missile             # official baseline protocol
python scripts/train.py missile                     # legacy small-config aggregator
```

Results land in `results/missile/` (`run.json` / `config.json` from training,
`baseline.json` from the official baseline protocol, `scenarios.json` from
the scenario harness).

## Honesty note

Only the numbers in the table above (plus `scenarios.json`) are measured; no
full RL training result exists in this repository yet. Nothing here should be
read as a benchmark claim beyond those measurements.
