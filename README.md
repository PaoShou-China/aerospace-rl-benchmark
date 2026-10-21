# pyflyt-genesis: Fully Tensorized Aerospace RL Environments

A suite of seven GPU-native reinforcement-learning environments spanning
atmospheric flight vehicles and spacecraft dynamics, written in the official
Genesis RL environment format (`__init__` cfg/buffers, `step()` returning
`(get_observations(), rew_buf, reset_buf, extras)`, `get_observations()`
wrapping `obs_buf` in a TensorDict under the `'policy'` group, `reset_idx()`
taking a device-resident boolean mask, `reset()`, `close()`).

All state lives on exactly one device as `float64` physics / `float32`
observation buffers. There is **no engine round-trip per replica, no
host-device synchronization, no gym/gymnasium dependency**: every operation
(physics integration, contact checks, reward, reset, RNG) is a masked tensor
kernel, so inactive replicas are frozen and consume no random numbers.
Training runs PPO from [rsl-rl](https://github.com/leggedrobotics/rsl_rl)
(`train.py`); evaluation via `eval.py`.

## Environments

| Task | Class | Dynamics model | Obs dim | Act dim | Policy rate | Episode | Success criterion |
|---|---|---|---|---|---|---|---|
| QuadX-Hover | `QuadXHoverEnv` | Genesis rigid body (Crazyflie 2.x, 240 Hz) + rate-PID | 21 | 4 | 40 Hz | 400 steps = 10 s | none (graded hover reward) |
| Fixedwing-Waypoints | `FixedwingWaypointsEnv` | Genesis rigid body (2.35 kg UAV, 240 Hz) + blade-element aero | 38 | 4 | 30 Hz | 3600 steps = 120 s | 4/4 waypoints captured (< 2 m) |
| Rocket-Landing | `RocketLandingEnv` | Genesis rigid body (F9-class booster, 240 Hz) + grid fins, gimbal | 30 | 7 | 40 Hz | 1200 steps = 30 s | pad contact, \|v\| < 0.02 m/s, \|ω\| < 0.02 rad/s, tilt < 0.1 rad |
| Satellite attitude (CMG) | `SatelliteAttitudeEnv` | Rigid body + 4- or 6-CMG cluster, RK4 2.5 ms | 23 / 29 | 4 / 6 | 40 Hz | 1200 steps = 30 s | att. error < 0.05 rad, \|ω\| < 0.02 rad/s at limit |
| Low-thrust transfer | `LowThrustTransferEnv` | Two-body gravity + 3-axis 10 mN EP, RK4 10 s | 8 | 3 | 300 s/step | 166 steps ≈ 5 revs | none (terminal reward) |
| Docking | `SpaceDockingEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 500 steps ≈ 8.3 min | range < 1 m & speed < 0.1 m/s |
| Rendezvous | `SpaceRendezvousEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 1000 steps ≈ 16.7 min | range < 10 m & speed < 0.5 m/s, keep-out respected |
| Pendulum (toy sanity check) | `PendulumEnv` | Classic single-DOF pendulum, explicit Euler 50 ms | 3 | 1 | 20 Hz | 200 steps = 10 s | completed-episode return > −200 |

Per-task quantitative details (action/observation layouts, reward
decompositions, termination conditions, spawn distributions) live in the
module docstring of each package: [src/quadx](src/quadx/__init__.py),
[src/fixedwing](src/fixedwing/__init__.py),
[src/rocket](src/rocket/__init__.py),
[src/satellite](src/satellite/__init__.py),
[src/transfer](src/transfer/__init__.py),
[src/docking](src/docking/__init__.py),
[src/pendulum](src/pendulum/__init__.py).

## Package layout

```
src/
├── common/          # shared device-resident machinery
│   ├── plant.py     # TensorPlant (vehicle physics, contacts, warmup),
│   │                #   TensorRandom (per-replica counter RNG streams)
│   ├── dynamics.py  # quaternion/euler helpers
│   ├── curriculum.py  # shared 0-1 difficulty ramp (success-rate gated)
│   └── aerodynamics.py  # batched lifting-surface model
├── quadx/           # QuadX-Hover (rotary-wing station keeping)
├── fixedwing/       # Fixedwing-Waypoints (aerodynamic navigation)
├── rocket/          # Rocket-Landing (propulsive vertical landing)
├── satellite/       # CMG attitude control (pure tensor)
├── transfer/        # low-thrust orbit raising (pure tensor)
├── docking/         # CWH docking + rendezvous (pure tensor)
├── pendulum/        # classic gym Pendulum-v1 toy (training-stack sanity check)
└── original_pyflyt/ # vendored PyFlyt vehicle URDF/YAML models
train.py             # rsl-rl PPO entry point for all 7 tasks
eval.py              # policy evaluation
```

The three atmospheric-vehicle tasks (`quadx`, `fixedwing`, `rocket`) run on
the Genesis rigid-body engine with fully tensorized actuation/contact
layers; the four space tasks (`satellite`, `transfer`, `docking`,
`rendezvous`) are pure-tensor integrators with no engine at all. All seven
share the identical `step`/`reset`/buffer contract, so `train.py` treats
them uniformly.

## Conventions

- `rew_buf` is `float32`, `obs_buf` is `float32`, all internal physics
  state is `float64`.
- `extras['time_outs']` marks pure time-limit truncations (for rsl-rl
  time-limit bootstrapping); `extras['terminal_observations']` carries the
  pre-reset observation.
- `completed_episodes` / `completed_successes` / `completed_returns`
  accumulate episode statistics on device; `complete` is the per-replica
  success flag.
- Actions arrive on `env.device`; implicit host/device copies raise.

## Curriculum difficulty (0–1)

Every environment exposes a unified difficulty setting in `[0, 1]` (0
easiest, 1 hardest) — the project-wide curriculum feature
([src/common/curriculum.py](src/common/curriculum.py)). Only initial-state /
task-parameter distributions are modulated; success criteria, terminations
and reward structure are untouched, so evaluation stays comparable at any
training difficulty.

- `difficulty=None` (default): published benchmark behavior, bit-for-bit
  (eval.py builds envs without kwargs and is unaffected).
- `difficulty=d` (float): train at a static difficulty.
- `curriculum=True`: the difficulty auto-ramps 0 → 1 under the shared
  `SuccessRateCurriculum` — +0.1 whenever a 1024-episode window closes with
  a rolling success rate ≥ 0.6 (forward-only). The current difficulty is
  recorded in `env.cfg` (`run.json`) and printed on each advance.

The per-task difficulty axes (d=0 → d=1):

| Task | Knob | d=0 (easy) → d=1 (benchmark) |
|---|---|---|
| Pendulum | reset ranges | th0 ±30° → ±180°, thdot0 ±0.25 → ±1 rad/s |
| QuadX-Hover | spawn perturbations | deterministic spawn (benchmark) → ±0.5 m xy / ±0.25 m z, ±0.5 m/s, ±0.35 rad |
| Fixedwing | waypoint shell | min radius 50 m → 20 m (max 65 m fixed) |
| Rocket-Landing | initial conditions | xy ±2 m, z 440–450 m, euler ±0.05 → xy ±20 m, z 400–450 m, euler ±0.3 |
| Satellite (CMG) | target attitude | ≤0.3 rad from home, no noise → uniform SO(3), ±0.001 noise |
| Transfer | initial eccentricity | e0 0.02 → 0.1 |
| Docking | spawn range/speed | 15–30 m, 0.1 m/s → 50–100 m, 0.5 m/s |
| Rendezvous | spawn range | 150–300 m → 800–1200 m |

All tasks place the published benchmark at d=1 except QuadX-Hover, whose
deterministic-spawn benchmark is already the easiest variant (its axis
extends toward harder spawn randomization). Train entries expose a single
`CURRICULUM` constant (`None` off / float static / `True` auto-ramp);
`train_rendezvous.py` ships with `CURRICULUM = True` (its proven default).

## Usage

```bash
python train.py --task quadx      # fixedwing | rocket | satellite
                                  # transfer | docking | rendezvous
```

Hyperparameters are hard-coded per task in `train.py`; results land in
`results/`.

## References

This suite reproduces or adapts the following published tasks; each
environment docstring lists the specific source of its defaults.

**Tasks and vehicle models**

1. Tai, J. K. *PyFlyt: UAV Simulation and Reinforcement Learning
   Environments*. github.com/jkuang421/PyFlyt — task definitions, reward
   shaping and the cf2x / fixedwing / rocket URDF-YAML vehicle models for
   QuadX-Hover, Fixedwing-Waypoints and Rocket-Landing (PyFlyt 0.29.0).
2. TorchRL contributors. *SatelliteEnv*. pytorch/rl (MIT) — reference for
   the CMG attitude-control task's reward weights, tolerances and reset
   distribution (`src/satellite`).
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital
   Transfer using Dynamics-Agnostic Reinforcement Learning"
   (arXiv:2211.08272, IEEE) and its companion code
   github.com/zampanteymedio/gym-satellite-trajectory — the
   `PerigeeRaisingEnv` task reproduced parameter-for-parameter
   (`src/transfer`).
4. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark
   for Deep Reinforcement Learning in Spacecraft Proximity Operations",
   CVPR 2026 Workshop on AI4Space (CC BY 4.0) — the docking and rendezvous
   benchmark's dynamics, task table and reward weights, reproduced from
   the paper's Section 3 (`src/docking`).

**Dynamics and GNC background**

5. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System
   for Satellite Rendezvous", *Journal of the Aerospace Sciences* 27(9),
   653–658 — the linearized relative-motion model used by
   `SpaceDockingEnv` / `SpaceRendezvousEnv`.
6. Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA Education
   Series — CMG momentum geometry and the singularity (manipulability)
   metric used by `SatelliteAttitudeEnv`.
7. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
   Kepler-to-Cartesian conversion and osculating-orbit recovery used by
   `LowThrustTransferEnv`.

**Infrastructure**

8. Genesis contributors. *Genesis*. github.com/Genesis-Embodied-AI/Genesis
   — rigid-body engine and tensor API backing the atmospheric-vehicle
   tasks.
9. Legged Robotics group, ETH Zürich. *rsl_rl*.
   github.com/leggedrobotics/rsl_rl — PPO training backend.
