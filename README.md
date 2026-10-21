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
(`train.py`, optional GRU backbone via `--rnn`); evaluation via `eval.py`.

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

Per-task quantitative details (action/observation layouts, reward
decompositions, termination conditions, spawn distributions) live in the
module docstring of each package: [src/quadx](src/quadx/__init__.py),
[src/fixedwing](src/fixedwing/__init__.py),
[src/rocket](src/rocket/__init__.py),
[src/satellite](src/satellite/__init__.py),
[src/transfer](src/transfer/__init__.py),
[src/docking](src/docking/__init__.py).

## Package layout

```
src/
├── common/          # shared device-resident machinery
│   ├── plant.py     # TensorPlant (vehicle physics, contacts, warmup),
│   │                #   TensorRandom (per-replica counter RNG streams)
│   ├── dynamics.py  # quaternion/euler helpers
│   └── aerodynamics.py  # batched lifting-surface model
├── quadx/           # QuadX-Hover (rotary-wing station keeping)
├── fixedwing/       # Fixedwing-Waypoints (aerodynamic navigation)
├── rocket/          # Rocket-Landing (propulsive vertical landing)
├── satellite/       # CMG attitude control (pure tensor)
├── transfer/        # low-thrust orbit raising (pure tensor)
├── docking/         # CWH docking + rendezvous (pure tensor)
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
