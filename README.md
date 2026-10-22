# AerospaceRL Benchmark: GPU-Parallel Aerospace Control Environments

A lightweight benchmark for GPU-parallel reinforcement learning in aerospace control. The repository contains seven aerospace tasks spanning atmospheric flight and spacecraft dynamics, together with a pendulum task used only as a training-pipeline check. All environments expose the same batched device-tensor interface without Gym or Gymnasium wrappers.

Each aerospace task includes a conventional guidance, navigation, or control baseline expressed in the same action convention as the learned policy. The controller can be evaluated directly, used to generate behavior-cloning data, or queried for nominal actions by an external residual-RL pipeline.

The three atmospheric vehicle tasks use URDF-based rigid-body plants; four spacecraft tasks use explicit tensor dynamics. Genesis is the current rigid-body backend for the URDF tasks and is treated as an implementation dependency rather than the benchmark abstraction.

## Environments

| Task | Class | Dynamics model | Obs dim | Act dim | Policy rate | Episode | Success criterion |
|---|---|---|---|---|---|---|---|
| QuadX-Hover | `QuadXHoverEnv` | URDF rigid body (Crazyflie 2.x, 240 Hz) + rate-PID | 21 | 4 | 40 Hz | 400 steps = 10 s | none (graded hover reward) |
| Fixedwing-Waypoints | `FixedwingWaypointsEnv` | URDF rigid body (2.35 kg UAV, 240 Hz) + blade-element aero | 38 | 4 | 30 Hz | 3600 steps = 120 s | 4/4 waypoints captured (< 2 m) |
| Rocket-Landing | `RocketLandingEnv` | URDF rigid body (F9-class booster, 240 Hz) + grid fins, gimbal | 30 | 7 | 40 Hz | 1200 steps = 30 s | pad contact, \|v\| < 0.02 m/s, \|ω\| < 0.02 rad/s, tilt < 0.1 rad |
| Satellite attitude (CMG) | `SatelliteAttitudeEnv` | Rigid body + 4- or 6-CMG cluster, RK4 2.5 ms | 23 / 29 | 4 / 6 | 40 Hz | 1200 steps = 30 s | att. error < 0.05 rad, \|ω\| < 0.02 rad/s at limit |
| Low-thrust transfer | `LowThrustTransferEnv` | Two-body gravity + 3-axis 10 mN EP, RK4 10 s | 8 | 3 | 300 s/step | 166 steps ≈ 5 revs | none (terminal reward) |
| Docking | `SpaceDockingEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 500 steps ≈ 8.3 min | range < 1 m & speed < 0.1 m/s |
| Rendezvous | `SpaceRendezvousEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 1000 steps ≈ 16.7 min | range < 10 m & speed < 0.5 m/s, keep-out respected |
| Pendulum (toy sanity check) | `PendulumEnv` | Classic single-DOF pendulum, explicit Euler 50 ms | 3 | 1 | 20 Hz | 200 steps = 10 s | completed-episode return > −200 |

Per-task quantitative details (action/observation layouts, reward
decompositions, termination conditions, spawn distributions, difficulty
axes) live in the module docstring of each package:
[src/quadx](src/quadx/__init__.py),
[src/fixedwing](src/fixedwing/__init__.py),
[src/rocket](src/rocket/__init__.py),
[src/satellite](src/satellite/__init__.py),
[src/transfer](src/transfer/__init__.py),
[src/docking](src/docking/__init__.py) (covers both docking and
rendezvous), [src/pendulum](src/pendulum/__init__.py).

## Package layout

```
src/
├── common/              # shared device-resident machinery
│   ├── plant.py         # TensorPlant (vehicle physics, contacts, warmup),
│   │                    #   TensorRandom (per-replica counter RNG streams)
│   ├── dynamics.py      # batched SI-unit QuadX model + quaternion/euler helpers
│   ├── aerodynamics.py  # batched lifting-surface model (incl. stall)
│   ├── curriculum.py    # shared 0-1 difficulty ramp (success-rate gated)
│   ├── consistency.py   # frozen reference assets, URDF inertial parsing,
│   │                    #   hash manifests, serializable plant contract
│   ├── train_utils.py   # shared run_training() for the per-task entries
│   ├── flash_sac.py     # FlashSAC (arXiv:2604.04539) on the batched protocol
│   ├── ppo_flashnet.py  # FlashSAC-trunk networks for the PPO A/B arm
│   ├── baselines.py     # classical GNC baselines, one per task
│   ├── fixedwing_autopilot.json # serialized L1+TECS autopilot parameters
│   ├── original_backend.py  # headless-Genesis backend for frozen PyFlyt code
│   └── original_batch.py    # shared-scene batching of the frozen NumPy path
├── quadx/               # QuadX-Hover (rotary-wing station keeping)
├── fixedwing/           # Fixedwing-Waypoints (aerodynamic navigation)
├── rocket/              # Rocket-Landing (propulsive vertical landing)
├── satellite/           # CMG attitude control (pure tensor)
├── transfer/            # low-thrust orbit raising (pure tensor)
├── docking/             # CWH docking + rendezvous (pure tensor)
├── pendulum/            # classic gym Pendulum-v1 toy (training-stack sanity check)
└── original_pyflyt/     # vendored PyFlyt vehicle URDF/YAML models
train.py                 # rsl-rl PPO entry point for all 7 real tasks
train_<task>.py          # self-contained per-task entries (hard-coded params)
train_rocket_v2.py       # rocket A/B arm with the redesigned reward
eval.py                  # deterministic evaluation of trained checkpoints
baseline_eval.py         # evaluation of the classical baselines
classic_pendulum.py      # pendulum energy-shaping + LQR baseline entry
docs/technical-highlights.md  # reviewer-oriented infrastructure write-up
```

The three atmospheric-vehicle tasks (`quadx`, `fixedwing`, `rocket`) run on
the Genesis rigid-body engine with fully tensorized actuation/contact
layers; the four space tasks (`satellite`, `transfer`, `docking`,
`rendezvous`) are pure-tensor integrators with no engine at all. All eight
share the identical `step`/`reset`/buffer contract, so the training
entries treat them uniformly.

## Algorithms

Training entries support two algorithms through `src/common/train_utils.py`,
sharing the identical warmup-cosine learning-rate schedule (peak 1e-3 →
1.5e-4, warmup 1e-6 × total updates) so cross-algorithm comparisons are not
schedule artifacts:

- **PPO** ([rsl-rl](https://github.com/leggedrobotics/rsl_rl)): the default
  on-policy arm. Optionally with the **FlashSAC trunk**
  (`src/common/ppo_flashnet.py`, `arch='flash_residual'`): input
  UnitBatchNorm → residual unit-norm blocks → UnitRMSNorm, norm-control
  semantics replicated on the PPO update path — architecture-matched against
  FlashSAC (the plain MLP trunk is kept as the A/B comparison arm).
- **FlashSAC** (`src/common/flash_sac.py`): a faithful port of FlashSAC
  (Kim et al., 2026, arXiv:2604.04539) to the batched GPU environment
  protocol — off-policy SAC with explicit weight/feature/gradient norm
  control and sharply reduced gradient updates compensated by larger models.

Every run records task configuration, real transition counts, physical-step
counts and device-side completion metrics in `results/<task>/run.json`.

## Classical (non-learned) baselines

Each task is paired with a classical GNC baseline
([src/common/baselines.py](src/common/baselines.py),
[classic_pendulum.py](classic_pendulum.py)) so learned results have a
non-learning reference on the same env, reward and success bar. Every
controller reads only the `'policy'` observation group, takes env constants
at construction, never touches internal state buffers, and leaves env
semantics untouched. Reference success rates (official `baseline_eval`
protocol, 1024 envs / 2000 steps / seed 7, per-task evaluation windows
declared in each report):

| Task | Baseline | Reference result |
|---|---|---|
| Pendulum | energy-shaping swing-up + discrete-time LQR | swing-up-and-hold, return ≈ −160 vs −200 bar |
| QuadX-Hover | SE(3) geometric control (Lee et al. 2010) with integral trim | 100% (5120 episodes) |
| Fixedwing | L1 lateral + TECS longitudinal autopilot | ~74.6% train / 73.1% test (3700-step window) |
| Rocket-Landing | ZEM/PN braking + virtual-pad manifold + pulse servo (PDG) | 66.8% (2000-step window) |
| Satellite (CMG) | quaternion PD + Schaub-Junkins damped pseudo-inverse | 100% (1024 episodes) |
| Transfer | Gauss one-step greedy tangential guidance | 100%, terminal R ≈ 0.86 |
| Docking | CWH discrete LQR (symplectic-exact) + shield/fuel derating | 100% |
| Rendezvous | two-impulse CWH ZEM/PN guidance, recursive re-solve | 100% (4096 episodes, benchmark envelope) |

These are honest references, not strawmen: the fixedwing and rocket
baselines retain documented residual failures (out-of-bound endgame limit
cycles, authority-saturated braking transients) that RL is expected to
attack. Task-specific deep-dive harnesses with per-scenario breakdowns and
ablation arms: [eval_docking_lqr.py](eval_docking_lqr.py),
[eval_fixedwing_autopilot.py](eval_fixedwing_autopilot.py),
[eval_rendezvous_baseline.py](eval_rendezvous_baseline.py),
[rocket_pdg_test.py](rocket_pdg_test.py),
[test_quadx_se3.py](test_quadx_se3.py).

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

Per-task entries (all parameters hard-coded in the file — edit constants,
no CLI arguments):

```bash
python train_quadx.py          # train_fixedwing.py | train_rocket.py
                               # train_satellite.py | train_transfer.py
                               # train_docking.py   | train_rendezvous.py
                               # train_pendulum.py
python train_rocket_v2.py      # rocket A/B arm with the redesigned reward
```

The original CLI aggregator still works for quick sweeps:

```bash
python train.py --task quadx   # fixedwing | rocket | satellite
                               # transfer | docking | rendezvous
```

Evaluation and baselines:

```bash
python eval.py quadx                     # deterministic checkpoint evaluation
python baseline_eval.py quadx            # classical baseline, any of the 8 tasks
python eval_docking_lqr.py               # per-task deep-dive harnesses:
python eval_fixedwing_autopilot.py       #   scenario grids, ablation arms,
python eval_rendezvous_baseline.py       #   per-scenario breakdown reports
python rocket_pdg_test.py
```

Results land in `results/<task>/` (`run.json` for training runs,
`baseline.json` / per-harness reports for baselines).

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
   `SpaceDockingEnv` / `SpaceRendezvousEnv` and the rendezvous two-impulse
   baseline.
6. Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA Education
   Series — CMG momentum geometry and the singularity (manipulability)
   metric used by `SatelliteAttitudeEnv` and the satellite CMG baseline.
7. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
   Kepler-to-Cartesian conversion and osculating-orbit recovery used by
   `LowThrustTransferEnv` and the transfer Gauss baseline.
8. Lee, T., Leok, M. & McClamroch, N. H. (2010). "Geometric Tracking
   Control of a Quadrotor UAV on SE(3)", CDC 2010 — the QuadX geometric
   baseline.
9. Park, S., Deyst, J. & How, J. P. (2004). "A New Nonlinear Guidance
   Logic for Trajectory Tracking", AIAA GNC — the fixedwing autopilot's
   L1 lateral guidance.

**Infrastructure**

10. Genesis contributors. *Genesis*. github.com/Genesis-Embodied-AI/Genesis
    — rigid-body engine and tensor API backing the atmospheric-vehicle
    tasks.
11. Legged Robotics group, ETH Zürich. *rsl_rl*.
    github.com/leggedrobotics/rsl_rl — PPO training backend.
12. Kim et al. (2026). *FlashSAC* (arXiv:2604.04539, RSS'26) — the
    norm-controlled off-policy algorithm ported in `src/common/flash_sac.py`
    and mirrored by the PPO trunk arm.
