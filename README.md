# AerospaceRL Benchmark: GPU-Parallel Aerospace Control Environments
A lightweight benchmark for GPU-parallel reinforcement learning in aerospace control. The repository contains eight aerospace tasks spanning atmospheric flight and spacecraft dynamics. All environments expose the same batched device-tensor interface without Gym or Gymnasium wrappers.

Each aerospace task includes a conventional guidance, navigation, or control baseline expressed in the same action convention as the learned policy. The controller can be evaluated directly or used to generate behavior-cloning data for the warm-started learning pipeline.

The four atmospheric vehicle tasks use URDF-based rigid-body plants; four spacecraft tasks use explicit tensor dynamics. Genesis is the current rigid-body backend for the URDF tasks and is treated as an implementation dependency rather than the benchmark abstraction.

All state lives on exactly one device as `float64` physics / `float32`
observation buffers. There is **no engine round-trip per replica, no
per-replica Python loop, no gym/gymnasium dependency**: every operation
(physics integration, contact checks, reward, reset, RNG) is a masked tensor
kernel, so inactive replicas are frozen and consume no random numbers.

Honesty note on synchronization: the *physics hot path* (per-physics-step
pause/restore, contact accumulation, rocket mass write) is free of host-device
syncs — data-dependent Python branching was removed in favor of unconditional
idempotent tensor writes. Each env `step()` still performs exactly ONE host
sync (`bool(done.any())`), needed to gate episode-boundary bookkeeping and the
reset path; console logging syncs a few extra scalars on
episode boundaries only.

A deeper, reviewer-oriented write-up of the infrastructure claims and their
verification conditions lives in
[docs/technical-highlights.md](docs/technical-highlights.md).

## Environments

| Task | Class | Dynamics model | Obs dim | Act dim | Policy rate | Episode | Success criterion |
|---|---|---|---|---|---|---|---|
| QuadX-Waypoints | `QuadXWaypointsEnv` | URDF rigid body (Crazyflie 2.x, 240 Hz) + rate-PID | 37 | 4 | 40 Hz | 800 steps = 20 s | 4/4 waypoints captured (< 0.2 m) |
| Fixedwing-Waypoints | `FixedwingWaypointsEnv` | URDF rigid body (2.35 kg UAV, 240 Hz) + blade-element aero | 39 | 4 | 30 Hz | 3600 steps = 120 s | 4/4 waypoints captured (< 10 m) |
| Rocket-Landing | `RocketLandingEnv` | URDF rigid body (F9-class booster, 240 Hz) + grid fins, gimbal | 30 | 7 | 60 Hz | 1800 steps = 30 s | v2: pad contact, \|v\| < 0.02 m/s, \|ω\| < 0.02 rad/s, tilt < 0.1 rad for 2 s (121 samples); success does not truncate |
| Missile-Intercept | `MissileInterceptEnv` | URDF rigid body (150 kg missile, 240 Hz) + tail-fin aero + boost-coast engine, kinematic 3-DOF target | 34 | 3 | 60 Hz | 900 steps = 15 s | miss distance < 15 m (proximity fuse, any physics substep) |
| Satellite attitude (CMG) | `SatelliteAttitudeEnv` | Rigid body + 4- or 6-CMG cluster, RK4 2.5 ms | 26 / 32 | 4 / 6 | 40 Hz | 1200 steps = 30 s | att. error < 0.05 rad & \|ω\| < 0.02 rad/s, checked every step; \|ω\| > 10 rad/s fails |
| Low-thrust transfer | `LowThrustTransferEnv` | Two-body gravity + 3-axis 10 mN EP, RK4 10 s | 8 | 3 | 300 s/step | 166 steps ≈ 5 revs | terminal orbital-element return R > 0 at the limit |
| Docking | `SpaceDockingEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 500 steps ≈ 8.3 min | range < 1 m & speed < 0.1 m/s |
| Rendezvous | `SpaceRendezvousEnv` | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 1000 steps ≈ 16.7 min | range < 10 m & speed < 0.5 m/s, keep-out respected |

Per-task quantitative details (action/observation layouts, reward
decompositions, termination conditions, spawn distributions, wind
response) live in the module docstring of each package:
[multirotor](environments/multirotor/__init__.py),
[fixedwing](environments/fixedwing/__init__.py),
[rocket](environments/rocket/__init__.py),
[missile](environments/missile/__init__.py),
[satellite/attitude_control](environments/satellite/attitude_control/__init__.py),
[satellite/orbit_transfer](environments/satellite/orbit_transfer/__init__.py),
[satellite/rendezvous_docking](environments/satellite/rendezvous_docking/__init__.py)
(covers both docking and rendezvous).

Paper-style method write-ups (notation, equations, exact observation/action
tables and constants) for every task and algorithm are indexed in
[docs/method/README.md](docs/method/README.md).

## Package layout

The repository is split into an **environments** part and an **algorithms**
part, each internally subdivided; generic entry points live in `scripts/`.

```
environments/             # ── everything that defines a task ──
├── multirotor/           # QuadX-Waypoints (rotary-wing waypoint navigation)
├── fixedwing/            # Fixedwing-Waypoints (aerodynamic navigation)
├── rocket/               # Rocket-Landing (propulsive vertical landing)
├── satellite/            # spacecraft tasks, grouped by mission phase
│   ├── attitude_control/     # CMG attitude control (pure tensor)
│   ├── orbit_transfer/       # low-thrust orbit raising (pure tensor)
│   └── rendezvous_docking/   # CWH docking + rendezvous (pure tensor)
├── missile/              # Missile-Intercept (6-DOF missile vs kinematic maneuvering target)
└── common/               # env-side shared machinery
    ├── plant.py            # TensorPlant/TensorRandom (physics, contacts, RNG)
    ├── dynamics.py         # batched SI-unit rigid-body model + quat helpers
    ├── aerodynamics.py     # batched lifting-surface model (incl. stall)
    ├── consistency.py      # frozen-reference assets, URDF inertial parsing
    ├── original_backend.py # headless-Genesis backend for frozen PyFlyt code
    ├── original_batch.py   # shared-scene batching of the frozen NumPy path
    └── original_pyflyt/    # vendored PyFlyt vehicle URDF/YAML models + source
algorithms/               # ── everything that produces actions (learned or not) ──
├── train_utils.py        # shared run_training() for the per-task entries
├── ppo/ppo_flashnet.py   # FlashSAC-trunk networks for the PPO A/B arm
├── flashsac/flash_sac.py # FlashSAC (arXiv:2604.04539) on the batched protocol
├── warmstart/bc_warmstart.py  # behavior-cloning warm start from baselines
└── baselines/            # classical (non-learned) GNC baselines, one per task
    ├── baselines.py        #   PN/PDG/LQR/autopilot/... controllers
    └── fixedwing_autopilot.json  # serialized L1+TECS autopilot parameters
scripts/                  # generic training / evaluation entry points
├── train.py              # legacy small-config PPO CLI sweep entry for all 8 tasks
├── eval.py               # deterministic evaluation of trained checkpoints
├── baseline_eval.py      # evaluation of the classical baselines
├── sweep_tune.py         # small hyperparameter sweep harness
├── summarize.py          # aggregate results/ into a summary table
└── visualize.py          # checkpoint replay / trajectory visualization
environments/<category>/train_<task>.py  # self-contained per-task entries
environments/<category>/eval_* / *_test.py  # per-task deep-dive harnesses
docs/technical-highlights.md  # reviewer-oriented infrastructure write-up
```

The four Genesis rigid-body tasks (`quadx`, `fixedwing`, `rocket`,
`missile`) run on the Genesis rigid-body engine with fully tensorized
actuation/contact layers (the missile target is a kinematic 3-DOF aircraft
advanced analytically each physics substep); the four space tasks
(`satellite`, `transfer`, `docking`, `rendezvous`) are pure-tensor
integrators with no engine at all. All eight share the identical
`step`/`reset`/buffer contract, so the training entries treat them uniformly.

## Algorithms

Training entries support two algorithms through `algorithms/train_utils.py`.
PPO can opt into the same warmup-cosine formula that FlashSAC always uses
(`run_training(lr_cosine=True)`; floor 1.5e-4, warmup 1e-6 × total updates)
so cross-algorithm comparisons are not schedule artifacts. Otherwise PPO
uses each entry's fixed learning rate (the library default is 1e-3; the
shipped entries pin 3e-4, transfer 2.5e-4) and FlashSAC uses its paper
defaults (peak 3e-4).

- **PPO** ([rsl-rl](https://github.com/leggedrobotics/rsl_rl)): the default
  on-policy arm, a state-independent Gaussian policy trained with clipped
  PPO. Optionally with the **FlashSAC trunk**
  (`algorithms/ppo/ppo_flashnet.py`, `arch='flash_residual'`): input whitening
  from rsl-rl `EmpiricalNormalization` → two residual unit-norm blocks
  (expansion 4) → `UnitRMSNorm`, with unit-weight renormalization applied at
  the first grad-enabled forward of each update — architecture-matched
  against FlashSAC (the plain MLP trunk is kept as the A/B comparison arm).
  BatchNorm is deliberately disabled on this arm: batch statistics make the
  rollout and update policies different functions and break PPO's importance
  ratio.
- **FlashSAC** (`algorithms/flashsac/flash_sac.py`): a faithful port of FlashSAC
  (Kim et al., 2026, arXiv:2604.04539) to the batched GPU environment
  protocol — off-policy SAC with unit-norm weights, sqrt(d)-bounded BatchNorm
  / RMSNorm affines and reward normalization, sharply reduced gradient
  updates compensated by larger models. Gradient control is implicit in those
  weight/feature bounds; no gradient clipping is applied. fp16 AMP is wired
  behind `use_amp` and is off by default.

Every run records task configuration, real transition counts, physical-step
counts and device-side completion metrics in `results/<task>/run.json`.

## Classical (non-learned) baselines

Each task is paired with a classical GNC baseline
([algorithms/baselines/baselines.py](algorithms/baselines/baselines.py)) so learned results have a
non-learning reference on the same env, reward and success bar. Every
controller reads only the `'policy'` observation group, takes env constants
at construction, never touches internal state buffers, and leaves env
semantics untouched. Reference success rates below are the shipped official
`scripts/baseline_eval.py` results in `results/<task>/baseline.json` (1024
envs, seed 7, GPU; each row's evaluation window and episode count come from
that file):

| Task | Baseline | Reference result |
|---|---|---|
| QuadX-Waypoints | SE(3) geometric control (Lee et al. 2010) with integral trim | 100% (1024 episodes, 523-step window) |
| Fixedwing-Waypoints | L1 lateral + TECS longitudinal autopilot | 95.9% (1024 episodes, 1361-step window) |
| Rocket-Landing | ZEM/PN braking + virtual-pad manifold + pulse servo (PDG) | historical v1: 82.8% instantaneous ever-success (not a v2 score) |
| Satellite (CMG) | quaternion PD + Schaub-Junkins damped pseudo-inverse | 100% (1024 episodes, 334-step window) |
| Transfer | Gauss one-step greedy tangential guidance | 100%, terminal return 0.855 (1024 episodes, 166 steps) |
| Docking | CWH discrete LQR (symplectic-exact) + shield/fuel derating | 100% (1033 episodes, 223-step window) |
| Rendezvous | two-impulse CWH ZEM/PN guidance, recursive re-solve | 100% (1024 episodes, 574-step window, mean return 195.5) |
| Missile-Intercept | augmented PN (N=15) → body-rate commands | 83.0% (1037 episodes, 433-step window) |

These are honest references, not strawmen: the fixedwing and rocket
baselines retain documented residual failures (out-of-bound endgame limit
cycles, authority-saturated braking transients) that RL is expected to
attack. Task-specific deep-dive harnesses with per-scenario breakdowns and
ablation arms: [eval_docking_lqr.py](environments/satellite/rendezvous_docking/eval_docking_lqr.py),
[eval_fixedwing_autopilot.py](environments/fixedwing/eval_fixedwing_autopilot.py),
[eval_rendezvous_baseline.py](environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py),
[rocket_pdg_test.py](environments/rocket/rocket_pdg_test.py),
[eval_missile_scenarios.py](environments/missile/eval_missile_scenarios.py),
[test_missile_apn.py](environments/missile/test_missile_apn.py),
[test_quadx_se3.py](environments/multirotor/test_quadx_se3.py).

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

## Wind disturbance

The four atmospheric vehicles (QuadX-Waypoints, Fixedwing-Waypoints,
Rocket-Landing, Missile-Intercept) accept an optional wind disturbance through
two mechanisms:

- `steady_wind` (tuple, m/s): constant world-frame wind velocity.
- `dryden` (0–1 scale of the severity presets) + `severity`
  (`'light' | 'moderate' | 'severe'`): Dryden gust turbulence from
  environments/common/disturbances.py, advancing once per control step.

Aerodynamic terms evaluate air-relative velocity `v_air = v - wind`
(lifting surfaces and body-drag laws), so the default calm air
(`steady_wind=None, dryden=0`) reproduces the benchmark bit-for-bit;
`eval.py` restores each run's recorded `env_kwargs` from
`results/<task>/config.json`. The quadx, fixedwing and rocket train entries
hard-code a `WIND` constant (pure Dryden gusts at 30% of the moderate
preset); the missile train entry runs calm air. Satellite tasks operate in
vacuum and have no wind.

## Dependencies

Exact versions are pinned in [requirements.txt](requirements.txt). The
reference environment is Python 3.10.20 with a CUDA 12.8 build of PyTorch
2.11.0, `genesis-world` 1.4.4, `rsl-rl-lib` 5.3.0, NumPy 2.2.6 and SciPy
1.15.3 on Linux (RTX 3090, driver 580.173.02).

```bash
python -m pip install -r requirements.txt
```

A CUDA-enabled PyTorch is required for the Genesis rigid-body tasks
(`quadx`, `fixedwing`, `rocket`, `missile`) when running the GPU backend;
the four tensor-dynamics spacecraft tasks and every classical baseline also
run with CPU-only PyTorch. `genesis-world` and `rsl-rl-lib` pull their own
transitive dependencies; `gymnasium` / `pettingzoo` are not needed by this
suite (they belong only to the frozen, unused upstream wrappers).

## Usage

Per-task entries (all parameters hard-coded in the file — edit constants,
no CLI arguments):

```bash
python environments/multirotor/train_quadx.py
python environments/fixedwing/train_fixedwing.py
python environments/rocket/train_rocket.py
python environments/missile/train_missile.py
python environments/satellite/attitude_control/train_satellite.py
python environments/satellite/orbit_transfer/train_transfer.py
python environments/satellite/rendezvous_docking/train_docking.py
python environments/satellite/rendezvous_docking/train_rendezvous.py
```

The original CLI aggregator still works for quick sweeps:

```bash
python scripts/train.py quadx   # fixedwing | missile | rocket | satellite
                                # transfer | docking | rendezvous
```

Evaluation and baselines:

```bash
python scripts/eval.py quadx                     # deterministic checkpoint evaluation
python scripts/baseline_eval.py quadx            # classical baseline, any of the 8 tasks
# per-task deep-dive harnesses live in their category directories
# (see the per-category READMEs):
python environments/satellite/rendezvous_docking/eval_docking_lqr.py
python environments/fixedwing/eval_fixedwing_autopilot.py    # scenario grids, ablation arms,
python environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py  # per-scenario breakdowns
python environments/rocket/rocket_pdg_test.py
python environments/missile/eval_missile_scenarios.py
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
   QuadX-Waypoints, Fixedwing-Waypoints and Rocket-Landing (PyFlyt 0.29.0).
2. TorchRL contributors. *SatelliteEnv*. pytorch/rl (MIT) — reference for
   the CMG attitude-control task's reward weights, tolerances and reset
   distribution (`satellite/attitude_control`).
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital
   Transfer using Dynamics-Agnostic Reinforcement Learning"
   (arXiv:2211.08272, IEEE) and its companion code
   github.com/zampanteymedio/gym-satellite-trajectory — the
   `PerigeeRaisingEnv` task reproduced parameter-for-parameter
   (`satellite/orbit_transfer`).
4. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark
   for Deep Reinforcement Learning in Spacecraft Proximity Operations",
   CVPR 2026 Workshop on AI4Space (CC BY 4.0) — the docking and rendezvous
   benchmark's dynamics, task table and reward weights, reproduced from
   the paper's Section 3 (`satellite/rendezvous_docking`).

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
    norm-controlled off-policy algorithm ported in `algorithms/flashsac/flash_sac.py`
    and mirrored by the PPO trunk arm.

## Roadmap: planned control tasks

All future tasks follow the same per-task recipe (tensorized env + spec
docstring → classical baseline registered in `algorithms/baselines/` →
self-contained `train_<task>.py` with 1024 envs / seed 7 →
`results/<task>/` → README update). No
multi-agent tasks are planned — every task below is a single agent.

### Priority 1

| Category | Task | Environment notes | Classical baseline | Interest |
|---|---|---|---|---|
| missile | Intercept — **landed** | 6-DOF rigid body + aero, boost-coast engine, maneuvering target (weave/barrel roll), RK4 1 ms | APN (N=15) | first missile env, fills the former `missile/` placeholder |
| rocket | Droneship Landing | Rocket-Landing + 6-DOF moving pad (sea-wave spectrum), 10 m height offset | PDG + lag-compensated retargeting | moving touchdown target |
| satellite | Docking-Fault | Docking + actuator failures (stuck/dead thruster) + measurement noise | LQR + control reallocation | fault injection, safe RL |
| fixedwing | Autoland | glide-slope tracking + crosswind + flare, runway-contact criteria | glide-slope guidance + TECS (reuse existing autopilot) | baseline already exists |
| multirotor | Slung Payload | QuadX + cable-suspended pendulum load; action must damp swing | SE(3) + swing-angle damping | underactuated coupling |

### Priority 2

| Category | Task | Environment notes | Classical baseline | Interest |
|---|---|---|---|---|
| fixedwing | Terrain Following | DEM sampling, radar-altimeter local observations | TF-lateral guidance + terrain matching | local observations, obstacle avoidance |
| satellite | Inspection | CWH + field-of-view constraint, coverage-rate reward | ZEM/PN fly-around law | long horizon, coverage-shaped reward |
| rocket | RTB Boost-back | ascent + 180° flip + landing, ~90 s | gravity turn + two-phase PDG | long-horizon reward shaping |
| missile | HGV Glide | hypersonic glide phase, heat-flux / load-factor corridor | BTT corridor guidance | constraint satisfaction |
| multirotor | Racing Gate | gate sequence, time penalty + speed cap | minimum-time trajectory tracking | aggressive flight, extends `test_quadx_se3` |

### Priority 3

| Category | Task | Environment notes | Classical baseline | Interest |
|---|---|---|---|---|
| satellite | GEO Station-keeping | perturbed two-body (J2 + sun/moon), ±0.05° deadband | impulsive corrections (Euler–Hill setpoint) | very low control budget |
| rocket | Ascent to Orbit | gravity turn + dynamic-pressure / heat-flux limits, orbit-insertion accuracy | linear-tangent guidance | completes launch → orbit → transfer → rendezvous chain |
| satellite | Detumble & Capture | tumbling non-cooperative target, contact dynamics | angular-momentum matching + attitude sync | contact-rich manipulation |
| multirotor | Precision Landing | moving cart pad + simulated vision latency | MPC-style trajectory tracking | latency-robust observation |

### Shared infrastructure (enabling work, single-agent only) — landed

1. `environments/common/disturbances.py` — **done**: unified
   `WindDisturbance` (steady wind + Dryden gust, MIL-F-8785C low-altitude
   shaping filters, seeded RNG stream); the atmospheric envs hand the
   returned velocity to the plant, whose aerodynamic terms evaluate
   v_air = v - wind (calm air is bit-for-bit with the benchmark)
2. `algorithms/baselines/` — **done**: dict-based `BUILDERS` registry plus
   a `@register_baseline('<task>')` decorator; a new classical baseline is
   wired into `baseline_eval.py` with one line

### Cadence

Batch 1 (parallel, independent): missile Intercept + Droneship Landing +
Docking-Fault. Batch 2: Autoland + Slung Payload. Later batches follow the
priority order above as infrastructure lands.

### Rocket benchmark v2（2026-10-26）

Rocket 默认使用独立版本的六球形脚垫接触资产，主指标为连续停稳 2 秒。
原始 `rocket.urdf`、旧模型和历史成绩保留；旧检查点缺少版本字段时按 v1 评估。
背景、结构性问题证据、物理扰动范围、版本边界及复现命令见
[benchmark-v2 中文说明](docs/rocket_benchmark_v2_zh.md)。
当前 main 保留 60 Hz 策略、120 Hz 控制、240 Hz 物理步长及已有 Dryden 风配置。
Claude 已完成的 v2 评估采用 40 Hz 冻结快照；版本、频率和源码哈希随结果一同保留。

```bash
# 每个环境的完整首回合，4 x 128，保留逐回合记录
python scripts/rocket_benchmark_eval.py --backend gpu --variation nominal --out results/rocket-v2-nominal
python scripts/rocket_benchmark_eval.py --backend gpu --variation shifted --out results/rocket-v2-shifted
# 无策略、关发动机的接触因果诊断（不计为策略成功率）
python scripts/rocket_passive_contact.py --version v1 --out results/passive-v1.json
python scripts/rocket_passive_contact.py --version v2 --out results/passive-v2.json
python -m unittest discover -s tests -v
```
