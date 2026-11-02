# Method Documentation

Paper-style method-section references for every RL environment and every
algorithm (learned or classical) in the AerospaceRL Benchmark. Each document
defines its notation, writes the dynamics / reward / control law as numbered
equations, tabulates the exact observation and action layouts, and lists every
constant with the value used in code. Where a package docstring or README
disagrees with the implementation, these documents follow the code.

## Environments

| Document | Task (id) | One-line summary |
|---|---|---|
| [env-quadx-waypoints.md](env-quadx-waypoints.md) | QuadX-Waypoints (`quadx`) | Crazyflie-scale quadrotor, 4-waypoint capture in a 5 m dome, potential-telescoping reward |
| [env-quadx-track.md](env-quadx-track.md) | QuadX-Track (`quadx_track`) | Crazyflie-scale quadrotor, persistent observation of a cooperative ground moving target under FOV/range/occlusion/noise |
| [env-fixedwing-waypoints.md](env-fixedwing-waypoints.md) | Fixedwing-Waypoints (`fixedwing`) | 2.35 kg UAV, 4-waypoint capture in a 20–65 m shell, blade-element aerodynamics with stall |
| [env.py](../../environments/vtol/env.py) | Quadplane-VTOL (`quadplane`) | 2.35 kg hybrid UAV (quad motors + fixedwing surfaces + pusher), waypoint capture with natural rotor→aero authority transition (module docstring is the method reference) |
| [env-rocket-landing.md](env-rocket-landing.md) | Rocket-Landing (`rocket`) | F9-class booster propulsive landing, benchmark v2 (40 Hz, 2 s settled): v5 dense reward, success without truncation |
| [env-missile-intercept.md](env-missile-intercept.md) | Missile-Intercept (`missile`) | 6-DOF tail-controlled missile vs kinematic maneuvering target, proximity-fuse kill |
| [env-satellite-attitude.md](env-satellite-attitude.md) | CMG attitude control (`satellite`) | Rigid bus + 4/6-CMG cluster, singularity-aware telescoping reward |
| [env-orbit-transfer.md](env-orbit-transfer.md) | Low-thrust transfer (`transfer`) | Two-body perigee raising, 10 mN/axis EP, PBRS telescoping onto a terminal orbital-element reward |
| [env-docking.md](env-docking.md) | Docking (`docking`) | CWH final approach 60–120 m, discrete-symplectic integration, 1 m / 0.1 m/s success gate |
| [env-rendezvous.md](env-rendezvous.md) | Rendezvous (`rendezvous`) | CWH medium approach 1000–1500 m with a 5 m keep-out sphere, 10 m / 0.5 m/s goal |

## Learning algorithms

| Document | Content |
|---|---|
| [algo-ppo.md](algo-ppo.md) | rsl-rl PPO over the 2x128 tanh MLP, shared warmup-cosine LR schedule, fair-comparison protocol |
| [algo-bc-warmstart.md](algo-bc-warmstart.md) | Behavior-cloning warm start from the classical baselines (PPO-only), data contract and fitting procedure |

## Classical (non-learned) baselines

Every environment registers one default controller in `BUILDERS` and exposes
alternative control laws as CLI-selectable arms in `BASELINE_VARIANTS`
(`scripts/baseline_eval.py --baseline <key>`). That is **10 defaults + 20
distinct variant controllers = 30 controller arms**
(`algorithms/baselines/baselines.py`); the 22 `--baseline` keys include two
aliases of default arms (marked below), so the tables list 32 rows covering
those 30 distinct controllers. `--baseline` keys are shown in backticks.

### 3.1 Default controller per task (`BUILDERS`)

| Document | Task | Default method |
|---|---|---|
| [baseline-quadx-se3.md](baseline-quadx-se3.md) | quadx | SE(3) geometric tracking control (Lee et al. 2010) with integral trim |
| [baseline-quadx-track-kf-pid.md](baseline-quadx-track-kf-pid.md) | quadx_track | Constant-velocity Kalman filter + fixed-offset SE(3) tracking with a public timeout hold |
| [baseline-fixedwing-l1-tecs.md](baseline-fixedwing-l1-tecs.md) | fixedwing | L1 lateral guidance (Park-Deyst-Howe 2004) + TECS longitudinal autopilot |
| [baselines/variants_quadplane.py](../../algorithms/baselines/variants_quadplane.py) (`QuadplaneSE3PusherBaseline`) | quadplane | Quad-rotor SE(3) geometric control for the whole waypoint task, pusher throttle closing the body-x ground-speed error (validated: 87.5% on 256 replicas / seed 7, 89.1%±3.1 over the four-seed protocol) |
| [baseline-rocket-pdg.md](baseline-rocket-pdg.md) | rocket | ZEM/ZEV powered-descent guidance, virtual-pad braking manifold, pulse servo |
| [baseline-missile-apn.md](baseline-missile-apn.md) | missile | Augmented proportional navigation mapped to body-rate commands |
| [baseline-satellite-cmg-pd.md](baseline-satellite-cmg-pd.md) | satellite | Quaternion PD + Schaub-Junkins damped pseudo-inverse CMG steering |
| [baseline-transfer-gauss.md](baseline-transfer-gauss.md) | transfer | Gauss one-step greedy tangential guidance with propellant-priced objective |
| [baseline-docking-cwh-lqr.md](baseline-docking-cwh-lqr.md) | docking | Symplectic-exact discrete CWH LQR + range shield / capture net / fuel derating |
| [baseline-rendezvous-two-impulse.md](baseline-rendezvous-two-impulse.md) | rendezvous | Recursively re-solved two-impulse CWH ZEM/ZEV guidance with feasibility guards |

`RendezvousCWHPDBaseline` is additionally registered under the separate
registry key `rendezvous_cwhpd` (a second controller for the same task, not
reachable through `--baseline`).

### 3.2 Per-task variants (`BASELINE_VARIANTS`)

| Task | `--baseline` key | Controller | Method reference |
|---|---|---|---|
| quadx | `cascaded_pid` | `QuadXCascadePIDBaseline` | [baseline-quadx-cascade-pid.md](baseline-quadx-cascade-pid.md): position PID → attitude P/PD cascade on the env's native rate PID |
| quadx | `smc` | `QuadXSMCBaseline` | [variants_quadx.py](../../algorithms/baselines/variants_quadx.py): sliding surface `s = e_v + λe_p` + saturating exponential reaching law, Euler attitude cascade |
| quadx_track | `pursuit` | `QuadXTrackPursuitBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py): raw-measurement pure pursuit, no estimator |
| quadx_track | `pid_ema` | `QuadXTrackPIDEMABaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py): EMA-filtered target + SE(3) cascade |
| fixedwing | `pid_cascade` | `FixedwingPIDBaseline` | [baseline-fixedwing-pid.md](baseline-fixedwing-pid.md): course→bank→bank-rate, altitude→pitch→pitch-rate, airspeed PI |
| fixedwing | `nlgl_tecs` | `FixedwingNLGLBaseline` | [variants_fw_rocket.py](../../algorithms/baselines/variants_fw_rocket.py): Park nonlinear guidance law + TECS longitudinal stack |
| rocket | `pd_landing` | `RocketPDLandingBaseline` | [baseline-rocket-pd-landing.md](baseline-rocket-pd-landing.md): brake-distance/threshold-height PD landing, hysteretic engine band, one-shot ignition |
| rocket | `eguidance` | `RocketEGuidanceBaseline` | [variants_fw_rocket.py](../../algorithms/baselines/variants_fw_rocket.py): Apollo E-guidance (quadratic polynomial targeting) mid-course |
| missile | `ppn` | `MissilePPNBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py): pure pursuit, no lead |
| missile | `tpn` | `MissileTPNBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py): true PN, no gravity augmentation |
| quadplane | `hybrid_cascade` | `QuadplaneHybridBaseline` | [baselines.py](../../algorithms/baselines/baselines.py): velocity-scheduled rotor/wing mode switching with a wing-lift unload term; retuned for the live-wing plant (26.6% success, ablation arm) |
| quadplane | `l1_wing` | `QuadplaneL1WingBaseline` | [variants_quadplane.py](../../algorithms/baselines/variants_quadplane.py): wing-mode L1 lateral guidance + quad low-speed assist (49.6%, ablation arm) |
| satellite | `quat_pd_damped_cmg` | `SatelliteCMGBaseline` | [baseline-satellite-cmg-pd.md](baseline-satellite-cmg-pd.md): the default arm, selectable by key |
| satellite | `quat_pd_mp_pinv_cmg` | `SatelliteOrdinaryPinvBaseline` | [baseline-satellite-mp-pinv.md](baseline-satellite-mp-pinv.md): same PD, ordinary Moore–Penrose pseudo-inverse steering |
| satellite | `quat_pd_wpinv_cmg` | `SatelliteWeightedPinvBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py): weighted pseudo-inverse + null-space singularity-avoidance steering |
| transfer | `gauss_tangential` | `TransferGaussBaseline` | [baseline-transfer-gauss.md](baseline-transfer-gauss.md): the default arm, selectable by key |
| transfer | `q_law` | `TransferQLawBaseline` | [baseline-transfer-qlaw.md](baseline-transfer-qlaw.md): Petropoulos Q-law feedback guidance on the planar (a, e) subset with effectivity coasting |
| transfer | `edelbaum` | `TransferEdelbaumBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py): Edelbaum open-loop Δv budget, thrust along local velocity |
| docking | `pole_placement` | `DockingCWHPlacementBaseline` | [baselines.py](../../algorithms/baselines/baselines.py): CWH full-state feedback, standard pole placement, no extra layers |
| docking | `zem_zev` | `DockingZEMZEVBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py): CWH ZEM/ZEV single-pulse guidance + fuel/range guards |
| rendezvous | `cwh_pd` | `RendezvousCWHPDBaseline` | [baselines.py](../../algorithms/baselines/baselines.py): CWH-compensated PD to the origin, no constraint layers |
| rendezvous | `glideslope` | `RendezvousGlideslopeBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py): range-rate braking corridor (glideslope) on the initial LOS |

The two "default arm, selectable by key" rows are duplicates of 3.1 under the
CLI name: they are not extra controllers. Per-arm success rates, Wilson
intervals, and failure-mode tables for all 30 distinct controllers are in
[baseline_test_report.md](../baseline_test_report.md).
