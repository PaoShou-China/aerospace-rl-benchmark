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
| [env-fixedwing-waypoints.md](env-fixedwing-waypoints.md) | Fixedwing-Waypoints (`fixedwing`) | 2.35 kg UAV, 4-waypoint capture in a 20–65 m shell, blade-element aerodynamics with stall |
| [env-rocket-landing.md](env-rocket-landing.md) | Rocket-Landing (`rocket`) | F9-class booster propulsive landing, v4 dense per-policy-step reward, success without truncation |
| [env-missile-intercept.md](env-missile-intercept.md) | Missile-Intercept (`missile`) | 6-DOF tail-controlled missile vs kinematic maneuvering target, proximity-fuse kill |
| [env-satellite-attitude.md](env-satellite-attitude.md) | CMG attitude control (`satellite`) | Rigid bus + 4/6-CMG cluster, singularity-aware telescoping reward |
| [env-orbit-transfer.md](env-orbit-transfer.md) | Low-thrust transfer (`transfer`) | Two-body perigee raising, 10 mN/axis EP, PBRS telescoping onto a terminal orbital-element reward |
| [env-docking.md](env-docking.md) | Docking (`docking`) | CWH final approach 50–100 m, discrete-symplectic integration, 1 m / 0.1 m/s success gate |
| [env-rendezvous.md](env-rendezvous.md) | Rendezvous (`rendezvous`) | CWH medium approach 800–1200 m with a 5 m keep-out sphere, 10 m / 0.5 m/s goal |

## Learning algorithms

| Document | Content |
|---|---|
| [algo-ppo.md](algo-ppo.md) | rsl-rl PPO, the FlashSAC-trunk (`flash_residual`) architecture option, shared warmup-cosine LR schedule, fair-comparison protocol |
| [algo-flashsac.md](algo-flashsac.md) | FlashSAC (arXiv:2604.04539) port: norm-controlled off-policy SAC on the batched GPU protocol |
| [algo-bc-warmstart.md](algo-bc-warmstart.md) | Behavior-cloning warm start from the classical baselines (PPO-only), data contract and fitting procedure |

## Classical (non-learned) baselines

| Document | Task | Method |
|---|---|---|
| [baseline-quadx-se3.md](baseline-quadx-se3.md) | quadx | SE(3) geometric tracking control (Lee et al. 2010) with integral trim |
| [baseline-fixedwing-l1-tecs.md](baseline-fixedwing-l1-tecs.md) | fixedwing | L1 lateral guidance (Park-Deyst-Howe 2004) + TECS longitudinal autopilot |
| [baseline-rocket-pdg.md](baseline-rocket-pdg.md) | rocket | ZEM/ZEV powered-descent guidance, virtual-pad braking manifold, pulse servo |
| [baseline-missile-apn.md](baseline-missile-apn.md) | missile | Augmented proportional navigation mapped to body-rate commands |
| [baseline-satellite-cmg-pd.md](baseline-satellite-cmg-pd.md) | satellite | Quaternion PD + Schaub-Junkins damped pseudo-inverse CMG steering |
| [baseline-transfer-gauss.md](baseline-transfer-gauss.md) | transfer | Gauss one-step greedy tangential guidance with propellant-priced objective |
| [baseline-docking-cwh-lqr.md](baseline-docking-cwh-lqr.md) | docking | Symplectic-exact discrete CWH LQR + range shield / capture net / fuel derating |
| [baseline-rendezvous-two-impulse.md](baseline-rendezvous-two-impulse.md) | rendezvous | Recursively re-solved two-impulse CWH ZEM/ZEV guidance with feasibility guards |
