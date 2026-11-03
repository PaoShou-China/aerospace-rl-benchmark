# References

This suite reproduces or adapts the following published tasks; each
environment docstring lists the source of its defaults.

## Tasks and vehicle models

1. Tai, J. K. *PyFlyt: UAV Simulation and Reinforcement Learning Environments*
   — task definitions, reward shaping and the cf2x / fixedwing / rocket
   URDF-YAML models (PyFlyt 0.29.0).
2. TorchRL contributors. *SatelliteEnv* — reward weights, tolerances and reset
   distribution of the CMG attitude task.
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital Transfer
   using Dynamics-Agnostic Reinforcement Learning" (arXiv:2211.08272) and its
   companion code — the perigee-raising task, reproduced parameter for
   parameter.
4. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark for
   Deep Reinforcement Learning in Spacecraft Proximity Operations", CVPR 2026
   Workshop on AI4Space (CC BY 4.0) — docking / rendezvous dynamics, task table
   and reward weights.

## Dynamics and GNC

5. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System for
   Satellite Rendezvous" — the linearized relative-motion model.
6. Wie, B. (2008). *Space Vehicle Dynamics and Control* — CMG momentum
   geometry and the singularity metric.
7. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
   Kepler-to-Cartesian conversion and osculating-orbit recovery.
8. Lee, T., Leok, M. & McClamroch, N. H. (2010). "Geometric Tracking Control of
   a Quadrotor UAV on SE(3)", CDC — the QuadX geometric baseline.
9. Park, S., Deyst, J. & How, J. P. (2004). "A New Nonlinear Guidance Logic for
   Trajectory Tracking", AIAA GNC — the fixed-wing L1 lateral guidance.

## Infrastructure

10. Genesis contributors. *Genesis* — rigid-body engine and tensor API.
11. Legged Robotics group, ETH Zürich. *rsl_rl* — PPO training backend.
