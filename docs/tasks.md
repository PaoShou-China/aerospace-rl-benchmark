# Task Overview

Ten single-agent aerospace control tasks: six atmospheric vehicles on the
Genesis rigid-body backend and four spacecraft missions on explicit tensor
dynamics. Every task exposes the same `step` / `reset` / buffer contract.

| Task | Class | Dynamics | Obs | Act | Policy rate | Episode | Success criterion |
|---|---|---|---|---|---|---|---|
| QuadX-Waypoints | `QuadXWaypointsEnv` | URDF Crazyflie 2.x (240 Hz) + rate PID | 37 | 4 | 40 Hz | 800 steps · 20 s | 4/4 waypoints < 0.2 m |
| QuadX-Track | `QuadXTrackEnv` | URDF Crazyflie 2.x (240 Hz) + rate PID, kinematic ground target, geometric camera | 25 | 4 | 40 Hz | 1200 steps · 30 s | visible-time fraction ≥ 0.85, no crash |
| Fixedwing-Waypoints | `FixedwingWaypointsEnv` | URDF 2.35 kg UAV (240 Hz) + blade-element aero | 39 | 4 | 30 Hz | 3600 steps · 120 s | 4/4 waypoints < 10 m |
| Quadplane-VTOL | `QuadplaneWaypointsEnv` | URDF hybrid UAV (240 Hz): 4 quad motors, fixed-wing surfaces, forward pusher | 47 | 8 | 40 Hz | 1200 steps · 30 s | 4/4 waypoints < 15 m |
| Rocket-Landing | `RocketLandingEnv` | URDF F9-class booster (240 Hz) + grid fins, gimbal | 30 | 7 | 40 Hz | 1200 steps · 30 s | pad contact, \|v\| < 0.02 m/s, \|ω\| < 0.02 rad/s, tilt < 0.1 rad for 2 s (81 samples) |
| Missile-Intercept | `MissileInterceptEnv` | URDF 150 kg missile (240 Hz) + tail-fin aero, boost-coast engine, kinematic 3-DOF target | 34 | 3 | 60 Hz | 900 steps · 15 s | miss distance < 15 m (proximity fuse, any physics substep) |
| Satellite attitude (CMG) | `SatelliteAttitudeEnv` | rigid body + 4/6-CMG cluster, RK4 2.5 ms | 26 / 32 | 4 / 6 | 40 Hz | 1200 steps · 30 s | att. error < 0.05 rad and \|ω\| < 0.02 rad/s; \|ω\| > 10 rad/s fails |
| Low-thrust transfer | `LowThrustTransferEnv` | two-body gravity + 3-axis 10 mN EP, RK4 10 s | 8 | 3 | 300 s/step | 166 steps ≈ 5 revs | terminal orbital-element return R > 0 at the limit |
| Docking | `SpaceDockingEnv` | Clohessy–Wiltshire–Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 500 steps ≈ 8.3 min | range < 1 m, speed < 0.1 m/s |
| Rendezvous | `SpaceRendezvousEnv` | Clohessy–Wiltshire–Hill, symplectic Euler 1 s | 6 | 3 | 1 s/step | 1000 steps ≈ 16.7 min | range < 10 m, speed < 0.5 m/s, keep-out respected |

The six Genesis rigid-body tasks (`quadx`, `quadx_track`, `fixedwing`,
`quadplane`, `rocket`, `missile`) use fully tensorized actuation and contact
layers; the missile's manoeuvring target and the QuadX-Track ground target plus
its camera are analytic geometry advanced each physics substep. The four space
tasks (`satellite`, `transfer`, `docking`, `rendezvous`) are pure-tensor
integrators with no engine.

Full layouts, reward decompositions, termination conditions and spawn
distributions live in each package docstring:
[multirotor](../environments/multirotor/__init__.py) ·
[fixedwing](../environments/fixedwing/__init__.py) ·
[vtol](../environments/vtol/__init__.py) ·
[rocket](../environments/rocket/__init__.py) ·
[missile](../environments/missile/__init__.py) ·
[attitude_control](../environments/satellite/attitude_control/__init__.py) ·
[orbit_transfer](../environments/satellite/orbit_transfer/__init__.py) ·
[rendezvous_docking](../environments/satellite/rendezvous_docking/__init__.py).
Paper-style method write-ups are indexed in
[method/README.md](method/README.md).
