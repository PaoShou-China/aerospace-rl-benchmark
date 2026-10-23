"""Spacecraft proximity operations: docking and rendezvous tasks
(Clohessy-Wiltshire-Hill relative motion).

Description
    Two benchmark tasks for spacecraft proximity operations in the LVLH
    frame of a circular 400 km target orbit (a = 6778 km, orbital rate
    n = 1.13e-3 rad/s, period ~92.6 min). A 500 kg chaser with three-axis
    10 N-per-axis RCS thrusters (Isp = 300 s; the paper leaves Isp
    symbolic) manages a task-scaled propellant budget, tracked in real
    time (docking 2 kg, rendezvous 4 kg). Relative
    motion follows the Clohessy-Wiltshire-Hill equations (x radial,
    y along-track, z cross-track):
        xddot = 3 n^2 x + 2 n ydot + Fx / m
        yddot = -2 n xdot + Fy / m
        zddot = -n^2 z + Fz / m
    integrated with the paper's symplectic Euler scheme at dt = 1 s
    (Algorithm 1). At reset the initial position is a uniform sphere
    direction scaled by the task's separation range and the initial
    velocity an independent uniform direction scaled to the task's
    |v0| bound. Fully tensorized: all replicas advance on one device,
    no engine, no gym.

    Docking (SpaceDockingEnv)
        Final approach from 50-100 m separation with |v0| <= 0.5 m/s.
        Success: range < 1 m and closing speed < 0.1 m/s. Failure: range
        > 200 m. Episode limit 500 steps (~8.3 min).
    Rendezvous (SpaceRendezvousEnv)
        Medium-range approach from 800-1200 m with |v0| <= 1 m/s to a
        < 10 m, < 0.5 m/s goal while avoiding a 5 m radius keep-out
        sphere around the target. Failure: keep-out violation or range
        > 2000 m. Episode limit 1000 steps (~16.7 min).

Action Space (3,)
    Thrust-direction commands in [-1, 1]^3, scaled to the 10 N per-axis
    RCS limit; the propellant budget bounds total impulse, though a full
    three-axis burn sustained for an entire episode consumes only ~2.9 kg
    (docking) / ~5.9 kg (rendezvous), so range and keep-out violations
    are the binding failure modes.

Observation Space (6,)
    float32, unbounded raw state, 'policy' TensorDict group:
    [relative position in LVLH (3), relative velocity in LVLH (3)].

Reward
    Telescoping potential shaping per control step (both tasks):
    r_t = progress_weight * (Phi(s_t) - Phi(s_{t+1})) - gamma * ||F|| / F_max
    with Phi = -(||p|| + ||v||^2 / (2 a_ref)) / d_norm, a_ref = 0.6 * F_max/m,
    progress_weight = 100, gamma = 0.01; d_norm = 100 m for docking and the
    per-episode initial separation d0 for rendezvous. The episode sum
    telescopes to Phi_T - Phi_0: hovering pays ~0, closing distance and
    braking pay +, retreating pays -, and round trips net 0 (the paper's
    Eq. 4/5 dense shaping made early failure cheaper than a careful
    approach and is not used). A +100 terminal bonus is granted on success;
    violations (including rendezvous keep-out entry) cost a one-time
    -100 terminal penalty instead of a per-step fee.

Episode End
    Success (truncation): docking range < 1 m & speed < 0.1 m/s;
    rendezvous range < 10 m & speed < 0.5 m/s.
    Termination (failure): docking range > 200 m; rendezvous keep-out
    sphere violation or range > 2000 m.
    Truncation: 500 steps (docking) / 1000 steps (rendezvous).

Difficulty
    None. The benchmark spawn distributions are fixed: docking 50-100 m,
    |v0| <= 0.5 m/s; rendezvous 800-1200 m, |v0| <= 1 m/s.

References
    - Supreeth, H. S. G. & Mehta, "SpaceGym: A Gymnasium-Based Benchmark
      for Deep Reinforcement Learning in Spacecraft Proximity
      Operations", CVPR 2026 Workshop on AI4Space (CC BY 4.0) --
      dynamics, task table and reward weights; reproduced from the
      paper's Section 3 as no public reference implementation exists.
    - Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance
      System for Satellite Rendezvous", Journal of the Aerospace
      Sciences, 27(9), 653-658 -- the linearized relative-motion model.
"""

from .env import SpaceDockingEnv, SpaceRendezvousEnv
