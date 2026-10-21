"""Fixedwing-Waypoints: fixed-wing sequential waypoint navigation task.

Description
    A 2.35 kg fixed-wing UAV (1.6 m span, 0.3 m chord main wing; 18 N
    tractor propeller, thrust-to-weight ~0.78) spawns at (0, 0, 10) m with
    a 20 m/s forward airspeed and must capture four waypoints sampled
    uniformly in a spherical shell (radius 1-90 m, elevation >= 0.5 m)
    around the origin, one at a time. Aerodynamics are blade-element
    lifting surfaces (Cl_alpha_2D = 2*pi, eta = 0.65, stall at +14/-9 deg)
    on the main wing, ±30 deg flapped ailerons and ±20 deg tail surfaces,
    each with a 50 ms first-order actuator lag. Genesis rigid-body engine
    at 240 Hz under a 120 Hz controller under a 30 Hz policy; 2% throttle
    noise. One episode spans 3600 policy steps (120 s). Fully tensorized:
    all replicas advance on one device, no host synchronization, no gym.

Action Space (4,)
    Policy output in [-1, 1]^4:
        aileron  [-1, 1] (differential flaps, roll command)
        elevator [-1, 1] (symmetric tail, pitch command)
        rudder   [-1, 1] (vertical tail, yaw command)
        throttle [-1, 1], remapped to [0, 1]

Observation Space (38,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (4), surface actuations (5),
    remaining-waypoint relative positions in body frame (4 x 3),
    waypoint validity flags (4)].

Reward
    Per control step: r_t = r_{t-1} + 3 * max(d_{t-1} - d_t, 0) + 1 / d_t,
    where d_t is the range to the active waypoint. Waypoint capture
    (d < 2 m) replaces the accumulated reward with 100; ground contact or
    out-of-bounds replaces it with -100. -0.1 per-step living cost.

Episode End
    Termination: ground contact, or ||p|| > 100 m.
    Success (truncation): all 4 waypoints captured.
    Truncation: 3600 policy steps (120 s).

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), Fixedwing-Waypoints
      task defaults and fixedwing vehicle model (task_defaults:
      PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
"""

from .env import FixedwingWaypointsEnv
