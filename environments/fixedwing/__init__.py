"""Fixedwing-Waypoints: fixed-wing sequential waypoint navigation task.

Description
    A 2.35 kg fixed-wing UAV (1.6 m span, 0.3 m chord main wing; 18 N
    tractor propeller, thrust-to-weight ~0.78) spawns at (0, 0, 10) m with
    a 20 m/s forward airspeed and must capture four waypoints sampled
    uniformly in a spherical shell (radius 20-65 m, altitude >= 20 m)
    around the origin, one at a time. The shell leaves 35 m of margin to
    the 100 m out-of-bounds sphere and keeps targets out of ground effect.
    Aerodynamics are blade-element lifting surfaces (Cl_alpha_2D = 2*pi,
    eta = 0.65, stall at +14/-9 deg)
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
    Channel a2 is dead by upstream design (it maps to the main wing, whose
    deflection limit is zero, so the plant multiplies it by 0): the
    effective action dimensionality is 3. The mapping is retained verbatim
    for parity and must not be "fixed".

Observation Space (39,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (4), surface actuations (5,
    ailerons L/R, horizontal/vertical tail, main wing), throttle (1),
    remaining-waypoint positions relative to the body (4 x 3, invalid
    slots zeroed), waypoint validity flags (4)].

Reward
    Potential-based telescoping shaping. With d_i the range to the active
    waypoint and L(i) the remaining leg-length suffix,
        Phi(s) = -(d_i + L(i)),  and 0 once all four waypoints are captured;
        r_t = Phi(s_{t+1}) - Phi(s_t) + 15 * c_t - 10 * kappa_t,
    where c_t counts captures during the policy step (capture is tested at
    120 Hz) and kappa_t marks a crash or out-of-bounds event. The shaping
    telescopes to the flown route length: progress pays its true value,
    orbiting a waypoint nets exactly zero, and a crash or timeout forfeits
    the unrealized potential. The +15 per waypoint keeps a crisp sparse
    event; the -10 sharpens failure. A capture is a 10 m sphere around the
    waypoint (about one minimum turn radius).

Episode End
    Termination: ground contact, or ||p|| > 100 m.
    Success (truncation): all 4 waypoints captured.
    Truncation: 3600 policy steps (120 s).

Difficulty
    None. Waypoints spawn on the 20-65 m shell (the published benchmark;
    altitude floor 20 m). Manual spawn-noise knobs are absent and an
    optional wind disturbance (steady_wind/dryden ctor kwargs) exists;
    calm air is bit-for-bit with the benchmark.

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), Fixedwing-Waypoints
      task defaults and fixedwing vehicle model (task_defaults:
      PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
    - Method documentation: docs/method/env-fixedwing-waypoints.md.
"""

from .env import FixedwingWaypointsEnv
