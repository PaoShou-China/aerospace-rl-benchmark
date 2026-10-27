"""QuadX-Waypoints: quadrotor sequential waypoint navigation task.

Description
    A 27 g Bitcraze Crazyflie 2.x quadrotor in "X" motor configuration
    (four rotors, 2.0 N aggregate thrust at full throttle, static
    thrust-to-weight ~7.6) spawns at rest at (0, 0, 1) m and must capture
    four waypoints in sequence. Waypoints follow the upstream PyFlyt polar
    sampling inside a 5 m flight dome: distance in [1.5, 4.8] m, altitude
    floored at 0.4 m, and a 0.2 m capture sphere. Rigid-body flight
    dynamics (Genesis rigid-body engine, 240 Hz) sit under an inner-loop
    body-rate PID (120 Hz) under a 40 Hz policy; rotor thrust is quadratic
    in a first-order (tau = 10 ms) filtered throttle with 2% multiplicative
    noise, plus body drag (-0.5 rho C_d A |v| v). One episode spans 800
    policy steps (20 s). A vel_clamp=(15, 30) guard caps world speed and
    body spin before each physics step so exploration-driven tumbling
    cannot NaN the contact solver; nominal waypoint flight stays well
    inside it. Fully tensorized: all replicas advance on one device, no
    engine host-synchronization, no gym.

Action Space (4,)
    Policy output in [-1, 1]^4, affinely remapped onto physical bounds:
        roll rate command   [-pi, pi] rad/s
        pitch rate command  [-pi, pi] rad/s
        yaw rate command    [-pi, pi] rad/s
        collective throttle [0, 0.8]
    Rate commands feed a body-rate PID (kp = [0.04, 0.04, 0.08]) whose
    outputs pass through a saturated per-motor PWM mixing matrix.

Observation Space (37,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (4), per-rotor throttle (4),
    remaining-waypoint positions relative to the body (4 x 3, invalid
    slots zeroed), waypoint validity flags (4)].

Reward
    Potential-based telescoping shaping. With d_i the range to the active
    waypoint and L(i) the remaining leg-length suffix,
        Phi(s) = -(d_i + L(i)),  and 0 once all four waypoints are captured;
        r_t = Phi(s_{t+1}) - Phi(s_t) + 15 * c_t - 10 * kappa_t,
    where c_t counts captures during the policy step (capture is tested at
    120 Hz, so one policy step may take several) and kappa_t marks a crash
    or out-of-bounds event. The shaping telescopes to the flown route
    length: progress pays its true value, orbiting a waypoint nets exactly
    zero, and a crash or timeout forfeits the unrealized potential. The
    +15 per waypoint keeps a crisp sparse event; the -10 sharpens failure.

Episode End
    Termination: ground contact, or ||p|| > 5 m.
    Success (truncation): all four waypoints captured ('complete').
    Truncation: 800 policy steps (20 s).

Difficulty
    None. Spawn is deterministic ((0, 0, 1) m, at rest, level); waypoints
    are the only reset randomness. Manual domain-randomization knobs
    (spawn_*_noise attrs) exist and are all zero in the benchmark, and an
    optional wind disturbance (steady_wind/dryden ctor kwargs) is
    available; calm air keeps the default path bit-for-bit with the
    benchmark.

Reference
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), QuadX-Waypoints task
      defaults and cf2x vehicle model (task_defaults: PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
    - Method documentation: docs/method/env-quadx-waypoints.md.

QuadX-Track
    The same cf2x plant, inner rate PID, action interface and 40 Hz policy
    also drive ``QuadXTrackEnv`` (task id ``quadx_track``): persistent
    observation of a cooperative ground moving target under geometric
    camera FOV, slant-range, occlusion and position-noise limits. See
    track_env.py and docs/method/env-quadx-track.md.
"""

from .env import QuadXWaypointsEnv
from .track_env import QuadXTrackEnv
