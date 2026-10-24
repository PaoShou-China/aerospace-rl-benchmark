"""QuadX-Hover: quadrotor station-keeping reinforcement-learning task.

Description
    A 27 g Bitcraze Crazyflie 2.x quadrotor in "X" motor configuration
    (four rotors, 2.0 N each at full throttle, aggregate 8.0 N, static
    thrust-to-weight ratio ~30) spawns at rest at (0, 0, 1) m and must hold
    station at 1 m altitude. Rigid-body flight dynamics (Genesis rigid-body
    engine, 240 Hz) sit under an inner-loop body-rate PID (120 Hz) under a
    40 Hz policy; rotor thrust is quadratic in a first-order (tau = 10 ms)
    filtered throttle with 2% multiplicative noise, plus body aerodynamic
    drag (-0.5 rho C_d A |v| v). One episode spans 400 policy steps (10 s).
    Fully tensorized: all replicas advance on one device, no engine
    host-synchronization, no gym.

Action Space (4,)
    Policy output in [-1, 1]^4, affinely remapped onto physical bounds:
        roll rate command   [-pi, pi] rad/s
        pitch rate command  [-pi, pi] rad/s
        yaw rate command    [-pi, pi] rad/s
        collective throttle [0, 0.8]
    Rate commands feed a body-rate PID (kp = [0.04, 0.04, 0.08]) whose
    outputs pass through a saturated per-motor PWM mixing matrix.

Observation Space (21,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (4), per-rotor throttle (4)].

Reward
    r_t = -0.1 + 1 - ||p - (0, 0, 1)||_2 - ||(phi, theta)||_2
          - 0.01 * psi_dot^2
    On ground contact or out-of-bounds, the -0.1 station-keeping term is
    replaced by -100 for that inner step. Cumulative hover reward is the
    performance metric; no discrete success flag exists (the original task
    has no win condition), so completed_successes stays 0.

Episode End
    Termination: ground contact, or ||p|| > 3 m.
    Truncation: 400 policy steps (10 s).

Difficulty
    None. The benchmark spawn is deterministic; optional manual
    domain-randomization knobs (spawn_*_noise attrs) and an optional wind
    disturbance (steady_wind/dryden ctor kwargs) exist; calm air keeps the
    default path bit-for-bit with the benchmark.

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), QuadX-Hover task
      defaults and cf2x vehicle model (task_defaults: PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
"""

from .env import QuadXWaypointsEnv
