"""Rocket-Landing: reusable-booster propulsive landing task.

Description
    Upstream-faithful port of the PyFlyt 0.29.0 Rocket-Landing task
    (gym_envs/rocket_envs/rocket_landing_env.py + rocket_base_env.py +
    core/drones/rocket.py): only the physics engine differs (Genesis
    rigid-body solver at 240 Hz under a 120 Hz controller under a 40 Hz
    policy, vs PyBullet). A Falcon-9-class booster (549.1 kg dry mass, 20.5
    kg initial propellant = 5% tank, 7.607 kN max / 2.967 kN min thrust =
    39-100% throttle authority, +-5 deg gimbal, four grid fins on dedicated
    URDF fin links) spawns at xy in +/-20 m, z in [400, 450] m, euler
    attitude in +-0.3 rad/axis, falling at vz = -100 m/s, and must relight,
    brake and settle on the landing pad at the origin. The upstream
    "randomize_drop" random linear/angular velocities never fire there (a
    misspelled option key), and that behaviour is reproduced. 1% throttle
    noise; body drag + grid-fin aerodynamics throughout; propellant
    depletion drains tank mass and inertia. One episode spans 1200 policy
    steps (30 s). Fully tensorized: all replicas advance on one device, no
    host synchronization, no gym.

Action Space (7,)
    Upstream Box: finlet roll/pitch/yaw [-1, 1], engine ignition [0, 1]
    (> 0.5 relights each physics step; reignitable), throttle [0, 1]
    (remapped to [0.39, 1.0] of max thrust), gimbal x/y [-1, 1] (mapped to
    +-5 deg).

Observation Space (30,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (7), grid-fin actuations (4),
    ignition flag (1), remaining fuel fraction (1), throttle (1),
    gimbal angles (2), landing-pad contact flag (1)].

Reward
    Upstream dense shaping, per 120 Hz control step:
    r = -0.3                                    stay-airborne cost
        + 0.3 / (||p_xy|| + 0.1)                over-pad holding
        + 10 * (||p_xy_prev|| - ||p_xy||)       lateral progress
        + 0.2 * (z_prev - z)                    descent progress
        + 4 * (vz - vz_prev + 1) / exp(z)       deceleration near the ground
            * (1 if vz < 0 else -1)             (up-drift penalized)
        - |omega_yaw| - ||(roll, pitch)||
    Pad contact pays (+5 - 0.3 |vz|) on every contact step; a landed success
    adds +3.

    Two arms behind the env's `reward_version` argument: 'upstream' (default,
    the reward above, verbatim) and 'v2' (2026-10-21 redesign -- impact-speed
    potential Phi=sqrt(vz^2+2*9.81*z) replaces the exp(z) deceleration term,
    an off-axis cost replaces the farmable holding term, first-touch quality
    bonus and success +100 replace the per-contact-step +5 and the +3; full
    rationale in src/rocket/env.py). All dense components are logged as
    Episode/rew_* scalars in both arms.

Episode End
    Success (truncation): pad contact with |v| < 0.02 m/s, |omega| <
    0.02 rad/s, tilt < 0.1 rad -- judged on the previous control-step speed,
    and the episode CONTINUES on intermediate-speed contact, so the rocket
    may settle on the pad (upstream touch semantics).
    Termination: any non-pad contact (ground) or z < 0, pad contact with
    previous |v| > 1.0 m/s or |omega| > 0.35 rad/s, or bounds ||p_xy|| >
    200 m / z > 500 m.
    Truncation: 1200 policy steps (30 s).

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), Rocket-Landing task
      defaults, rocket and landing-pad vehicle models (task_defaults:
      PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
"""

from .env import RocketLandingEnv
