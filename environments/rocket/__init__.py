"""Rocket-Landing: reusable-booster propulsive landing task.

Description
    Upstream-faithful port of the PyFlyt 0.29.0 Rocket-Landing task
    (gym_envs/rocket_envs/rocket_landing_env.py + rocket_base_env.py +
    core/drones/rocket.py): only the physics engine differs (Genesis
    rigid-body solver at 240 Hz under a 120 Hz controller under a 40 Hz
    policy, vs PyBullet). A Falcon-9-class booster (138.2 kg dry
    structural mass; the 5% fuel spawn fill adds 20.5 kg, so ~158.7 kg at
    spawn and 549.1 kg fully fuelled; 7.607 kN max / 2.967 kN min thrust =
    39-100% throttle authority, +-5 deg gimbal, four grid fins on
    dedicated URDF fin links) spawns at xy in +/-20 m, z in [400, 450] m,
    euler attitude in +-0.3 rad/axis, falling at vz = -100 m/s, and must
    relight, brake and settle on the landing pad at the origin. The
    upstream "randomize_drop" random linear/angular velocities never fire
    there (a misspelled option key), and that behaviour is reproduced. 1%
    throttle noise; body drag + grid-fin aerodynamics throughout;
    propellant depletion drains tank mass and inertia. One episode spans
    1200 policy steps (30 s). Fully tensorized: all replicas advance on
    one device, no host synchronization, no gym.

Action Space (7,)
    Upstream finlet-mixer Box: finlet x/y/yaw [-1, 1], mixed by the plant
    onto the four grid-fin hinge servos (fin_pos_x <- y+yaw,
    fin_neg_x <- y-yaw, fin_pos_y <- x-yaw, fin_neg_y <- x+yaw); engine
    ignition [0, 1] (> 0.5 relights each physics step; reignitable);
    throttle [0, 1] (remapped to [0.39, 1.0] of max thrust); gimbal x/y
    [-1, 1] (mapped to +-5 deg).

Observation Space (30,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (7), grid-fin actuations (4),
    ignition flag (1), remaining fuel fraction (1), throttle (1),
    gimbal angles (2), landing-pad contact flag (1)].

Reward
    Dense shaping, one term per POLICY step (40 Hz) from whole-step state
    deltas (2026-10-24 'v4'; the earlier v2 potential design was farmed by
    minimum-speed descent):
    r = -HOVER_COST                              0.3, time pressure
        + HOLD_COEF / (||p_xy|| + 0.1)           0.005, capped 0.05
        + LATERAL_COEF * (||p_xy_prev|| - ||p_xy||)     10
        + DESCENT_COEF * (z_prev - z)            0.1
        + DECEL_COEF * (dvz + 1) exp(-max(z,0)) sign(vz)  0.1
        - YAW_COEF * |w_yaw| - TILT_COEF * ||(roll, pitch)||  0.5 / 0.5
        + CONTACT_BASE - CONTACT_VEL_COEF * |vz|  0.5 - 1.0 |vz| on any
          pad-contact policy step, |vz| at the first contact inner step
        + SUCCESS_BONUS once on first settle (5000)
    All coefficients are constructor-overridable via the `reward` dict and
    recorded in cfg. The hold term cannot farm hover, the descent term
    keeps a pure fast dive net-negative, and the success bonus dominates
    any delayed-contact stream.

Episode End
    Success (NO truncation): pad contact with the PREVIOUS inner-step
    speed < 0.02 m/s and spin < 0.02 rad/s, plus a current tilt < 0.1 rad.
    The episode CONTINUES, so the rocket may settle and keep earning the
    contact stream until the time limit.
    Termination: any non-pad contact (ground) or z < 0; a pad-contact
    step whose previous inner-step speed > 1.0 m/s or spin > 0.35 rad/s;
    bounds ||p_xy|| > 200 m or z > 500 m.
    Truncation: 1200 policy steps (30 s).

Difficulty
    None. The initial-condition distribution is the upstream drop (xy ±20 m,
    altitude 400-450 m, initial euler ±0.3 rad); the published benchmark.
    An optional wind disturbance (steady_wind/dryden ctor kwargs) exists;
    calm air is bit-for-bit with the benchmark.

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), Rocket-Landing task
      defaults, rocket and landing-pad vehicle models (task_defaults:
      PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
    - Method documentation: docs/method/env-rocket-landing.md.
"""

from .env import RocketLandingEnv
