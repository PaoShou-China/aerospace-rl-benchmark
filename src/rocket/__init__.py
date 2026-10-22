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
    Dense shaping, per 120 Hz control step (2026-10-22: the 'v2' redesign is
    the only arm; the verbatim-upstream shaping was deleted after a full
    1000-update run converged to a farming local optimum on its farmable
    holding term -- rationale in src/rocket/env.py):
    r = -3.0                                    stay-airborne cost
        + 10 * (Phi_prev - Phi)                 impact-speed potential,
            Phi = sqrt(vz^2 + 2*9.81*z)         free fall conserves Phi
        + 10 * (||p_xy_prev|| - ||p_xy||)       lateral progress
        - 0.1 * ||p_xy||                        off-axis centring pressure
        + 0.2 * (z_prev - z)                    descent progress
        - |omega_yaw| - ||(roll, pitch)||       attitude cost
    First pad touch pays 50 * max(0, 1 - touch_speed); while SETTLED on the
    pad after a landed success the env pays +10 per inner step (a success no
    longer truncates -- the episode runs to the 1200-step time limit, so
    landing softly early maximizes the settled stream).

Episode End
    Success (NO truncation since 2026-10-22): pad contact with |v| < 0.02
    m/s, |omega| <
    0.02 rad/s, tilt < 0.1 rad -- judged on the previous control-step speed,
    and the episode CONTINUES on intermediate-speed contact, so the rocket
    may settle on the pad (upstream touch semantics).
    Termination: any non-pad contact (ground) or z < 0, pad contact with
    previous |v| > 1.0 m/s or |omega| > 0.35 rad/s, or bounds ||p_xy|| >
    200 m / z > 500 m.
    Truncation: 1200 policy steps (30 s).

Difficulty
    Constructor kwarg `difficulty` (float in [0, 1]) modulates
    only the initial-condition distribution; the pad, success
    and fatal criteria and reward are untouched. d=0 -> d=1: horizontal
    offset xy ±2 m -> ±20 m, altitude 440-450 m -> 400-450 m (span
    shrinking with difficulty), initial euler ±0.05 -> ±0.3 rad; d=1 is
    the published benchmark.

References
    - PyFlyt (J. K. Tai, github.com/jkuang421/PyFlyt), Rocket-Landing task
      defaults, rocket and landing-pad vehicle models (task_defaults:
      PyFlyt 0.29.0).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
"""

from .env import RocketLandingEnv
