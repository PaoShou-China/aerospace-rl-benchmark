"""Missile-Intercept: 6-DOF tail-controlled missile vs kinematic target.

Description
    Project-authored air-to-air intercept task (task_defaults: project,
    informed by BVR Gym (arXiv:2403.17533), LAG (arXiv:2506.14164) and the
    PPO-guidance literature's overload-command/LOS-rate convention). A
    150 kg missile (body +x nose, four cross tail fins, 20 kN boost for the
    first 4 s then unpowered coast, body drag throughout) flies on the shared
    Genesis TensorPlant at 240 Hz physics / 120 Hz rate-PID / 60 Hz policy.
    The target is a kinematic 3-DOF aircraft (250 m/s around 4 km altitude)
    advanced analytically each physics substep; a collision-free visual
    entity mirrors it in the scene for rendering only. One episode spans 900
    policy steps (15 s). Fully tensorized: all replicas advance on one
    device, no host synchronization, no gym.

Action Space (3,)
    Body-axis rate commands in [-1, 1], affinely mapped per axis: roll to
    +-6 rad/s (env.OMEGA_MAX), pitch/yaw to +-1.0 rad/s (env.OMEGA_LAT --
    transient terminal guidance authority above the steady a/V turn rate).
    The plant's rate PID converts them to
    normalized (roll, pitch, yaw) fin commands expanded by the skid-to-turn
    mixer onto the four tail fins. No throttle: the boost-coast profile is
    autonomous (ignition fires on the first active physics step; burn ends
    after boost_params.burn_time).

Observation Space (34,)
    float32, unbounded, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), position (3), last action (3), fin actuations (4),
    ignited flag (1), engine burning (1), burn-time left fraction (1),
    target relative
    position in body frame (3), relative velocity in body frame (3), LOS
    rate in body frame (3), closing speed Vc (1), target speed (1)].

Reward
    Per inner (120 Hz) step, active replicas only:
    r = +0.01 * (d_prev - d)      telescoping closing-progress potential
        - 0.5  * ||LOS rate||_body    PN-consistency pressure
        - 0.01 * ||action||^2         rate-command effort
        + 50   on kill (once)         -10 on unrecoverable miss (once)
    Components are logged as Episode/rew_{progress,los,ctrl,kill,miss}.

Episode End
    Success (completes + truncates immediately): miss distance <
    environments.missile.env.KILL_RADIUS = 15 m (proximity fuse) at ANY
    physics substep.
    Termination (failures): the closest-approach point passes (closing rate
    positive after 2 s of flight), |p| > 15 km, or z < 0.
    Truncation: 900 policy steps (15 s).

Engagement Geometry
    Single fixed medium operating point (2026-10-24 user decision): initial
    range 2500 m, target-heading cone
    half-width +-37.5 deg around the missile bearing, target lateral
    acceleration 7.5 g, weave frequency 0.75 Hz. The 15 m kill radius is
    FIXED and never scaled.

References
    - BVR Gym (arXiv:2403.17533): PN-guided BVR missile modelling practice.
    - LAG (arXiv:2506.14164): missile dodge/intercept RL environment design.
    - IEEE "Design of Impact Angle Constrained Guidance Law Based on Deep
      Reinforcement Learning" (PPO, overload/LOS state convention).
    - Genesis (github.com/Genesis-Embodied-AI/Genesis), rigid-body engine.
    - rsl_rl (github.com/leggedrobotics/rsl_rl), PPO training backend.
"""

from .env import MissileInterceptEnv
