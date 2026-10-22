"""Toy pendulum swing-up (gym-classic Pendulum-v1 spec) for RL-algorithm
sanity checks.

Description
    Minimal single-DOF control task used to verify the correctness of the
    training stack (rsl-rl PPO / FlashSAC on the fully tensorized batched
    env protocol) before spending compute on the real tasks. A point-mass
    pendulum starts at a random angle and must be swung up and held
    upright by applying torque at the pivot:
        th'' = -3 g / (2 l) sin(th + pi) + 3 u / (m l^2)
    with g = 10, m = 1, l = 1, torque u = 2 * clip(a, -1, 1) N m, |th'| <=
    8 rad/s, explicit Euler at dt = 0.05 exactly as gym (the position
    integrates the *unclipped* derivative before the velocity clamp).
    th = 0 is the upright position. Fully tensorized: all replicas advance
    on one device, no engine, no gym.

Action Space (1,)
    Normalized torque command in [-1, 1], scaled to the 2 N m torque limit.

Observation Space (3,)
    float32, 'policy' TensorDict group: [cos(th), sin(th), th'].

Reward
    r = -(th_norm^2 + 0.1 th'^2 + 0.001 u^2) with th_norm the angle
    wrapped to [-pi, pi] (gym Pendulum-v1 reward, negated cost). Hovering
    at the bottom pays about -pi^2/3 per step; holding upright pays ~0.

Episode End
    No early termination: every episode truncates at 200 steps and the
    runner bootstraps the timeout values via extras['time_outs'] /
    terminal_observations, so the toy exercises the truncation-bootstrap
    path of the training stack. Success metric: completed-episode return
    > -200 (do-nothing baseline ~ -680, swing-up-and-hold ~ -160); a
    healthy implementation clears the bar within ~100-200 updates.

Difficulty / Curriculum
    Constructor kwargs `difficulty` (float in [0, 1]) and `curriculum`
    (bool) follow the project-wide contract of src/common/curriculum.py:
    only the reset distribution is modulated; dynamics and reward are
    untouched. d=0 -> d=1: th0 ±30° -> ±180°, thdot0 ±0.25 -> ±1 rad/s;
    d=1 is the published benchmark.
"""

from .env import PendulumEnv
