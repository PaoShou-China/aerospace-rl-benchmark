"""Satellite attitude control via control moment gyro (CMG) steering.

Description
    A rigid spacecraft bus carries a cluster of single-gimbal control
    moment gyros -- either a 4-CMG pyramid (gimbal skew 54.7356 deg =
    arctan(sqrt 2), bus inertia (75, 60, 113) kg m^2, 0.2 kg m^2 rotors
    spun at 100 rad/s, stored angular momentum h = 20 kg m^2/s) or a
    6-CMG orthogonal cluster (one CMG per bus face, inertia
    (0.055, 0.055, 0.085) kg m^2, rotors at 200 rad/s, h = 0.0125 kg
    m^2/s). At reset the bus attitude, body rate and gimbal angles are
    perturbed by 1e-2-scale noise from identity (gimbals about the
    momentum-neutral home configuration), and a target attitude is
    sampled uniformly on SO(3). The agent slews the bus to the target
    while steering around internal cluster singularities, measured by the
    manipulability m(J) = sqrt(det(J J^T)) / h^3 of the momentum Jacobian
    J with columns h (g_i x r_i). Reduced-order rigid-body + CMG dynamics,
    I wdot = -w x (I w) - w x (h sum r_i) - h sum (dtheta_i (g_i x r_i)),
    integrated with RK4 at 2.5 ms over 10 substeps per control step
    (40 Hz policy; the torchrl reference runs the same physics at 100 Hz).
    Fully tensorized: all replicas advance on one device, no engine,
    no gym. One episode spans 1200 policy steps (30 s).

Action Space (num_cmgs,)
    Gimbal-rate commands in [-1, 1]^N (N = 4 or 6), scaled to
    +-3 rad/s.

Observation Space (14 + 3 * num_cmgs,)
    float32, unbounded, 'policy' TensorDict group:
    [current quaternion wxyz (4), attitude-error log vector (3), body
    rate (3), sin(gimbal angles) (N), cos(gimbal angles) (N), gimbal
    rates (N), normalized manipulability (1), unit-torque allocation
    direction along the attitude-error axis (3)]. The bus quaternion is
    scalar-first (self.q is initialized to [1, 0, 0, 0]).
    26-dim for the 4-CMG pyramid, 32-dim for the 6-CMG cluster.

Reward
    Telescoping potential-based progress bonus (docking/rendezvous
    'progress' convention): r_t = Phi(s_t) - Phi(s_t+1) - 0.0005 |u|^2,
    plus the one-time terminal terms below, with the positive
    distance-to-go potential
    Phi = 30 att_err + 70 att_err g_n(att_err)
          + 30 |omega|^2/(2 alpha_ref) g(att_err) + 0.5/(m(J) + 0.5)
    where alpha_ref = 1.0 rad/s^2, g = sigmoid((0.8 - att_err)/0.15)
    switches the brake-distance term on inside a 0.8-rad neighborhood
    of the target, g_n = sigmoid((0.5 - att_err)/0.15) is the
    near-field boost gate (error slope 30 -> 100 inside 0.5 rad) and
    m(J) is the cluster manipulability (singularity-avoidance term).
    Success pays a one-time settle bonus of +1.0 and a runaway spin a
    one-time penalty of -10; the |u|^2 control cost uses the scaled
    gimbal rates (0.0045/step per gimbal at full scale). Approaching
    the target, braking near it and escaping low-manipulability
    configurations earn positive reward; the episode sum telescopes
    to Phi_0 - Phi_T regardless of path.

Episode End
    Success is checked every step: attitude error < 0.05 rad and body
    rate < 0.02 rad/s complete the episode immediately (marked
    truncated but not terminated) and pay the one-time settle bonus.
    A runaway spin |omega| > 10 rad/s terminates as a failure with the
    one-time penalty; the telescoping reward alone is failure-neutral.
    Truncation: 1200 policy steps (30 s).
    extras['time_outs'] = truncated & ~terminated & ~complete, so only
    plain horizon truncations are bootstrapped.

Difficulty
    None. The target attitude is sampled uniformly on SO(3); the
    initial attitude, body rate and gimbal angles get ±0.01 state
    perturbations (2026-10-24: raised from ±0.001 -- the baseline sat
    at 100% success, so initial |omega| up to ~0.017 rad/s now sits
    just under the 0.02 success rate tolerance and must be actively
    damped).

References
    - torchrl SatelliteEnv (pytorch/rl, MIT), task reference for reward
      weights, tolerances and reset distribution.
    - Wie, B. (2008). Space Vehicle Dynamics and Control, AIAA
      Education Series -- CMG momentum geometry and the singularity
      (manipulability) metric.
    - Method documentation: docs/method/env-satellite-attitude.md.
"""

from .env import SatelliteAttitudeEnv
