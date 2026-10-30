"""Classical guidance/control baselines for every task, one per environment.

Each controller is a batched, device-resident policy that reads *only* the
'policy' observation group of the corresponding environment and returns an
action in the env's native [-1, 1] command convention.  Plant constants
(mass, inertia, thrust limits, CMG geometry, nominal orbit) are taken from the
env at construction time, mirroring the model knowledge a classical GNC
designer would have; no internal state buffer is read.

Methods
    quadx       SE(3) geometric control (Lee et al. 2010): force/torque wrench
                allocated onto the four rotors, mapped to rate+collective cmds
    fixedwing   L1 lateral guidance + TECS total-energy control
    rocket      powered-descent guidance (ZEM/ZEV) + TVC/fin attitude hold
    satellite   quaternion PD + singularity-robust damped CMG steering
    transfer    Gauss planetary-equation tangential-thrust guidance
    docking     CWH-LQR terminal approach (exact symplectic discretization,
                Bryson Q/R on the task tolerances, per-axis saturation,
                200 m range-corridor shield, terminal capture net, fuel derate)
    rendezvous  receding-horizon CWH two-impulse terminal guidance (exact
                symplectic STM, online transfer-time ratchet) to a hold point
                between the keep-out and success spheres
    missile     augmented proportional navigation (APN, N=15, half-N gravity
                compensation, first-order LOS-rate filter) mapped through the
                skid-to-turn kinematic inverse to body-rate commands

All formulas are analytic; no training, no tuning loop, no host-device
synchronization.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import logm, solve_discrete_are
from scipy.optimize import linear_sum_assignment
from scipy.signal import place_poles


# --------------------------------------------------------------------------- #
# small batched helpers
# --------------------------------------------------------------------------- #
def _vec(obs):
    """Raw policy tensor from a TensorDict or a plain tensor."""
    return obs['policy'] if hasattr(obs, 'keys') and 'policy' in obs.keys() else obs


def _quat_to_R(q):
    """xyzw -> body-to-world rotation matrix, (..., 3, 3)."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack((
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)), -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def _vee(S):
    """Skew-symmetric part of a 3x3 matrix, as a 3-vector."""
    return torch.stack((S[..., 2, 1] - S[..., 1, 2],
                        S[..., 0, 2] - S[..., 2, 0],
                        S[..., 1, 0] - S[..., 0, 1]), -1) * .5


def _cross_cols(a, b):
    """Row-wise cross product, (..., 3)."""
    return torch.linalg.cross(a, b, dim=-1)


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _rot_to_rpy(R):
    """ZYX Euler angles (roll, pitch, yaw) from a body-to-world matrix."""
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1., 1.))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw


def _cwh_lqr(env, q_pos, q_vel, r_acc, dt=None, method='symplectic'):
    """Discrete CWH-LQR on x=[p, v] with u = acceleration [m/s^2].

    Continuous plant (LVLH: x radial, y along-track, z cross-track):
        acc = G p + H v + u,  G = diag(3n^2, 0, -n^2),
        H = [[0, 2n, 0], [-2n, 0, 0], [0, 0, 0]].

    Discretization
        'symplectic'  exact match of the env's symplectic Euler integrator
                      (v+ = v + dt(Gp + Hv + u); p+ = p + dt v+), i.e. the
                      LQR model has ZERO discretization error against the
                      plant for piecewise-constant u:
                          Ad = [[I + dt^2 G, dt(I + dt H)], [dt G, I + dt H]]
                          Bd = [[dt^2 I], [dt I]]
        'taylor2'     truncated matrix exponential (I + A dt + (A dt)^2 / 2),
                      kept only to reproduce the legacy incumbent gains.

    Returns a float64 torch gain matrix K (3x6) such that u = -K x.
    """
    n = env.n
    dt = env.dt if dt is None else dt
    G = np.zeros((3, 3)); G[0, 0] = 3 * n * n; G[2, 2] = -n * n
    H = np.zeros((3, 3)); H[0, 1] = 2 * n; H[1, 0] = -2 * n
    I3 = np.eye(3)
    if method == 'symplectic':
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = I3 + dt * dt * G
        Ad[:3, 3:] = dt * (I3 + dt * H)
        Ad[3:, :3] = dt * G
        Ad[3:, 3:] = I3 + dt * H
        Bd = np.zeros((6, 3))
        Bd[:3] = dt * dt * I3
        Bd[3:] = dt * I3
    else:
        Ac = np.zeros((6, 6)); Ac[:3, 3:] = I3
        Ac[3:, :3] = G; Ac[3:, 3:] = H
        Bc = np.zeros((6, 3)); Bc[3:] = I3
        M = Ac * dt
        Ad = np.eye(6) + M + .5 * M @ M
        Bd = (np.eye(6) * dt + .5 * Ac * dt * dt) @ Bc
    Q = np.diag([q_pos] * 3 + [q_vel] * 3)
    R = np.eye(3) * r_acc
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return torch.as_tensor(K, device=env.device, dtype=torch.float64)


def _cwh_symplectic_Ad(env):
    """One-step transition matrix of the env's symplectic Euler CWH plant.

    The plant integrates (LVLH: x radial, y along-track, z cross-track)
        v+ = v + dt (G p + H v + u),   p+ = p + dt v+,
    with G = diag(3n^2, 0, -n^2) and H = [[0, 2n, 0], [-2n, 0, 0], [0, 0, 0]],
    so the unthrusted step map is
        Ad = [[I + dt^2 G, dt(I + dt H)], [dt G, I + dt H]].
    Powers Ad^k propagate the unthrusted plant EXACTLY over k control steps
    (zero model error, unlike the continuous exponential).
    """
    n = env.n
    dt = env.dt
    G = np.diag((3 * n * n, 0., -n * n))
    H = np.array(((0., 2 * n, 0.), (-2 * n, 0., 0.), (0., 0., 0.)))
    I3 = np.eye(3)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = I3 + dt * dt * G
    Ad[:3, 3:] = dt * (I3 + dt * H)
    Ad[3:, :3] = dt * G
    Ad[3:, 3:] = I3 + dt * H
    return Ad


def _cwh_discrete_model(env, dt=None):
    """Exact symplectic-Euler CWH model (Ad, Bd) with u = acceleration.

    Stacked state x = [p, v]; the plant advances
        v+ = v + dt (G p + H v + u),   p+ = p + dt v+,
    hence
        Ad = [[I + dt^2 G, dt (I + dt H)], [dt G, I + dt H]],
        Bd = [[dt^2 I], [dt I]].
    This is bit-for-bit the env's integrator for piecewise-constant u, so the
    design model carries no discretization error.
    """
    n = env.n
    dt = env.dt if dt is None else dt
    G = np.diag((3. * n * n, 0., -n * n))
    H = np.array(((0., 2. * n, 0.), (-2. * n, 0., 0.), (0., 0., 0.)))
    I3 = np.eye(3)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = I3 + dt * dt * G
    Ad[:3, 3:] = dt * (I3 + dt * H)
    Ad[3:, :3] = dt * G
    Ad[3:, 3:] = I3 + dt * H
    Bd = np.zeros((6, 3))
    Bd[:3] = dt * dt * I3
    Bd[3:] = dt * I3
    return Ad, Bd


def _cwh_pole_gain(env, wn, zeta=.9, ratios=(.8, 1.), dt=None):
    """Full-state pole placement on the exact discrete CWH model.

    The six-state model is block-decoupled into the in-plane pair (x, y) and
    the out-of-plane axis z (the CWH coupling matrix H only links x and y).
    Each block gets distinct damped pairs s = -zeta w +- j w sqrt(1-zeta^2),
    with w = wn * ratios[i] in-plane and w = wn out-of-plane, mapped to the
    discrete plane by z = exp(s dt) (exact for the zero-order-hold
    acceleration command).  ``place_poles`` (Kautsky-Nichols-Van Dooren,
    method 'YT') yields K for u = -K x with u in m/s^2.  Placing the blocks
    separately keeps the gain block-diagonal, so a radial error never
    commands spurious cross-track thrust.

    Controllability is checked on the full six-state pair (rank 6) and on
    each block.  Returns (K float64 torch gain 3x6, design report dict);
    raises if a rank is deficient, the placed eigenvalues do not match the
    requested poles, or the closed loop leaves the unit disc.
    """
    Ad, Bd = _cwh_discrete_model(env, dt)
    dt = env.dt if dt is None else dt
    ctrb = np.hstack([np.linalg.matrix_power(Ad, i) @ Bd for i in range(6)])
    rank = int(np.linalg.matrix_rank(ctrb))
    if rank != 6:
        raise ValueError(f'CWH pair is not controllable (rank {rank}/6)')
    ii, oo = [0, 1, 3, 4], [2, 5]
    Aii, Bii = Ad[np.ix_(ii, ii)], Bd[ii][:, [0, 1]]
    Aoo, Boo = Ad[np.ix_(oo, oo)], Bd[oo][:, [2]]
    ctrb_i = np.hstack([np.linalg.matrix_power(Aii, i) @ Bii for i in range(4)])
    ctrb_o = np.hstack([Boo, Aoo @ Boo])
    rank_i = int(np.linalg.matrix_rank(ctrb_i))
    rank_o = int(np.linalg.matrix_rank(ctrb_o))
    if rank_i != 4 or rank_o != 2:
        raise ValueError(f'CWH block controllability deficient '
                         f'(in-plane {rank_i}/4, out-of-plane {rank_o}/2)')

    def _poles(w_list):
        out = []
        for w in w_list:
            s = -zeta * w + 1j * w * math.sqrt(max(0., 1. - zeta * zeta))
            out.extend((np.exp(s * dt), np.exp(s.conjugate() * dt)))
        return np.array(out)

    w_in = [wn * r for r in ratios]
    w_out = [wn]
    poles_in, poles_out = _poles(w_in), _poles(w_out)
    K = np.zeros((3, 6))
    K[np.ix_([0, 1], ii)] = place_poles(Aii, Bii, poles_in, method='YT').gain_matrix
    K[np.ix_([2], oo)] = place_poles(Aoo, Boo, poles_out, method='YT').gain_matrix
    eig = np.linalg.eigvals(Ad - Bd @ K)
    poles = np.concatenate((poles_in, poles_out))
    # order-invariant eigenvalue matching (two blocks share the wn pair)
    cost = np.abs(poles[:, None] - eig[None, :])
    rows, cols = linear_sum_assignment(cost)
    err = float(cost[rows, cols].max())
    if err > 1e-6:
        raise ValueError(f'pole placement residual {err:.2e} exceeds 1e-6')
    max_eig = float(np.max(np.abs(eig)))
    if max_eig >= 1.:
        raise ValueError(f'placed closed loop is unstable (max |z| = {max_eig:.6f})')
    cont_poles = [[float(-zeta * w), float(w * math.sqrt(max(0., 1. - zeta * zeta)))]
                  for w in w_in + w_out]
    report = {'wn': float(wn), 'zeta': float(zeta), 'ratios': list(ratios),
              'controllability_rank': rank, 'controllability_cond': float(np.linalg.cond(ctrb)),
              'block_ranks': {'in_plane': rank_i, 'out_of_plane': rank_o},
              'poles_continuous': cont_poles,
              'poles_discrete': [[float(z.real), float(z.imag)] for z in poles],
              'placement_residual': err, 'max_abs_eig': max_eig,
              'gain': K.tolist()}
    return torch.as_tensor(K, device=env.device, dtype=torch.float64), report


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
class Baseline:
    name = 'baseline'

    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions

    def actions(self, obs):
        raise NotImplementedError

    def reset_idx(self, mask):
        """Mirror the env reset mask (only stateful controllers override)."""


# --------------------------------------------------------------------------- #
# quadx: SE(3) geometric control
# --------------------------------------------------------------------------- #
class QuadXSE3Baseline(Baseline):
    """Geometric tracking on SE(3) (Lee, Leok & McClamroch 2010).

    Position PD + gravity feedforward produce the desired force F_des; the
    collective is its projection on the body thrust axis f = F_des . b3 and the
    desired attitude R_d is built from b3_d = F_hat_des and the commanded yaw.
    Attitude error e_R = 1/2 vee(R_d' R - R' R_d) and body-rate error
    e_O = w - R' R_d w_d (= w for the stationary waypoint setpoint) drive the
    torque law M = -kR e_R - kOm e_O + w x Jw.  The wrench (f, M) is allocated
    onto the four rotors with the frozen cf2x geometry,
        t_i = f/4 + ay_i Mx/(4d) - ax_i My/(4d) + spin_i Mz/(4c),
    (d = rotor arm, c = km/kf), clipped to the plant's total-thrust window and
    per-rotor pwm window [0.05, 1], and mapped onto the env's rate-command +
    collective interface by inverting its mixer (tau_pid = Mmix' diff / 4 with
    diff = pwm - mean(pwm)) and its rate-loop P gain (w_cmd = w + tau/kp).

    One practical augmentation beyond the paper: the commanded pwm vector is
    divided by PWM_GAIN so delivered thrust/differentials match the ideal
    wrench.  The plant's motor-lag state passes through the multiplicative
    motor noise every physics step; with the zero-mean noise fix the lag
    fixed point is exactly the commanded pwm (dc gain 1.0), so PWM_GAIN = 1.
    A small integral on the b3-projected position error then trims residuals,
    freezing under total-thrust saturation (anti-windup) and clearing on
    reset_idx.
    Saturation accounting lives in self.sat_total / self.sat_rotor /
    self.sat_rate (cumulative clipped-command fractions per replica).
    """
    name = 'se3_geometric'

    # frozen cf2x plant constants (reference URDF inertials + vehicle yaml)
    MASS = 0.027                   # kg
    G = 9.81                       # m/s^2, world -z
    ARM = 0.028                    # rotor distance from the body z axis [m]
    KM_KF = 7.94e-12 / 3.16e-10    # reaction-torque / thrust coefficient
    INERTIA = (1.4e-5, 1.4e-5, 2.17e-5)   # base_link diagonal [kg m^2]
    TOTAL_THRUST = 2.0             # N at pwm = 1, all four rotors
    PWM_MIN, PWM_MAX, THR_MAX = 0.05, 1.0, 0.8   # plant mixer / env command
    KP_RATE = (0.04, 0.04, 0.08)   # plant inner rate-PID P gains
    PWM_GAIN = 1.0                 # motor-lag dc gain = 1 under zero-mean noise

    # controller gains
    KX, KV = 4.0, 3.0              # position PD [1/s^2, 1/s]
    V_MAX = 1.5                    # cruise speed cap [m/s]
    KR, KOM = 8e-4, 2.5e-4         # attitude PD [N m/rad, N m s/rad]
    KI_POS = 0.15                  # axial position-error integral [N/m]
    I_MAX = 0.35                   # integral clamp [N]

    def __init__(self, env):
        super().__init__(env)
        d, n = self.device, env.num_envs
        self.dt = 1. / env.agent_hz           # agent control period [s]
        # setpoint = the env's active waypoint (env.current_target); the same
        # PD law tracks it as a stationary setpoint per control step
        self.e1 = torch.tensor((1., 0., 0.), device=d, dtype=torch.float64).expand(n, 3)
        self.e3 = torch.tensor((0., 0., 1.), device=d, dtype=torch.float64).expand(n, 3)
        self.J = torch.tensor(self.INERTIA, device=d, dtype=torch.float64)
        self.kp = torch.tensor(self.KP_RATE, device=d, dtype=torch.float64)
        # rotor geometry: motor order (+x,-y), (-x,+y), (+x,+y), (-x,-y)
        ay = torch.tensor((-1., 1., 1., -1.), device=d, dtype=torch.float64)   # arm_y / d
        ax = torch.tensor((1., -1., 1., -1.), device=d, dtype=torch.float64)   # arm_x / d
        sp = torch.tensor((-1., -1., 1., 1.), device=d, dtype=torch.float64)   # spin signs
        self.rotor_xy = torch.stack((ay, -ax), -1)          # (4, 2) unit arms
        self.mixer_t = torch.stack((ay, -ax, sp), -1)       # (4, 3) mixer torque columns
        self.t_min = self.TOTAL_THRUST * self.PWM_MIN ** 2  # per-rotor floor
        self.t_max = self.TOTAL_THRUST * .25                # per-rotor ceiling
        self.f_min = 4. * self.t_min
        self.f_max = self.TOTAL_THRUST * self.THR_MAX ** 2
        self.trim = torch.zeros(n, device=d, dtype=torch.float64)
        self.sat_total = torch.zeros(n, device=d, dtype=torch.float64)
        self.sat_rotor = torch.zeros((n, 4), device=d, dtype=torch.float64)
        self.sat_rate = torch.zeros((n, 3), device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.trim.masked_fill_(mask, 0.)

    def actions(self, obs):
        return self._track(self.env.current_target, obs)

    def _track(self, reference, obs, v_ff=None):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        R = _quat_to_R(q)                       # body -> world
        b3 = R[:, :, 2]                         # body thrust axis in world
        v_w = torch.einsum('nij,nj->ni', R, vb)
        e_x = p - reference
        # cascade outer loop: position error -> speed-capped velocity setpoint
        # (raw PD accelerates to ~KX/KV*|e_x| ~ 6 m/s toward a 4.5 m waypoint,
        # overshooting the 5 m dome; the cap keeps transit <= V_MAX).  v_ff is
        # an optional world-frame velocity feedforward (used by the moving
        # target tracker; stationary waypoint tasks leave it at zero).
        v_d = -self.KX / self.KV * e_x
        if v_ff is not None:
            v_d = v_d + v_ff
        v_d = v_d * (self.V_MAX / v_d.norm(dim=-1, keepdim=True).clamp_min(self.V_MAX))
        e_v = v_w - v_d            # waypoint setpoint with cruise feedforward

        F_des = self.MASS * (self.G * self.e3 - self.KV * e_v)
        f_raw = (F_des * b3).sum(-1) + self.trim
        # axial trim integral; freeze while the collective command is clipped
        # and outside the capture neighborhood (a multi-meter transit error
        # would wind the trim to its clamp -- ~1.3x hover thrust -- and loft
        # every approach; the trim only exists to cancel millimeter-scale
        # hover residuals)
        near = e_x.norm(dim=-1) < 0.5
        trim_dot = -self.KI_POS * (e_x * b3).sum(-1) * near.to(torch.float64)
        trim_dot = torch.where(f_raw > self.f_max, trim_dot.clamp_max(0.), trim_dot)
        trim_dot = torch.where(f_raw < self.f_min, trim_dot.clamp_min(0.), trim_dot)
        self.trim += trim_dot * self.dt
        self.trim = self.trim.clamp(-self.I_MAX, self.I_MAX)
        sat = (f_raw < self.f_min) | (f_raw > self.f_max)
        self.sat_total += sat.to(torch.float64)
        f = f_raw.clamp(self.f_min, self.f_max)

        # desired attitude: b3_d from the (unclipped) force direction, yaw = 0
        n2 = F_des.norm(dim=-1)
        b3_d = torch.where((n2 > 1e-6)[:, None], F_des / n2[:, None].clamp_min(1e-6), b3)
        e2 = torch.tensor((0., 1., 0.), device=o.device, dtype=torch.float64).expand_as(b3_d)
        b1 = self.e1 - (self.e1 * b3_d).sum(-1, keepdim=True) * b3_d
        # heading projection degenerates only for b3_d ~ +-e1; fall back to e2
        b1 = torch.where((b1 * b1).sum(-1, keepdim=True) < 1e-8,
                         e2 - (e2 * b3_d).sum(-1, keepdim=True) * b3_d, b1)
        b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Rd = torch.stack((b1, torch.linalg.cross(b3_d, b1, dim=-1), b3_d), -1)
        e_R = _vee(Rd.transpose(-1, -2) @ R - R.transpose(-1, -2) @ Rd)
        M = -self.KR * e_R - self.KOM * wb + torch.linalg.cross(wb, self.J * wb, dim=-1)

        # wrench -> rotors (frozen geometry), per-rotor clip, mixer inversion
        t = f[:, None] * .25 + M[:, :2] @ (self.rotor_xy / (4. * self.ARM)).T \
            + M[:, 2:3] * (self.mixer_t[:, 2:3] / (4. * self.KM_KF)).T
        r_sat = (t < self.t_min) | (t > self.t_max)
        self.sat_rotor += r_sat.to(torch.float64)
        t = t.clamp(self.t_min, self.t_max)
        pwm = torch.sqrt(4. * t / self.TOTAL_THRUST) / self.PWM_GAIN
        thr_raw = pwm.mean(-1)
        thr = thr_raw.clamp(0., self.THR_MAX)         # collective channel bound
        self.sat_total += (thr_raw > self.THR_MAX).to(torch.float64)
        diff = pwm - thr[:, None]
        tau = diff @ self.mixer_t * .25               # normalized pid torque
        w_cmd = wb + tau / self.kp
        w_sat = w_cmd.abs() > math.pi
        self.sat_rate += w_sat.to(torch.float64)
        w_cmd = w_cmd.clamp(-math.pi, math.pi)
        return torch.cat((w_cmd / math.pi, (thr / self.THR_MAX * 2. - 1.)[:, None]),
                         -1).clamp(-1., 1.).to(torch.float32)


class QuadXTrackKFBaseline(QuadXSE3Baseline):
    """Constant-velocity Kalman filter + fixed-offset SE(3) tracking.

    Method
        State x = [p_t, v_t] in R^6 (target position and velocity, world
        frame).  Standard linear constant-velocity prediction
            F = [[I, dt I], [0, I]],
            Q = q_a [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]],
        with q_a = SIGMA_ACCEL^2 (continuous white-acceleration PSD), and a
        position-only measurement H = [I, 0], R = SIGMA_MEAS^2 I.
        The filter predicts on every policy step.  When the visibility flag
        is set it updates with y = p_uav + rel_meas (the same 3-D camera
        measurement every other method receives, in world axes); when the
        flag is clear it only predicts.  It starts from the first detection
        with a broad prior -- no target state is read from the environment.

    Control reference
        p_ref = [x_hat + REF_OFFSET_X, y_hat + REF_OFFSET_Y, ALT_REF] with
        the environment's fixed altitude and fixed horizontal offset.  The
        SE(3) position loop tracks p_ref with the estimated target velocity
        as world-frame feedforward, i.e. v_d = v_hat + -(KX/KV)(p - p_ref),
        speed-capped at V_MAX.

    Fixed timeout hold (public, no active search)
        If no detection has arrived for HOLD_TIMEOUT seconds, the reference
        freezes at the estimate held at that instant and the vehicle hovers
        there until a new detection arrives.  There is no search, information
        gathering, or replanning behaviour.

    All filter noise parameters and control gains are constants selected on
    an independent tuning seed set (see environments/multirotor/eval_quadx_track_kf.py).
    """

    name = 'cv_kf_pid'
    SIGMA_ACCEL = 1.5        # m/s^2, white-acceleration std (process model), tuned
    SIGMA_MEAS = 0.15        # m, camera position noise std (matches the sensor)
    VEL_PRIOR_STD = 0.5      # m/s, first-detection velocity prior std
    HOLD_TIMEOUT = 3.0       # s without detection before the fixed hold
    V_MAX = 3.0              # m/s, tracking cruise cap
    KX, KV = 4.0, 2.0        # position PD, tuned on the seed-101 set

    def __init__(self, env, sigma_accel=None, sigma_meas=None, kx=None, kv=None,
                 v_max=None, hold_timeout=None):
        super().__init__(env)
        d, n = self.device, env.num_envs
        if sigma_accel is not None:
            self.SIGMA_ACCEL = float(sigma_accel)
        if sigma_meas is not None:
            self.SIGMA_MEAS = float(sigma_meas)
        if kx is not None:
            self.KX = float(kx)
        if kv is not None:
            self.KV = float(kv)
        if v_max is not None:
            self.V_MAX = float(v_max)
        if hold_timeout is not None:
            self.HOLD_TIMEOUT = float(hold_timeout)

        self.offset = torch.tensor(env.REF_OFFSET_XY, device=d, dtype=torch.float64)
        self.home = torch.tensor((0., 0., env.ALT_REF), device=d, dtype=torch.float64)
        self.alt = torch.tensor(env.ALT_REF, device=d, dtype=torch.float64)
        self.eye3 = torch.eye(3, device=d, dtype=torch.float64)
        self.mu = torch.zeros((n, 6), device=d, dtype=torch.float64)
        self.P = torch.eye(6, device=d, dtype=torch.float64).expand(n, 6, 6).clone() * 1e6
        self.initialized = torch.zeros(n, device=d, dtype=torch.bool)
        self.since_seen = torch.zeros(n, device=d, dtype=torch.float64)
        self.frozen = torch.zeros(n, device=d, dtype=torch.bool)
        self.hold = self.home.expand(n, 3).clone()
        self._build_filter()

    def _build_filter(self):
        dt = self.dt
        F = torch.eye(6, device=self.device, dtype=torch.float64)
        F[:3, 3:] = torch.eye(3, device=self.device, dtype=torch.float64) * dt
        q = self.SIGMA_ACCEL ** 2
        I3 = torch.eye(3, device=self.device, dtype=torch.float64)
        Q = torch.zeros((6, 6), device=self.device, dtype=torch.float64)
        Q[:3, :3] = I3 * (dt ** 3 / 3.)
        Q[:3, 3:] = I3 * (dt ** 2 / 2.)
        Q[3:, :3] = I3 * (dt ** 2 / 2.)
        Q[3:, 3:] = I3 * dt
        self.F = F
        self.Q = Q * q
        prior = torch.zeros((6, 6), device=self.device, dtype=torch.float64)
        prior[:3, :3] = I3 * (self.SIGMA_MEAS ** 2)
        prior[3:, 3:] = I3 * (self.VEL_PRIOR_STD ** 2)
        self.prior = prior

    def reset_idx(self, mask):
        super().reset_idx(mask)
        self.mu.masked_fill_(mask[:, None], 0.)
        self.P = torch.where(mask[:, None, None],
                             torch.eye(6, device=self.device, dtype=torch.float64).expand_as(self.P) * 1e6,
                             self.P)
        self.initialized.masked_fill_(mask, False)
        self.since_seen.masked_fill_(mask, 0.)
        self.frozen.masked_fill_(mask, False)
        self.hold.copy_(torch.where(mask[:, None], self.home.expand_as(self.hold), self.hold))

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p = o[:, 10:13]
        rel = o[:, 21:24]
        vis = o[:, 24] > .5
        dt = self.dt

        # ---- prediction (every step) -------------------------------------
        self.mu = self.mu @ self.F.T
        FP = self.F @ self.P
        self.P = FP @ self.F.T + self.Q

        # ---- first detection: initialize from the measurement --------------
        fresh = vis & ~self.initialized
        y = p + rel
        init_mu = torch.cat((y, torch.zeros((y.shape[0], 3), device=o.device, dtype=torch.float64)), -1)
        self.mu = torch.where(fresh[:, None], init_mu, self.mu)
        self.P = torch.where(fresh[:, None, None], self.prior.expand_as(self.P), self.P)
        self.initialized |= vis

        # ---- measurement update (detections only) -------------------------
        upd = vis & self.initialized & ~fresh
        S = self.P[:, :3, :3] + self.SIGMA_MEAS ** 2 * self.eye3
        K = self.P[:, :, :3] @ torch.linalg.inv(S)
        innov = y - self.mu[:, :3]
        mu_new = self.mu + (K @ innov[..., None])[..., 0]
        P_new = self.P - K @ self.P[:, :3, :]
        P_new = .5 * (P_new + P_new.transpose(-1, -2))
        self.mu = torch.where(upd[:, None], mu_new, self.mu)
        self.P = torch.where(upd[:, None, None], P_new, self.P)

        # ---- fixed timeout hold -------------------------------------------
        self.since_seen = torch.where(vis, torch.zeros_like(self.since_seen),
                                     self.since_seen + dt)
        timed_out = self.initialized & (self.since_seen > self.HOLD_TIMEOUT)
        freeze_now = timed_out & ~self.frozen
        ref_est = torch.cat((self.mu[:, :2] + self.offset,
                             self.alt.expand(self.mu.shape[0], 1)), -1)
        self.hold = torch.where(freeze_now[:, None], ref_est, self.hold)
        self.frozen = timed_out

        ref = torch.where(self.frozen[:, None], self.hold, ref_est)
        ref = torch.where(self.initialized[:, None], ref, self.home.expand_as(ref))
        v_ff = torch.cat((self.mu[:, 3:5], torch.zeros((self.mu.shape[0], 1), device=o.device,
                                                       dtype=torch.float64)), -1)
        v_ff = torch.where((self.initialized & ~self.frozen)[:, None], v_ff,
                           torch.zeros_like(v_ff))
        return self._track(ref, obs, v_ff)


# --------------------------------------------------------------------------- #
# quadx: position -> attitude cascade with the env's inner rate PID
# --------------------------------------------------------------------------- #
class QuadXCascadePIDBaseline(Baseline):
    """Position-attitude cascaded PID on the native rate+collective interface.

    Outer loop (world frame).  A PID on the active-waypoint position error
    produces a desired acceleration
        a_des = -(Kp e_p + Ki int e_p dt + Kd v_w),
    with the derivative taken on the measured world velocity (so a waypoint
    switch is not a derivative kick), a norm clamp A_MAX, an integral clamp
    I_MAX, and conditional integration frozen while the command is clipped.

    Guidance.  Gravity compensation gives the specific-force command
        a_cmd = a_des + g e3.
    With the yaw reference psi_r fixed, resolving a_cmd in the heading frame
    yields the desired roll/pitch and collective
        u = ( a_x cos psi_r + a_y sin psi_r,
             -a_x sin psi_r + a_y cos psi_r,
              a_z ),
        theta_d = atan2(u_x, u_z),   phi_d = atan2(-u_y, sqrt(u_x^2 + u_z^2)),
        f = m |a_cmd|.
    f maps back through the plant's quadratic thrust law and the measured
    motor-lag/noise dc gain
        throttle = sqrt(f / F_total) / G_pwm,
    which is the physical collective (pwm) the env expects.

    Attitude loop.  A P/PD on the ZYX Euler errors produces body-rate
    commands that the env's existing 120 Hz rate PID executes:
        w_cmd = K_att (phi_d - phi, theta_d - theta, psi_r - psi) - K_d w_b,
    clipped to the +-pi rate-command window.

    Action map.  The returned four-vector is the exact inverse of the env's
    affine action scaling: rates / pi and 2 throttle / T_max - 1, clipped to
    [-1, 1].  There is no learning, no prediction, no waypoint lookahead and
    no command-safety layer; the only internal state is the clamped
    position-error integral, cleared on reset and on a waypoint-index change.
    """

    name = 'cascaded_pid'

    # frozen cf2x plant constants (same reference model as QuadXSE3Baseline)
    MASS = 0.027                   # kg
    G = 9.81                       # m/s^2, world -z
    TOTAL_THRUST = 2.0             # N at pwm = 1 on all four rotors
    THR_MAX = 0.8                  # env collective command ceiling
    PWM_GAIN = 1.0                 # motor-lag dc gain = 1 under zero-mean noise

    # outer position PID [s^-2, s^-3, s^-1] and output/integral clamps
    KP_POS = 3.0                   # position error -> acceleration [1/s^2]
    KI_POS = 0.10                  # position integral [1/s^3]
    KD_POS = 4.0                   # measured-velocity damping [1/s]
    A_MAX = 4.0                    # desired-acceleration norm cap [m/s^2]
    I_MAX = 0.5                    # integral clamp [m/s^2]
    # attitude P/PD -> body-rate commands [1/s, dimensionless]
    KP_ATT = 4.0                   # roll/pitch attitude P [1/s]
    KD_ATT = 0.35                  # roll/pitch rate damping [-]
    KP_YAW = 2.0                   # yaw attitude P [1/s]
    KD_YAW = 0.20                  # yaw rate damping [-]
    YAW_REF = 0.0                  # fixed yaw reference [rad]
    RATE_MAX = math.pi             # env rate-command window [rad/s]

    def __init__(self, env, gains=None):
        """``gains`` optionally overrides the hard-coded defaults (tuning tests)."""
        super().__init__(env)
        d, n = self.device, env.num_envs
        self.dt = 1. / env.agent_hz
        g = dict(kp_pos=self.KP_POS, ki_pos=self.KI_POS, kd_pos=self.KD_POS,
                 a_max=self.A_MAX, i_max=self.I_MAX, kp_att=self.KP_ATT,
                 kd_att=self.KD_ATT, kp_yaw=self.KP_YAW, kd_yaw=self.KD_YAW)
        if gains:
            g.update(gains)
        self.kp_pos, self.ki_pos, self.kd_pos = g['kp_pos'], g['ki_pos'], g['kd_pos']
        self.a_max, self.i_max = g['a_max'], g['i_max']
        self.rate_max = self.RATE_MAX
        self.e3 = torch.tensor((0., 0., 1.), device=d, dtype=torch.float64)
        # attitude gains per axis: roll, pitch, yaw
        self.k_att = torch.tensor((g['kp_att'], g['kp_att'], g['kp_yaw']),
                                  device=d, dtype=torch.float64)
        self.kd_att = torch.tensor((g['kd_att'], g['kd_att'], g['kd_yaw']),
                                   device=d, dtype=torch.float64)
        self.f_max = self.TOTAL_THRUST * self.THR_MAX ** 2
        self.i_acc = torch.zeros((n, 3), device=d, dtype=torch.float64)
        self.last_target_index = env.target_index.clone()
        # cumulative clipped-command fractions (diagnostics only)
        self.sat_acc = torch.zeros(n, device=d, dtype=torch.float64)
        self.sat_rate = torch.zeros(n, device=d, dtype=torch.float64)
        self.sat_thrust = torch.zeros(n, device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.i_acc.masked_fill_(mask[:, None], 0.)
        self.last_target_index.masked_fill_(mask, 0)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        R = _quat_to_R(q)                       # body -> world
        roll, pitch, yaw = _rot_to_rpy(R)
        v_w = torch.einsum('nij,nj->ni', R, vb)

        # waypoint handoff: clear the integral when the env advances its index
        idx = self.env.target_index
        switched = idx != self.last_target_index
        self.i_acc.masked_fill_(switched[:, None], 0.)
        self.last_target_index.copy_(idx)
        e_p = p - self.env.current_target

        # outer PID -> desired acceleration (conditional anti-windup)
        a_prev = -self.kp_pos * e_p - self.kd_pos * v_w + self.i_acc
        clipped_prev = a_prev.norm(dim=-1) > self.a_max
        i_new = (self.i_acc + self.ki_pos * e_p * self.dt).clamp(-self.i_max, self.i_max)
        self.i_acc.copy_(torch.where(clipped_prev[:, None], self.i_acc, i_new))
        a_raw = -self.kp_pos * e_p - self.kd_pos * v_w + self.i_acc
        a_norm = a_raw.norm(dim=-1)
        clipped = a_norm > self.a_max
        self.sat_acc += clipped.to(torch.float64)
        a_des = a_raw * (self.a_max / a_norm.clamp_min(self.a_max))[:, None]

        # gravity compensation -> desired roll/pitch and total thrust
        a_cmd = a_des + self.G * self.e3
        f_cmd = self.MASS * a_cmd.norm(dim=-1)
        self.sat_thrust += (f_cmd > self.f_max).to(torch.float64)
        f = f_cmd.clamp_max(self.f_max)
        # env collective: quadratic thrust with the measured lag/noise dc gain
        throttle = (torch.sqrt(f / self.TOTAL_THRUST) / self.PWM_GAIN) \
            .clamp(0., self.THR_MAX)
        cy, sy = math.cos(self.YAW_REF), math.sin(self.YAW_REF)
        u_x = a_cmd[:, 0] * cy + a_cmd[:, 1] * sy
        u_y = -a_cmd[:, 0] * sy + a_cmd[:, 1] * cy
        u_z = a_cmd[:, 2]
        pitch_d = torch.atan2(u_x, u_z)
        roll_d = torch.atan2(-u_y, torch.sqrt(u_x * u_x + u_z * u_z))

        # attitude P/PD -> body-rate command executed by the env's rate PID
        e_att = torch.stack((_wrap(roll_d - roll), _wrap(pitch_d - pitch),
                             _wrap(torch.full_like(yaw, self.YAW_REF) - yaw)), -1)
        w_cmd = self.k_att * e_att - self.kd_att * wb
        self.sat_rate += (w_cmd.abs() > self.rate_max).to(torch.float64).sum(-1)
        w_cmd = w_cmd.clamp(-self.rate_max, self.rate_max)

        # inverse of the env action scaling: [-pi, pi] rates, [0, 0.8] throttle
        return torch.cat((w_cmd / self.rate_max,
                          (2. * throttle / self.THR_MAX - 1.)[:, None]),
                         -1).clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# fixedwing: L1 lateral guidance + TECS + attitude inner loops
# --------------------------------------------------------------------------- #
class FixedwingAutopilotBaseline(Baseline):
    """Classical autopilot: L1 guidance, TECS, PD attitude loops.

    Environment facts this design is built on (verified in code and in
    simulation on 2026-10-21, see trash/fw_verify*.py history):

    * Action interface (TensorPlant fixedwing mapping, action in [-1, 1]^4):
      a0 -> differential ailerons (a0 > 0 rolls right bank), a1 -> horizontal
      tail (+, nose-down moment) AND vertical tail (opposite sign, rudder
      coupling treated as a disturbance), a2 -> main wing with
      deflection_limit = 0 (dead channel, driven with exactly 0), a3 ->
      throttle (a3 + 1) / 2 with thrust = throttle^2 * 18 N.
    * Observation: wb(0:3), quat xyzw(3:7), body velocity vb(7:10), world
      position p(10:13), target deltas in body frame(23:35, slot k = target
      target_index + k), per-target validity(35:39). The tensorized plant
      supports a wind field; aero evaluates v_air = v - wind, so in calm air
      (the default) airspeed = ground speed = |vb|.
    * Env euler convention (x fwd, y left, z up): positive roll = right bank,
      positive pitch = nose down, so nose-up commands are negative pitch.
    * Guidance laws: L1 (Park, Deyst & Howe 2004) turns the line-of-sight
      angle to the active waypoint into a bank command, with a full-strength
      turn when the waypoint is behind (|eta| > 90 deg would otherwise vanish
      through sin(eta)) and a stall guard that trades bank authority for
      airspeed.  TECS (Lambregts 1983, simplified): specific total energy
      error E = h + V^2/2g drives throttle (P + I), energy balance
      B = h - V^2/2g drives the pitch command, with a load-factor feedforward
      for banked turns.  Inner loops are PD on angle with body-rate damping.

    Parameters live in fixedwing_autopilot.json next to this file; they were
    tuned exclusively on training/development initial conditions.
    """
    name = 'l1_tecs'
    CONFIG_PATH = Path(__file__).parent / 'fixedwing_autopilot.json'

    def __init__(self, env):
        super().__init__(env)
        cfg = json.loads(self.CONFIG_PATH.read_text())
        lat, lon = cfg['lateral'], cfg['longitudinal']
        self.g = 9.81
        self.l1 = float(lat['l1_distance'])
        self.l1_dist_gain = float(lat['l1_dist_gain'])
        self.l1_dist_offset = float(lat['l1_dist_offset'])
        self.l1_min = float(lat['l1_min_distance'])
        self.max_bank = math.radians(float(lat['max_bank_deg']))
        self.v_stall = float(lat['v_stall_guard'])
        self.k_roll, self.kd_roll = float(lat['k_roll']), float(lat['kd_roll'])
        self.v_ref = float(lon['v_ref'])
        self.v_turn_min = float(lon['v_turn_min'])
        self.k_speed_bleed = float(lon['k_speed_bleed'])
        self.k_dist_slow = float(lon['k_dist_slow'])
        self.d_slow_ref = float(lon['d_slow_ref'])
        self.k_aw_speed = float(lon['k_aw_speed'])
        self.k_pp = float(lon['k_pp'])
        self.d_pp_entry = float(lon['d_pp_entry'])
        self.eta_pp_gate = math.radians(float(lon['eta_pp_gate_deg']))
        self.bank_climb_rad = math.radians(float(lon['bank_climb_deg']))
        self.v_climb_ref = float(lon['v_climb_ref'])
        self.h_climb_gate = float(lon['h_climb_gate'])
        self.pn_shield = float(lon['pn_shield'])
        self.v_shield_min = float(lon['v_shield_min'])
        self.k_shield = float(lon['k_shield'])
        self.k_energy = float(lon['k_energy'])
        self.k_energy_i = float(lon['k_energy_i'])
        self.integral_max = float(lon['integral_max'])
        self.throttle_trim = float(lon['throttle_trim'])
        self.k_balance = float(lon['k_balance'])
        self.k_speed_keep = float(lon['k_speed_keep'])
        self.k_geo_up = float(lon['k_geo_up'])
        self.h_loiter_gate = float(lon['h_loiter_gate'])
        self.pn_loiter = float(lon['pn_loiter'])
        self.pitch_trim = math.radians(float(lon['pitch_trim_deg']))
        self.pitch_up_max = math.radians(float(lon['pitch_up_max_deg']))
        self.pitch_up_v = (float(lon['pitch_up_v_min']), float(lon['pitch_up_v_full']))
        self.pitch_down_max = math.radians(float(lon['pitch_down_max_deg']))
        self.alpha_trim = math.radians(float(lon['alpha_trim_deg']))
        self.k_pitch, self.kd_pitch = float(lon['k_pitch']), float(lon['kd_pitch'])
        self.k_pitch_i = float(lon['k_pitch_i'])
        self.dt = 1. / 30.                       # agent control period
        self.integral = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.i_pitch = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)

    def reset_idx(self, mask):
        self.integral.masked_fill_(mask, 0.)
        self.i_pitch.masked_fill_(mask, 0.)

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 23:35].reshape(-1, 4, 3)
        R = _quat_to_R(q)
        v_w = torch.einsum('nij,nj->ni', R, vb)
        V = vb.norm(dim=-1)
        roll, pitch, _ = _rot_to_rpy(R)
        target = p + torch.einsum('nij,nj->ni', R, delta[:, 0])

        # ---- L1 lateral guidance on the active waypoint ----
        d3 = delta[:, 0].norm(dim=-1)
        h_err = target[:, 2] - p[:, 2]
        pn = p.norm(dim=-1)
        # Climb-phase containment: with the target far above and the aircraft
        # already far from the origin, aiming laterally at the waypoint turns
        # the climb into a pursuit curve that passes UNDER the target and
        # reverses on arcs too wide for the 100 m sphere (dominant OOB mode:
        # terminal pn ~ 87, dz +20, V pinned ~23).  Blend the lateral aim
        # toward the ORIGIN so the climb becomes a centered spiral; waypoint
        # guidance resumes as the height deficit closes or the aircraft comes
        # back inside the loiter radius.
        pn_h = p[:, :2].norm(dim=-1)
        w_loit = (torch.sigmoid((pn_h - self.pn_loiter) / 3.)
                  * torch.sigmoid((h_err - self.h_loiter_gate) / 3.))
        d_h = target[:, :2] - p[:, :2]
        d_h_norm = (1. - w_loit) * d_h.norm(dim=-1) + w_loit * pn_h
        # Shortest-path angular blend of the aim bearing (continuous through
        # the atan2 seam): waypoint LOS -> origin LOS as w_loit rises.
        los = torch.atan2(d_h[:, 1], d_h[:, 0])
        los_home = torch.atan2(-p[:, 1], -p[:, 0])
        los = _wrap(los + w_loit * _wrap(los_home - los))
        v_h_norm = v_w[:, :2].norm(dim=-1)
        course = torch.where(v_h_norm > 2.,
                             torch.atan2(v_w[:, 1], v_w[:, 0]),
                             torch.atan2(R[:, 1, 0], R[:, 0, 0]))   # heading fallback
        eta = _wrap(los - course)
        # |eta| > 90 deg: sin(eta) would weaken exactly when a hard turn is
        # needed; command the full-strength turn instead.
        eta_c = eta.clamp(-math.pi / 2, math.pi / 2)
        # Distance-scheduled L1: a constant L1 sustains a stable limit-cycle
        # orbit around a point waypoint at r ~ L1/(2 sin eta) (observed at
        # ~27-31 m with L1 = 40, exactly the 60-deg-bank minimum turn radius
        # at cruise speed).  Shrinking L1 with range pushes that equilibrium
        # inside the 10 m capture sphere so the pursuit spiral converges.
        l1_eff = (self.l1_dist_gain * d_h_norm + self.l1_dist_offset).clamp(self.l1_min, self.l1)
        a_lat = 2. * V ** 2 * torch.sin(eta_c) / l1_eff
        # Near-field deadband: with the target almost overhead the LOS azimuth
        # is noise; hold course and let the vertical channel capture it.
        a_lat = a_lat * (d_h_norm > 3.).to(a_lat.dtype)
        # Aerodynamic bank limit: the accelerated-stall boundary n_max =
        # (V/v_stall)^2 gives cos(bank_max) = 1/n_max = (v_stall/V)^2, i.e.
        # MORE bank authority at higher speed.  (A linear low-speed ramp does
        # the opposite -- it inflates the minimum turn radius exactly when
        # the terminal pursuit needs it tight.)
        bank_lim = torch.acos(((self.v_stall / V) ** 2).clamp(0., 1.)).clamp_max(self.max_bank)
        bank_aero = bank_lim
        # Altitude-priority bank gate: this airframe is power-limited
        # (max thrust 18 N < weight 23 N), so sustained steep banks sink the
        # aircraft 0.5-1 m/s (observed: permanent 55-60 deg hold 10 m BELOW
        # the waypoint, dz never closing).  While a meaningful height error
        # remains, cap the bank so the climb proceeds on straight-ish legs;
        # once on altitude, release the full aerodynamic bank for the
        # horizontal capture.
        bank_climb = torch.minimum(bank_lim, torch.full_like(bank_lim, self.bank_climb_rad))
        w_climb = torch.sigmoid((h_err - self.h_climb_gate) / 3.)
        bank_lim = bank_lim + (bank_climb - bank_lim) * w_climb
        # Env roll convention: positive roll = right bank turns the course
        # negative (clockwise), so a target to the left (eta > 0) needs bank < 0.
        bank_l1 = -torch.atan(a_lat / self.g)
        # Terminal pure-pursuit branch: when close and roughly aligned, blend
        # from L1 to bank = k_pp * eta.  Pure pursuit of a stationary point
        # bends the path less and less as eta -> 0, so the pass goes straight
        # THROUGH the capture sphere; L1 at full gain keeps demanding near
        # max bank which, in a sustained 55-60 deg hold at endgame speeds,
        # starves the wing of lift and the aircraft sinks 0.5-1 m/s below the
        # waypoint forever (observed orbit at d3 ~ 12-46 m, dz ~ +10 m).
        # Smooth sigmoid gates on both conditions avoid chattering.
        bank_pp = -self.k_pp * eta
        w_pp = (torch.sigmoid((self.d_pp_entry - d3) / 3.)
                * torch.sigmoid((eta.abs() - self.eta_pp_gate) / -.1)
                * (1. - w_loit))
        bank_cmd = (w_pp * bank_pp + (1. - w_pp) * bank_l1).clamp(-bank_lim, bank_lim)
        # Out-of-bounds shield: the 100 m failure sphere is small relative to
        # wide max-bank return loops around far-side waypoints.  When close to
        # the boundary AND moving outward, override with a bank toward the
        # origin (classical boundary avoidance).
        v_toward = -(p * v_w).sum(-1) / pn.clamp_min(1e-6)
        w_shield = (torch.sigmoid((pn - self.pn_shield) / 4.)
                    * torch.sigmoid((v_toward - self.v_shield_min) / 1.))
        if bool(w_shield.any()):
            # Safety overrides the altitude-priority bank cap: a boundary
            # reversal at the climb-capped 45 deg needs a ~2x wider arc and
            # crosses the failure sphere before the heading comes around.
            eta_home = _wrap(torch.atan2(-p[:, 1], -p[:, 0]) - course)
            bank_shield = -self.k_shield * eta_home
            bank_cmd = (1. - w_shield) * bank_cmd + w_shield * bank_shield
            bank_cmd = bank_cmd.clamp(-bank_aero, bank_aero)
        else:
            bank_cmd = bank_cmd.clamp(-bank_lim, bank_lim)
        aileron = self.k_roll * _wrap(bank_cmd - roll) - self.kd_roll * wb[:, 0]

        # ---- TECS: total energy -> throttle, energy balance -> pitch ----
        # Waypoint-bearing-dependent speed reference: bleed airspeed with the
        # track error so hard course reversals happen at a small turn radius
        # (R = V^2 / g tan(bank)) that fits inside the 100 m out-of-bounds
        # sphere.
        v_eff = (self.v_ref - self.k_speed_bleed * eta.abs()).clamp(self.v_turn_min, self.v_ref)
        # Distance-based terminal bleed: below d_slow_ref the reference decays
        # toward v_turn_min, shrinking the minimum turn radius so the endgame
        # pursuit stays convergent inside the capture geometry.
        v_dist = (self.v_turn_min + self.k_dist_slow * (d3 - self.d_slow_ref)).clamp(self.v_turn_min, self.v_ref)
        v_eff = torch.minimum(v_eff, v_dist)
        # Climbing legs also slow down: the altitude-priority gate caps the
        # bank, so the speed must come down with it or the climb spiral
        # radius (V^2 / (g tan bank)) swells past the OOB sphere (observed
        # 42 m arcs out to |p| ~ 80 on the way up).
        v_climb = self.v_ref + (self.v_climb_ref - self.v_ref) * w_climb
        v_eff = torch.minimum(v_eff, v_climb)
        dv = (v_eff ** 2 - V ** 2) / (2. * self.g)
        dE = (h_err + dv).clamp(-25., 25.)
        dB = h_err - dv
        # Integral with saturation/speed anti-windup: the pure energy integral
        # stores trim margin during climbs and then holds the throttle up
        # while the endgame needs the airframe to DECELERATE (observed: I
        # stuck at ~0.17 kept throttle at 0.5 and airspeed pinned at 21 m/s
        # in the terminal orbit).  The speed-error bleed only acts on energy
        # SURPLUS (dE < 0): during a genuine climb the aircraft is 'too fast'
        # relative to the scheduled reference by design, and bleeding the
        # integral there cancels the climb power entirely.
        aw = self.k_aw_speed * (V - v_eff).clamp_min(0.) * (dE < 0.).to(dE.dtype)
        integral = (self.integral + self.k_energy_i * dE * self.dt
                    - aw * self.dt).clamp(0., self.integral_max)
        self.integral.copy_(integral)
        throttle = (self.throttle_trim + self.k_energy * dE + integral).clamp(0., 1.)
        # Banked level flight needs extra alpha: delta-alpha ~ (n - 1) alpha_trim.
        load_extra = (1. / torch.cos(roll).clamp_min(0.2) - 1.) * self.alpha_trim
        # Speed keeping: the airframe is underpowered (max thrust < weight), so
        # when airspeed falls below the reference the pitch channel gives back
        # climb demand to protect energy (airspeed priority).  Clamped to the
        # underspeed side: pushing the nose DOWN on the overspeed side would
        # accelerate the airframe exactly when the turn geometry needs it slow.
        v_overspeed = (V - v_eff).clamp_min(0.)
        # Overspeed energy trade: when the scheduled airspeed sits below the
        # current one AND altitude is wanted (w_climb gate), pitch up beyond
        # trim to convert kinetic energy into climb.  Without it the TECS
        # balance demand is neutralised by the coexisting speed deficit
        # (h_err=+20 with dv=-17 gives dE~=+3 -> cruise throttle) and the
        # airframe cruises overspeed at ~23 m/s, so the 45-deg-bank climb
        # spiral radius V^2/(g tan bank) swells past the OOB sphere.
        pitch_cmd = (self.pitch_trim - self.k_balance * dB - load_extra
                     + self.k_speed_keep * (v_eff - V).clamp_min(0.)
                     - self.k_geo_up * v_overspeed * w_climb)
        # Nose-up authority shrinks with airspeed (stall protection).
        pv_lo, pv_hi = self.pitch_up_v
        pitch_up = self.pitch_up_max * ((V - pv_lo) / (pv_hi - pv_lo)).clamp(0., 1.)
        pitch_cmd = pitch_cmd.clamp_min(-pitch_up).clamp_max(self.pitch_down_max)
        # Inner pitch loop: PD on attitude, but the rate damping acts only on
        # the deviation from the coordinated level-turn pitch rate.  A steady
        # banked turn needs a continuous body pitch rate wb_y = wb_z tan(roll)
        # (kinematic identity for a pure yaw rotation); damping the full
        # wb_y would cancel ~40% of the elevator in a 60-deg bank and force a
        # spiral descent.
        q_coord = wb[:, 2] * torch.tan(roll.clamp(-1.45, 1.45))
        # PID: the P-only loop leaves a steady-state offset equal to the
        # trim elevator of the current flight condition divided by k_pitch
        # (observed: in a 45 deg bank the trim a1 ~ -0.14 held the nose 4 deg
        # below the command, exactly cancelling the climb angle -- the
        # aircraft cruised overspeed at 22.5 m/s forever 15-20 m below the
        # waypoint).  The integral closes that offset.
        i_pitch = (self.i_pitch + self.k_pitch_i * (pitch_cmd - pitch) * self.dt).clamp(-.35, .35)
        self.i_pitch.copy_(i_pitch)
        elevator = (self.k_pitch * (pitch_cmd - pitch) - self.kd_pitch * (wb[:, 1] - q_coord)
                    + i_pitch)

        a = torch.stack((aileron, elevator, torch.zeros_like(aileron), 2. * throttle - 1.), -1)
        return a.clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# fixedwing: conventional cascaded PID autopilot
# --------------------------------------------------------------------------- #
class FixedwingPIDBaseline(Baseline):
    """Conventional cascaded PID autopilot (no guidance heuristics).

    Five single-input/single-output loops, each closed on the public
    ``'policy'`` observation group only:

    1. course error -> P/I -> bank-angle reference, clamped to
       ``max_bank_deg``;
    2. bank-angle error -> P -> roll-rate reference -> P/I -> aileron;
    3. altitude error -> P/I -> pitch-angle reference, clamped to the
       nose-up/nose-down authority;
    4. pitch-angle error -> P -> pitch-rate reference -> P/I -> elevator;
    5. airspeed -> P/I -> throttle.

    Interface facts (verified against the plant; see also
    ``docs/method/baseline-fixedwing-l1-tecs.md``):

    * Actions a in [-1,1]^4: a0 > 0 rolls right (positive bank), a1 > 0 is a
      nose-down elevator, a2 is a dead channel (main-wing deflection limit 0)
      and is driven with exactly 0, a3 is throttle remapped by the env as
      (a3+1)/2.
    * Euler convention (x fwd, y left, z up): positive roll = right bank,
      positive pitch = nose down.  A left turn therefore needs negative bank
      and a climb needs a negative pitch reference.
    * Airspeed feedback: the observation exposes body-frame velocity v_b and
      attitude q.  This controller uses V = ||v_b||, i.e. GROUND speed.  In
      calm air (the benchmark default) it equals true airspeed; when the
      optional wind disturbance is enabled it remains a public ground-speed
      measurement, deliberately NOT the internal air-relative truth
      (v_air = v - wind), which the controller never reads.  Course is
      likewise the ground track atan2(v_y, v_x) when the horizontal speed
      exceeds 2 m/s (yaw fallback below that).

    No path planning, NMPC, stall-protection layer, learning compensation or
    adaptive scheduling is present: every gain is constant and the only
    state is five clamp-limited integrators, cleared by ``reset_idx``.
    """

    name = 'pid_cascade'
    CONFIG_PATH = Path(__file__).parent / 'fixedwing_pid.json'

    def __init__(self, env):
        super().__init__(env)
        cfg = json.loads(self.CONFIG_PATH.read_text())
        lat, lon, spd = cfg['lateral'], cfg['longitudinal'], cfg['speed']
        self.g = 9.81
        self.dt = 1. / 30.
        # lateral
        self.k_course = float(lat['k_course'])
        self.k_i_course = float(lat['k_i_course'])
        self.i_course_max = math.radians(float(lat['i_course_max_deg']))
        self.max_bank = math.radians(float(lat['max_bank_deg']))
        self.k_roll_angle = float(lat['k_roll_angle'])
        self.max_roll_rate = math.radians(float(lat['max_roll_rate_deg_s']))
        self.k_roll_rate = float(lat['k_roll_rate'])
        self.k_i_roll_rate = float(lat['k_i_roll_rate'])
        self.i_roll_rate_max = float(lat['i_roll_rate_max'])
        # longitudinal
        self.k_altitude = float(lon['k_altitude'])
        self.k_i_altitude = float(lon['k_i_altitude'])
        self.i_altitude_max = math.radians(float(lon['i_altitude_max_deg']))
        self.pitch_trim = math.radians(float(lon['pitch_trim_deg']))
        self.pitch_up_max = math.radians(float(lon['pitch_up_max_deg']))
        self.pitch_down_max = math.radians(float(lon['pitch_down_max_deg']))
        self.k_pitch_angle = float(lon['k_pitch_angle'])
        self.max_pitch_rate = math.radians(float(lon['max_pitch_rate_deg_s']))
        self.k_pitch_rate = float(lon['k_pitch_rate'])
        self.k_i_pitch_rate = float(lon['k_i_pitch_rate'])
        self.i_pitch_rate_max = float(lon['i_pitch_rate_max'])
        # speed
        self.v_ref = float(spd['v_ref'])
        self.v_min_valid = float(spd['v_min_valid'])
        self.k_speed = float(spd['k_speed'])
        self.k_i_speed = float(spd['k_i_speed'])
        self.i_speed_max = float(spd['i_speed_max'])
        self.throttle_trim = float(spd['throttle_trim'])
        self.throttle_min = float(spd.get('throttle_min', 0.))
        self.throttle_max = float(spd.get('throttle_max', 1.))
        # integrators (one scalar per replica, float64, device-resident)
        self.i_course = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.i_roll_rate = torch.zeros_like(self.i_course)
        self.i_altitude = torch.zeros_like(self.i_course)
        self.i_pitch_rate = torch.zeros_like(self.i_course)
        self.i_speed = torch.zeros_like(self.i_course)

    def reset_idx(self, mask):
        for state in (self.i_course, self.i_roll_rate, self.i_altitude,
                      self.i_pitch_rate, self.i_speed):
            state.masked_fill_(mask, 0.)

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 23:35].reshape(-1, 4, 3)
        R = _quat_to_R(q)
        v_w = torch.einsum('nij,nj->ni', R, vb)
        # Public speed feedback: body-frame speed = world ground speed.
        V = vb.norm(dim=-1)
        roll, pitch, yaw = _rot_to_rpy(R)
        target = p + torch.einsum('nij,nj->ni', R, delta[:, 0])

        # ---- lateral: bearing -> course -> bank -> roll rate -> aileron ----
        d = target[:, :2] - p[:, :2]
        bearing = torch.atan2(d[:, 1], d[:, 0])
        v_h = v_w[:, :2].norm(dim=-1)
        course = torch.where(v_h > self.v_min_valid, torch.atan2(v_w[:, 1], v_w[:, 0]), yaw)
        e_course = _wrap(bearing - course)
        i_course = (self.i_course + self.k_i_course * e_course * self.dt).clamp(
            -self.i_course_max, self.i_course_max)
        drive = self.k_course * e_course + i_course
        # Conditional integration: do not wind up while the bank reference is
        # saturated by this same error.
        over = (drive.abs() > self.max_bank) & (drive * e_course > 0.)
        i_course = torch.where(over, self.i_course, i_course)
        self.i_course.copy_(i_course)
        # Env convention: positive roll = right bank turns the course negative,
        # so a positive (left) course error needs a negative bank reference.
        roll_ref = (-drive).clamp(-self.max_bank, self.max_bank)

        e_roll = _wrap(roll_ref - roll)
        p_ref = (self.k_roll_angle * e_roll).clamp(-self.max_roll_rate, self.max_roll_rate)
        e_p = p_ref - wb[:, 0]
        i_roll = (self.i_roll_rate + self.k_i_roll_rate * e_p * self.dt).clamp(
            -self.i_roll_rate_max, self.i_roll_rate_max)
        raw_ail = self.k_roll_rate * e_p + i_roll
        over = ((raw_ail.abs() > 1.) & (raw_ail * e_p > 0.))
        i_roll = torch.where(over, self.i_roll_rate, i_roll)
        self.i_roll_rate.copy_(i_roll)
        aileron = (self.k_roll_rate * e_p + i_roll).clamp(-1., 1.)

        # ---- longitudinal: altitude -> pitch -> pitch rate -> elevator ----
        h_err = target[:, 2] - p[:, 2]
        i_alt = (self.i_altitude + self.k_i_altitude * h_err * self.dt).clamp(
            -self.i_altitude_max, self.i_altitude_max)
        drive_h = self.k_altitude * h_err + i_alt
        # Climb = negative pitch reference (positive pitch is nose down).
        raw_pitch = self.pitch_trim - drive_h
        over = (((raw_pitch < -self.pitch_up_max) & (h_err > 0.))
                | ((raw_pitch > self.pitch_down_max) & (h_err < 0.)))
        i_alt = torch.where(over, self.i_altitude, i_alt)
        self.i_altitude.copy_(i_alt)
        pitch_ref = (self.pitch_trim - drive_h).clamp(-self.pitch_up_max, self.pitch_down_max)

        e_pitch = pitch_ref - pitch
        q_ref = (self.k_pitch_angle * e_pitch).clamp(-self.max_pitch_rate, self.max_pitch_rate)
        # Coordinated-turn kinematics: in a banked turn the body pitch rate
        # contains q_coord = r tan(phi) that does not change the Euler pitch.
        # Removing it lets the inner loop regulate the angle-of-attack-bearing
        # pitch instead of chasing the turn itself (constant, non-adaptive).
        q_coord = wb[:, 2] * torch.tan(roll)
        e_q = q_ref - (wb[:, 1] - q_coord)
        i_q = (self.i_pitch_rate + self.k_i_pitch_rate * e_q * self.dt).clamp(
            -self.i_pitch_rate_max, self.i_pitch_rate_max)
        raw_ele = self.k_pitch_rate * e_q + i_q
        over = ((raw_ele.abs() > 1.) & (raw_ele * e_q > 0.))
        i_q = torch.where(over, self.i_pitch_rate, i_q)
        self.i_pitch_rate.copy_(i_q)
        elevator = (self.k_pitch_rate * e_q + i_q).clamp(-1., 1.)

        # ---- airspeed (ground speed) -> throttle ----
        e_v = self.v_ref - V
        i_v = (self.i_speed + self.k_i_speed * e_v * self.dt).clamp(
            -self.i_speed_max, self.i_speed_max)
        raw_thr = self.throttle_trim + self.k_speed * e_v + i_v
        over = (((raw_thr > self.throttle_max) & (e_v > 0.))
                | ((raw_thr < self.throttle_min) & (e_v < 0.)))
        i_v = torch.where(over, self.i_speed, i_v)
        self.i_speed.copy_(i_v)
        throttle = (self.throttle_trim + self.k_speed * e_v + i_v).clamp(
            self.throttle_min, self.throttle_max)

        # a2 is the dead upstream channel and stays exactly zero.
        a = torch.stack((aileron, elevator, torch.zeros_like(aileron),
                         2. * throttle - 1.), -1)
        return a.clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# rocket: powered-descent guidance (ZEM/ZEV) + attitude hold
# --------------------------------------------------------------------------- #
class RocketPDGBaseline(Baseline):
    """ZEM/ZEV powered-descent guidance + TVC/fin attitude inner loop.

    Environment contract (confirmed 2026-10-21 from rocket/env.py,
    environments/common/plant.py, rocket.urdf/.yaml and live probes on the plant):
      * Frames: obs position is world-frame (z up, gravity -9.81 m/s^2);
        linear/angular velocities in the obs are body-frame.  Thrust acts
        along body +z (nose up); gimbal (dims 5-6) tilts it within +/-5 deg
        with a 10 ms lag; grid fins (dims 0-2) are lift surfaces 2.64 m above
        the CM (torque ~ V^2, 50 ms actuation lag).  All actuator signs were
        verified with open-loop probes.
      * Mass: dry 138.2 kg (base 91 + booster 47 + fins 0.2; legs/flame
        massless) + fuel*410.9 kg on the fueltank link.  Spawn fuel is 5%
        (20.5 kg), so the booster lifts off at ~158.7 kg: full thrust gives
        T/W = 4.9 (net braking ~43 m/s^2 with drag, probe-measured), and
        MIN thrust/weight = 1.91 > 1 (net +8.9 m/s^2 at the spawn mass) --
        once lit the engine cannot follow a slow descent, so the terminal
        phase is a min-thrust pulse train with coasting between pulses
        (hysteresis on the observed ignited flag).
      * Attitude authority (probe-measured): the grid fins act as a strong
        AERO RESTORING SPRING pulling body +z toward the retrograde flow
        axis, k_aero ~ 3.6*V^2 N m/rad (23e3 N m/rad at 80 m/s); at full
        deflection the vehicle can hold at most ~9.6 deg of tilt against the
        flow, at any speed.  A plain PD cannot follow large tilt commands
        (steady-state error = k_aero*tilt/KP), so the loop adds a classical
        aerodynamic-moment feed-forward cancelling the spring, torques are
        allocated by ceiling (fast TVC first, fins for the remainder), and
        the lateral guidance is capped to tilt-consistent demands
        (|a_lat| <= 0.14 * vertical acceleration, i.e. <= ~8 deg).
      * Control: 40 Hz policy command held for 3 inner steps (120 Hz) at
        240 Hz physics; 1200 policy steps = 30 s episode limit; the engine
        relights per physics step (cmd > 0.5).
      * Landing (unchanged criteria): pad contact (radius 2 m, top z = 0.15,
        rest base height ~2.575 m) is fatal iff the pre-contact speed
        exceeds 1.0 m/s or spin 0.35 rad/s; success requires speed/spin < 0.02
        with tilt < 0.1 rad, judged on the state at the start of the contact
        inner step (i.e. the previous inner-step state).  Success does not
        truncate: the rocket keeps earning the settled contact stream until
        the 30 s limit.

    Guidance: closed-form ZEM/ZEV to the pad rest point with zero terminal
    velocity.  Time-to-go is re-solved each step from a constant-decel descent
    profile (60% of the max net braking) that bleeds airspeed early; the
    engine self-ignites when the commanded thrust crosses min thrust.  Below
    25 m/s a two-stage suicide-burn endgame takes over: stage 1 brakes along
    the min-thrust manifold of a virtual pad raised 1 m above the real one
    (with EPS margin over the floor so thrust noise cannot clip it), stage 2
    is a constant-rate pulse servo (ignition hysteresis on airspeed, band
    fixed in speed rather than shrinking with height) that flies the last
    metre at ~0.5 m/s and touches down at <= ~0.65 m/s by construction.  The
    lateral demand is split between thrust tilting and the fins (independent
    lift actuators, alive even with the engine off), with the fin-force
    coupling torque pre-compensated through the gimbal whenever it burns.
    When any demand exceeds authority the corresponding actuator simply
    saturates -- the deterministic fallback of an analytic law.  There is NO
    online optimizer, hence no solver and no solver-failure mode; the honest
    analogs reported by the test entry are the saturation / coast / pulse
    statistics.

    Fully batched: every replica is computed on-device from the 'policy' obs
    only; the ignition hysteresis state is read from the observed ignited
    flag, so no per-env Python loop and no host synchronization is needed.
    """
    name = 'zem_zev_pdg'

    # vehicle constants (rocket.urdf / rocket.yaml; probed against the plant)
    M_DRY = 138.2                    # base 91 + booster 47 + 4 fins 0.2 [kg]
    FUEL_FULL = 410.9                # fueltank mass at fuel = 1 [kg]
    T_MAX = 7607.0                   # booster_params.max_thrust [N]
    MIN_RATIO = 2966.7 / 7607.0      # min_thrust/max_thrust
    GIMBAL_RAD = math.radians(5.)    # gimbal_range_degrees
    G = 9.81
    # finlet torque per unit raw deflection: 2 fins * dcl/drad (slope 1.45 *
    # eta .65) * 45 deg * 0.5*rho*S(0.25) * z_fin * V^2 [N m / unit]
    FIN_K = 0.22668
    FIN_YAW_K = 0.1587               # 4-fin differential yaw torque [N m/unit/V^2]
    AERO_K = 3.6                     # fin restoring spring k_aero [N m/rad/V^2]
    # guidance / control gains
    V_TOUCH = 0.35                   # pulse band centre / manifold terminal speed
                                     # (lowered from 0.45: softer touchdown leaves
                                     # less rocking energy; with S_ON/S_OFF
                                     # lowered to 0.40/0.30 the 25 ms band drift
                                     # still bounds the touch at <= ~0.55 m/s)
    TGO_FLOOR = 0.5                  # ZEM/ZEV singularity guard [s]
    S_GATE = 25.                     # airspeed below which the suicide-burn
                                     # endgame tracker replaces the ZEM law
    EPS_END = 1.0                    # endgame profile decel margin over the
                                     # min-thrust net braking a_min: commanded
                                     # thrust rides ~5% above T_min so the
                                     # plant's multiplicative thrust noise
                                     # (1 + 0.01*N(1,1)) can neither clip the
                                     # floor nor out-brake the profile into
                                     # cut-relight chatter
    H_VIRT = 2.0                     # virtual pad raise [m]: the braking
                                     # manifold aims H_VIRT above the real pad
                                     # and hands a ~0.5 m/s vehicle to the pulse
                                     # servo, which flies the last ~2 m at
                                     # constant rate.  Riding the real-pad
                                     # manifold to h = 0 is unflyable: any
                                     # hysteresis cut below ~5 m re-enters free
                                     # fall at 2-4 m/s and the >1.0 m/s
                                     # pre-contact check kills the rocket
                                     # (probed: 1.38 m/s touchdown).  The 2 m
                                     # pulse leg (~4 s at the band rate) also
                                     # gives the ignition-transient spin pumped
                                     # in during the suicide burn time to damp
                                     # below the 0.02 on-pad settle threshold
                                     # (at H_VIRT=1 the leg is ~1 s and corner
                                     # cases touched down with spin ~0.076,
                                     # toppled, and never settled)
    S_PULSE = 1.5                    # airspeed below which the pulse servo runs
                                     # (set above the burn-out speed so the
                                     # servo catches the vehicle right at
                                     # suicide-burn end; no coast dead zone)
    S_OFF = 0.30                     # pulse hysteresis: cut when descending
                                     # slower than S_OFF -- while lit the net
                                     # accel is +a_min (min thrust > weight),
                                     # so the engine can only brake, never
                                     # hold a slow descent: pulsing is the
                                     # ONLY way to fly the last metre
    S_ON = 0.40                      # ... and relight when faster than S_ON
    H_PULSE = 2.0                    # pulse servo altitude ceiling [m]
    A_LAT_PULSE = 1.5                # pulse-phase lateral accel cap [m/s^2]
                                     # (~1.6 deg tilt): nulls the <=0.5 m/s
                                     # entry drift inside the 2 m pulse leg
                                     # without asking the half-blind attitude
                                     # loop for real tilt work
    LAT_TILT = 0.11                  # lateral cap = 0.11 * vertical accel (~6 deg,
                                     # inside the ~9.6 deg full-fin aero hold limit)
    A_LAT_MAX = 4.5                  # absolute lateral accel cap [m/s^2]
    KP_T = 12000.                    # tilt torque gain [N m/rad] (I_eff ~600:
                                     # Genesis scales the fueltank inertia with
                                     # fuel mass, so the URDF sum 2243 is ~4x
                                     # the true pitch inertia; free-osc probe)
    KD_T = 6000.                     # tilt rate damping [N m s/rad] (~zeta .8;
                                     # raised from 4300 in the 2026-10-22 tuning
                                     # sweep: stronger terminal tilt-rate damping
                                     # improves the arrival state (touch v/spin/
                                     # tilt) far more than any pulse-band change
                                     # -- settle failures are rocking/topple
                                     # deaths, and KD_T is the only actuator
                                     # path that damps rocking while lit.  8000
                                     # is marginally better alone (83.9%) but
                                     # degrades with the soft band; 10000
                                     # overshoots (79.2%))
    KD_Z = 3.0                       # yaw spin damping [N m s/rad]
    REST_Z = 2.575                   # pad top 0.15 + leg bottom offset 2.425 [m]

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        f64 = torch.float64
        self.ez = torch.tensor((0., 0., 1.), device=d, dtype=f64)
        self.rf = torch.tensor((0., 0., self.REST_Z), device=d, dtype=f64)
        self.gvec = torch.tensor((0., 0., self.G), device=d, dtype=f64)
        # on-device diagnostics (analogs of solver statistics; host-synced only
        # by the test entry): max-thrust saturation, engine-off coasting,
        # ignition toggles, guidance demand above authority, t_go floor hits
        z = torch.zeros(self.num_envs, device=d, dtype=f64)
        self.diag = {k: z.clone() for k in
                     ('steps', 'sat_max', 'coast', 'ignitions', 'infeasible',
                      'floor', 'fin_sat')}

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        # aux tail: [13:20] 7-dim action echo, [20:24] FOUR finlet actuation
        # states, then ignited=24, fuel=25, throttle=26, gimbal=27:29, pad=29
        ignited_obs = o[:, 24] > .5
        fuel = o[:, 25].clamp(0., 1.)
        m = self.M_DRY + fuel * self.FUEL_FULL
        R = _quat_to_R(q)
        v_w = torch.einsum('nij,nj->ni', R, vb)
        s = (-v_w[:, 2]).clamp_min(0.)                  # descent speed

        # ---- time-to-go from a constant-decel descent profile: the vehicle
        # tracks v(z) = sqrt(2*a_brake*z + v_touch^2) with a_brake = 60% of the
        # max net decel.  Braking EARLY (instead of the fuel-optimal coast-
        # then-max-brake schedule) collapses airspeed while |v_xy| is still
        # tiny, which keeps the airflow -- the fin restoring spring's alignment
        # target -- near vertical for the whole burn; the spare 40% authority
        # absorbs the lateral tilt component and drag model error.  Re-solved
        # closed-form every step, so no optimizer and no solver failure mode.
        a_net = (self.T_MAX / m - self.G).clamp_min(1.)
        a_brake = .6 * a_net
        t_go = ((s - self.V_TOUCH) / a_brake).clamp(self.TGO_FLOOR, 30.)
        self.diag['floor'] += (t_go <= self.TGO_FLOOR + 1e-9).to(torch.float64)

        # ---- ZEM/ZEV acceleration toward rf with zero terminal velocity.
        # Optimal quadratic solution for fixed final position and zero final
        # velocity: a*(0) = 6*zem/t_go^2 - 2*v/t_go (the velocity term SUBTRACTS
        # -- it brakes motion toward the target and stiffens motion away from
        # it; with a plus sign the law accelerates into the target and demands
        # free-fall while descending, which is how the earliest builds crashed).
        zem = self.rf - p - v_w * t_go[:, None]
        a_cmd = 6. * zem / t_go.square()[:, None] - 2. * v_w / t_go[:, None]
        a_vert = (a_cmd[:, 2] + self.G).clamp_min(0.)
        lat_cap = (self.LAT_TILT * a_vert).clamp(1., self.A_LAT_MAX)[:, None]
        a_cmd = torch.cat((a_cmd[:, :2].clamp(-lat_cap, lat_cap), a_cmd[:, 2:]), -1)
        a_thr = a_cmd + self.gvec
        # the engine can only push along +z: never reward a downward-accel
        # demand with thrust (prevents a full-thrust ascent runaway)
        a_thr = torch.cat((a_thr[:, :2], a_thr[:, 2:].clamp_min(0.)), -1)
        a_mag = a_thr.norm(dim=-1).clamp_min(1e-6)
        T_raw = m * a_mag
        T_cmd = T_raw.clamp(0., self.T_MAX)
        T_min = self.MIN_RATIO * self.T_MAX
        self.diag['infeasible'] += (T_raw > self.T_MAX).to(torch.float64)
        ign_mid = torch.where(ignited_obs, T_cmd >= .97 * T_min, T_cmd >= 1.03 * T_min)

        # ---- endgame: two-stage suicide burn.  Stage 1 brakes along the
        # min-thrust manifold of a VIRTUAL pad H_VIRT above the real one,
        #     a_req = (vz^2 - s_t^2)/(2 h_virt) + g + A,
        #     s_t^2 = V_TOUCH^2 + 2 A h_virt,  A = a_min + EPS_END,
        # i.e. the closed-form decel that removes the speed error over the
        # remaining virtual height (the error decays like e ~ e0 h/h0) and
        # coasts below it -- with min thrust > weight the engine cannot
        # descend slower than the manifold, so coasting IS the below-line
        # branch.  EPS_END keeps the commanded thrust ~5% above T_min so the
        # plant's thrust noise neither clips the floor nor out-brakes it.
        # Stage 2 (pulse servo, active for s <= S_PULSE and h <= H_PULSE)
        # flies the last metres at a constant ~0.5 m/s descent with an
        # ignition hysteresis on airspeed: cut below S_OFF, relight above
        # S_ON.  While lit the net accel is +a_min (brake only), while cut
        # -g, and the 25 ms policy quantization bounds the in-band drift to
        # ~0.25 m/s per sample, so the touch speed is ~[0.15, 0.65] m/s by
        # construction versus the 1.0 m/s fatal threshold.  The virtual-pad
        # raise is what creates this settle zone; the h_v floor (5 cm) makes
        # the brake branch saturate rather than limit-cycle when the vehicle
        # arrives fast just above it.  Pad contact (obs tail) forces the
        # engine off for the settle phase.
        pad_c = o[:, 29] > .5
        h_agl = (p[:, 2] - self.REST_Z).clamp_min(0.)
        endg = (s < self.S_GATE) & (h_agl > 0.)
        a_min = (self.MIN_RATIO * self.T_MAX / m - self.G).clamp_min(.5)
        A_END = a_min + self.EPS_END
        h_v = (h_agl - self.H_VIRT).clamp_min(.05)
        s_t = (self.V_TOUCH ** 2 + 2. * A_END * h_v).clamp_min(0.).sqrt()
        a_req = (v_w[:, 2].square() - s_t.square()) / (2. * h_v) \
            + self.G + A_END
        pulse = endg & (s <= self.S_PULSE) & (h_agl <= self.H_PULSE)
        a_vert_end = torch.where(pulse, self.G + A_END, a_req)
        T_end = (m * a_vert_end).clamp(0., self.T_MAX)
        self.diag['infeasible'] += (endg & ~pulse
                                    & (m * a_req > self.T_MAX)).to(torch.float64)
        T_min = self.MIN_RATIO * self.T_MAX
        a_lat_end = (-1.5 * v_w[:, :2] - (p[:, :2] - self.rf[:2]))
        lat_end_cap = (self.LAT_TILT * a_vert_end).clamp(1., self.A_LAT_MAX)
        # pulse phase: damp LATERAL VELOCITY only, never chase position, and
        # with a tight cap.  The fins are dead below ~2 m/s (force ~ V^2) and
        # the engine is off half the time, so every degree of commanded tilt is
        # attitude work the gimbal must redo after every relight -- the spin
        # that pumps is exactly what fails the settle check (probed: with the
        # full position+velocity law the lateral loop went into a
        # grow-and-reverse limit cycle, tilt diverged to 0.06 rad with the
        # gimbal saturated, and the tilted min-thrust pulses skated the legs
        # across the pad until a leg snag kicked the spin to 0.69 rad/s).
        # Velocity damping converges (no position oscillation) and stops the
        # drift within the first ~0.5 s of the pulse leg, keeping the touchdown
        # point on the pad -- the drift must stop because ANY ground contact
        # beside the pad is an instant fatal, and with zero tilt demand the
        # residual attitude work stays inside the ignition-transient budget.
        a_lat_end = torch.where(
            pulse[:, None],
            (-1.5 * v_w[:, :2]).clamp(-self.A_LAT_PULSE, self.A_LAT_PULSE),
            a_lat_end.clamp(-lat_end_cap[:, None], lat_end_cap[:, None]))
        a_thr_end = torch.cat((a_lat_end, a_vert_end[:, None]), -1)
        # NOTE: no rising/rebound relight on purpose.  The engine can only
        # push UP (min thrust > weight), so lighting during an ascent is a
        # BOOST, not a catch: probed at 1024-env scale it turns the touchdown
        # bounce into a sustained bounce-boost oscillator that ends in the
        # 30 s env timeout for ~25% of episodes (reset-state forensics:
        # timeouts at h median 29 m, speed 25 m/s).  The falling-side band
        # relight below is the only rebound mechanism: an uncaught leg launch
        # re-enters the band on the way down and the S_ON relight brakes it
        # back inside the 1.0 m/s pre-contact budget.
        ign_pulse = torch.where(ignited_obs, s >= self.S_OFF, s >= self.S_ON)
        ign_end = torch.where(pulse, ign_pulse,
                              torch.where(ignited_obs, T_end >= .97 * T_min,
                                          T_end >= 1.03 * T_min))
        # keep the band servo alive through the settle phase (pad_c only cuts
        # the engine on a sensed contact): a micro-bounce launch off the
        # padded-contact rest zone comes back down at 1-1.5 m/s and the
        # S_ON relight brakes it at ~a_min net, putting it back on the pad
        # inside the 1.0 m/s pre-contact budget -- the band IS the rebound
        # catcher.  A hard cut-after-touchdown latch removes that catch and
        # turns the rebound into a fatal re-contact (probed: 1.36 m/s).
        ign = torch.where(endg, ign_end, ign_mid) & ~pad_c
        self.diag['ignitions'] += (ign & ~ignited_obs).to(torch.float64)
        self.diag['coast'] += (~ign).to(torch.float64)
        T_cmd = torch.where(endg, T_end, T_cmd)
        a_thr = torch.where(endg[:, None], a_thr_end, a_thr)
        a_mag = torch.where(endg, a_thr_end.norm(dim=-1).clamp_min(1e-6), a_mag)
        u_thr = ((T_cmd / self.T_MAX - self.MIN_RATIO)
                 / (1. - self.MIN_RATIO)).clamp(0., 1.)
        self.diag['sat_max'] += (ign & (u_thr >= .999)).to(torch.float64)

        # ---- desired thrust direction.  Keep the guidance tilt across
        # coast/burn transitions: forcing the setpoint upright at every
        # min-thrust cutoff makes the attitude loop chase a bang-bang
        # reference it cannot track, and the endgame pulse train then wobbles
        # the pad point by metres.  At high airspeed the restoring spring
        # simply overrides the coast tilt (the FF saturates), so this is free;
        # at low speed the spring is weak and the tilt holds smoothly.
        b3 = torch.where((a_mag > 1e-6)[:, None],
                         a_thr / a_mag[:, None], self.ez.expand(self.num_envs, 3))
        b3 = torch.where((b3[:, 2] > .15)[:, None], b3,
                         self.ez.expand(self.num_envs, 3))
        u = torch.einsum('nji,nj->ni', R, b3)           # desired body z (body frame)
        e_tilt = torch.stack((-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])), -1)

        # ---- aerodynamic-moment feed-forward: cancel the fin restoring spring
        # (the spring pulls body z toward the retrograde flow axis with
        # k_aero*V^2; cancel it about the tilt axes so the PD sees a free body).
        # The linear spring model only holds to ~10 deg (fins stall beyond the
        # ~9.6 deg trim point), so the FF is capped at the fin hold capability:
        # past that the uncapped FF would over-cancel and diverge the loop.
        V2 = vb.square().sum(-1)                        # airspeed^2
        Vn = v_w.norm(dim=-1).clamp_min(1.)
        flow = -v_w / Vn[:, None]                       # retrograde axis (world)
        f_b = torch.einsum('nji,nj->ni', R, flow)       # ...in body frame
        e_flow = torch.stack((-f_b[:, 1], f_b[:, 0], torch.zeros_like(f_b[:, 0])), -1)
        cm_z = (47. * -2. + 0.2 * 2.051) / m            # fueltank sits at z=0
        z_fin = 2.051 - cm_z
        h = 2. + cm_z                                   # booster lever below CM
        ff_cap = (.95 * self.FIN_K * z_fin * V2)[:, None]
        tau_ff = ((self.AERO_K * V2)[:, None] * e_flow).clamp(-ff_cap, ff_cap)

        # ---- attitude PD (tilt tracking + spin damping, yaw free)
        tau_cmd = torch.cat((self.KP_T * e_tilt[:, :2] - self.KD_T * wb[:, :2],
                             -self.KD_Z * wb[:, 2:3]), -1)
        tau_cmd[:, :2] -= tau_ff[:, :2]

        # ---- lateral force split.  The world-frame lateral budget comes from
        # three actuators, two of which are OBSERVABLE (throttle and gimbal
        # angle states ride the aux obs), so the fins -- the only ones left to
        # command -- only ever make up the true residual:
        #   * thrust tilting: T_act along the actual body z (world xy),
        #   * TVC gimbal side-force: T_act*(sin gy, -sin gx cos gy) in body
        #     frame (plant's thrust stack).  This term peaks at T*sin(5 deg)
        #     ~ 4.2 m/s^2 -- LARGER than the tilt contribution, and it is a
        #     pure disturbance for the guidance if left unmodeled (the torque
        #     allocation treats the gimbal as a torque actuator and injects
        #     this force as a side effect of every attitude correction).
        #   * fins: independent lift actuators, alive even with the engine
        #     off, capped at 30% of their force authority.
        # A fin deflected for force also torques the body (force at the z_fin
        # lever): with the engine on the gimbal absorbs that coupling torque
        # (pre-compensated into tau_cmd below), while coasting it is left to
        # the aero spring, which holds the resulting ~2 deg tilt bias without
        # cancelling the force: fins lock torque to force (tau = z_fin x J F),
        # so a coast-phase pre-compensation would close that lock against the
        # very force this channel exists to create.
        # observed actuator states (aux tail): o[27] is the throttle state --
        # ALREADY the fraction of max thrust (plant applies thrust = throttle *
        # T_MAX, and the state keeps decaying after cutoff, so no ignition
        # gate); o[27:29] are the gimbal states in NORMALIZED action units
        # (the plant scales by gimbal_range_degrees only when building the
        # thrust direction).
        u_obs = o[:, 26]
        T_act = self.T_MAX * u_obs
        bz_w = R[..., 2]                                # body z in world
        a_lat_thr = (T_act / m)[:, None] * bz_w[:, :2]
        g_ob = o[:, 27:29] * self.GIMBAL_RAD
        f_gim_b = torch.stack((T_act * torch.sin(g_ob[:, 1]),
                               -T_act * torch.sin(g_ob[:, 0]) * torch.cos(g_ob[:, 1]),
                               torch.zeros_like(g_ob[:, 0])), -1)
        a_lat_gim = torch.einsum('nij,nj->ni', R, f_gim_b)[:, :2] / m[:, None]
        k_fin = (self.FIN_K * z_fin * V2).clamp_min(1e-9)
        T_now = torch.where(ign, T_cmd, torch.zeros_like(T_cmd))
        fin_a_cap = (.3 * self.FIN_K * V2 / m)[:, None]
        a_fin_w = (a_thr[:, :2] - a_lat_thr - a_lat_gim).clamp(-fin_a_cap, fin_a_cap)
        self.diag['fin_sat'] += (a_fin_w.abs() >= fin_a_cap - 1e-12).any(-1).to(torch.float64)
        a_fin_b = torch.einsum('nji,nj->ni', R, torch.cat(
            (a_fin_w, torch.zeros_like(a_fin_w[:, :1])), -1))[:, :2]
        d_fin = a_fin_b * (m / (self.FIN_K * V2).clamp_min(1e-9))[:, None]
        tau_coup = torch.stack((-a_fin_b[:, 1], a_fin_b[:, 0],
                                torch.zeros_like(a_fin_b[:, 0])), -1)
        tau_coup = tau_coup * (m * z_fin)[:, None]
        tau_cmd = tau_cmd + torch.where(ign[:, None], -tau_coup,
                                        torch.zeros_like(tau_coup))

        # ---- torque allocation by ceiling: fast TVC first, fins for the rest.
        k_gim = (h * T_now * self.GIMBAL_RAD).clamp_min(1e-9)
        gim_cap = .95 * k_gim
        # Fin authority is deliberately de-rated for the ATTITUDE loop: grid-fin
        # deflection at high airspeed carries a large LATERAL FORCE (~0.1 N per
        # unit deflection per (m/s)^2 -- over 2 m/s^2 of sideways accel at
        # 100 m/s).  The force is no longer purely parasitic -- the split above
        # now commands part of it on purpose -- but the attitude loop still
        # gets only a third of the fins so that attitude deflections plus the
        # force deflections stay well inside the raw +-1 window.
        fin_cap = .35 * k_fin
        tau_gim = tau_cmd.clamp(-gim_cap[:, None], gim_cap[:, None])
        tau_fin = (tau_cmd - tau_gim).clamp(-fin_cap[:, None], fin_cap[:, None])
        k_yaw = (self.FIN_YAW_K * V2).clamp_min(1e-9)
        # sign map verified by probes: finlet_x -> +tau_y, finlet_y -> -tau_x,
        # gimbal_x -> -tau_x, gimbal_y -> -tau_y, finlet_yaw -> +tau_z;
        # finlet_x -> +F_x, finlet_y -> +F_y (actuator-map probe), so the force
        # deflections simply add on top of the attitude deflections
        fx = tau_fin[:, 1] / k_fin + d_fin[:, 0]
        fy = -tau_fin[:, 0] / k_fin + d_fin[:, 1]
        gx = torch.where(T_now > 1., -tau_gim[:, 0] / k_gim,
                         torch.zeros_like(tau_gim[:, 0]))
        gy = torch.where(T_now > 1., -tau_gim[:, 1] / k_gim,
                         torch.zeros_like(tau_gim[:, 1]))
        yaw = -self.KD_Z * wb[:, 2] / k_yaw
        # Unified 7-dim finlet mixer convention (plant.py applies the mixer
        # internally: finlet x,y,yaw -> 4 hinge servos). Ignition/throttle in
        # [-1,1] — the env _scale_actions maps to [0,1] at step time.
        raw = torch.stack((fx, fy, yaw,
                           torch.where(ign, torch.ones_like(u_thr), -torch.ones_like(u_thr)),
                           2. * u_thr - 1., gx, gy), -1)
        self.diag['steps'] += 1.
        return raw.clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# rocket: plain position/velocity PD landing (second, simpler rocket baseline)
# --------------------------------------------------------------------------- #
class RocketPDLandingBaseline(Baseline):
    """Simple conventional PD landing law (no guidance optimisation).

    Structure (one policy step, 40 Hz):
      1. horizontal position/velocity PD -> desired world lateral acceleration
         a_xy = -K_xy (p_xy - pad_xy) - D_xy v_xy, norm-capped;
      2. height-scheduled descent-speed reference
         v_z_ref = -min(V_MAX, K_H max(h, 0)), h = max(p_z - REST_Z, 0),
         vertical velocity feedback plus gravity compensation
         a_z = K_v (v_z_ref - v_z), specific force f_z = a_z + g;
      3. desired thrust vector sigma = (a_xy, f_z); thrust magnitude
         T = clamp(m|sigma|, T_min, T_MAX) and desired thrust axis from the
         specific-thrust vector s = T/m: sin(tilt) = a_xy/s, so the delivered
         lateral acceleration equals the command at unit plant gain (the
         min-thrust clamp must enter the tilt reference, or it multiplies
         the lateral loop gain by s/f_z);
      4. attitude/rate PD -> torque demand tau = K_att e_tilt - D_att w_xy
         (plus yaw damping); torque is split with fixed physical gains:
         engine TVC first (k_gim = lever T sin(gimbal_max)), grid fins carry
         the remainder (k_fin = FIN_K z_fin V^2) and the yaw channel;
      5. ignition: the braking-distance switch h <= h_need + M_ON is an EDGE
         trigger for the first descent burn (descending only, v_dn =
         max(0,-v_z), h_need = max(0, v_dn^2 - v_ref_dn^2)/(2 a_brake),
         a_brake = BRAKE_FRAC (T_MAX/m - g)); afterwards a fixed speed-band
         hysteresis modulates the burn (cut when v_dn < lo, relight when
         v_dn > hi; lo/hi = reference -/+ V_BAND, replaced by the fixed
         terminal band [V_TERM_OFF, V_TERM_ON] below H_TERM).  The engine is
         off (zero thrust) when cut and clamps to the plant's min-thrust
         floor when lit; no learning anywhere in the switch;
      6. the public pad-contact flag (obs[29]) is the engine switch on the
         pad: the engine is forced off while ANY contact is sensed (per
         policy step; no permanent cut latch -- after a micro-bounce launch
         the flag clears and the speed band re-catches the rebound on the
         way down, keeping every re-contact inside the 1.0 m/s budget);
         and the episode is allowed to run to its own end, so any post-touch
         bounce/topple is observed rather than hidden.

    Deliberate approximations versus ``RocketPDGBaseline`` (documented, not
    hidden): no fin aerodynamic-spring feed-forward, no fin lateral-force
    split / coupling-torque pre-compensation, no ZEM/ZEV or virtual-pad
    manifold, and no terminal pulse servo beyond the fixed speed-band
    hysteresis above -- the only ignition logic is the edge-triggered
    braking-distance hysteresis above.  Torque allocation reuses the
    sign-verified plant map and the same physical gains (lever, FIN_K,
    FIN_YAW_K) as the PDG baseline, but without its force-split terms.

    Constants verified against the plant/YAML: dry mass 138.2 kg, fuel-tank
    mass 410.9 kg at f=1 (spawn 5% -> 158.7 kg), max thrust 7607 N, min thrust
    2966.7 N (ratio 0.39), reignitable ignition (command > 0.5 each 240 Hz
    physics step), throttle remap 0.39..1 with tau 0.01 s and 1% noise,
    two-axis gimbal +/-5 deg with tau 0.01 s, four grid fins with 45 deg
    deflection, tau 0.05 s and the upstream mixing
    (ax+/-ay, ay-/+ayaw) already applied by the plant.  The raw action
    convention is (fx, fy, fyaw, ign, thr, gx, gy) in [-1,1]^7; the env maps
    it affinely onto the upstream box (ignition/throttle in [0,1]).
    """

    name = 'pd_landing'

    # ---- vehicle constants (same probe-verified values as RocketPDGBaseline)
    M_DRY = 138.2
    FUEL_FULL = 410.9
    T_MAX = 7607.0
    MIN_RATIO = 2966.7 / 7607.0
    GIMBAL_RAD = math.radians(5.)
    G = 9.81
    FIN_K = 0.22668
    FIN_YAW_K = 0.1587
    REST_Z = 2.575

    # ---- tunable guidance gains (selected in rocket_pd_tune.py)
    KP_XY = 0.25         # horizontal position gain [1/s^2]
    KD_XY = 1.00         # horizontal velocity damping [1/s]
    V_MAX = 20.0         # descent-speed reference saturation [m/s]
    K_H = 0.50           # descent-speed slope [1/s]
    KV_Z = 3.0           # vertical velocity-error gain [1/s]
    A_LAT_MAX = 5.0      # absolute lateral acceleration cap [m/s^2]
    LAT_TILT = 0.14      # lateral cap as a fraction of the vertical specific force
    KP_ATT = 12000.0     # attitude (tilt) proportional gain [N m/rad]
    KD_ATT = 8000.0      # attitude (tilt) rate damping [N m s/rad] (sweep:
                         # 6000 -> 0.50, 8000 -> 0.58, 10000 -> 0.45 at 64
                         # envs; without the aero FF the loop wants this)
    KD_YAW = 6.0         # yaw-rate damping [N m s/rad]
    BRAKE_FRAC = 0.90    # braking-distance uses this fraction of max net decel
    M_ON = 15.0          # ignition margin [m] (lights earlier -> safety margin)
    V_BAND = 0.15        # fixed speed-band hysteresis on the descent-speed error
    V_CUT = 0.10         # absolute cut-off floor (never thrust while stopped)
    H_TERM = 2.0         # terminal pulse-leg altitude [m]
    V_TERM_OFF = 0.30    # terminal pulse cut speed [m/s]
    V_TERM_ON = 0.40     # terminal pulse relight speed [m/s] (fixed hysteresis)
    KV_LAT = 1.5         # pulse-leg lateral VELOCITY damping [1/s]
    A_LAT_PULSE = 1.0    # pulse-leg lateral accel cap [m/s^2] (tune grid:
                         # 1.0 -> 0.531 vs 1.5 -> 0.484 tuning-seed success)
    FIN_CAP = 0.95       # share of the fin torque authority used by the loop
    GIM_CAP = 0.95       # share of the gimbal torque authority used by the loop

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        f64 = torch.float64
        self.ez = torch.tensor((0., 0., 1.), device=d, dtype=f64)
        # started: edge latch for the braking-distance ignition switch (cleared
        # by reset_idx, mirroring the env)
        self.started = torch.zeros(self.num_envs, device=d, dtype=torch.bool)
        z = torch.zeros(self.num_envs, device=d, dtype=f64)
        self.diag = {k: z.clone() for k in
                     ('steps', 'sat_max', 'coast', 'ignitions', 'pad_cut',
                      'fin_sat', 'gim_sat', 'lat_cap')}

    def reset_idx(self, mask):
        self.started.masked_fill_(mask, False)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        ignited_obs = o[:, 24] > .5
        fuel = o[:, 25].clamp(0., 1.)
        pad_c = o[:, 29] > .5
        m = self.M_DRY + fuel * self.FUEL_FULL
        R = _quat_to_R(q)
        v_w = torch.einsum('nij,nj->ni', R, vb)
        h = (p[:, 2] - self.REST_Z).clamp_min(0.)

        # ---- 1. horizontal position-velocity PD
        a_xy = -self.KP_XY * p[:, :2] - self.KD_XY * v_w[:, :2]

        # ---- 2. height-scheduled descent speed + vertical velocity feedback
        v_ref = -torch.minimum(torch.full_like(h, self.V_MAX), self.K_H * h)
        a_z = self.KV_Z * (v_ref - v_w[:, 2])
        f_z = a_z + self.G                       # specific force along world +z
        f_z_att = f_z.clamp_min(self.G)          # attitude reference stays upright
        lat_cap = (self.LAT_TILT * f_z_att).clamp(1., self.A_LAT_MAX)
        n_xy = a_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a_xy = a_xy * (lat_cap[:, None] / n_xy).clamp_max(1.)
        # pulse leg (below H_TERM): damp lateral VELOCITY only, never chase
        # position.  With min thrust > weight every degree of commanded tilt is
        # attitude work the gimbal must redo after every relight, and a
        # position chase goes into a grow-and-reverse limit cycle that skates
        # the legs across the pad -- the drift must simply stop.
        a_lat_pulse = (-self.KV_LAT * v_w[:, :2]).clamp(
            -self.A_LAT_PULSE, self.A_LAT_PULSE)
        a_xy = torch.where((h < self.H_TERM)[:, None], a_lat_pulse, a_xy)
        self.diag['lat_cap'] += (n_xy[:, 0] > lat_cap).to(torch.float64)

        # ---- 5. ignition by braking distance (fixed hysteresis, no learning)
        v_dn = (-v_w[:, 2]).clamp_min(0.)
        v_ref_dn = (-v_ref).clamp_min(0.)
        a_brake = (self.BRAKE_FRAC * (self.T_MAX / m - self.G)).clamp_min(1.)
        h_need = ((v_dn.square() - v_ref_dn.square()).clamp_min(0.)
                  / (2. * a_brake))
        # the distance switch is an EDGE trigger: it fires the first descent
        # burn (only while descending, so the altitude margin cannot relight
        # the engine on a bounce); afterwards the speed band alone decides
        # cut/relight as a proper hysteresis (cut at lo < relight at hi), so
        # the engine cannot chatter between lo and the distance floor.
        first = ((h <= h_need + self.M_ON) & (v_dn > self.V_BAND)
                 & ~self.started)
        lo = (v_ref_dn - self.V_BAND).clamp_min(self.V_CUT)
        hi = (v_ref_dn + self.V_BAND).clamp_min(self.V_CUT + self.V_BAND)
        # close to the pad the linear speed reference is below the 40 Hz pulse
        # resolution, so the same switch uses a fixed terminal band instead of
        # a vanishing distance; it is still a fixed hysteresis, not a tuner.
        term = h < self.H_TERM
        lo = torch.where(term, torch.full_like(lo, self.V_TERM_OFF), lo)
        hi = torch.where(term, torch.full_like(hi, self.V_TERM_ON), hi)
        ign = torch.where(ignited_obs,
                          v_dn >= lo,
                          first | (self.started & (v_dn > hi)))
        # the public pad-contact flag is the engine switch on the pad: the
        # engine is forced off while ANY contact is sensed.  There is NO
        # permanent cut latch on purpose: after a micro-bounce launch the flag
        # clears and the speed band re-catches the rebound on the way down
        # (relight at hi, brake at min thrust) so it can never re-contact past
        # the 1.0 m/s pre-contact budget -- a hard cut-after-touchdown latch
        # turns the rebound into an uncaught fatal re-contact.
        ign &= ~pad_c
        self.started |= ign
        n_on = ign.sum().to(torch.float64)
        self.diag['ignitions'] += (ign & ~ignited_obs).to(torch.float64).sum()
        self.diag['coast'] += (self.num_envs - n_on)
        self.diag['pad_cut'] += pad_c.sum().to(torch.float64)

        # ---- 3. thrust magnitude and desired axis
        sig = torch.cat((a_xy, f_z.clamp_min(0.)[:, None]), -1)
        T_req = m * sig.norm(dim=-1)
        T_min = self.MIN_RATIO * self.T_MAX
        T_cmd = torch.where(ign, T_req.clamp(T_min, self.T_MAX),
                            torch.zeros_like(T_req))
        u_thr = ((T_cmd / self.T_MAX - self.MIN_RATIO)
                 / (1. - self.MIN_RATIO)).clamp(0., 1.)
        self.diag['sat_max'] += (ign & (u_thr >= .999)).to(torch.float64).sum()

        # ---- desired thrust axis: the specific-thrust vector the engine
        # produces when lit, s = T/m = clamp(|sigma|, T_min/m, T_MAX/m) with
        # sigma = (a_xy, f_z).  The body must tilt sin(tilt) = a_xy / s so the
        # DELIVERED lateral acceleration equals the command at a unit plant
        # gain.  Referencing f_z instead (atan(a_xy/f_z)) leaves the
        # min-thrust clamp to multiply the loop gain by s/f_z: ~2x overgained
        # in the pulse leg (s ~ 20.5 vs f_z ~ 9.8 specific force, so up to
        # 8.7 deg of stale tilt per relight -- the tilted min-thrust pulses
        # skate the legs across the pad in a grow-and-reverse limit cycle)
        # and ~3x undergained at full brake.  The reference is kept across
        # coast/burn transitions (the next light is on the same clamp), so
        # the pulse train never chases an upright setpoint it cannot track.
        s_ref = (T_req / m).clamp(self.MIN_RATIO * self.T_MAX / m,
                                  self.T_MAX / m)
        a_lat2 = a_xy.square().sum(-1, keepdim=True)
        f_vert = (s_ref[:, None].square() - a_lat2).clamp_min(1e-6).sqrt()
        f_att = torch.cat((a_xy, f_vert), -1)
        n_att = f_att.norm(dim=-1).clamp_min(1e-9)
        b3 = f_att / n_att[:, None]
        b3 = torch.where((b3[:, 2] > .15)[:, None], b3, self.ez.expand_as(b3))
        u = torch.einsum('nji,nj->ni', R, b3)     # desired body +z in body frame
        e_tilt = torch.stack((-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])), -1)

        # ---- 4. attitude/rate PD -> torque demand
        tau = torch.cat((self.KP_ATT * e_tilt[:, :2] - self.KD_ATT * wb[:, :2],
                         -self.KD_YAW * wb[:, 2:3]), -1)

        # ---- torque allocation with fixed physical gains (TVC first, fins rest)
        V2 = vb.square().sum(-1)
        cm_z = (47. * -2. + 0.2 * 2.051) / m
        z_fin = 2.051 - cm_z
        lever = 2. + cm_z
        k_gim = (lever * T_cmd * self.GIMBAL_RAD).clamp_min(1e-9)
        k_fin = (self.FIN_K * z_fin * V2).clamp_min(1e-9)
        k_yaw = (self.FIN_YAW_K * V2).clamp_min(1e-9)
        tau_gim = tau[:, :2].clamp(-self.GIM_CAP * k_gim[:, None],
                                   self.GIM_CAP * k_gim[:, None])
        tau_fin = (tau[:, :2] - tau_gim).clamp(-self.FIN_CAP * k_fin[:, None],
                                               self.FIN_CAP * k_fin[:, None])
        self.diag['fin_sat'] += ((tau_fin.abs() >= self.FIN_CAP * k_fin[:, None]
                                  - 1e-12).any(-1) & (V2 > .04)).to(torch.float64)
        self.diag['gim_sat'] += ((tau_gim.abs() >= self.GIM_CAP * k_gim[:, None]
                                  - 1e-12).any(-1) & (T_cmd > 1.)).to(torch.float64)
        fx = tau_fin[:, 1] / k_fin
        fy = -tau_fin[:, 0] / k_fin
        gx = torch.where(T_cmd > 1., -tau_gim[:, 0] / k_gim,
                         torch.zeros_like(k_gim))
        gy = torch.where(T_cmd > 1., -tau_gim[:, 1] / k_gim,
                         torch.zeros_like(k_gim))
        yaw = -self.KD_YAW * wb[:, 2] / k_yaw
        # Unified 7-dim finlet mixer convention (plant.py applies the mixer
        # internally). Ignition/throttle in [-1,1]; env scales to [0,1].
        raw = torch.stack((fx, fy, yaw,
                           torch.where(ign, torch.ones_like(u_thr),
                                       -torch.ones_like(u_thr)),
                           2. * u_thr - 1., gx, gy), -1)
        self.diag['steps'] += 1.
        return raw.clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# satellite: quaternion PD + singularity-robust CMG steering
# --------------------------------------------------------------------------- #
class _SatelliteCMGPDBase(Baseline):
    """Shared quaternion-PD torque command + CMG geometry for satellite
    steering-law A/Bs (``SatelliteCMGBaseline`` damped inverse vs
    ``SatelliteOrdinaryPinvBaseline`` Moore-Penrose inverse).

    Convention audit (every item verified against SatelliteAttitudeEnv):
    - quaternions are (w,x,y,z) Hamilton with qdot = 0.5*q(x)omega and omega in
      the BODY frame (env _dynamics); the 'policy' obs layout is
      [q, quat_err, omega, sin(theta), cos(theta), gimbal_rate, m_norm, alloc].
    - quat_err = log(q^{-1} (x) q_target) lives in the body frame and obeys
      edot ~ -omega for a static target, so the rate command is +kp*err
      (NOT -kp): a positive rate about the err axis shrinks the error.
    - the env plant is I*wdot = -w x (I w) - w x (h sum r) + tau with diagonal
      I and stored rotor momentum h*sum r_i(theta); the required CMG torque is
      tau = I*wd + w x (I w) + w x (h sum r) (exact cancellation feedforward).
    - the action is NEVER a body torque: the plant maps gimbal rates to
      tau = -h_dyn * G delta with G columns g_i x r_i(theta).  In the scaled
      factorization used below gr = rotor_speed*G and kappa = h/rotor_speed, so
      tau = -kappa gr^T delta; G is reconstructed from the observed gimbal
      angles bit-exactly against env._manipulability (probe-verified, 4- and
      6-CMG).
    - gimbal limits: the plant has no hard stops (angles wrap at +-pi); the
      +-action_scale (3 rad/s) rate limit is enforced by the final clamp in
      ``_apply_rate_limit`` and every saturation is counted.
    - torque accounting: the reported torque is the ACTUAL torque produced
      after the rate clamp, tau_actual = -kappa gr^T delta_applied, evaluated
      at the pre-step gimbal angles.  ``tau_cmd`` is the PD command and
      tau_err = tau_actual - tau_cmd is reported (not silently dropped).
    - numerical protection: tau and the observed m_norm are passed through
      nan_to_num (nonfinite inputs steer toward zero rate instead of poisoning
      the solve) and every nonfinite intermediate is counted.
    """
    kp, kd = 3., 4.

    def __init__(self, env):
        super().__init__(env)
        self.n_cmg = env.num_gimbals
        self.h_dyn = env.rotor_h_dyn
        self.rotor_speed = env.rotor_speed
        self.kappa = self.h_dyn / self.rotor_speed      # tau = -kappa * gr^T delta
        self.I = env.inertia.to(torch.float64)
        self.rotor_ref = env.rotor_axes_ref.to(torch.float64)
        self.g_cross_r0 = env.g_cross_r0.to(torch.float64)
        self.action_scale = env.action_scale
        # Standard numerical rank threshold for the Moore-Penrose solve: the
        # LAPACK/pinv convention rtol = max(m, n) * eps(dtype), atol = 0.  The
        # threshold is RELATIVE to the largest singular value, so it is
        # independent of the rotor-speed scaling of gr.
        self.pinv_rtol = max(3, self.n_cmg) * torch.finfo(torch.float64).eps
        self.pinv_atol = 0.0
        self.last_rate_request = torch.zeros(env.num_envs, device=self.device,
                                             dtype=torch.float64)
        # on-device diagnostic accumulators (no host sync in the step loop)
        z = lambda: torch.zeros((), device=self.device, dtype=torch.float64)
        self.metrics = {'steps': z(), 'clamp_envsteps': z(), 'nonfinite_envsteps': z(),
                        'm_below_envsteps': z(),
                        'm_min': torch.full((), float('inf'), device=self.device,
                                            dtype=torch.float64),
                        'hsum_max': z(),
                        # steering-law / torque accounting
                        'sat_gimbal_envsteps': z(), 'rank_def_envsteps': z(),
                        'rate_request_max': z(), 'rate_request_norm_sum': z(),
                        'rate_applied_norm_sum': z(),
                        'tau_cmd_norm_sum': z(), 'tau_act_norm_sum': z(),
                        'tau_err_norm_sum': z(), 'tau_err_max': z(),
                        'tau_err_rel_sum': z(), 'm_norm_sum': z(),
                        'sigma_min_ratio_sum': z(),
                        'sigma_min_ratio_min': torch.full((), float('inf'),
                                                          device=self.device,
                                                          dtype=torch.float64)}

    def _pd_command(self, o):
        """Quaternion PD + exact gyroscopic feed-forward (shared, gains fixed)."""
        err, w = o[:, 4:7], o[:, 7:10]
        n = self.n_cmg
        sin, cos = o[:, 10:10 + n], o[:, 10 + n:10 + 2 * n]
        m_norm = o[:, 10 + 3 * n]

        # gimbal-frame cross products from the observed angles, scaled exactly as
        # the plant does (gr = rotor_speed * g x r, bit-exact vs the env)
        gr = self.rotor_speed * (cos[:, :, None] * self.g_cross_r0[None, :, :]
                                 - sin[:, :, None] * self.rotor_ref[None, :, :])
        # stored rotor momentum sum r_i
        r = (cos[:, :, None] * self.rotor_ref[None, :, :]
             + sin[:, :, None] * self.g_cross_r0[None, :, :])
        h_sum = self.h_dyn * r.sum(1)

        # PD + gyroscopic compensation (I diagonal, body frame)
        # quat_err is log(current^-1 * target) in the body frame: a positive
        # rotation about that axis reduces the error, hence +kp * err
        wd = self.kp * err - self.kd * w
        tau = self.I[None, :] * wd + torch.linalg.cross(w, self.I[None, :] * w, dim=-1) \
            + torch.linalg.cross(w, h_sum, dim=-1)
        return tau, gr, m_norm, h_sum

    def _steer(self, tau, gr, m_norm):
        raise NotImplementedError

    def _record_rank(self, gr):
        """Rank diagnostics for the 3xN Jacobian (no effect on the command)."""
        s = torch.linalg.svdvals(gr)
        ratio = s[:, -1] / s[:, 0].clamp_min(torch.finfo(torch.float64).tiny)
        self.metrics['sigma_min_ratio_sum'] += ratio.sum()
        self.metrics['sigma_min_ratio_min'] = torch.minimum(
            self.metrics['sigma_min_ratio_min'], ratio.min())
        self.metrics['rank_def_envsteps'] += (ratio < self.pinv_rtol).sum()

    def _apply_rate_limit(self, delta, tau_cmd, gr):
        """Clip to the env's gimbal-rate bound and account the ACTUAL torque."""
        self.metrics['steps'] += 1
        over = delta.abs() > self.action_scale
        self.metrics['clamp_envsteps'] += over.any(-1).sum()
        self.metrics['sat_gimbal_envsteps'] += over.sum()
        self.metrics['rate_request_max'] = torch.maximum(
            self.metrics['rate_request_max'], delta.abs().max())
        self.metrics['rate_request_norm_sum'] += delta.norm(dim=-1).sum()
        # per-replica peak PRE-clamp rate, exposed for external harnesses that
        # need the requested (unclipped) rate of the last call
        self.last_rate_request = delta.abs().max(dim=-1).values.detach()

        action = (delta / self.action_scale).clamp(-1., 1.)
        # the env casts the action to float32 then scales by action_scale; use
        # exactly the delivered rate for the torque accounting
        delta_applied = action.to(torch.float32).to(torch.float64) * self.action_scale
        self.metrics['rate_applied_norm_sum'] += delta_applied.norm(dim=-1).sum()
        tau_act = -self.kappa * (gr.transpose(1, 2) @ delta_applied.unsqueeze(-1)).squeeze(-1)

        err = tau_act - tau_cmd
        err_norm = err.norm(dim=-1)
        self.metrics['tau_cmd_norm_sum'] += tau_cmd.norm(dim=-1).sum()
        self.metrics['tau_act_norm_sum'] += tau_act.norm(dim=-1).sum()
        self.metrics['tau_err_norm_sum'] += err_norm.sum()
        self.metrics['tau_err_max'] = torch.maximum(self.metrics['tau_err_max'], err_norm.max())
        self.metrics['tau_err_rel_sum'] += (
            err_norm / tau_cmd.norm(dim=-1).clamp_min(1e-12)).sum()
        return action.to(torch.float32)

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        tau, gr, m_norm, h_sum = self._pd_command(o)

        # diagnostics: nonfinite inputs, manipulability floor, stored momentum
        bad = ~torch.isfinite(tau).all(-1) | ~torch.isfinite(m_norm)
        self.metrics['nonfinite_envsteps'] += bad.sum()
        self.metrics['m_min'] = torch.minimum(
            self.metrics['m_min'],
            torch.where(torch.isfinite(m_norm), m_norm,
                        torch.full_like(m_norm, float('inf'))).min())
        self.metrics['m_below_envsteps'] += (m_norm < .1).sum()
        self.metrics['hsum_max'] = torch.maximum(self.metrics['hsum_max'], h_sum.norm(dim=-1).max())
        self.metrics['m_norm_sum'] += torch.nan_to_num(m_norm).sum()

        tau = torch.nan_to_num(tau)
        delta = self._steer(tau, gr, m_norm)
        self.metrics['nonfinite_envsteps'] += (~torch.isfinite(delta).all(-1)).sum()
        delta = torch.nan_to_num(delta)
        self._record_rank(gr)
        return self._apply_rate_limit(delta, tau, gr)

    def config(self):
        """Reproducibility record for reports (no effect on the control law)."""
        return {'name': self.name, 'steering_law': 'damped_moore_penrose_pinv',
                'num_cmgs': self.n_cmg, 'kp': self.kp, 'kd': self.kd,
                'pinv_rtol': self.pinv_rtol, 'pinv_atol': self.pinv_atol,
                'rank_threshold_rule': 'max(m,n)*eps(float64) (LAPACK/pinv default)',
                'rate_limit_rad_s': self.action_scale,
                'rotor_speed_rad_s': self.rotor_speed,
                'rotor_h_dyn': self.h_dyn, 'kappa': self.kappa,
                'inertia': self.I.tolist(),
                'torque_metric': 'tau_actual = -kappa * gr^T @ delta_applied at pre-step gimbal angles'}


class SatelliteCMGBaseline(_SatelliteCMGPDBase):
    """Quaternion PD + singularity-robust damped-pseudoinverse CMG steering.

    The damped pseudoinverse is delta = -(1/kappa) gr (gr^T gr + lam^2 I)^-1 tau
    (Schaub & Junkins), written in the rotor_speed-scaled factorization that is
    algebraically identical to delta = -(1/h) G (G G^T + lam~^2 I)^-1 tau.
    Damping lam~ = lam0 + 0.15*exp(-3 m_norm) grows as manipulability
    collapses: torque relay error 0.05% far from singularities, ~3% fully
    damped, with gimbal rates bounded throughout (SR steering).
    """
    name = 'quat_pd_damped_cmg'

    def __init__(self, env):
        super().__init__(env)
        self.lam0 = .02

    def _steer(self, tau, gr, m_norm):
        # singularity-robust damped pseudoinverse (Schaub & Junkins):
        # tau = -kappa gr^T delta  =>  delta = -(1/kappa) gr (gr^T gr + lam^2 I)^-1 tau
        lam = self.rotor_speed * (self.lam0 + .15 * torch.exp(-3. * m_norm))
        JtJ = gr.transpose(1, 2) @ gr
        JtJ = JtJ + torch.eye(3, device=self.device, dtype=torch.float64)[None] * (lam ** 2)[:, None, None]
        return -(1. / self.kappa) * (gr @ torch.linalg.solve(JtJ, tau.unsqueeze(-1))).squeeze(-1)

    def config(self):
        cfg = super().config()
        cfg.update(steering_law='damped_moore_penrose_pinv', lam0=self.lam0,
                   damping_schedule='lam = rotor_speed*(lam0 + 0.15*exp(-3*m_norm))')
        return cfg


class SatelliteOrdinaryPinvBaseline(_SatelliteCMGPDBase):
    """Quaternion PD + ORDINARY Moore-Penrose CMG steering (ablation baseline).

    Same quaternion-error convention, same PD gains (kp=3, kd=4) and same exact
    gyroscopic feed-forward as ``SatelliteCMGBaseline``; only the torque ->
    gimbal-rate steering law differs, so the comparison isolates the steering
    law.  The Moore-Penrose pseudoinverse is the SVD-based ``torch.linalg.pinv``
    with the standard numerical rank threshold rtol = max(m, n)*eps(float64),
    atol = 0 (LAPACK/pinv convention), applied to the 3 x N scaled Jacobian:

        delta = -(1/kappa) pinv(gr) tau,   tau = -kappa gr^T delta.

    No damping schedule, no null-space motion, no singularity avoidance, no
    predictive control and no reference time scaling.  When the Jacobian is
    near rank-deficient the pseudoinverse legitimately produces large rates;
    those rates are clipped to the env's +-action_scale bound and the resulting
    torque shortfall (tau_actual - tau_cmd) is recorded rather than remedied.
    Works for the 4-CMG pyramid and the 6-CMG orthogonal cluster directly from
    the matrix dimensions (3 x N -> N x 3).
    """
    name = 'quat_pd_mp_pinv_cmg'

    def _steer(self, tau, gr, m_norm):
        # ordinary Moore-Penrose pseudoinverse (SVD, truncated at the standard
        # relative rank threshold).  gr holds the N Jacobian ROWS (n, N, 3), so
        # the torque map is tau = -kappa * gr^T delta and the steering uses
        # pinv(gr^T) = pinv(gr)^T -> (n, N, 3).
        gr_pinv = torch.linalg.pinv(gr.transpose(1, 2),
                                     rtol=self.pinv_rtol, atol=self.pinv_atol)
        return -(1. / self.kappa) * (gr_pinv @ tau.unsqueeze(-1)).squeeze(-1)

    def config(self):
        cfg = super().config()
        cfg.update(steering_law='ordinary_moore_penrose_pinv',
                   damping_schedule=None, null_space_motion=False,
                   singularity_avoidance=False, predictive_control=False,
                   reference_time_scaling=False)
        return cfg


# --------------------------------------------------------------------------- #
# transfer: Gauss planetary-equation tangential-thrust guidance
# --------------------------------------------------------------------------- #
class TransferGaussBaseline(Baseline):
    """Perigee-raising guidance from the Gauss variational equations.

    Each step the osculating elements are extracted from the observed [r, v]
    and the per-step element changes under a transverse thrust S_t are formed
    from the Gauss rates,

        da = 2 a^2/h (p/r) S_t dt
        de = ((p + r) cos(nu) + r e)/h S_t dt
        drp = (1-e) da - a de,    dra = (1+e) da + a de,

    so a prograde transverse burn near apogee raises rp at (first-order)
    fixed ra -- the classical perigee-raise burn -- while near perigee the
    same thrust only lifts ra, and mid-course it trades rp for ra.  The
    controller is a one-step-greedy maximiser of the reference terminal
    objective J = (rp - rp0) - |ra - ra0| - 2e5 (m0 - m): coast vs saturated
    prograde vs saturated retrograde transverse burn (+-f_max along h^ x r^)
    are scored over one control step and the burn is taken only when its
    predicted J gain exceeds the propellant cost 2e5 dm, with the mass flow
    dm = f_max |u|_1 dt / (isp g0) of the three orthogonal thruster pairs
    evaluated on the observed mass.  An optional target_rp (meters) coasts
    once the raise is achieved.

    Near-circular numerical stability: nu is recovered from the O(e) pair
    (e sin nu, e cos nu) = (h vr/mu, p/r - 1) through atan2, so no quantity
    is divided by e and every rate stays bounded as e -> 0.  The task's
    eccentricity spans [0.02, 0.1], five orders above the float32
    observation-noise floor (~1e-6) below which the apsidal direction is
    unresolvable and the gate degrades to bounded, objective-anchored
    chatter (ra is driven toward ra0 by the |ra - ra0| term) rather than to
    divergence.
    """
    name = 'gauss_tangential'

    def __init__(self, env, target_rp=None):
        super().__init__(env)
        self.mu, self.dt = env.mu, env.dt
        self.f_max, self.isp, self.g0 = env.f_max, env.isp, env.g0
        self.rp0 = env.sma0 * (1. - env.ecc0)     # nominal reference orbit
        self.ra0 = env.sma0 * (1. + env.ecc0)
        self.prop_cost = 2e5                       # J weight of propellant [1/kg]
        self.target_rp = target_rp                 # optional perigee target [m]

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        r, v, m = o[:, 1:4], o[:, 4:7], o[:, 7:8]
        rm = r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        hvec = torch.linalg.cross(r, v, dim=-1)
        h = hvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a = 1. / (2. / rm - (v * v).sum(-1, keepdim=True) / self.mu)
        evec = torch.linalg.cross(v, hvec, dim=-1) / self.mu - r / rm
        e = evec.norm(dim=-1, keepdim=True)
        p = h * h / self.mu
        that = torch.linalg.cross(hvec / h, r / rm, dim=-1)
        vr = (v * r).sum(-1, keepdim=True) / rm

        # nu from the O(e) pair (h vr/mu, p/r - 1) = (e sin nu, e cos nu):
        # atan2 divides by nothing, so the pair vanishing together as e -> 0
        # keeps sin/cos bounded and consistent
        nu = torch.atan2(vr * h / self.mu, p / rm - 1.)

        # Gauss per-step radii rates for a saturated transverse burn
        sdt = self.f_max / m * self.dt
        da = 2. * a ** 2 / h * (p / rm) * sdt
        de = ((p + rm) * nu.cos() + rm * e) / h * sdt
        drp = (1. - e) * da - a * de
        dra = (1. + e) * da + a * de
        dm = self.f_max * that.abs().sum(-1, keepdim=True) / (self.isp * self.g0) * self.dt

        # one-step objective gain: coast (0) vs prograde vs retrograde, net of
        # the propellant cost; the |ra - ra0| kink is carried exactly
        ra = a * (1. + e)
        d0 = (ra - self.ra0).abs()
        net = torch.stack((drp - ((ra + dra - self.ra0).abs() - d0),
                           -drp - ((ra - dra - self.ra0).abs() - d0)), -1) \
            - self.prop_cost * dm.unsqueeze(-1)
        cand = torch.cat((torch.zeros_like(net[..., :1]), net), -1)   # coast, +t, -t
        pick = cand.argmax(-1)
        s = torch.where(pick == 1, 1., torch.where(pick == 2, -1., 0.))
        if self.target_rp is not None:
            s = torch.where(a * (1. - e) < self.target_rp, s, torch.zeros_like(s))
        return (s * that).to(torch.float32)


# --------------------------------------------------------------------------- #
# transfer variant: Petropoulos Q-law Lyapunov guidance
# --------------------------------------------------------------------------- #
class TransferQLawBaseline(Baseline):
    """Q-law proximity-quotient guidance (Petropoulos, AAS 04-278, "Low-Thrust
    Orbit Transfers Using Candidate Lyapunov Functions with a Mechanism for
    Coasting"; AAS 05-130, "Simple Control Laws for Low-Thrust Orbit
    Transfers") for the perigee-raising transfer task, restricted to the
    planar element subset (a, e): the task requires no plane change, so
    i, Omega, omega are left free (their Gauss rates carry only out-of-plane
    or e-singular terms, irrelevant here).

        Q = W_a [S_a (a - a_T)/a_xx]^2 + W_e [(e - e_T)/e_xx]^2,
        S_a = (1 + ((a - a_T)/(3 a_T))^4)^(1/2),

    where z_xx = max over thrust direction AND true anomaly of z-dot on the
    current osculating orbit -- the best-case local velocity of each element,
    making Q a weighted sum of optimistic per-element transfer times.  The
    maxima are evaluated on a dense true-anomaly grid of the osculating orbit
    (batched; the semi-major-axis maximum coincides with the analytic
    (1 + e) form, checked in the validation harness).  The thrust direction
    is the analytic steepest descent of Q through the Gauss variational
    equations,

        u_hat = -(D_r, D_t)/|D|,   D = Q_a G_a(nu) + Q_e G_e(nu),

    with G_a/G_e the Gauss rate row vectors in the radial-transverse frame
    (evaluated at the current true anomaly nu) and Q_a, Q_e the exact
    partials of Q w.r.t. (a, e), obtained by autodiff through the grid maxima
    (envelope theorem; finite-difference checked in the harness).  The thrust
    acceleration f_max/m multiplies every rate, so it cancels in both the
    normalized direction and the coast ratio and is omitted from the
    scalings (standard Q-law treatment: mass is not a Q coordinate).

    Coasting (Petropoulos's effectivity mechanism): the instantaneous optimal
    decrease rate is compared against the best rate achievable anywhere on
    the osculating orbit,

        eta = |Q_dot_min(nu)| / max_nu |Q_dot_min| = |D| / D_xx,

    and the thruster shuts off when eta < eta_min, concentrating the burn on
    the efficient arc (near apogee for a perigee raise).  eta_min and the
    weights are fixed scalars -- no scheduling, no learning.

    Task-oriented references (the gain-type objective R has no unique target
    orbit): the perigee reference rp_ref is a public fixed parameter
    (constructor argument, frozen after selection on independent tuning
    seeds); the apogee reference is the INITIAL apogee of the spawn orbit, so
    the implied target elements a_T = (ra0 + rp_ref)/2 and
    e_T = (ra0 - rp_ref)/(ra0 + rp_ref) encode "raise perigee while holding
    apogee".  This is a Q-law configuration FOR THIS TASK and is not claimed
    to be equivalent to the untargeted gain-type objective.

    Interface contract (identical to TransferGaussBaseline): reads only the
    'policy' group o = [t, r, v, m] (float32 buffer, cast to float64);
    constants mu/dt/f_max/isp/g0 and the spawn radii are taken from the env
    at construction; the emitted action is the Euclidean-unit thrust
    direction rotated from the orbital (RSW) frame to inertial,
    u = u_r r^ + u_t t^ with t^ = h^ x r^, so every component lies within the
    env's per-axis [-1, 1] bound and propellant follows the env's own
    f_max |u|_1/(isp g0) mass-flow model.  Stateless; reset_idx is the
    base-class no-op.
    """
    name = 'q_law_petropoulos'

    A_SCALE_M, A_SCALE_N, A_SCALE_R = 3., 4., 2.  # S_a hyperparameters (Petropoulos nominal m, n, r)
    N_GRID = 2048      # true-anomaly grid for the xx scalings / coast reference
    RP_RAISE = 20e3    # perigee-reference raise above rp0 [m] (frozen via tuning seeds)
    W_A, W_E = 1., 1.  # element weights (frozen via tuning seeds)
    ETA_MIN = .1       # effectivity coast threshold (frozen via tuning seeds)

    def __init__(self, env, rp_ref=None, w_a=None, w_e=None, eta_min=None):
        super().__init__(env)
        self.mu, self.dt = env.mu, env.dt
        self.f_max, self.isp, self.g0 = env.f_max, env.isp, env.g0
        self.rp0 = env.sma0 * (1. - env.ecc0)
        self.ra_ref = env.sma0 * (1. + env.ecc0)     # apogee reference = initial apogee
        self.rp_ref = float(rp_ref) if rp_ref is not None else self.rp0 + self.RP_RAISE
        self.a_T = .5 * (self.ra_ref + self.rp_ref)
        self.e_T = (self.ra_ref - self.rp_ref) / (self.ra_ref + self.rp_ref)
        self.w_a = self.W_A if w_a is None else float(w_a)
        self.w_e = self.W_E if w_e is None else float(w_e)
        self.eta_min = self.ETA_MIN if eta_min is None else float(eta_min)
        g = torch.arange(self.N_GRID, device=self.device, dtype=torch.float64)
        self._sin_g = torch.sin(2. * math.pi * g / self.N_GRID)
        self._cos_g = torch.cos(2. * math.pi * g / self.N_GRID)

    def config(self):
        """Frozen law parameters, recorded verbatim in evaluation reports."""
        return {'law': 'Petropoulos Q-law, planar (a, e) subset',
                'rp_ref_m': self.rp_ref, 'ra_ref_m': self.ra_ref,
                'a_T_m': self.a_T, 'e_T': self.e_T,
                'w_a': self.w_a, 'w_e': self.w_e, 'eta_min': self.eta_min,
                'n_grid': self.N_GRID,
                's_a_scale_mnr': (self.A_SCALE_M, self.A_SCALE_N, self.A_SCALE_R)}

    def _a_xx_grid(self, a, e):
        """Detached grid maximum of da/dt over (thrust direction, anomaly);
        exposed for the validation harness (analytic form: (2a^2/h)(1+e))."""
        with torch.no_grad():
            p = a * (1. - e * e)
            h = torch.sqrt(self.mu * p)
            r_k = p[:, None] / (1. + e[:, None] * self._cos_g)
            da = (2. * a[:, None] ** 2 / h[:, None]) * torch.sqrt(
                (e[:, None] * self._sin_g) ** 2 + (p[:, None] / r_k) ** 2)
            return da.max(-1).values

    def _q_partials(self, a, e):
        """Q(a, e) and its exact partials (autodiff through the grid maxima)."""
        a = a.detach().requires_grad_(True)
        e = e.detach().requires_grad_(True)
        with torch.enable_grad():
            p = a * (1. - e * e)
            h = torch.sqrt(self.mu * p)
            sg, cg = self._sin_g, self._cos_g
            r_k = p[:, None] / (1. + e[:, None] * cg)     # osculating-orbit radius grid
            pr = p[:, None] / r_k
            # element rates maximized over the thrust direction at each anomaly
            da = (2. * a[:, None] ** 2 / h[:, None]) * torch.sqrt(
                (e[:, None] * sg) ** 2 + pr ** 2)
            de = (1. / h[:, None]) * torch.sqrt((p[:, None] * sg) ** 2
                + ((p[:, None] + r_k) * cg + r_k * e[:, None]) ** 2)
            a_xx = da.max(-1).values                      # ... then over true anomaly
            e_xx = de.max(-1).values
            da_err = a - self.a_T
            x = da_err / (self.A_SCALE_M * self.a_T)
            s_a = (1. + x ** self.A_SCALE_N) ** (1. / self.A_SCALE_R)
            q = self.w_a * (s_a * da_err / a_xx) ** 2 \
                + self.w_e * ((e - self.e_T) / e_xx) ** 2
            q_a, q_e = torch.autograd.grad(q.sum(), (a, e))
        return q.detach(), q_a.detach(), q_e.detach()

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        r, v = o[:, 1:4], o[:, 4:7]
        rm = r.norm(dim=-1).clamp_min(1e-9)
        hvec = torch.linalg.cross(r, v, dim=-1)
        h = hvec.norm(dim=-1).clamp_min(1e-9)
        a = 1. / (2. / rm - (v * v).sum(-1) / self.mu)
        evec = torch.linalg.cross(v, hvec, dim=-1) / self.mu - r / rm[:, None]
        e = evec.norm(dim=-1)
        p = h * h / self.mu
        rhat = r / rm[:, None]
        that = torch.linalg.cross(hvec / h[:, None], rhat, dim=-1)  # t^ = h^ x r^
        vr = (v * r).sum(-1) / rm
        nu = torch.atan2(vr * h / self.mu, p / rm - 1.)             # e-safe atan2 pair

        _, q_a, q_e = self._q_partials(a, e)

        # Gauss rate row vectors at the current anomaly (radial / transverse)
        sn, cn = nu.sin(), nu.cos()
        d_r = q_a * (2. * a * a / h) * e * sn + q_e * (p * sn / h)
        d_t = q_a * (2. * a * a / h) * (p / rm) + q_e * (((p + rm) * cn + rm * e) / h)
        d_norm = torch.sqrt(d_r * d_r + d_t * d_t)

        # effectivity reference: max over the osculating orbit of |Phi' Q_z|
        r_k = p[:, None] / (1. + e[:, None] * self._cos_g)
        ga_r = (2. * a[:, None] ** 2 / h[:, None]) * (e[:, None] * self._sin_g)
        ga_t = (2. * a[:, None] ** 2 / h[:, None]) * (p[:, None] / r_k)
        ge_r = (p[:, None] * self._sin_g) / h[:, None]
        ge_t = ((p[:, None] + r_k) * self._cos_g + r_k * e[:, None]) / h[:, None]
        mag = torch.sqrt((q_a[:, None] * ga_r + q_e[:, None] * ge_r) ** 2
                         + (q_a[:, None] * ga_t + q_e[:, None] * ge_t) ** 2)
        d_xx = mag.max(-1).values

        burn = (d_norm >= self.eta_min * d_xx) & (d_norm > 0.)
        scale = torch.where(burn, 1. / d_norm.clamp_min(1e-300), torch.zeros_like(d_norm))
        u_r, u_t = -d_r * scale, -d_t * scale
        return (u_r[:, None] * rhat + u_t[:, None] * that).to(torch.float32)


# --------------------------------------------------------------------------- #
# docking: CWH-LQR terminal approach
# --------------------------------------------------------------------------- #
class DockingCWHLQRBaseline(Baseline):
    """CWH-LQR final approach with explicit constraint handling.

    Task contract (SpaceDockingEnv): LVLH frame (x radial, y along-track,
    z cross-track) of a 400 km circular orbit; state [p; v] in m / m/s;
    action = normalized per-axis thrust in [-1, 1] -> F = a * f_max
    (10 N/axis on m = 500 kg, i.e. 0.02 m/s^2 per-axis authority); target
    the origin; success |p| < pos_goal (1 m) AND |v| < vel_goal (0.1 m/s)
    at the same step; hard failure |p| > range_fail (200 m) or fuel
    >= fuel0 (2 kg); no keep-out sphere on docking (rendezvous-only);
    500-step horizon; symplectic Euler at dt = 1 s.

    LQR design
        The continuous CWH model is discretized EXACTLY as the plant's
        symplectic Euler step (zero discretization error, see _cwh_lqr)
        and the gain follows from the discrete Riccati equation
        (scipy solve_discrete_are), K = (R + Bd' P Bd)^-1 Bd' P Ad.
        Q/R follow Bryson normalization by the task tolerances:
            Q = diag(pos_goal^-2 I3, w_vel * vel_goal^-2 I3)
            R = (kappa * a_max)^-2 I3,   a_max = f_max / m
        With q_pos * pos_goal^2 = 1 the pure position feedback demands
        exactly kappa * a_max at the success boundary (kappa = 0.35 ->
        65% terminal authority headroom), and w_vel = 3 places the
        closed loop near critical damping (double-integrator limit:
        k2 = sqrt(w_vel + 2 sqrt(q_pos/r_acc) * ... )).  Verified
        capture margin: |v| ~ 0.066 m/s when crossing |p| = 1 m, 34%
        under vel_goal.  The far-field approach saturates by physics
        (braking 100 m of closure at 0.02 m/s^2 authority), which the
        per-axis output saturation absorbs as a bang-brake phase.

    Constraint handling (the LQR itself assumes no obstacles/constraints)
        1. Range-corridor shield: docking has no keep-out sphere, so the
           only state constraint is the 200 m range-fail bound.  With
           outward radial rate v_out, the stopping distance is
           v_out^2 / (2 a_b); when |p| + stopping distance crosses the
           0.95 * range_fail buffer, the command is overridden by a
           full-authority brake along -p_hat; inside the 0.8 * range_fail
           corridor any outward thrust component is projected out.
        2. Terminal capture net: within 2 * pos_goal, if the closing
           speed already exceeds vel_goal the command brakes the approach
           so the success sphere is crossed below the velocity tolerance.
        3. Fuel derate: the controller integrates its own realized
           impulse (exact for this plant); past 75% of fuel0 the
           per-step thrust norm is capped so the remaining budget
           covers the remaining horizon (never below half authority).
    """
    name = 'cwh_lqr'

    def __init__(self, env, kappa=.35, w_vel=3., shield=True, capture_net=True,
                 fuel_guard=True, legacy=False):
        super().__init__(env)
        self.mass = env.mass
        self.f_max = env.f_max
        self.a_max = env.f_max / env.mass
        self.dt = env.dt
        self.isp, self.g0 = env.isp, env.g0
        self.pos_goal, self.vel_goal = env.pos_goal, env.vel_goal
        self.range_fail, self.fuel0 = env.range_fail, env.fuel0
        self.max_steps = env.max_steps
        self.shield = shield and not legacy
        self.capture_net = capture_net and not legacy
        self.fuel_guard = fuel_guard and not legacy
        if legacy:  # exact pre-upgrade incumbent (taylor2 gains, no layers)
            self.K = _cwh_lqr(env, q_pos=1e-4, q_vel=1., r_acc=1., method='taylor2')
        else:
            self.K = _cwh_lqr(env, q_pos=1. / self.pos_goal ** 2,
                              q_vel=w_vel / self.vel_goal ** 2,
                              r_acc=1. / (kappa * self.a_max) ** 2)
        # shield / guard thresholds
        self.d_corridor = .8 * self.range_fail
        self.d_brake = .95 * self.range_fail
        self.a_brake = .5 * self.a_max
        self.fuel_soft = .75 * self.fuel0
        # per-episode bookkeeping (reset with the env)
        n = env.num_envs
        d = self.device
        self.fuel_est = torch.zeros(n, device=d, dtype=torch.float64)
        self.steps_left = torch.full((n,), self.max_steps, device=d, dtype=torch.long)
        self.stat_shield_hard = torch.zeros(n, device=d, dtype=torch.float64)
        self.stat_shield_soft = torch.zeros(n, device=d, dtype=torch.float64)
        self.stat_capture = torch.zeros(n, device=d, dtype=torch.float64)
        self.stat_fuel_cap = torch.zeros(n, device=d, dtype=torch.float64)
        self.stat_sat = torch.zeros(n, device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.fuel_est.masked_fill_(mask, 0.)
        self.steps_left.masked_fill_(mask, self.max_steps)
        for s in (self.stat_shield_hard, self.stat_shield_soft, self.stat_capture,
                  self.stat_fuel_cap, self.stat_sat):
            s.masked_fill_(mask, 0.)

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p, v = o[:, :3], o[:, 3:6]
        d = p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        p_hat = p / d
        a = -(self.K @ torch.cat((p, v), -1).unsqueeze(-1)).squeeze(-1)

        # 1) range-corridor shield (state constraint: |p| < range_fail)
        if self.shield:
            v_out = (v * p_hat).sum(-1, keepdim=True)
            stop = d + v_out.clamp_min(0.) ** 2 / (2 * self.a_brake)
            hard = (v_out > 0) & (stop > self.d_brake)
            u_shield = -self.a_max * p_hat.squeeze(-1)
            a = torch.where(hard, u_shield, a)
            self.stat_shield_hard += hard.squeeze(-1).to(torch.float64)
            # corridor projection: never thrust outward inside the corridor
            a_out = (a * p_hat).sum(-1, keepdim=True)
            soft = (d > self.d_corridor) & (a_out > 0)
            a = torch.where(soft, a - a_out * p_hat, a)
            self.stat_shield_soft += soft.squeeze(-1).to(torch.float64)

        # 2) terminal capture net: brake an over-fast crossing of the goal sphere
        if self.capture_net:
            v_close = -(v * p_hat).sum(-1, keepdim=True)
            near = (d < 2 * self.pos_goal) & (v_close > self.vel_goal)
            a = torch.where(near, self.a_max * p_hat.squeeze(-1), a)
            self.stat_capture += near.squeeze(-1).to(torch.float64)

        # 3) fuel derate: cap the per-step impulse so the budget outlasts the horizon
        if self.fuel_guard:
            remaining = self.steps_left.clamp_min(1).to(torch.float64)
            f_allow = (self.fuel0 - self.fuel_est).clamp_min(0.) * self.isp * self.g0 \
                / (remaining * self.dt) * self.mass
            f_allow = f_allow.clamp_min(.5 * self.f_max).unsqueeze(-1)  # never below half authority
            f_cmd = a.norm(dim=-1, keepdim=True) * self.mass
            over = (f_cmd > f_allow) & (self.fuel_est > self.fuel_soft).unsqueeze(-1)
            a = torch.where(over, a * (f_allow / f_cmd.clamp_min(1e-12)), a)
            self.stat_fuel_cap += over.squeeze(-1).to(torch.float64)

        # output saturation: per-axis normalized thrust in [-1, 1]
        action = (a * self.mass / self.f_max).clamp(-1., 1.)
        self.stat_sat += (action.abs() >= 1.).to(torch.float64).sum(-1)
        self.fuel_est += action.norm(dim=-1) * self.f_max * self.dt / (self.isp * self.g0)
        self.steps_left -= 1
        return action.to(torch.float32)


# --------------------------------------------------------------------------- #
# rendezvous: CWH two-impulse terminal guidance
# --------------------------------------------------------------------------- #
class RendezvousTwoImpulseBaseline(Baseline):
    """Receding-horizon CWH two-impulse terminal guidance to a hold point.

    Task contract (SpaceRendezvousEnv): LVLH frame (x radial, y along-track,
    z cross-track), state [p; v]; action = per-axis normalized thrust in
    [-1, 1] -> 10 N/axis on 500 kg, i.e. 0.02 m/s^2 per-axis authority and
    dv_step = 0.02 m/s of per-axis Delta-v per 1 s step (the env accepts
    continuous control only, so both impulses are realized as finite burns
    through an explicit per-axis saturation map); success |p| < pos_goal
    (10 m) AND |v| < vel_goal (0.5 m/s) at the same step; failure |p| <
    keep_out (5 m), |p| > range_fail (2000 m) or fuel >= 4 kg; 1000-step
    horizon; symplectic Euler at dt = 1 s, n = 1.13e-3 rad/s.

    Target selection: the hold point r_f = hold_radius * p0/|p0| is latched
    on the initial line of sight, between the keep-out and success spheres,
    and the terminal velocity target is null.  The transfer time is not a
    magic constant: it is seeded with the acceleration-limited straight-line
    bound 2 sqrt(|p0|/a_axis) and then RATCHETED ONLINE whenever the two
    sequential burns no longer fit before arrival; hitting the episode
    horizon flags the case infeasible (degraded to velocity nulling).

    Impulse solve (every step, from the current state, for the remaining
    time t_go -- the receding-horizon re-solve absorbs saturation timing,
    the float32 observation quantization and the dt^2 position slip of a
    one-step thrust application):
        ZEM  = r_f - (Phi_rr p + Phi_rv v)                 position error at T
        ZEV  = 0    - (Phi_vr p + Phi_vv v)                velocity error at T
        dv1 = Phi_rv^-1 ZEM                               position pulse, now
        dv2 = ZEV - Phi_vv dv1                             velocity pulse, at T
    with Phi = Ad^t_go the exact symplectic-Euler STM (Ad^k = expm(logm(Ad)/dt
    * k dt), computed batched with torch.matrix_exp), so coasting arcs are
    predicted with ZERO model error against the plant.

    Impulse -> finite-thrust mapping: the pair (dv1, dv2) is spread over the
    remaining time as the min-energy thrust program whose first step is the
    ZEM/ZEV optimal guidance law a = 6 ZEM/t_go^2 - 2 ZEV/t_go (exact first
    step of the min-energy profile for the impulse pair; CWH gravity bias is
    carried inside ZEM/ZEV by the STM, and the receding-horizon re-solve
    corrects the residual each step).  Per-axis action = clamp(a/a_axis)
    obeys the env action limit.  The program naturally burns early (pulse 1
    work), coasts mid-course where the profile crosses zero, and brakes into
    the hold point at arrival (pulse 2 work), crossing the success sphere
    below vel_goal and stopping 4 m inside the keep-out boundary.
    Ill-conditioned Phi_rv (singular manifold at n t_go = k*pi, i.e.
    half-orbital multiples far beyond this task's horizon: cond <= 1.8 for
    all t_go <= 1000 s, verified numerically) is handled by a
    Tikhonov-regularized solve plus a schedule snap off the manifold.
    """
    name = 'cwh_two_impulse'

    def __init__(self, env):
        super().__init__(env)
        self.dt = env.dt
        self.max_steps = env.max_steps
        self.mass = env.mass
        self.a_axis = env.f_max / env.mass            # per-axis accel authority [m/s^2]
        self.dv_step = self.a_axis * env.dt           # per-axis per-step Delta-v [m/s]
        self.hold = env.keep_out + .8 * (env.pos_goal - env.keep_out)   # 9 m hold radius
        self.ext_margin = 10.                         # burn-schedule slack [steps]
        self.t_cap = env.max_steps - 8.               # latest admissible arrival [steps]
        # exact discrete generator: Ad^k = expm(gen * k * dt), batched per env
        Ad = _cwh_symplectic_Ad(env)
        L = logm(Ad)
        if np.iscomplexobj(L):
            if np.abs(L.imag).max() > 1e-9:
                raise ValueError('symplectic Ad has eigenvalues on the negative real axis')
            L = L.real
        self.gen = torch.as_tensor(L / env.dt, device=self.device, dtype=torch.float64)
        n = env.num_envs
        self.rf = torch.zeros((n, 3), device=self.device, dtype=torch.float64)
        self.t_arr = torch.zeros(n, device=self.device, dtype=torch.float64)   # [steps]
        self.infeasible = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.count = torch.zeros(n, device=self.device, dtype=torch.long)

    def reset_idx(self, mask):
        self.count.masked_fill_(mask, 0)
        self.infeasible.masked_fill_(mask, False)
        self.t_arr.masked_fill_(mask, 0.)
        self.rf.masked_fill_(mask[:, None], 0.)

    def _stm(self, t_go):
        """Exact symplectic STM blocks for per-env horizons t_go [s]: (N,6,6)."""
        Phi = torch.matrix_exp(self.gen[None] * t_go[:, None, None])
        return Phi[:, :3, :3], Phi[:, :3, 3:], Phi[:, 3:, :3], Phi[:, 3:, 3:]

    def _plan(self, p, v, rf, t_go):
        """STM blocks, zero-effort errors and both impulses for arrival at rf
        with null velocity in t_go seconds."""
        Phi_rr, Phi_rv, Phi_vr, Phi_vv = self._stm(t_go)
        # ill-conditioning guard: Tikhonov-regularize near the singular
        # manifold of Phi_rv (det -> 0 as n*t_go -> k*pi)
        det = torch.linalg.det(Phi_rv).abs()
        eye = torch.eye(3, device=self.device, dtype=torch.float64).expand_as(Phi_rv)
        bad = det < 1e-10 / self.env.n ** 3
        Phi_rv = torch.where(bad[:, None, None], Phi_rv + (1e-6 / self.env.n) * eye, Phi_rv)
        r_ze = torch.bmm(Phi_rr, p.unsqueeze(-1)).squeeze(-1) \
            + torch.bmm(Phi_rv, v.unsqueeze(-1)).squeeze(-1)
        v_ze = torch.bmm(Phi_vr, p.unsqueeze(-1)).squeeze(-1) \
            + torch.bmm(Phi_vv, v.unsqueeze(-1)).squeeze(-1)
        ZEM = rf - r_ze
        ZEV = -v_ze
        dv1 = torch.linalg.solve(Phi_rv, ZEM.unsqueeze(-1)).squeeze(-1)
        dv2 = ZEV - torch.bmm(Phi_vv, dv1.unsqueeze(-1)).squeeze(-1)
        return dv1, dv2, ZEM, ZEV

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p, v = o[:, :3], o[:, 3:6]
        first = self.count == 0
        # ---- latch the hold point on the initial line of sight and seed the
        # transfer time with the acceleration-limited straight-line bound
        rf0 = self.hold * p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.rf.copy_(torch.where(first[:, None], rf0, self.rf))
        t_seed = (2. * torch.sqrt(p.norm(dim=-1) / self.a_axis) / self.dt).clamp_min(10.)
        self.t_arr.copy_(torch.where(first, torch.maximum(self.t_arr, t_seed), self.t_arr))
        t_now = self.count.to(torch.float64)

        # ---- solve; ratchet the arrival time when the sequential burns no
        # longer fit before it; flag infeasible at the horizon and degrade
        t_go = ((self.t_arr - t_now).clamp_min(1.)) * self.dt
        dv1, dv2, ZEM, ZEV = self._plan(p, v, self.rf, t_go)
        need = t_now + dv1.abs().amax(-1) / self.dv_step + dv2.abs().amax(-1) / self.dv_step \
            + self.ext_margin
        ext = (need > self.t_arr) & ~self.infeasible
        self.infeasible |= ext & (need > self.t_cap)
        self.t_arr.copy_(torch.where(ext, torch.minimum(need, torch.full_like(need, self.t_cap)),
                                     self.t_arr))
        # snap the schedule off the Phi_rv singular manifold (n*t = k*pi)
        nt = self.env.n * self.t_arr * self.dt
        bad = (nt - torch.round(nt / math.pi) * math.pi).abs() < 1e-3
        self.t_arr.copy_(torch.where(bad, self.t_arr + 8., self.t_arr))

        # ---- re-solve and realize the impulse pair as the min-energy finite-
        # thrust program (ZEM/ZEV optimal guidance first step, per-axis clamp)
        t_go = ((self.t_arr - t_now).clamp_min(1.)) * self.dt
        dv1, dv2, ZEM, ZEV = self._plan(p, v, self.rf, t_go)
        a_cmd = 6. * ZEM / (t_go * t_go)[:, None] - 2. * ZEV / t_go[:, None]
        action = (a_cmd / self.a_axis).clamp(-1., 1.)
        brake = (-v / self.dv_step).clamp(-1., 1.)
        give_up = self.infeasible | (self.t_arr <= t_now)
        action = torch.where(give_up[:, None], brake, action)
        self.count += 1
        return action.to(torch.float32)


class RendezvousCWHPDBaseline(Baseline):
    """CWH-compensated PD approach to the origin (exact feedback linearization).

    Same task contract as ``RendezvousTwoImpulseBaseline`` (SpaceRendezvousEnv;
    LVLH x radial / y along-track / z cross-track, [p; v] obs, per-axis
    normalized thrust in [-1, 1] -> 10 N/axis on 500 kg = 0.02 m/s^2 per-axis
    authority, dt = 1 s symplectic Euler, n = 1.13e-3 rad/s, success
    |p| < 10 m & |v| < 0.5 m/s, 5 m keep-out sphere, 2000 m range-fail,
    4 kg fuel, 1000-step horizon).

    Control law (fixed diagonal gains, no integral/adaptive/avoidance terms):
        a_des = -Kp p - Kd v,         Kp = wn^2 I3, Kd = 2 zeta wn I3
        F/m_x = a_des_x - 3 n^2 x - 2 n v_y       (cancel plant coupling)
        F/m_y = a_des_y + 2 n v_x
        F/m_z = a_des_z + n^2 z
    so the thrust cancels the CWH coupling terms of the plant exactly and the
    unsaturated closed loop reduces to three decoupled double integrators
    p'' = -wn^2 p - 2 zeta wn p' (per axis, symplectic-Euler discretized with
    the plant).  The command is multiplied by the mass, clamped to the
    per-axis thrust limit and mapped to the normalized action
    u = clamp(F / f_max).  No keep-out shield, planner, fuel manager or
    learning term is added: the target is the origin and the environment
    applies its own termination rules.

    Gain selection: (wn, zeta) are chosen on an independent dev-scenario set
    (eval_rendezvous_pd.py, seeds disjoint from the formal seed-7 evaluation)
    and then frozen in WN/ZETA below.  The gains must keep the unsaturated
    approach slow enough that the 10 m goal sphere is crossed below the
    0.5 m/s capture speed (the 5 m keep-out lies inside the goal sphere, so
    an over-fast crossing is a failure, not a success), while still
    converging from 1500 m within the 1000-step horizon on the 0.02 m/s^2
    per-axis authority.
    """
    name = 'cwh_compensated_pd'

    # frozen 2026-10-25 by the dev-scenario sweep (eval_rendezvous_pd.py)
    WN = .01        # closed-loop natural frequency [rad/s]
    ZETA = 1.       # damping ratio

    def __init__(self, env, wn=None, zeta=None):
        super().__init__(env)
        self.mass = env.mass
        self.f_max = env.f_max
        self.n = env.n
        self.dt = env.dt
        self.wn = self.WN if wn is None else float(wn)
        self.zeta = self.ZETA if zeta is None else float(zeta)
        self.kp = self.wn ** 2
        self.kd = 2. * self.zeta * self.wn
        d = self.device
        self.stat_sat = torch.zeros(env.num_envs, device=d, dtype=torch.float64)
        self.stat_steps = torch.zeros(env.num_envs, device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.stat_sat.masked_fill_(mask, 0.)
        self.stat_steps.masked_fill_(mask, 0.)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p, v = o[:, :3], o[:, 3:6]
        a_des = -self.kp * p - self.kd * v
        n2 = self.n * self.n
        # commanded specific force = desired accel minus the plant's own
        # CWH coupling, so the net acceleration equals a_des exactly
        a_cmd = torch.stack((a_des[:, 0] - 3. * n2 * p[:, 0] - 2. * self.n * v[:, 1],
                             a_des[:, 1] + 2. * self.n * v[:, 0],
                             a_des[:, 2] + n2 * p[:, 2]), -1)
        # F = m a_cmd, per-axis thrust limit, normalized action
        action = (a_cmd * (self.mass / self.f_max)).clamp(-1., 1.)
        self.stat_sat += (action.abs() >= 1.).to(torch.float64).sum(-1)
        self.stat_steps += 1.
        return action.to(torch.float32)


class DockingCWHPlacementBaseline(Baseline):
    """CWH full-state pole-placement final approach (no constraint layers).

    Task contract (SpaceDockingEnv): LVLH frame (x radial, y along-track,
    z cross-track), observation state x = [p, v] in m and m/s; action =
    per-axis normalized thrust in [-1, 1] -> F = u * f_max (10 N/axis on
    m = 500 kg, a_max = 0.02 m/s^2); target state x_ref = 0; success
    |p| < 1 m AND |v| < 0.1 m/s; 500-step horizon; symplectic Euler at
    dt = 1 s.

    Design
        The exact discrete CWH model (Ad, Bd) of ``_cwh_discrete_model`` is
        used, with the acceleration command u in m/s^2.  Pole placement is
        applied block-wise on the same model (in-plane 4-state/2-input,
        out-of-plane 2-state/1-input), so K is block-diagonal: in-plane
        poles are two damped pairs at wn * {0.8, 1.0}, the cross-track pair
        sits at wn, all at damping zeta.  Poles are mapped by z = exp(s dt)
        and realized with ``place_poles``.  The gain K (3x6) obeys the
        standard state-feedback law

            u = -K (x - x_ref),   x_ref = 0.

        The command is converted to the env's normalized action by dividing
        by the per-axis authority, u_norm = u / (f_max/m), and clipped to
        [-1, 1] (the env's thrust limit).  No shield, capture net, fuel
        derate or other extra mechanism is applied.
    """
    name = 'cwh_pole_placement'

    def __init__(self, env, wn=.05, zeta=.9, ratios=(.8, 1.)):
        super().__init__(env)
        self.mass = env.mass
        self.f_max = env.f_max
        self.a_max = env.f_max / env.mass
        self.K, self.design = _cwh_pole_gain(env, wn=wn, zeta=zeta, ratios=ratios)
        self.stat_sat = torch.zeros(env.num_envs, device=self.device, dtype=torch.float64)

    def reset_idx(self, mask):
        self.stat_sat.masked_fill_(mask, 0.)

    def actions(self, obs):
        x = _vec(obs).to(torch.float64)[:, :6]
        # acceleration command [m/s^2], u = -K(x - 0)
        a = -(x @ self.K.T)
        # force is m*a; normalized action = force / f_max = a / (f_max/m)
        action = (a / self.a_max).clamp(-1., 1.)
        self.stat_sat += (action.abs() >= 1.).to(torch.float64).sum(-1)
        return action.to(torch.float32)


# --------------------------------------------------------------------------- #
# quadplane: velocity-scheduled hybrid cascade PID (quads + forward pusher
# + aerodynamic surfaces, single control law that blends with airspeed)
# --------------------------------------------------------------------------- #
class QuadplaneHybridBaseline(Baseline):
    """Cascaded PID + forward pusher baseline for the quadplane VTOL.

    Direct adaptation of QuadXCascadePIDBaseline (position PD → roll/pitch/collective
    via heading-frame force resolution → attitude PD → body-rate commands executed
    by the env's rate PID), with constants swapped for the quadplane's 2.35 kg
    mass, four identical quad motors, and 8-channel action space (rates +
    collective + 3 surfaces + forward pusher).

    The plant mixer is physically derived (pseudoinverse of the rotor torque
    mapping τ = A @ ΔT, linearized around hover pwm 0.626).  Rows = motors,
    cols = inputs (roll, pitch, yaw, collective), collective row amplitude 1.0,
    with simple direct clamp (no collective redistribution) to preserve
    intended torque directions.  Total quad thrust is capped at 58.9 N
    (T/W ≈ 2.55).  The CG sits 3.2 mm behind the rotor centroid, a small
    inherent nose-up pitch bias (0.07 N·m) that the rate-PID integral preload
    cancels from the first step.

    Plant constants mirrored from environments/common/plant.py (frozen):
        total_mass = 2.35 kg, weight = 23.05 N
        quad motors (4, identical): 14.715 N each @ pwm=1.0 (t2w 1.5 * g)
        total quad thrust @ collective=1: 58.86 N; hover pwm = 0.626
        forward pusher: 58.86 N @ throttle=1 (t2w 6 * g), along body +x
        rate PID: kp=[0.9,0.8,4.0], ki=[0.005,0.005,0.02],
                  kd=[0,0,0] (D-on-error is discretely unstable here),
                  lim=[2.0,2.0,2.0]
        mixer: rows=motors, cols=inputs (roll,pitch,yaw,col), s=0.09504
               (+-1 PID unit -> +-3.5 N.m roll/pitch, +-0.176 N.m yaw)
        simple clamp (no normalization)
    """

    name = 'hybrid_cascade'

    MASS = 2.35
    G = 9.81
    TOTAL_THRUST = 58.86         # N at collective=1 (4 x 14.715 N, col row 1.0)
    THR_MAX = 1.0                 # env collective command ceiling [0, 1]
    PWM_GAIN = 1.0                # motor-lag dc gain = 1 under zero-mean noise

    # outer position PID [1/s^2, 1/s^3, 1/s] and clamps.  The quadplane has
    # ~59 N collective thrust (T/W ≈ 2.55) plus the pusher; gains are tuned
    # for the 30 s / 4-waypoint / 40-75 m task budget (≈6 s per leg).
    KP_POS = 1.8; KI_POS = 0.01; KD_POS = 2.2
    A_MAX = 5.5; I_MAX = 0.3
    # attitude P/PD -> body-rate commands
    KP_ATT = 6.0; KD_ATT = 0.6
    # yaw: physical authority is only 0.88 rad/s inner-loop bandwidth, so the
    # outer yaw gain must stay near it (KP_YAW=3 rang the loop at ~4 deg
    # phase margin); 1.0 gives ~41 deg PM and turns 40 deg in ~1.5 s.
    KP_YAW = 1.0; KD_YAW = 0.3
    YAW_REF = 0.0
    RATE_MAX = math.pi * 0.7

    # Fixed roll-rate feedforward: plant PID integral now handles the
    # steady-state bias, so zero feedforward needed.
    ROLL_BIAS = 0.0

    # forward pusher: holds ~15 m/s cruise against ~13 N drag at speed
    V_CRUISE = 15.0
    K_FF = 0.15
    FF_MAX = 0.80                 # raw 0.80 → thrust = 58.86*0.80^2 ≈ 37.7 N

    # Rate-PID integral preload canceling the airframe's inherent torque bias
    # (measured open-loop at hover pwm=0.6258 with a pass-through PID:
    # w_dot = [0, -0.242, 0] rad/s^2 from the 3.2 mm CG offset behind the
    # rotor centroid; mixer authority [roll 6.81, pitch 11.40, yaw 0.219]
    # rad/s^2 per unit input).  pitch preload = 0.242/11.40 = 0.0213.
    # Without the preload the tiny ki=0.005 integral takes seconds to wind up.
    INTEGRAL_PRELOAD = (0.0, 0.0213, 0.0)

    def __init__(self, env):
        super().__init__(env)
        d, n = self.device, env.num_envs
        self.dt = 1. / env.agent_hz
        self.rate_max = self.RATE_MAX
        self.e3 = torch.tensor((0., 0., 1.), device=d, dtype=torch.float64)
        self.k_att = torch.tensor((self.KP_ATT, self.KP_ATT, self.KP_YAW),
                                  device=d, dtype=torch.float64)
        self.kd_att = torch.tensor((self.KD_ATT, self.KD_ATT, self.KD_YAW),
                                   device=d, dtype=torch.float64)
        self.f_max = self.TOTAL_THRUST * self.THR_MAX ** 2  # 58.86 N
        self.i_acc = torch.zeros((n, 3), device=d, dtype=torch.float64)
        # the env's constructor reset ran before this controller existed --
        # preload every replica now so the FIRST episode also flies
        self._preload(torch.ones(n, dtype=torch.bool, device=d))

    def _preload(self, mask):
        # Pre-load the plant rate PID integral with the steady-state bias
        # compensation (see INTEGRAL_PRELOAD).  PID lim=2.0, preload fits.
        if mask.any():
            for axis, val in enumerate(self.INTEGRAL_PRELOAD):
                self.env.plant.pid.integral[mask, axis] = val

    def reset_idx(self, mask):
        self.i_acc.masked_fill_(mask[:, None], 0.)
        self._preload(mask)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 31:43].reshape(-1, 4, 3)
        R = _quat_to_R(q)
        roll, pitch, yaw = _rot_to_rpy(R)
        v_w = torch.einsum('nij,nj->ni', R, vb)
        target = p + torch.einsum('nij,nj->ni', R, delta[:, 0])
        e_p = p - target
        V_forward = vb[:, 0]

        # outer PID -> desired acceleration (conditional anti-windup)
        a_prev = -self.KP_POS * e_p - self.KD_POS * v_w + self.i_acc
        clipped_prev = a_prev.norm(dim=-1) > self.A_MAX
        i_new = (self.i_acc + self.KI_POS * e_p * self.dt).clamp(-self.I_MAX, self.I_MAX)
        self.i_acc.copy_(torch.where(clipped_prev[:, None], self.i_acc, i_new))
        a_raw = -self.KP_POS * e_p - self.KD_POS * v_w + self.i_acc
        a_norm = a_raw.norm(dim=-1)
        clipped = a_norm > self.A_MAX
        a_des = a_raw * (self.A_MAX / a_norm.clamp_min(self.A_MAX))[:, None]

        # gravity compensation -> desired roll/pitch and total thrust
        a_cmd = a_des + self.G * self.e3
        f_cmd = self.MASS * a_cmd.norm(dim=-1)
        f = f_cmd.clamp_max(self.f_max)
        throttle = (torch.sqrt(f / self.TOTAL_THRUST) / self.PWM_GAIN).clamp(0., self.THR_MAX)
        # Resolve the desired acceleration into the CURRENT heading frame
        # (not the waypoint-bearing frame): the quad's yaw authority is weak
        # (0.22 rad/s^2/unit), so the yaw loop lags seconds behind; resolving
        # against the bearing would misdirect the tilt for the whole transient.
        # YAW_REF (bearing to waypoint) is used for the yaw channel only.
        target_vec = target - p
        YAW_REF = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        u_x = a_cmd[:, 0] * cy + a_cmd[:, 1] * sy
        u_y = -a_cmd[:, 0] * sy + a_cmd[:, 1] * cy
        u_z = a_cmd[:, 2]
        # ZYX Euler: positive pitch = nose DOWN (thrust tilts forward),
        # positive roll = right bank (thrust tilts right/-y) — same
        # convention as the quadx cascade baseline.
        pitch_d = torch.atan2(u_x, u_z)
        roll_d = torch.atan2(-u_y, torch.sqrt(u_x * u_x + u_z * u_z))

        # attitude P/PD -> body-rate command executed by the env's rate PID,
        # plus a roll-rate feedforward to cancel the inherent URDF bias.
        e_att = torch.stack((_wrap(roll_d - roll), _wrap(pitch_d - pitch),
                             _wrap(YAW_REF - yaw)), -1)
        w_cmd = self.k_att * e_att - self.kd_att * wb
        w_cmd[:, 0] += self.ROLL_BIAS   # fixed roll bias compensation
        w_cmd = w_cmd.clamp(-self.rate_max, self.rate_max)

        # surfaces: all conserved (zero) for now — quads handle everything
        a4_cmd = torch.zeros((p.shape[0],), device=p.device, dtype=torch.float64)
        a5_cmd = torch.zeros_like(a4_cmd)
        a6_cmd = torch.zeros_like(a4_cmd)

        # forward pusher: gentle speed-proportional throttle
        forward_err = self.V_CRUISE - V_forward
        throttle7 = (self.K_FF * forward_err).clamp(0., self.FF_MAX)

        # assemble action vector: 8 channels, each in [-1, 1]
        return torch.cat((
            w_cmd / self.rate_max,                          # a0..a2: body rates / pi
            (2. * throttle / self.THR_MAX - 1.)[:, None],   # a3: collective
            a4_cmd[:, None], a5_cmd[:, None], a6_cmd[:, None],  # a4..a6: surfaces (all 0)
            (2. * throttle7 - 1.)[:, None],                 # a7: forward throttle
        ), -1).clamp(-1., 1.).to(torch.float32)


BUILDERS = {
    'quadx': QuadXSE3Baseline,
    'quadx_track': QuadXTrackKFBaseline,
    'fixedwing': FixedwingAutopilotBaseline,
    'rocket': RocketPDGBaseline,
    'satellite': SatelliteCMGBaseline,
    'transfer': TransferGaussBaseline,
    'docking': DockingCWHLQRBaseline,
    'rendezvous': RendezvousTwoImpulseBaseline,
    'quadplane': QuadplaneHybridBaseline,
    # second rendezvous controller under the same task (registry key is
    # explicit): CWH-compensated PD to the origin, no constraint layers
    'rendezvous_cwhpd': RendezvousCWHPDBaseline,
}

# Per-task controller variants that share the task's env but implement an
# alternative control law (A/B arms).  The first entry of each task is the
# default already wired into BUILDERS; variants are selected explicitly by
# ``build_baseline(vehicle, env, variant=...)`` / ``baseline_eval --baseline``.
BASELINE_VARIANTS = {
    'quadx': {
        # position -> attitude cascade with the env's native rate PID
        'cascaded_pid': QuadXCascadePIDBaseline,
    },
    'transfer': {
        # Gauss one-step greedy tangential guidance (default, in BUILDERS)
        'gauss_tangential': TransferGaussBaseline,
        # Petropoulos Q-law Lyapunov guidance, planar (a, e) subset
        'q_law': TransferQLawBaseline,
    },
    'fixedwing': {
        # conventional cascaded PID (no L1/TECS guidance heuristics)
        'pid_cascade': FixedwingPIDBaseline,
    },
    'rocket': {
        'pd_landing': RocketPDLandingBaseline,
    },
    'satellite': {
        'quat_pd_damped_cmg': SatelliteCMGBaseline,
        'quat_pd_mp_pinv_cmg': SatelliteOrdinaryPinvBaseline,
    },
    'docking': {
        # CWH full-state feedback, standard pole placement, no extra layers
        'pole_placement': DockingCWHPlacementBaseline,
    },
    'rendezvous': {
        # CWH-compensated PD to the origin, no constraint layers
        'cwh_pd': RendezvousCWHPDBaseline,
    },
}


def register_baseline(vehicle):
    """Class decorator: wire a Baseline subclass into the registry.

    A new task's classical baseline becomes available to ``baseline_eval``,
    ``build_baseline`` and the BC warm start with exactly one line::

        @register_baseline('missile')
        class MissilePNBaseline(Baseline): ...

    (the seven shipped entries below register explicitly at definition
    time, keeping the BUILDERS dict as the single source of truth).
    """
    def deco(cls):
        if vehicle in BUILDERS:
            raise KeyError(f"baseline for task {vehicle!r} already registered")
        BUILDERS[vehicle] = cls
        return cls
    return deco


def build_baseline(vehicle, env, variant=None):
    """Instantiate the classical baseline for a task name.

    variant: optional controller name from BASELINE_VARIANTS[vehicle]; None
    keeps the task's registered default in BUILDERS.
    """
    if variant is None:
        try:
            return BUILDERS[vehicle](env)
        except KeyError:
            raise KeyError(f'no classical baseline registered for task '
                           f'{vehicle!r}; known: {sorted(BUILDERS)}') from None
    try:
        return BASELINE_VARIANTS[vehicle][variant](env)
    except KeyError:
        known = sorted(BASELINE_VARIANTS.get(vehicle, {}))
        raise KeyError(f'no baseline variant {variant!r} for task {vehicle!r}; '
                       f'known: {known}') from None


# The classical baselines double as the conventional (expert) policies for the BC
# warm start (algorithms/warmstart/bc_warmstart.py): each task env declares its
# ``conventional_controller`` class attribute naming the Baseline subclass above,
# and the BC collector instantiates its own controller against the env and
# drives it directly.  The envs themselves stay plain tensor environments --
# they neither own a controller instance nor expose one through info.


# --------------------------------------------------------------------------- #
# missile: augmented proportional navigation -> body-rate commands
# --------------------------------------------------------------------------- #
def _quat_to_matrix(q_xyzw):
    """Batched xyzw quaternion -> 3x3 rotation matrices (body -> world)."""
    x, y, z, w = q_xyzw.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).unflatten(-1, (3, 3))


@register_baseline('missile')
class MissileAPNBaseline(Baseline):
    """APN guidance (N=15, retuned from the N=4/N=10 lineage in the constant
    below) with half-N gravity compensation, mapped to the env's body-rate
    command interface through the skid-to-turn kinematic inverse
    ``omega = cross(b_x, a_lat) / V``: rotating the thrust axis toward
    the commanded lateral-acceleration direction at the turn rate a/V that
    curves the velocity vector accordingly.  Sole internal state: a
    first-order low-pass filter on the body-frame LOS rate (reset_idx-clear).

    Guards: LOS-rate filter freezes below Vc < 50 m/s (terminal noise
    blow-up), the relative range is clamped away from zero, and the output is
    normalized by the per-axis rate limits (roll OMEGA_MAX, pitch/yaw
    OMEGA_LAT) into [-1, 1].
    """
    name = 'apn_rate'
    # N=15 measured (1024-ep, seed 7): SR 0.812 vs 0.770 at N=10 on the fixed
    # medium operating point; N>=18 overshoots and diverges the LOS loop.
    N = 15.0
    LOS_FILTER = 0.8

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        from environments.missile.env import OMEGA_MAX, OMEGA_LAT
        # per-axis rate-command limits (roll, pitch, yaw)
        self.omega_scale = torch.tensor((OMEGA_MAX, OMEGA_LAT, OMEGA_LAT),
                                        device=d, dtype=torch.float64)
        self.lam_filt = torch.zeros((env.num_envs, 3), device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.lam_filt.masked_fill_(mask[:, None], 0.)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        q = o[:, 3:7]
        vb = o[:, 7:10]
        r_b = o[:, 23:26]
        v_rel_b = o[:, 26:29]
        lam = o[:, 29:32]
        vc = o[:, 32].clamp_min(0.)
        R = _quat_to_matrix(q)
        # first-order LOS-rate filter, frozen in the terminal Vc<50 regime
        alpha = torch.where(vc < 50., torch.zeros_like(vc), torch.full_like(vc, self.LOS_FILTER))
        self.lam_filt += alpha[:, None] * (lam - self.lam_filt)
        lam_w = torch.einsum('nij,nj->ni', R, self.lam_filt)
        # APN: a_cmd = N*Vc*(lambda_dot x lambda_hat) + (N/2)*g_perp_LOS.
        # lambda_w = cross(r, v_rel)/r^2 is the LOS ANGULAR VELOCITY (rotation
        # axis), which is perpendicular to the LOS but is NOT the acceleration
        # direction: the command must be rotated 90 deg about the LOS into the
        # plane the LOS tip sweeps through, i.e. cross(lambda_w, r_hat_w).
        # Dropping the cross product tilts the command and produces persistent
        # cross-range divergence (verified: monotonically growing miss).
        rng = r_b.norm(dim=-1).clamp_min(1e-3)
        g = torch.tensor([0., 0., -9.81], device=self.device, dtype=torch.float64).expand_as(r_b)
        r_hat_w = torch.einsum('nij,nj->ni', R, r_b / rng[:, None])
        a_cmd = self.N * vc[:, None] * torch.linalg.cross(lam_w, r_hat_w) \
            + 0.5 * self.N * (g - (g * r_hat_w).sum(-1, keepdim=True) * r_hat_w)
        # express in body frame, drop the thrust-axis (b_x) component
        a_lat_b = torch.einsum('nji,nj->ni', R, a_cmd)
        a_lat_b = a_lat_b - a_lat_b[:, 0:1] * torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float64)
        # skid-to-turn kinematic inverse: omega = cross(b_x, a_lat) / V
        V = vb.norm(dim=-1).clamp_min(50.)
        b_x = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float64).expand_as(a_lat_b)
        w_cmd = torch.linalg.cross(b_x, a_lat_b) / V[:, None]
        return (w_cmd / self.omega_scale).clamp(-1., 1.).to(torch.float32)
