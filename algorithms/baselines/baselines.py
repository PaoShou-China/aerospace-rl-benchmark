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

All formulas are analytic; no training, no tuning loop, no host-device
synchronization.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import logm, solve_discrete_are


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
    e_O = w - R' R_d w_d (= w for the stationary hover setpoint) drive the
    torque law M = -kR e_R - kOm e_O + w x Jw.  The wrench (f, M) is allocated
    onto the four rotors with the frozen cf2x geometry,
        t_i = f/4 + ay_i Mx/(4d) - ax_i My/(4d) + spin_i Mz/(4c),
    (d = rotor arm, c = km/kf), clipped to the plant's total-thrust window and
    per-rotor pwm window [0.05, 1], and mapped onto the env's rate-command +
    collective interface by inverting its mixer (tau_pid = Mmix' diff / 4 with
    diff = pwm - mean(pwm)) and its rate-loop P gain (w_cmd = w + tau/kp).

    One practical augmentation beyond the paper: the frozen plant's motor lag
    state passes through the multiplicative motor noise every physics step,
    E[1 + 0.02*N(4,1)] = 1.08, giving the commanded pwm a measured dc gain of
    1.216 (matches the lag-filter fixed point) and delivered thrust a 1.216^2
    ~ 1.48 gain.  The commanded pwm vector is divided by PWM_GAIN so delivered
    thrust/differentials match the ideal wrench; a small integral on the
    b3-projected position error then trims residuals, freezing under
    total-thrust saturation (anti-windup) and clearing on reset_idx.
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
    PWM_GAIN = 1.216               # motor-lag dc gain under noise mean 1.08

    # controller gains
    KX, KV = 4.0, 3.0              # position PD [1/s^2, 1/s]
    KR, KOM = 8e-4, 2.5e-4         # attitude PD [N m/rad, N m s/rad]
    KI_POS = 0.15                  # axial position-error integral [N/m]
    I_MAX = 0.35                   # integral clamp [N]

    def __init__(self, env):
        super().__init__(env)
        d, n = self.device, env.num_envs
        self.dt = 1. / env.agent_hz           # agent control period [s]
        self.target = env.hover_target.to(torch.float64)
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
        o = _vec(obs).to(torch.float64)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        R = _quat_to_R(q)                       # body -> world
        b3 = R[:, :, 2]                         # body thrust axis in world
        v_w = torch.einsum('nij,nj->ni', R, vb)
        e_x, e_v = p - self.target, v_w         # hover setpoint: v_d = 0

        F_des = self.MASS * (self.G * self.e3 - self.KX * e_x - self.KV * e_v)
        f_raw = (F_des * b3).sum(-1) + self.trim
        # axial trim integral; freeze while the collective command is clipped
        trim_dot = -self.KI_POS * (e_x * b3).sum(-1)
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
      target_index + k), validity(35:39). No wind field exists in the
      tensorized plant, so airspeed = ground speed = |vb|.
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
        MIN thrust/weight = 1.91 > 1 -- once lit the engine cannot follow a
        slow descent, so the terminal phase is a min-thrust pulse train with
        coasting between pulses (hysteresis on the observed ignited flag).
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
        240 Hz physics; the engine relights per physics step (cmd > 0.5).
      * Landing (unchanged criteria): pad contact (radius 2 m, top z = 0.15,
        rest base height ~2.575 m) is fatal iff the pre-contact speed
        exceeds 1.0 m/s or spin 0.35 rad/s; success requires settling on the
        pad to speed/spin < 0.02 with tilt < 0.1 rad.

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
        # aux tail: [13:20] raw-action echo, [20:24] the FOUR finlet actuation
        # states, then ignited=24, fuel=25, throttle=26, gimbal=27:29
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
        # observed actuator states (aux tail): o[26] is the throttle state --
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
        raw = torch.stack((fx, fy, yaw,
                           torch.where(ign, torch.ones_like(u_thr), -torch.ones_like(u_thr)),
                           2. * u_thr - 1., gx, gy), -1)
        self.diag['steps'] += 1.
        return raw.clamp(-1., 1.).to(torch.float32)


# --------------------------------------------------------------------------- #
# satellite: quaternion PD + singularity-robust CMG steering
# --------------------------------------------------------------------------- #
class SatelliteCMGBaseline(Baseline):
    """Quaternion PD + singularity-robust damped-pseudoinverse CMG steering.

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
      tau = -h_dyn * G delta with G columns g_i x r_i(theta).  The steering is
      the damped pseudoinverse delta = -(1/h) G (G G^T + lam~^2 I)^-1 tau,
      written below in a rotor_speed-scaled factorization that is algebraically
      identical (gr = rotor_speed*G, kappa = h/rotor_speed, lam = rotor_speed*
      lam~); G is reconstructed from the observed gimbal angles bit-exactly
      against env._manipulability (probe-verified, 4- and 6-CMG).
    - damping lam~ = lam0 + 0.15*exp(-3 m_norm) grows as manipulability
      collapses: torque relay error 0.05% far from singularities, ~3% fully
      damped, with gimbal rates bounded throughout (Schaub & Junkins SR steering).
    - gimbal limits: the plant has no hard stops (angles wrap at +-pi); the
      +-action_scale (3 rad/s) rate limit is enforced by the final clamp and
      every saturation is counted in metrics['clamp_envsteps'].
    - numerical protection: tau and the observed m_norm are passed through
      nan_to_num (nonfinite inputs steer toward zero rate instead of poisoning
      the solve), the damped 3x3 system stays SPD with lam~ >= lam0, and every
      nonfinite intermediate is counted in metrics['nonfinite_envsteps'].
    """
    name = 'quat_pd_damped_cmg'

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
        self.kp, self.kd = 3., 4.
        self.lam0 = .02
        # on-device diagnostic accumulators (no host sync in the step loop)
        z = lambda: torch.zeros((), device=self.device, dtype=torch.float64)
        self.metrics = {'steps': z(), 'clamp_envsteps': z(), 'nonfinite_envsteps': z(),
                        'm_below_envsteps': z(),
                        'm_min': torch.full((), float('inf'), device=self.device,
                                            dtype=torch.float64),
                        'hsum_max': z()}

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
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

        # diagnostics: nonfinite inputs, manipulability floor, stored momentum
        bad = ~torch.isfinite(tau).all(-1) | ~torch.isfinite(m_norm)
        self.metrics['nonfinite_envsteps'] += bad.sum()
        self.metrics['m_min'] = torch.minimum(
            self.metrics['m_min'],
            torch.where(torch.isfinite(m_norm), m_norm,
                        torch.full_like(m_norm, float('inf'))).min())
        self.metrics['m_below_envsteps'] += (m_norm < .1).sum()
        self.metrics['hsum_max'] = torch.maximum(self.metrics['hsum_max'], h_sum.norm(dim=-1).max())

        # singularity-robust damped pseudoinverse (Schaub & Junkins):
        # tau = -kappa gr^T delta  =>  delta = -(1/kappa) gr (gr^T gr + lam^2 I)^-1 tau
        # lam grows as the manipulability m -> 0, trading torque accuracy for
        # bounded gimbal rates near singular configurations.
        tau = torch.nan_to_num(tau)
        lam = self.rotor_speed * (self.lam0 + .15 * torch.exp(-3. * m_norm))
        JtJ = gr.transpose(1, 2) @ gr
        JtJ = JtJ + torch.eye(3, device=self.device, dtype=torch.float64)[None] * (lam ** 2)[:, None, None]
        delta = -(1. / self.kappa) * (gr @ torch.linalg.solve(JtJ, tau.unsqueeze(-1))).squeeze(-1)
        self.metrics['nonfinite_envsteps'] += (~torch.isfinite(delta).all(-1)).sum()
        delta = torch.nan_to_num(delta)

        # gimbal-rate limit: the env multiplies the action by action_scale, so
        # |action| = 1 is exactly the +-action_scale rad/s rate bound
        self.metrics['steps'] += 1
        self.metrics['clamp_envsteps'] += (delta.abs() > self.action_scale).any(-1).sum()
        return (delta / self.action_scale).clamp(-1., 1.).to(torch.float32)


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


BUILDERS = {
    'quadx': QuadXSE3Baseline,
    'fixedwing': FixedwingAutopilotBaseline,
    'rocket': RocketPDGBaseline,
    'satellite': SatelliteCMGBaseline,
    'transfer': TransferGaussBaseline,
    'docking': DockingCWHLQRBaseline,
    'rendezvous': RendezvousTwoImpulseBaseline,
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


def build_baseline(vehicle, env):
    """Instantiate the classical baseline for a task name."""
    try:
        return BUILDERS[vehicle](env)
    except KeyError:
        raise KeyError(f'no classical baseline registered for task '
                       f'{vehicle!r}; known: {sorted(BUILDERS)}') from None


# The classical baselines double as the conventional (expert) policies for the BC
# warm start (algorithms/warmstart/bc_warmstart.py): each task env declares its
# ``conventional_controller`` class attribute naming the Baseline subclass above,
# and the BC collector instantiates its own controller against the env and
# drives it directly.  The envs themselves stay plain tensor environments --
# they neither own a controller instance nor expose one through info.
