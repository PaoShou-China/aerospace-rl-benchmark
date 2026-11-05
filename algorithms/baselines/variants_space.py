"""Classical (non-learning) baseline VARIANTS for the four space tasks.

Companion module to ``algorithms/baselines/baselines.py``: same contract
(batched, device-resident, reads ONLY the 'policy' observation group,
float64 internal math, actions in the env's native [-1, 1] convention,
plant constants mirrored from the env at construction), but alternative
control laws so the orchestrator can A/B them against the registered
defaults through ``BASELINE_VARIANTS``.  This module deliberately does NOT
touch the registries -- wiring is the orchestrator's job.

Contents
    satellite  ``SatelliteWeightedPinvBaseline``  ('quat_pd_wpinv_cmg')
               quaternion PD (shared ``_SatelliteCMGPDBase`` torque loop) +
               weighted damped minimum-norm CMG steering with a Vadali-style
               null-space singularity-avoidance gradient.
    transfer   ``TransferEdelbaumBaseline``       ('edelbaum')
               Edelbaum-type budgeted tangential low-thrust guidance: an
               open-loop Delta-v budget from the circular-speed-equivalent
               approximation, spent prograde along the local velocity inside
               a fixed apogee arc (perigee-raise phasing).
    docking    ``DockingZEMZEVBaseline``          ('zem_zev')
               CWH ZEM/ZEV single-pulse targeting to the docking port over a
               receding horizon, with range shield, capture net and fuel
               derate (constraint layers mirror the CWH-LQR default).
    rendezvous ``RendezvousGlideslopeBaseline``   ('glideslope')
               classical glideslope/range-rate braking corridor along the
               initial line of sight to the hold point between the keep-out
               and success spheres, with transverse line-of-sight keeping.

All formulas are analytic; no training, no tuning loop, no host-device
synchronization.
"""

import math

import numpy as np
import torch
from scipy.linalg import logm

from environments.common.dynamics import COMPUTE_DTYPE

from algorithms.baselines.baselines import Baseline, _vec, _SatelliteCMGPDBase

__all__ = [
    "SatelliteWeightedPinvBaseline",
    "TransferEdelbaumBaseline",
    "DockingZEMZEVBaseline",
    "RendezvousGlideslopeBaseline",
]


# --------------------------------------------------------------------------- #
# satellite variant: weighted pseudo-inverse + null-space singularity avoidance
# --------------------------------------------------------------------------- #
class SatelliteWeightedPinvBaseline(_SatelliteCMGPDBase):
    """Quaternion PD + WEIGHTED damped minimum-norm CMG steering with
    null-space singularity avoidance.

    The torque loop (PD gains kp=3, kd=4, exact gyroscopic feed-forward,
    obs parsing, gimbal-rate limiting and torque accounting) is inherited
    unchanged from ``_SatelliteCMGPDBase``; only ``_steer`` differs, exactly
    like the shipped ``quat_pd_damped_cmg`` / ``quat_pd_mp_pinv_cmg`` A/B arms.
    With the module's scaled factorization gr = rotor_speed * (g_i x r_i)
    (n, N, 3) the torque map is tau = -kappa * J delta with J = gr^T (3 x N),
    so the steering solves J delta = b, b = -tau/kappa, for the gimbal rates.

    Steering law (one concrete classical formulation, Vadali/Oh-Fleming
    family), all 3 x 3 solves Tikhonov-regularized to stay SPD:

    1. Weighted damped minimum-norm (Oh & Fleming 1989): minimize the weighted
       norm delta^T W delta subject to J delta = b,

           delta_los = W^-1 J^T (J W^-1 J^T + lam^2 I)^-1 b,

       with the per-CMG weight W = diag(1 + K_W alpha_i^2), alpha_i the
       alignment |c_i^T b_hat| of CMG i's unit output direction c_i with the
       required torque direction b_hat.  Highly aligned CMGs cost more, so the
       solution loads the CMGs far from the (torque-dependent) singular
       geometry and preserves geometric diversity.  lam keeps the 3 x 3 system
       SPD and provides singularity-robust damping with the SAME proven
       schedule as the default damped pinv: lam = rotor_speed * (LAM0 +
       0.15 exp(-3 m_norm)), growing as the observed manipulability collapses.

    2. Null-space singularity-avoidance gradient (Vadali et al.): the spare
       degrees of freedom are spent moving the gimbal angles UP the
       manipulability gradient m_norm(theta) = sqrt(det(C C^T)) (unit output
       columns c_i = g_i x r_i),

           delta_ns = K_NS * P grad_m,   P = I_N - W^-1 J^T (J W^-1 J^T +
                     lam^2 I)^-1 J   (damped weighted null-space projector),

       so the avoidance motion produces (to first order, and exactly in the
       lam -> 0 limit) zero commanded torque and cannot fight the PD loop.
       The gradient has the closed form (dc_i/dtheta_i = g_i x (g_i x r_i) =
       -r_i for rotor axes orthogonal to the gimbal axes)

           d m_norm / d theta_i = -m_norm * r_i^T (C C^T)^-1 c_i,

       evaluated batched from the observed geometry (c_i reconstructed from
       gr, r_i = -g_i x c_i, g_i = rotor_ref x g_cross_r0 -- both stored by
       the shared base; the reconstruction is exact unit algebra).  The
       finite-difference identity is probe-checked in the smoke harness.
       delta_ns is norm-clamped to NS_MAX (half the +-3 rad/s rate bound),
       leaving the torque term full headroom; the residual second-order torque
       leakage of the damped projector is O((lam/sigma_min)^2) (<= ~1e-3 of
       the momentum capacity far from singularities, where it is active).

    Tuning: LAM0 and the damping schedule are inherited verbatim from the
    100%-success default so the comparison isolates the weighting + null-space
    terms; K_W = 1, K_NS = 0.5 and NS_MAX = 1 rad/s were fixed a priori
    (conservative: null-space rates stay under 1/3 of the rate bound) and
    frozen -- no per-task sweep.  Works for the 4-CMG pyramid and the 6-CMG
    orthogonal cluster directly from the matrix dimensions.
    """

    name = "quat_pd_wpinv_cmg"

    LAM0 = 0.02  # base Tikhonov damping (identical to the default arm)
    K_W = 1.0  # Oh-Fleming alignment weight gain (dimensionless)
    K_NS = 0.5  # null-space gradient gain [rad/s per unit gradient]
    NS_MAX = 1.0  # null-space rate clamp [rad/s] (action_scale is 3)

    def __init__(self, env, design=None):
        super().__init__(env, design)
        self.lam0 = self.LAM0
        # gimbal axes recovered exactly from the base's stored reference
        # geometry: rotor_ref x g_cross_r0 = g_i (both unit, mutually
        # orthogonal, and g_i = r_i0 x (g_i x r_i0) by the vector identity).
        self.g_axes = torch.linalg.cross(self.rotor_ref, self.g_cross_r0, dim=-1)

        def z():
            return torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE)

        self.metrics.update({"ns_clamp_envsteps": z(), "ns_norm_sum": z(), "w_align_max": z()})

    @staticmethod
    def _nullspace_grad(chat, r_ax, m_norm, eye3):
        """Closed-form gradient of m_norm(theta) = sqrt(det(C C^T)) w.r.t. the
        gimbal angles: (n, N), with chat = (n, N, 3) unit output directions,
        r_ax = (n, N, 3) rotor axes, d c_i/d theta_i = -r_i."""
        M = torch.bmm(chat.transpose(1, 2), chat) + eye3[None] * 1e-9
        Minv_c = torch.linalg.solve(M, chat.transpose(1, 2))  # (n, 3, N)
        return -m_norm[:, None] * (r_ax * Minv_c.transpose(1, 2)).sum(-1)

    def _steer(self, tau, gr, m_norm):
        eye3 = torch.eye(3, device=self.device, dtype=COMPUTE_DTYPE)
        eye_n = torch.eye(self.n_cmg, device=self.device, dtype=COMPUTE_DTYPE)
        J = gr.transpose(1, 2)  # (n, 3, N)
        b = (-tau / self.kappa).unsqueeze(-1)  # (n, 3, 1)

        # ---- Oh-Fleming alignment weight W = diag(1 + K_W alpha_i^2)
        chat = J / J.norm(dim=1, keepdim=True).clamp_min(1e-12)  # unit cols
        bhat = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
        alpha = (chat * bhat).sum(1).abs()  # (n, N)
        self.metrics["w_align_max"] = torch.maximum(
            self.metrics["w_align_max"], alpha.max().detach()
        )
        w = 1.0 + self.K_W * alpha * alpha
        JwT = J.transpose(1, 2) / w[:, :, None]  # W^-1 J^T
        # ---- damped weighted min-norm: SPD-safe by the lam^2 Tikhonov term
        lam = self.rotor_speed * (self.lam0 + 0.15 * torch.exp(-3.0 * m_norm))
        A = (
            torch.bmm(J / w[:, None, :], J.transpose(1, 2)) + eye3[None] * (lam**2)[:, None, None]
        )  # J W^-1 J^T + lam^2 I
        delta_los = torch.bmm(JwT, torch.linalg.solve(A, b)).squeeze(-1)

        # ---- null-space singularity-avoidance gradient (Vadali)
        chat2 = gr / self.rotor_speed  # (n, N, 3) unit
        r_ax = -torch.linalg.cross(self.g_axes[None], chat2, dim=-1)
        grad_m = self._nullspace_grad(chat2, r_ax, m_norm, eye3)  # (n, N)
        AJ = torch.linalg.solve(A, J)  # A^-1 J
        P = eye_n[None] - torch.bmm(JwT, AJ)  # (n, N, N)
        delta_ns = torch.bmm(P, (self.K_NS * grad_m).unsqueeze(-1)).squeeze(-1)
        ns_norm = delta_ns.norm(dim=-1)
        over = ns_norm > self.NS_MAX
        delta_ns = torch.where(
            over[:, None], delta_ns * (self.NS_MAX / ns_norm.clamp_min(1e-12))[:, None], delta_ns
        )
        self.metrics["ns_clamp_envsteps"] += over.sum()
        self.metrics["ns_norm_sum"] += ns_norm.sum()
        return delta_los + delta_ns

    def config(self):
        cfg = super().config()
        cfg.update(
            steering_law="weighted_damped_min_norm_plus_nullspace_gradient",
            lam0=self.lam0,
            damping_schedule="lam = rotor_speed*(lam0 + 0.15*exp(-3*m_norm))",
            weight="W = diag(1 + K_W alpha_i^2), alpha_i = |c_i^T b_hat|",
            k_w=self.K_W,
            k_ns=self.K_NS,
            ns_max_rad_s=self.NS_MAX,
            null_space_motion=True,
            singularity_measure="m_norm = sqrt(det(C C^T)), unit output columns",
        )
        return cfg


# --------------------------------------------------------------------------- #
# transfer variant: Edelbaum budgeted tangential guidance
# --------------------------------------------------------------------------- #
class TransferEdelbaumBaseline(Baseline):
    """Edelbaum-type OPEN-LOOP budgeted tangential guidance for perigee
    raising (Edelbaum 1961, "Optimum Low-Thrust Trajectories to Circular
    Orbit"; the classical time-optimal circular-to-circular approximation
    treats the transfer as a velocity increment Delta-v spent along the
    velocity vector).

    Exactly what is implemented (and how it differs from the one-step-greedy
    Gauss default, which re-scores coast/prograde/retrograde against the task
    objective EVERY step):

    1. Budget (pre-computed once per episode from the latched initial orbit,
       read from the env's own ra0/rp0 reset latches -- no controller state
       needed for the plan): target perigee rp_t = rp0 + RP_TARGET_RAISE with
       apogee held at ra0, i.e. a_1 = (rp_t + ra0)/2 vs a_0 = (rp0 + ra0)/2.
       The Edelbaum circular-speed-equivalent increment is
           Delta_v_E = |sqrt(mu/a_1) - sqrt(mu/a_0)|,
       the analytic acceleration-magnitude schedule a(t) = Delta_v_E/T spread
       over the episode reduces here to: burn at saturated magnitude while the
       REMAINING budget is positive, coast after (the budget, not a per-step
       score, is the decision variable).
    2. Steering: thrust always ALONG the local (inertial, world-frame)
       velocity vector v/|v| from the observed [r, v], magnitude saturated at
       f_max (action = v_hat, per-axis within [-1, 1]).
    3. Phasing gate (the eccentric-orbit extension of the tangential Edelbaum
       profile, required by the boundary condition "raise rp, hold ra"): the
       budget is spent only inside a FIXED open-loop apogee arc
       |nu - pi| <= APOGEE_HALF_ARC and only while rp < rp_t.  Off-arc
       transverse thrust trades perigee for apogee (Gauss: mid-course prograde
       burn gives dra > 0, drp ~ 0), so the arc gate is what makes the
       budgeted profile perigee-productive; nu is recovered with the same
       O(e)-safe atan2 pair as the Gauss baseline (no division by e).
    4. Budget accounting: the realized per-step Delta-v f_max |u| dt / m(obs)
       is integrated per replica; burn stops when the integral reaches
       Delta_v_E (typical budget for the 4 km reference raise: ~0.6 m/s ~ 21
       saturated steps, spent over the apogee arcs of the 5-revolution
       episode).  The one-step-greedy Gauss arm instead coasts whenever its
       scored J gain net of propellant is negative -- including mid-course and
       retrograde burns this law never commands.

    Tuning: RP_TARGET_RAISE = 4 km and the 60 deg half-arc were fixed from the
    task's first-order Gauss rates (apogee-arc transverse burn: drp ~ +163 m
    per saturated step at nu = pi, propellant cost 20 dm ~ 0.08 km-equivalent
    for the whole budget) and frozen; no seed sweep.  Interface: obs
    o = [t, r, v, m] float32 cast to float64; constants mu/dt/f_max/isp/g0
    from the env; stateful member dv_used only (mirrored in reset_idx).
    """

    name = "edelbaum"

    RP_TARGET_RAISE = 4000.0  # perigee reference raise above rp0 [m]
    APOGEE_HALF_ARC_DEG = 60.0  # open-loop burn arc half-width about apogee

    def __init__(self, env, design=None):
        super().__init__(env, design)
        self.mu, self.dt = env.mu, env.dt
        self.f_max = self._design_scalar("f_max", env.f_max)
        self.isp = self._design_scalar("isp", env.isp)
        self.g0 = self._design_scalar("g0", env.g0)
        self.arc_cos = -math.cos(math.radians(self.APOGEE_HALF_ARC_DEG))
        self.dv_used = torch.zeros(env.num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        # per-replica latched initial perigee/apogee, computed from the
        # FIRST obs frame after each reset (avoids env.rp0/ra0 side-channel).
        self.rp0 = torch.zeros(env.num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self.ra0 = torch.zeros(env.num_envs, device=self.device, dtype=COMPUTE_DTYPE)

    def reset_idx(self, mask):
        self.dv_used.masked_fill_(mask, 0.0)
        self.rp0.masked_fill_(mask, 0.0)
        self.ra0.masked_fill_(mask, 0.0)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        r, v, m = o[:, 1:4], o[:, 4:7], o[:, 7:8]
        rm = r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        hvec = torch.linalg.cross(r, v, dim=-1)
        h = hvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a = 1.0 / (2.0 / rm - (v * v).sum(-1, keepdim=True) / self.mu)
        evec = torch.linalg.cross(v, hvec, dim=-1) / self.mu - r / rm
        e = evec.norm(dim=-1, keepdim=True)
        p = h * h / self.mu
        vr = (v * r).sum(-1, keepdim=True) / rm
        nu = torch.atan2(vr * h / self.mu, p / rm - 1.0)  # O(e)-safe

        # Edelbaum budget from initial orbit (latched at first frame after reset).
        # We compute rp0/ra0 from the obs itself (no env side-channel):
        #   a = 1/(2/r - v^2/mu),   e = |(v x h)/mu - r/r|,
        #   rp0 = a(1 - e),   ra0 = a(1 + e)
        fresh = self.dv_used == 0
        if fresh.any():
            rm0 = r[fresh].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            hvec0 = torch.linalg.cross(r[fresh], v[fresh], dim=-1)
            a0 = 1.0 / (2.0 / rm0 - (v[fresh] * v[fresh]).sum(-1, keepdim=True) / self.mu)
            evec0 = torch.linalg.cross(v[fresh], hvec0, dim=-1) / self.mu - r[fresh] / rm0
            e0 = evec0.norm(dim=-1, keepdim=True)
            self.rp0[fresh] = (a0 * (1.0 - e0)).squeeze(-1)
            self.ra0[fresh] = (a0 * (1.0 + e0)).squeeze(-1)

        rp_t = self.rp0 + self.RP_TARGET_RAISE
        a0_t = 0.5 * (self.rp0 + self.ra0)
        a1_t = 0.5 * (rp_t + self.ra0)
        dv_budget = (torch.sqrt(self.mu / a0_t) - torch.sqrt(self.mu / a1_t)).abs()  # (n,)

        rp_now = a * (1.0 - e)
        in_arc = nu.cos() <= self.arc_cos  # |nu - pi| <= arc
        budget_left = (self.dv_used < dv_budget)[:, None]
        below = rp_now < rp_t[:, None]
        burn = budget_left & below & in_arc
        vhat = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        u = torch.where(burn, vhat, torch.zeros_like(vhat))
        # realized Delta-v accounting (exact for this plant: |u| = 1 burns)
        self.dv_used += self.f_max * u.norm(dim=-1) * self.dt / m.squeeze(-1)
        return u.to(torch.float32)

    def config(self):
        return {
            "name": self.name,
            "law": "Edelbaum budgeted tangential",
            "rp_target_raise_m": self.RP_TARGET_RAISE,
            "apogee_half_arc_deg": self.APOGEE_HALF_ARC_DEG,
            "thrust_direction": "local velocity (world frame)",
            "decision_variable": "open-loop Delta-v budget (no per-step scoring)",
        }


# --------------------------------------------------------------------------- #
# docking variant: CWH ZEM/ZEV receding-horizon single-pulse guidance
# --------------------------------------------------------------------------- #
class DockingZEMZEVBaseline(Baseline):
    """CWH ZEM/ZEV single-pulse guidance for final docking approach.

    Task contract (SpaceDockingEnv, identical to the CWH-LQR default): LVLH
    x radial / y along-track / z cross-track, obs x = [p; v] in m / m/s;
    action = per-axis normalized thrust in [-1, 1] -> 10 N/axis on 500 kg
    (a_max = 0.02 m/s^2); success |p| < 1 m AND |v| < 0.1 m/s at one step;
    failure |p| > 200 m or fuel >= 2 kg; 500-step horizon; symplectic Euler
    dt = 1 s; no keep-out sphere on docking.

    Guidance: at every step the two-pulse CWH targeting problem to the port
    (r_f = 0, null arrival velocity) is solved over a receding horizon T_go
    with the EXACT symplectic-Euler STM Phi = Ad^{t_go} (Ad^k = expm(logm(Ad)
    / dt * k dt) via the batched matrix exponential of the discrete
    generator -- the construction is duplicated self-contained from the
    two-impulse rendezvous baseline, giving ZERO coast-arc model error
    against the plant):

        ZEM = r_f - (Phi_rr p + Phi_rv v)      position error at T_go
        ZEV = 0    - (Phi_vr p + Phi_vv v)     velocity error at T_go
        dv1 = Phi_rv^-1 ZEM                    first pulse, applied NOW
        (the second pulse ZEV - Phi_vv dv1 is never commanded explicitly:
        the receding-horizon re-solve each step absorbs it, so the closed
        loop tracks the equilibrium closing speed v ~ -p/T_go and arrives
        below the 0.1 m/s capture tolerance)

    Horizon schedule (hard-coded constants): T_go = clamp(K_T sqrt(d / a_max),
    T_MIN, T_MAX) with K_T = 2, T_MIN = 20 s, T_MAX = 150 s -- the
    acceleration-limited braking-corridor schedule (full-authority stop from
    the profile speed covers sqrt(2 d a_max) < d at K_T = 2, so the profile is
    always trackable) and T_MIN = 20 s caps the terminal equilibrium speed at
    p/T_MIN = 0.05 m/s at the 1 m success boundary.  Ill-conditioned Phi_rv
    (n t_go = k pi manifold, unreachable for T_go <= 150 s << 1/n) is handled
    by the same Tikhonov guard as the two-impulse baseline.

    Constraint layers (mirroring the LQR default's, since the raw guidance
    assumes no constraints):
        1. Range shield: with outward radial rate v_out the stopping distance
           is v_out^2/(2 a_brake); when |p| + stopping distance crosses
           0.95 * range_fail the command is overridden by a full-authority
           brake along -p_hat; inside the 0.8 * range_fail corridor any
           outward thrust component is projected out.
        2. Terminal capture net: within 2 * pos_goal, an over-tolerant
           closing speed (v_close > vel_goal) is braked at full authority so
           the success sphere is crossed below the velocity tolerance.
        3. Fuel derate: the controller integrates its own realized impulse
           (exact for this plant); past 75% of fuel0 the per-step thrust norm
           is capped so the remaining budget outlasts the remaining horizon
           (never below half authority).

    Impulse -> thrust map: the current-step pulse is realized as a one-step
    finite burn F = m dv1/dt (dt = 1 s, i.e. action = dv1 * mass / f_max per
    axis, per-axis clamped).  Stateful members fuel_est / steps_left / stats
    are mirrored in reset_idx.
    """

    name = "zem_zev"

    K_T = 2.0  # braking-corridor schedule gain
    T_MIN = 20.0  # shortest admissible horizon [s] (terminal-speed cap)
    T_MAX = 150.0  # longest admissible horizon [s]
    # Orbital mean motion: 400 km LEO, a = 6.778 Mm, mu = 3.986e14 m^3/s^2
    # n = sqrt(mu / a^3) = 0.0011314 rad/s (docking and rendezvous share this orbit).
    # Hard-coded (RL agent cannot recover n from CWH relative-state obs alone --
    # the LVLH orbital parameters are env-private physics, unavailable to the
    # policy).  Same constant used by every CWH baseline in this repo.
    N = 0.0011314003283155144

    def __init__(self, env, design=None):
        super().__init__(env, design)
        self.mass = self._design_scalar("mass", env.mass)
        self.f_max = self._design_scalar("f_max", env.f_max)
        self.a_max = self.f_max / self.mass
        self.dt = env.dt
        self.isp, self.g0 = self._design_scalar("isp", env.isp), self._design_scalar("g0", env.g0)
        self.pos_goal, self.vel_goal = 1.0, 0.1  # docking hard targets [m, m/s]
        self.range_fail, self.fuel0 = 200.0, 2.0  # docking env hard limits
        self.max_steps = env.max_steps
        # exact one-step transition matrix of the env's symplectic Euler CWH
        # plant (self-contained duplicate of the two-impulse construction).
        # N hard-coded (RL cannot read LVLH n from CWH obs alone).
        n, dt = self.N, env.dt
        G = np.diag((3.0 * n * n, 0.0, -n * n))
        H = np.array(((0.0, 2.0 * n, 0.0), (-2.0 * n, 0.0, 0.0), (0.0, 0.0, 0.0)))
        I3 = np.eye(3)
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = I3 + dt * dt * G
        Ad[:3, 3:] = dt * (I3 + dt * H)
        Ad[3:, :3] = dt * G
        Ad[3:, 3:] = I3 + dt * H
        L = logm(Ad)
        if np.iscomplexobj(L):
            if np.abs(L.imag).max() > 1e-9:
                raise ValueError("symplectic Ad has eigenvalues on the negative real axis")
            L = L.real
        self.gen = torch.as_tensor(L / dt, device=self.device, dtype=COMPUTE_DTYPE)
        # shield / guard thresholds (LQR-default values)
        self.d_corridor = 0.8 * self.range_fail
        self.d_brake = 0.95 * self.range_fail
        self.a_brake = 0.5 * self.a_max
        self.fuel_soft = 0.75 * self.fuel0
        d = self.device
        self.fuel_est = torch.zeros(env.num_envs, device=d, dtype=COMPUTE_DTYPE)
        self.steps_left = torch.full((env.num_envs,), self.max_steps, device=d, dtype=torch.long)
        self.stat_shield = torch.zeros(env.num_envs, device=d, dtype=COMPUTE_DTYPE)
        self.stat_capture = torch.zeros(env.num_envs, device=d, dtype=COMPUTE_DTYPE)
        self.stat_fuel_cap = torch.zeros(env.num_envs, device=d, dtype=COMPUTE_DTYPE)

    def reset_idx(self, mask):
        self.fuel_est.masked_fill_(mask, 0.0)
        self.steps_left.masked_fill_(mask, self.max_steps)
        for s in (self.stat_shield, self.stat_capture, self.stat_fuel_cap):
            s.masked_fill_(mask, 0.0)

    def _stm(self, t_go):
        """Exact symplectic STM blocks for per-env horizons t_go [s]."""
        Phi = torch.matrix_exp(self.gen[None] * t_go[:, None, None])
        return Phi[:, :3, :3], Phi[:, :3, 3:], Phi[:, 3:, :3], Phi[:, 3:, 3:]

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        p, v = o[:, :3], o[:, 3:6]
        d = p.norm(dim=-1)
        d_k = d[:, None]

        # ---- receding-horizon schedule + two-pulse solve
        t_go = (self.K_T * torch.sqrt(d.clamp_min(1e-6) / self.a_max)).clamp(self.T_MIN, self.T_MAX)
        Phi_rr, Phi_rv, _, _ = self._stm(t_go)
        eye = torch.eye(3, device=self.device, dtype=COMPUTE_DTYPE).expand_as(Phi_rv)
        det = torch.linalg.det(Phi_rv).abs()
        bad = det < 1e-10 / self.N**3
        Phi_rv = torch.where(bad[:, None, None], Phi_rv + (1e-6 / self.N) * eye, Phi_rv)
        ZEM = -(torch.bmm(Phi_rr, p.unsqueeze(-1)) + torch.bmm(Phi_rv, v.unsqueeze(-1))).squeeze(-1)
        dv1 = torch.linalg.solve(Phi_rv, ZEM.unsqueeze(-1)).squeeze(-1)
        # current-step impulse as a one-step finite burn: F = m dv1 / dt
        a_cmd = dv1 / self.dt

        # ---- 1) range-corridor shield (state constraint: |p| < range_fail)
        p_hat = p / d_k.clamp_min(1e-9)
        v_out = (v * p_hat).sum(-1, keepdim=True)
        stop = d_k + v_out.clamp_min(0.0) ** 2 / (2 * self.a_brake)
        hard = (v_out > 0) & (stop > self.d_brake)
        a_cmd = torch.where(hard, -self.a_max * p_hat, a_cmd)
        self.stat_shield += hard.squeeze(-1).to(COMPUTE_DTYPE)
        a_out = (a_cmd * p_hat).sum(-1, keepdim=True)
        soft = (d_k > self.d_corridor) & (a_out > 0)
        a_cmd = torch.where(soft, a_cmd - a_out * p_hat, a_cmd)

        # ---- 2) terminal capture net (over-fast crossing of the goal sphere)
        v_close = -(v * p_hat).sum(-1, keepdim=True)
        near = (d_k < 2 * self.pos_goal) & (v_close > self.vel_goal)
        a_cmd = torch.where(near, self.a_max * p_hat, a_cmd)
        self.stat_capture += near.squeeze(-1).to(COMPUTE_DTYPE)

        # ---- 3) fuel derate (budget must outlast the horizon)
        remaining = self.steps_left.clamp_min(1).to(COMPUTE_DTYPE)[:, None]
        f_allow = (
            (self.fuel0 - self.fuel_est[:, None]).clamp_min(0.0)
            * self.isp
            * self.g0
            / (remaining * self.dt)
            * self.mass
        ).clamp_min(0.5 * self.f_max)
        f_cmd = a_cmd.norm(dim=-1, keepdim=True) * self.mass
        over = (f_cmd > f_allow) & (self.fuel_est > self.fuel_soft)[:, None]
        a_cmd = torch.where(over, a_cmd * (f_allow / f_cmd.clamp_min(1e-12)), a_cmd)
        self.stat_fuel_cap += over.squeeze(-1).to(COMPUTE_DTYPE)

        action = (a_cmd * self.mass / self.f_max).clamp(-1.0, 1.0)
        self.fuel_est += action.norm(dim=-1) * self.f_max * self.dt / (self.isp * self.g0)
        self.steps_left -= 1
        return action.to(torch.float32)

    def config(self):
        return {
            "name": self.name,
            "law": "CWH ZEM/ZEV receding-horizon single pulse",
            "k_t": self.K_T,
            "t_min_s": self.T_MIN,
            "t_max_s": self.T_MAX,
            "discretization": "exact symplectic STM (matrix_exp of logm(Ad)/dt)",
            "shield": "0.95*range_fail hard brake + 0.8 corridor projection",
            "capture_net": True,
            "fuel_soft_frac": 0.75,
        }


# --------------------------------------------------------------------------- #
# rendezvous variant: glideslope / range-rate braking approach
# --------------------------------------------------------------------------- #
class RendezvousGlideslopeBaseline(Baseline):
    """Classical glideslope (range-rate braking corridor) approach along the
    initial line of sight to the hold point between the keep-out and success
    spheres.

    Task contract (SpaceRendezvousEnv, identical to the two-impulse default):
    LVLH x radial / y along-track / z cross-track, obs x = [p; v]; per-axis
    normalized thrust in [-1, 1] -> 10 N/axis on 500 kg (0.02 m/s^2 per-axis
    authority); success |p| < 10 m AND |v| < 0.5 m/s (violation-free); 5 m
    keep-out sphere; 2000 m range-fail; 4 kg fuel; 1000-step horizon;
    symplectic Euler dt = 1 s.

    Guidance: the hold radius mirrors the two-impulse default,
        hold = keep_out + 0.8 (pos_goal - keep_out) = 9 m,
    and the approach flies the line of sight latched at reset (u0 = p0/|p0|,
    re-latched after every controller reset).  The commanded range-rate is
    the linear braking corridor with a floor and an inside-hold stop:

        v_prof(d) = V_MAX * clamp((d - hold)/D_BRACKET, V_FRAC_MIN, 1)  d > hold
        v_prof(d) = 0                                                   d <= hold
        v_cmd    = -v_prof(d) * p_hat        (inward along the LOS)

    with V_MAX = 4 m/s, D_BRACKET = 500 m, V_FRAC_MIN = 0.08.  Constants are
    derived, not swept: D_BRACKET >= V_MAX^2/(2 a_max) = 400 m makes the
    corridor head trackable under full-authority braking (a full stop from
    V_MAX covers 400 m, so the commanded profile can never drive the vehicle
    through the hold sphere); the V_FRAC_MIN floor bounds the exponential
    tail crawl of the linear profile (0.32 m/s < vel_goal, so the 10 m goal
    sphere is crossed below the capture tolerance); the full stop inside the
    hold sphere keeps the 5 m keep-out sphere unreachable by construction
    (braking distance at the floor speed is 2.6 m << the 4 m margin).

    Tracking (velocity PD + line keeping):
        a_des = K_V (v_cmd - v) - K_LOS * gate * e_perp,
        e_perp = p - (p^T u0) u0,  gate = clamp((|e_perp| - DEADBAND)/DEADBAND, 0, 1),
    i.e. PD tracking of the commanded LOS velocity whose transverse component
    is exactly the off-LOS velocity damper, plus a soft-deadband position
    feedback that pulls the trajectory back onto the latched LOS line (this
    absorbs the up-to-1 m/s transverse reset velocity: worst-case lateral
    drift ~25 m decays with the K_LOS/K_V slow pole over the ~600 s cruise).
    The CWH coupling terms (3 n^2 x + 2 n ydot etc.) are cancelled exactly as
    in the compensated-PD default, so the net acceleration equals a_des.
    Per-axis action = clamp(m a_cmd / f_max).

    Stateful members u0 (latched LOS) and count (first-call latch) are
    mirrored in reset_idx; all constants are class attributes, hard-coded.
    """

    name = "glideslope"

    V_MAX = 4.0  # corridor entry speed [m/s]
    D_BRACKET = 500.0  # braking-corridor length [m] (>= V_MAX^2 / (2 a_max))
    V_FRAC_MIN = 0.08  # corridor speed floor fraction (0.32 m/s < vel_goal)
    K_V = 0.2  # velocity-tracking gain [1/s]
    K_LOS = 0.001  # lateral line-keeping gain [1/s^2]
    DEADBAND = 1.0  # lateral deadband [m]

    def __init__(self, env, design=None):
        super().__init__(env, design)
        self.mass = self._design_scalar("mass", env.mass)
        self.f_max = self._design_scalar("f_max", env.f_max)
        self.n = 0.0011314003283155144  # LEO mean motion, hardcoded (400 km orbit)
        self.hold = 5.0 + 0.8 * (10.0 - 5.0)  # 9 m hold radius (env.keep_out=5, pos_goal=10)
        d = self.device
        self.u0 = torch.zeros((env.num_envs, 3), device=d, dtype=COMPUTE_DTYPE)
        self.count = torch.zeros(env.num_envs, device=d, dtype=torch.long)

    def reset_idx(self, mask):
        self.count.masked_fill_(mask, 0)
        self.u0.masked_fill_(mask[:, None], 0.0)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        p, v = o[:, :3], o[:, 3:6]
        # latch the initial line of sight (first call of every episode)
        first = self.count == 0
        u_new = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.u0.copy_(torch.where(first[:, None], u_new, self.u0))
        self.count += 1

        d = p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        p_hat = p / d
        # linear braking corridor with floor; full stop inside the hold sphere
        frac = ((d - self.hold) / self.D_BRACKET).clamp(0.0, 1.0)
        v_prof = self.V_MAX * frac
        v_prof = torch.where(d > self.hold, v_prof.clamp_min(self.V_FRAC_MIN * self.V_MAX), v_prof)
        v_cmd = -v_prof * p_hat
        a_des = self.K_V * (v_cmd - v)
        # transverse line keeping with a soft deadband (latched LOS)
        e_perp = p - (p * self.u0).sum(-1, keepdim=True) * self.u0
        gate = ((e_perp.norm(dim=-1, keepdim=True) - self.DEADBAND) / self.DEADBAND).clamp(0.0, 1.0)
        a_des = a_des - self.K_LOS * gate * e_perp
        # exact CWH coupling feedforward: net accel == a_des
        n2 = self.n * self.n
        a_cmd = torch.stack(
            (
                a_des[:, 0] - 3.0 * n2 * p[:, 0] - 2.0 * self.n * v[:, 1],
                a_des[:, 1] + 2.0 * self.n * v[:, 0],
                a_des[:, 2] + n2 * p[:, 2],
            ),
            -1,
        )
        return (a_cmd * (self.mass / self.f_max)).clamp(-1.0, 1.0).to(torch.float32)

    def config(self):
        return {
            "name": self.name,
            "law": "glideslope range-rate braking corridor",
            "hold_radius_m": self.hold,
            "v_max_mps": self.V_MAX,
            "d_bracket_m": self.D_BRACKET,
            "v_frac_min": self.V_FRAC_MIN,
            "k_v": self.K_V,
            "k_los": self.K_LOS,
            "deadband_m": self.DEADBAND,
            "keep_out_avoidance": "by construction (stop on the 9 m hold sphere)",
        }
