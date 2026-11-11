"""Classical baseline VARIANTS for the fixedwing-waypoints and rocket-landing
tasks.

Two non-learning A/B arms against the shipped task defaults
(``FixedwingAutopilotBaseline`` / ``RocketPDGBaseline`` in baselines.py, which
is intentionally untouched -- the orchestrator wires these classes into its
variant registry itself):

  fixedwing  ``nlgl_tecs``  Park's Nonlinear Guidance Law (NLGL, Park, Deyst &
                            Howe 2004) replacing the L1 lateral law, over the
                            SAME duplicated TECS / attitude inner-loop
                            structure as the default autopilot
  rocket     ``eguidance``  Apollo-era E-guidance (quadratic thrust-acceleration
                            polynomial a(t) = a0 + a1 t + a2 t^2, closed-form
                            polynomial targeting re-solved every step)
                            replacing the ZEM/ZEV mid-course law, over the SAME
                            duplicated terminal pulse-servo + TVC/fin attitude
                            stack as the default PDG baseline

Both classes are SELF-CONTAINED: the longitudinal/TECS and terminal/attitude
code is duplicated here (minimal needed subset) rather than imported from the
default classes, so this file can evolve without touching baselines.py.

Every controller reads only ``obs['policy']``, runs batched torch on
env.device in float64 internally, returns actions in the env's native [-1, 1]
convention with the right shape, and keeps its stateful parts clearable
through ``reset_idx``.  All constants are hard-coded class attributes (no
CLI, no config file): as A/B arms they are expected to sit at or below the
heavily tuned defaults, and that gap is the measurement being taken.
"""

import math

import torch

from algorithms.dtypes import COMPUTE_DTYPE

from algorithms.baselines.baselines import Baseline, _quat_to_R, _rot_to_rpy, _vec, _wrap


# --------------------------------------------------------------------------- #
# fixedwing: Park NLGL lateral guidance + duplicated TECS longitudinal stack
# --------------------------------------------------------------------------- #
class FixedwingNLGLBaseline(Baseline):
    """L1 replacement: Park's Nonlinear Guidance Law on the waypoint line.

    Lateral law (Park, Deyst & Howe 2004), replacing the L1 circle of the
    default ``FixedwingAutopilotBaseline``:

      * The desired path is the straight LINE from the previous waypoint to
        the active waypoint (world frame, recovered exactly from the policy
        obs: target = p + R @ delta[:, 0]).  The previous waypoint is tracked
        in controller state: at reset it is the spawn point (0, 0, 10) and it
        rotates forward whenever the observed active target changes.
      * The reference point R is the closest point of that line to the
        PREDICTED position p_hat = p_h + v_h * T_LA (horizontal ground
        velocity, lookahead time T_LA), with the projection CLAMPED to the
        segment [0, |seg|]: the unclamped infinite-line reference lands far
        outside the leg after a waypoint capture and produces large
        wrong-direction commands (observed in traces).  The d vector runs
        from R to the ACTUAL aircraft position.  With the signed angle eta
        between the ground velocity and d (sign via the z-component of
        d_hat x v_hat, left-positive in the env's x-fwd/y-left frame),
            a_lat = 2 V^2 sin(eta) / |d|        (|d| floored at D_MIN)
        and the bank command is bank = -atan(a_lat / g) (env convention:
        positive roll = right bank turns the course negative, so a
        left-positive lateral accel needs bank < 0).  Because d always
        contains the backward-looking -V*T_LA*v_h component, the law is
        self-damping: flying parallel to the line through R commands zero
        lateral acceleration, and the correction grows with the predicted
        cross-track error a_lat = -2 V^2 eps / (L^2 + eps^2), eps = predicted
        left-offset, L = lookahead distance.
      * Anti-parallel fallback (needed for waypoint LEGS, observed in traces):
        Park's law commands zero lateral accel when the velocity is
        (anti)parallel to d -- exactly the post-capture course-reversal
        geometry (next waypoint behind the line start) and the
        flew-past-the-waypoint geometry (tracking the extended line away from
        a vertically-missed waypoint).  Whenever the aircraft makes no
        forward progress along the line (smooth gate on dot(v_h, line)) or
        the predicted position passes the segment end (smooth gate on the
        projection), the law blends into the L1-style pursuit command
        2 V^2 sin(bearing error) / d_h toward the ACTIVE waypoint, whose sin
        self-amplifies from eta ~ 180 deg where the cross-track-driven NLGL
        term is blind.  The bearing error is clamped to +/-90 deg so a
        behind waypoint commands a full-strength turn (the L1 default's
        eta_c trick), with the shipped distance-scheduled denominator and
        near-field deadband.
      * The same aero bank limit as the L1 default is kept verbatim
        (accelerated-stall boundary bank_max = acos((v_stall/V)^2), capped at
        max_bank), together with the altitude-priority climb gate (bank capped
        to 45 deg while a meaningful height error remains -- the airframe is
        power-limited, 18 N thrust < 23 N weight), the identical
        out-of-bounds shield (bank override toward the origin when close to
        the 100 m boundary AND moving outward, clamped to the pure aero
        limit), and the L1 default's terminal pure-pursuit branch
        (-K_PP * eta inside D_PP_ENTRY with the eta gate -- documented fix
        for the sustained-bank sinking orbit at endgame speeds).  The
        (shipped-config-disabled) climb loiter containment is dropped: the
        line law flies THROUGH the 10 m capture sphere by construction and
        does not orbit the waypoint.

    Longitudinal: the TECS / attitude stack of the default autopilot,
    duplicated (not imported) with the identical tuned constants from
    fixedwing_autopilot.json (physics identical): total-energy error drives
    the throttle (P + anti-windup I), energy balance drives the pitch command
    with load-factor feedforward, underspeed protection and the
    overspeed-climb trade; the inner pitch loop is PD with the coordinated
    turn-rate damping correction plus an integral; the inner roll loop is PD
    on the bank command.  The TECS speed scheduling consumes the waypoint-LOS
    angle exactly as the L1 default did (it is a longitudinal scheduling
    input, not the lateral law).

    Stateful parts (cleared by reset_idx): previous/current waypoint memory
    and the throttle/pitch integrators.
    """

    name = "nlgl_tecs"

    # ---- lateral (NLGL) constants ----
    T_LA = 1.2  # lookahead time [s] (L = V_h * T_LA)
    D_MIN = 3.0  # |d| floor [m] (guidance singularity)
    WP_EPS = 0.1  # waypoint-advance detection [m]
    SPAWN_P = (0.0, 0.0, 10.0)  # first-leg previous waypoint
    # aero bank limit + inner roll loop (identical to the L1 default)
    MAX_BANK = math.radians(60.0)
    V_STALL = 11.5  # v_stall_guard [m/s]
    K_ROLL, KD_ROLL = 3.0, 0.5
    # pursuit-regime constants (fixedwing_autopilot.json lateral/longitudinal)
    K_PP = 1.2  # terminal pure-pursuit bank gain
    D_PP_ENTRY = 35.0  # terminal branch entry distance [m]
    ETA_PP_GATE = math.radians(40.0)  # terminal branch bearing gate

    # ---- longitudinal TECS / attitude (fixedwing_autopilot.json verbatim) --
    G = 9.81
    V_REF = 20.0
    V_TURN_MIN = 14.0
    K_SPEED_BLEED = 2.5
    K_DIST_SLOW = 0.35
    D_SLOW_REF = 10.0
    K_AW_SPEED = 0.012
    BANK_CLIMB = math.radians(45.0)
    V_CLIMB_REF = 14.0
    H_CLIMB_GATE = 5.0
    PN_SHIELD = 72.0
    V_SHIELD_MIN = 0.5
    K_SHIELD = 2.0
    K_ENERGY = 0.008
    K_ENERGY_I = 0.005
    INTEGRAL_MAX = 0.2
    THROTTLE_TRIM = 0.35
    K_BALANCE = 0.008
    K_SPEED_KEEP = 0.05
    K_GEO_UP = 0.6
    PITCH_TRIM = math.radians(-2.0)
    PITCH_UP_MAX = math.radians(16.0)
    PITCH_UP_V = (11.0, 15.0)
    PITCH_DOWN_MAX = math.radians(12.0)
    ALPHA_TRIM = math.radians(2.5)
    K_PITCH, KD_PITCH, K_PITCH_I = 4.0, 0.45, 2.0

    # tunable gain space for online gain scheduling
    TUNABLE_PARAMS = {
        "T_LA": dict(low=0.5, high=2.5, default=1.2),
        "K_ROLL": dict(low=1.5, high=5.0, default=3.0),
        "KD_ROLL": dict(low=0.2, high=1.2, default=0.5),
        "K_PP": dict(low=0.5, high=2.5, default=1.2),
        "K_ENERGY": dict(low=0.003, high=0.02, default=0.008, log=True),
        "K_BALANCE": dict(low=0.003, high=0.02, default=0.008, log=True),
        "K_PITCH": dict(low=2.0, high=7.0, default=4.0),
        "KD_PITCH": dict(low=0.2, high=1.0, default=0.45),
    }

    def __init__(self, env, design=None):
        super().__init__(env, design)
        # tunable gains as instance attributes
        for _k in self.TUNABLE_PARAMS:
            setattr(self, _k, self._param(_k))
        d = self.device
        f64 = COMPUTE_DTYPE
        self.dt = 1.0 / 30.0  # agent control period
        self.spawn_w = torch.tensor(self.SPAWN_P, device=d, dtype=f64)
        n = self.num_envs
        self.prev_wp = torch.zeros((n, 3), device=d, dtype=f64)
        self.cur_wp = torch.zeros((n, 3), device=d, dtype=f64)
        self.has_wp = torch.zeros(n, device=d, dtype=torch.bool)
        self.integral = torch.zeros(n, device=d, dtype=f64)
        self.i_pitch = torch.zeros(n, device=d, dtype=f64)

    def reset_idx(self, mask):
        self.prev_wp.masked_fill_(mask[:, None], 0.0)
        self.cur_wp.masked_fill_(mask[:, None], 0.0)
        self.has_wp.masked_fill_(mask, False)
        self.integral.masked_fill_(mask, 0.0)
        self.i_pitch.masked_fill_(mask, 0.0)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 23:35].reshape(-1, 4, 3)
        R = _quat_to_R(q)
        v_w = torch.einsum("nij,nj->ni", R, vb)
        V = vb.norm(dim=-1)
        roll, pitch, _ = _rot_to_rpy(R)
        target = p + torch.einsum("nij,nj->ni", R, delta[:, 0])

        # ---- waypoint memory: prev -> cur defines the NLGL line ----
        fresh = ~self.has_wp
        advanced = (~fresh) & ((target - self.cur_wp).norm(dim=-1) > self.WP_EPS)
        self.prev_wp = torch.where(
            fresh[:, None], self.spawn_w, torch.where(advanced[:, None], self.cur_wp, self.prev_wp)
        )
        self.cur_wp = torch.where((fresh | advanced)[:, None], target, self.cur_wp)
        self.has_wp |= fresh

        # ---- NLGL lateral guidance on the prev -> cur line ----
        prev_h, cur_h = self.prev_wp[:, :2], self.cur_wp[:, :2]
        seg = cur_h - prev_h
        seg_n = seg.norm(dim=-1)
        v_h = v_w[:, :2]
        V_h = v_h.norm(dim=-1)
        course = torch.where(
            V_h > 2.0, torch.atan2(v_w[:, 1], v_w[:, 0]), torch.atan2(R[:, 1, 0], R[:, 0, 0])
        )  # heading fb
        v_hat = torch.stack((course.cos(), course.sin()), -1)
        # unit line direction (degenerate segment: bearing to the waypoint)
        n_hat = seg / seg_n.clamp_min(1e-6)[:, None]
        to_wp = cur_h - p[:, :2]
        n_hat = torch.where(
            (seg_n < 1.0)[:, None], to_wp / to_wp.norm(dim=-1).clamp_min(1e-6)[:, None], n_hat
        )
        # predicted position -> closest point R on the segment (projection
        # clamped to [0, |seg|]: the unclamped infinite-line reference lands
        # far outside the leg after a waypoint capture, where the law produces
        # large wrong-direction commands -- observed in traces); d = R ->
        # aircraft.  With R clamped at the segment end the law degenerates to
        # pursuit of the waypoint, so the terminal regime is pursuit-like by
        # construction.
        p_hat = p[:, :2] + v_hat * (V_h * self.T_LA)[:, None]
        s_raw = ((p_hat - prev_h) * n_hat).sum(-1)
        s_ref = s_raw.clamp_min(0.0).clamp_max(seg_n)
        r_ref = prev_h + n_hat * s_ref[:, None]
        d_vec = p[:, :2] - r_ref
        d_norm = d_vec.norm(dim=-1)
        d_hat = d_vec / d_norm.clamp_min(1e-9)[:, None]
        # signed sin(eta), eta = angle between ground velocity and d
        # (z-component of d_hat x v_hat; left-positive in the env frame)
        sin_eta = d_hat[:, 0] * v_hat[:, 1] - d_hat[:, 1] * v_hat[:, 0]
        a_nlgl = 2.0 * V_h**2 * sin_eta / d_norm.clamp_min(self.D_MIN)

        d3 = delta[:, 0].norm(dim=-1)
        # ---- pursuit regime (L1 default's proven mid-course law toward the
        # ACTIVE waypoint), used where the line law is structurally blind:
        # (a) flying AGAINST the line (post-capture 180-deg course reversal:
        # d anti-parallel to v commands zero lateral accel), and (b) the
        # terminal/past regime (predicted position beyond the segment end:
        # d anti-parallel to v again once receding).  eta_c clamps the
        # bearing error to +/-90 deg so a behind waypoint commands a
        # full-strength turn (the L1 default's trick -- sin(eta) would vanish
        # exactly when a hard turn is needed); the distance-scheduled
        # denominator and the near-field deadband are the shipped defaults.
        fwd = (v_hat * n_hat).sum(-1)
        w_back = torch.sigmoid((-fwd - 0.3) / 0.2)
        w_term = torch.sigmoid((s_raw - seg_n + 3.0) / 3.0)
        w_purs = (w_back + w_term).clamp(0.0, 1.0)
        d_h_norm = to_wp.norm(dim=-1)
        los = torch.atan2(to_wp[:, 1], to_wp[:, 0])
        eta = _wrap(los - course)
        eta_c = eta.clamp(-math.pi / 2, math.pi / 2)
        l1_eff = (0.6 * d_h_norm + 8.0).clamp(12.0, 40.0)
        a_pp = 2.0 * V**2 * torch.sin(eta_c) / l1_eff
        a_pp = a_pp * (d_h_norm > 3.0).to(a_pp.dtype)  # near-field deadband
        a_lat = (1.0 - w_purs) * a_nlgl + w_purs * a_pp
        bank_nlgl = -torch.atan(a_lat / self.G)
        # Terminal pure-pursuit branch of the L1 default (documented fix for
        # the sustained-bank sinking orbit at endgame speeds): bend the path
        # less when close and misaligned.
        bank_pp = -self.K_PP * eta
        w_pp = torch.sigmoid((self.D_PP_ENTRY - d3) / 3.0) * torch.sigmoid(
            (eta.abs() - self.ETA_PP_GATE) / -0.1
        )
        bank_purs = w_pp * bank_pp + (1.0 - w_pp) * (-torch.atan(a_pp / self.G))
        bank_raw = (1.0 - w_purs) * bank_nlgl + w_purs * bank_purs

        # ---- bank limits: aero (stall) + climb gate, as the L1 default ----
        h_err = target[:, 2] - p[:, 2]
        bank_lim = torch.acos(((self.V_STALL / V) ** 2).clamp(0.0, 1.0)).clamp_max(self.MAX_BANK)
        bank_aero = bank_lim
        w_climb = torch.sigmoid((h_err - self.H_CLIMB_GATE) / 3.0)
        bank_climb = torch.minimum(bank_lim, torch.full_like(bank_lim, self.BANK_CLIMB))
        bank_lim = bank_lim + (bank_climb - bank_lim) * w_climb
        # env convention: positive roll = right bank turns the course negative,
        # so a left-positive a_lat needs bank < 0.
        bank_cmd = bank_raw.clamp(-bank_lim, bank_lim)
        # out-of-bounds shield (identical to the L1 default): near the 100 m
        # boundary AND moving outward, override with a bank toward the origin.
        pn = p.norm(dim=-1)
        v_toward = -(p * v_w).sum(-1) / pn.clamp_min(1e-6)
        w_shield = torch.sigmoid((pn - self.PN_SHIELD) / 4.0) * torch.sigmoid(
            (v_toward - self.V_SHIELD_MIN) / 1.0
        )
        if bool(w_shield.any()):
            # safety overrides the climb bank cap: clamp to the pure aero limit
            eta_home = _wrap(torch.atan2(-p[:, 1], -p[:, 0]) - course)
            bank_shield = -self.K_SHIELD * eta_home
            bank_cmd = (1.0 - w_shield) * bank_cmd + w_shield * bank_shield
            bank_cmd = bank_cmd.clamp(-bank_aero, bank_aero)
        aileron = self.K_ROLL * _wrap(bank_cmd - roll) - self.KD_ROLL * wb[:, 0]

        # ---- TECS: total energy -> throttle, energy balance -> pitch ----
        # (duplicated from FixedwingAutopilotBaseline with identical constants;
        # the speed scheduling consumes the waypoint-LOS angle as before)
        d_h = target[:, :2] - p[:, :2]
        los = torch.atan2(d_h[:, 1], d_h[:, 0])
        eta = _wrap(los - course)
        v_eff = (self.V_REF - self.K_SPEED_BLEED * eta.abs()).clamp(self.V_TURN_MIN, self.V_REF)
        v_dist = (self.V_TURN_MIN + self.K_DIST_SLOW * (d3 - self.D_SLOW_REF)).clamp(
            self.V_TURN_MIN, self.V_REF
        )
        v_eff = torch.minimum(v_eff, v_dist)
        v_climb = self.V_REF + (self.V_CLIMB_REF - self.V_REF) * w_climb
        v_eff = torch.minimum(v_eff, v_climb)
        dv = (v_eff**2 - V**2) / (2.0 * self.G)
        dE = (h_err + dv).clamp(-25.0, 25.0)
        dB = h_err - dv
        aw = self.K_AW_SPEED * (V - v_eff).clamp_min(0.0) * (dE < 0.0).to(dE.dtype)
        integral = (self.integral + self.K_ENERGY_I * dE * self.dt - aw * self.dt).clamp(
            0.0, self.INTEGRAL_MAX
        )
        self.integral.copy_(integral)
        throttle = (self.THROTTLE_TRIM + self.K_ENERGY * dE + integral).clamp(0.0, 1.0)
        load_extra = (1.0 / torch.cos(roll).clamp_min(0.2) - 1.0) * self.ALPHA_TRIM
        v_overspeed = (V - v_eff).clamp_min(0.0)
        pitch_cmd = (
            self.PITCH_TRIM
            - self.K_BALANCE * dB
            - load_extra
            + self.K_SPEED_KEEP * (v_eff - V).clamp_min(0.0)
            - self.K_GEO_UP * v_overspeed * w_climb
        )
        pv_lo, pv_hi = self.PITCH_UP_V
        pitch_up = self.PITCH_UP_MAX * ((V - pv_lo) / (pv_hi - pv_lo)).clamp(0.0, 1.0)
        pitch_cmd = pitch_cmd.clamp_min(-pitch_up).clamp_max(self.PITCH_DOWN_MAX)
        q_coord = wb[:, 2] * torch.tan(roll.clamp(-1.45, 1.45))
        i_pitch = (self.i_pitch + self.K_PITCH_I * (pitch_cmd - pitch) * self.dt).clamp(-0.35, 0.35)
        self.i_pitch.copy_(i_pitch)
        elevator = (
            self.K_PITCH * (pitch_cmd - pitch) - self.KD_PITCH * (wb[:, 1] - q_coord) + i_pitch
        )

        a = torch.stack((aileron, elevator, torch.zeros_like(aileron), 2.0 * throttle - 1.0), -1)
        return a.clamp(-1.0, 1.0).to(torch.float32)


# --------------------------------------------------------------------------- #
# rocket: Apollo E-guidance mid-course + duplicated terminal pulse-servo stack
# --------------------------------------------------------------------------- #
class RocketEGuidanceBaseline(Baseline):
    """ZEM/ZEV replacement: quadratic-polynomial (Apollo E-) guidance.

    Environment contract identical to ``RocketPDGBaseline`` (verified there
    against rocket/env.py, environments/common/plant.py and the URDF/YAML):
    world-frame position (z up, gravity -9.81), body-frame velocities, thrust
    along body +z with a +/-5 deg gimbal, grid fins 2.64 m above the CM with a
    V^2-scaled restoring spring, min thrust/weight 1.91 > 1 at spawn mass (once
    lit the engine can only brake, never hold a slow descent), 7-dim finlet
    mixer action in [-1, 1] (ignition/throttle remapped to [0, 1] by the env),
    success = on-pad settle (speed/spin < 0.02, tilt < 0.1) and success does
    NOT truncate.

    Mid-course guidance (replacing the ZEM/ZEV law): classical Apollo-era
    E-guidance.  The thrust acceleration is the quadratic polynomial
        u(t) = a0 + a1 t + a2 t^2
    targeted on the pad rest point rf with ZERO terminal velocity at the
    time-to-go t_go (dynamics vdot = u + grav, grav = -9.81 zhat):
        c1 = -v - grav * t_go                      (velocity to gain)
        c2 = rf - p - v * t_go - grav * t_go^2 / 2 (position to gain)
        a0 T + a1 T^2/2 + a2 T^3/3   = c1
        a0 T^2/2 + a1 T^3/6 + a2 T^4/12 = c2 .
    This Chandler/Smith boundary-value system leaves one free vector
    parameter, closed per axis as follows:
      * Lateral (x, y): the Apollo landing boundary condition u_xy(T) = 0
        (thrust vertical at the target) -- a 3x3 closed-form solve per axis
        (fixed confluent system, well conditioned for T in [0.5, 30] s).
        Versus ZEM/ZEV the lateral correction is front-loaded and the
        terminal thrust direction is vertical.
      * Vertical (z): the 2-condition closed-form fit (a2_z = 0,
        gravity-inclusive -- note the PDG default uses the gravity-free ZEM
        convention instead, with the velocity term sign-flipped for
        receding-horizon stability; see its inline comment).  The
        terminal-accel-conditioned vertical fit was evaluated and REJECTED:
        it demands initial free-fall so deep (u_z(0) ~ -108 m/s^2 at spawn)
        that the receding-horizon loop destabilizes.
    The command applied at each step is u(0) = a0; t_go is RE-SOLVED every
    step from the POSITION-CONSISTENT constant-decel descent profile
    v(z) = sqrt(2 a_brake h + V_TOUCH^2) (h = altitude above the pad,
    a_brake = 60% of max net braking, floored at TGO_FLOOR), making the law a
    closed-loop receding-horizon targeting.  This t_go pairing is
    load-bearing for the gravity-inclusive fit: with the PDG default's
    velocity-schedule t_go the spawn boundary ("at rest at the pad in 4.4 s
    from 430 m up") is unreachable, the exact polynomial solution dives
    first and procrastinates the burn, and the measured result was 9%
    success with ~119 m/s fatal impacts; with the reachable boundary the fit
    yields the classical coast-until-the-profile-catches-you-then-brake
    profile (u_z(0) ~ +6 m/s^2 at spawn -- a gentle thrust demand below the
    engine's minimum, so the vehicle honestly coasts out until ignition at
    h ~ 380 m) and 26.5% success.  The min-energy closure of the free
    parameter was also evaluated and rejected: the true energy-optimal
    acceleration is LINEAR in t (a2 = 0 identically), i.e. it degenerates to
    the ZEM/ZEV family member and provides no A/B contrast.

    Thrust saturation, handled honestly:
      * ABOVE max thrust: the demand is clamped at T_MAX and counted in
        diag['infeasible'] (same as the PDG default; the receding-horizon
        re-solve picks the error up on later steps).
      * BELOW min thrust: the engine cannot throttle under
        MIN_RATIO * T_MAX, so the ignition hysteresis simply keeps it CUT and
        the vehicle coasts ballistically; because t_go and the polynomial are
        re-solved from the actual state every step, the coast error is
        corrected closed-loop (no integrator windup exists in the law).

    Terminal phase and attitude stack: duplicated verbatim (minimal needed
    code) from ``RocketPDGBaseline`` -- the two-stage suicide-burn endgame
    (virtual-pad min-thrust manifold, then the constant-rate velocity-band
    pulse servo with ignition hysteresis S_OFF/S_ON flying the last metres at
    ~0.5 m/s), the aerodynamic-moment feed-forward, the tilt PD + yaw damping,
    the observed-actuator lateral force split, and the TVC-first torque
    allocation.  Same probe-verified vehicle constants.

    Stateful parts: none (the ignition state is read from the observed
    ignited flag; every guidance quantity is re-solved from the current obs),
    so the inherited no-op reset_idx is correct.
    """

    name = "eguidance"

    # vehicle constants (rocket.urdf / rocket.yaml; same probe-verified values
    # as RocketPDGBaseline)
    M_DRY = 138.2
    FUEL_FULL = 410.9
    T_MAX = 7607.0
    MIN_RATIO = 2966.7 / 7607.0
    GIMBAL_RAD = math.radians(5.0)
    G = 9.81
    FIN_K = 0.22668
    FIN_YAW_K = 0.1587
    AERO_K = 3.6
    REST_Z = 2.575

    # guidance / control gains (t_go schedule + endgame identical to the PDG
    # default; the A/B is the mid-course law shape)
    V_TOUCH = 0.35
    TGO_FLOOR = 0.5
    S_GATE = 25.0
    EPS_END = 1.0
    H_VIRT = 2.0
    S_PULSE = 1.5
    S_OFF = 0.30
    S_ON = 0.40
    H_PULSE = 2.0
    A_LAT_PULSE = 1.5
    LAT_TILT = 0.11
    A_LAT_MAX = 4.5
    KP_T = 12000.0
    KD_T = 6000.0
    KD_Z = 3.0

    def __init__(self, env, design=None):
        super().__init__(env, design)
        if getattr(env, "f9_real", False):
            # v1 is the real-scale Falcon 9 asset: its fuel link is in kg and
            # its thrust/gimbal/fin constants differ from the 1:10 v2 model.
            self.M_DRY = 25600.0
            self.FUEL_FULL = 410900.0
            self.T_MAX = 845000.0
            self.MIN_RATIO = 0.57
            self.FIN_K = 45.0
            self.FIN_YAW_K = 30.0
            self.AERO_K = 400.0
            self.REST_Z = 0.65
            self.V_TOUCH = 1.0
            self.S_GATE = 60.0
            self.H_VIRT = 0.0
            self.S_PULSE = 0.0
            self.H_PULSE = 0.0
            self.KP_T = 4.0e6
            self.KD_T = 4.0e6
            self.KD_Z = 2000.0
        d = self.device
        f64 = COMPUTE_DTYPE
        self.ez = torch.tensor((0.0, 0.0, 1.0), device=d, dtype=f64)
        self.rf = torch.tensor((0.0, 0.0, self.REST_Z), device=d, dtype=f64)
        self.gvec = torch.tensor((0.0, 0.0, self.G), device=d, dtype=f64)
        # on-device diagnostics (analogs of solver statistics)
        z = torch.zeros(self.num_envs, device=d, dtype=f64)
        self.diag = {
            k: z.clone()
            for k in ("steps", "sat_max", "coast", "ignitions", "infeasible", "floor", "fin_sat")
        }

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        # aux tail: [13:20] 7-dim action echo, [20:24] FOUR finlet actuation
        # states, then ignited=24, fuel=25, throttle=26, gimbal=27:29, pad=29
        ignited_obs = o[:, 24] > 0.5
        fuel = o[:, 25].clamp(0.0, 1.0)
        m = self.M_DRY + fuel * self.FUEL_FULL
        R = _quat_to_R(q)
        v_w = torch.einsum("nij,nj->ni", R, vb)
        s = (-v_w[:, 2]).clamp_min(0.0)  # descent speed

        # ---- time-to-go from the POSITION-CONSISTENT constant-decel descent
        # profile v(z) = sqrt(2 a_brake h + v_touch^2), a_brake = 60% of the
        # max net decel: t_go is the profile time from the current ALTITUDE.
        # This pairing is load-bearing for the gravity-inclusive polynomial
        # fit below: with the default's velocity-schedule t_go the spawn
        # boundary ("at rest at the pad in 4.4 s from 430 m up") is
        # unreachable, and the exact BVP solution dives first and brakes
        # later -- procrastination that the receding-horizon re-solve repeats
        # until the burn is infeasible (measured: 9% success, ~119 m/s fatal
        # impacts).  With the reachable boundary the fit yields the classical
        # coast-until-the-profile-catches-you-then-brake suicide profile, and
        # the commanded vertical accel never goes strongly negative.
        h_agl = (p[:, 2] - self.REST_Z).clamp_min(0.0)
        a_net = (self.T_MAX / m - self.G).clamp_min(1.0)
        a_brake = 0.6 * a_net
        v_prof = (2.0 * a_brake * h_agl + self.V_TOUCH**2).clamp_min(0.0).sqrt()
        t_go = ((v_prof - self.V_TOUCH) / a_brake).clamp(self.TGO_FLOOR, 30.0)
        self.diag["floor"] += (t_go <= self.TGO_FLOOR + 1e-9).to(COMPUTE_DTYPE)

        # ---- E-guidance: closed-form quadratic targeting on rf with zero
        # terminal velocity at t_go (Chandler/Smith boundary-value system,
        # re-solved every step from the current state).  Thrust accel
        # u(t) = a0 + a1 t + a2 t^2, dynamics vdot = u + grav.
        T = t_go
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        grav = -self.gvec
        c1 = -v_w - grav * T[:, None]
        c2 = self.rf - p - v_w * T[:, None] - grav * (T2 / 2.0)[:, None]
        # lateral: free parameter closed by u_xy(T) = 0 (thrust vertical at
        # the target) -> 3x3 solve per axis
        M = torch.stack(
            (
                torch.stack((T, T2 / 2.0, T3 / 3.0), -1),
                torch.stack((T2 / 2.0, T3 / 6.0, T4 / 12.0), -1),
                torch.stack((torch.ones_like(T), T, T2), -1),
            ),
            -2,
        )
        rhs = torch.stack((c1[:, :2], c2[:, :2], torch.zeros_like(c1[:, :2])), 1)
        sol = torch.linalg.solve(M, rhs)
        # vertical: 2-condition closed-form fit (a2_z = 0, gravity-inclusive)
        a0_z = -2.0 * c1[:, 2:3] / T[:, None] + 6.0 * c2[:, 2:3] / T2[:, None]
        a_cmd = torch.cat((sol[:, 0, :], a0_z), -1)  # u(0) = a0

        a_vert = (a_cmd[:, 2] + self.G).clamp_min(0.0)
        lat_cap = (self.LAT_TILT * a_vert).clamp(1.0, self.A_LAT_MAX)[:, None]
        a_cmd = torch.cat((a_cmd[:, :2].clamp(-lat_cap, lat_cap), a_cmd[:, 2:]), -1)
        a_thr = a_cmd + self.gvec
        # the engine can only push along +z: never reward a downward-accel
        # demand with thrust
        a_thr = torch.cat((a_thr[:, :2], a_thr[:, 2:].clamp_min(0.0)), -1)
        a_mag = a_thr.norm(dim=-1).clamp_min(1e-6)
        T_raw = m * a_mag
        T_cmd = T_raw.clamp(0.0, self.T_MAX)
        T_min = self.MIN_RATIO * self.T_MAX
        self.diag["infeasible"] += (T_raw > self.T_MAX).to(COMPUTE_DTYPE)
        ign_mid = torch.where(ignited_obs, T_cmd >= 0.97 * T_min, T_cmd >= 1.03 * T_min)

        # ---- endgame: two-stage suicide burn (duplicated from the PDG
        # default): stage 1 brakes along the min-thrust manifold of a virtual
        # pad H_VIRT above the real one; stage 2 is the constant-rate pulse
        # servo (ignition hysteresis on airspeed) flying the last metres at
        # ~0.5 m/s.  Pad contact (obs tail) forces the engine off for settle.
        pad_c = o[:, 29] > 0.5
        endg = (s < self.S_GATE) & (h_agl > 0.0)
        a_min = (self.MIN_RATIO * self.T_MAX / m - self.G).clamp_min(0.5)
        A_END = a_min + self.EPS_END
        h_v = (h_agl - self.H_VIRT).clamp_min(0.05)
        s_t = (self.V_TOUCH**2 + 2.0 * A_END * h_v).clamp_min(0.0).sqrt()
        a_req = (v_w[:, 2].square() - s_t.square()) / (2.0 * h_v) + self.G + A_END
        pulse = endg & (s <= self.S_PULSE) & (h_agl <= self.H_PULSE)
        a_vert_end = torch.where(pulse, self.G + A_END, a_req)
        T_end = (m * a_vert_end).clamp(0.0, self.T_MAX)
        self.diag["infeasible"] += (endg & ~pulse & (m * a_req > self.T_MAX)).to(COMPUTE_DTYPE)
        a_lat_end = -1.5 * v_w[:, :2] - (p[:, :2] - self.rf[:2])
        lat_end_cap = (self.LAT_TILT * a_vert_end).clamp(1.0, self.A_LAT_MAX)
        # pulse phase: damp LATERAL VELOCITY only, never chase position
        a_lat_end = torch.where(
            pulse[:, None],
            (-1.5 * v_w[:, :2]).clamp(-self.A_LAT_PULSE, self.A_LAT_PULSE),
            a_lat_end.clamp(-lat_end_cap[:, None], lat_end_cap[:, None]),
        )
        a_thr_end = torch.cat((a_lat_end, a_vert_end[:, None]), -1)
        ign_pulse = torch.where(ignited_obs, s >= self.S_OFF, s >= self.S_ON)
        ign_end = torch.where(
            pulse, ign_pulse, torch.where(ignited_obs, T_end >= 0.97 * T_min, T_end >= 1.03 * T_min)
        )
        # keep the band servo alive through the settle phase (pad_c only cuts
        # the engine on a sensed contact): the band IS the rebound catcher
        ign = torch.where(endg, ign_end, ign_mid) & ~pad_c
        self.diag["ignitions"] += (ign & ~ignited_obs).to(COMPUTE_DTYPE)
        self.diag["coast"] += (~ign).to(COMPUTE_DTYPE)
        T_cmd = torch.where(endg, T_end, T_cmd)
        a_thr = torch.where(endg[:, None], a_thr_end, a_thr)
        a_mag = torch.where(endg, a_thr_end.norm(dim=-1).clamp_min(1e-6), a_mag)
        u_thr = ((T_cmd / self.T_MAX - self.MIN_RATIO) / (1.0 - self.MIN_RATIO)).clamp(0.0, 1.0)
        self.diag["sat_max"] += (ign & (u_thr >= 0.999)).to(COMPUTE_DTYPE)

        # ---- desired thrust direction (keep the guidance tilt across
        # coast/burn transitions)
        b3 = torch.where(
            (a_mag > 1e-6)[:, None], a_thr / a_mag[:, None], self.ez.expand(self.num_envs, 3)
        )
        b3 = torch.where((b3[:, 2] > 0.15)[:, None], b3, self.ez.expand(self.num_envs, 3))
        u = torch.einsum("nji,nj->ni", R, b3)  # desired body z
        e_tilt = torch.stack((-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])), -1)

        # ---- aerodynamic-moment feed-forward: cancel the fin spring
        V2 = vb.square().sum(-1)  # airspeed^2
        Vn = v_w.norm(dim=-1).clamp_min(1.0)
        flow = -v_w / Vn[:, None]  # retrograde axis
        f_b = torch.einsum("nji,nj->ni", R, flow)  # ...in body frame
        e_flow = torch.stack((-f_b[:, 1], f_b[:, 0], torch.zeros_like(f_b[:, 0])), -1)
        cm_z = (47.0 * -2.0 + 0.2 * 2.051) / m  # fueltank at z=0
        z_fin = 2.051 - cm_z
        h_lev = 2.0 + cm_z  # booster lever below CM
        ff_cap = (0.95 * self.FIN_K * z_fin * V2)[:, None]
        tau_ff = ((self.AERO_K * V2)[:, None] * e_flow).clamp(-ff_cap, ff_cap)

        # ---- attitude PD (tilt tracking + spin damping, yaw free)
        tau_cmd = torch.cat(
            (self.KP_T * e_tilt[:, :2] - self.KD_T * wb[:, :2], -self.KD_Z * wb[:, 2:3]), -1
        )
        tau_cmd[:, :2] -= tau_ff[:, :2]

        # ---- lateral force split: thrust tilt + gimbal side-force observed,
        # fins make up the residual (capped at 30% of fin force authority)
        u_obs = o[:, 26]
        T_act = self.T_MAX * u_obs
        bz_w = R[..., 2]  # body z in world
        a_lat_thr = (T_act / m)[:, None] * bz_w[:, :2]
        g_ob = o[:, 27:29] * self.GIMBAL_RAD
        f_gim_b = torch.stack(
            (
                T_act * torch.sin(g_ob[:, 1]),
                -T_act * torch.sin(g_ob[:, 0]) * torch.cos(g_ob[:, 1]),
                torch.zeros_like(g_ob[:, 0]),
            ),
            -1,
        )
        a_lat_gim = torch.einsum("nij,nj->ni", R, f_gim_b)[:, :2] / m[:, None]
        k_fin = (self.FIN_K * z_fin * V2).clamp_min(1e-9)
        T_now = torch.where(ign, T_cmd, torch.zeros_like(T_cmd))
        fin_a_cap = (0.3 * self.FIN_K * V2 / m)[:, None]
        a_fin_w = (a_thr[:, :2] - a_lat_thr - a_lat_gim).clamp(-fin_a_cap, fin_a_cap)
        self.diag["fin_sat"] += (a_fin_w.abs() >= fin_a_cap - 1e-12).any(-1).to(COMPUTE_DTYPE)
        a_fin_b = torch.einsum(
            "nji,nj->ni", R, torch.cat((a_fin_w, torch.zeros_like(a_fin_w[:, :1])), -1)
        )[:, :2]
        d_fin = a_fin_b * (m / (self.FIN_K * V2).clamp_min(1e-9))[:, None]
        tau_coup = torch.stack((-a_fin_b[:, 1], a_fin_b[:, 0], torch.zeros_like(a_fin_b[:, 0])), -1)
        tau_coup = tau_coup * (m * z_fin)[:, None]
        tau_cmd = tau_cmd + torch.where(ign[:, None], -tau_coup, torch.zeros_like(tau_coup))

        # ---- torque allocation by ceiling: fast TVC first, fins for the rest
        k_gim = (h_lev * T_now * self.GIMBAL_RAD).clamp_min(1e-9)
        gim_cap = 0.95 * k_gim
        fin_cap = 0.35 * k_fin
        tau_gim = tau_cmd.clamp(-gim_cap[:, None], gim_cap[:, None])
        tau_fin = (tau_cmd - tau_gim).clamp(-fin_cap[:, None], fin_cap[:, None])
        k_yaw = (self.FIN_YAW_K * V2).clamp_min(1e-9)
        # sign map verified by probes (see RocketPDGBaseline)
        fx = tau_fin[:, 1] / k_fin + d_fin[:, 0]
        fy = -tau_fin[:, 0] / k_fin + d_fin[:, 1]
        gx = torch.where(T_now > 1.0, -tau_gim[:, 0] / k_gim, torch.zeros_like(tau_gim[:, 0]))
        gy = torch.where(T_now > 1.0, -tau_gim[:, 1] / k_gim, torch.zeros_like(tau_gim[:, 1]))
        yaw = -self.KD_Z * wb[:, 2] / k_yaw
        # Unified 7-dim finlet mixer convention; ignition/throttle in [-1, 1]
        # (the env _scale_actions maps to [0, 1] at step time).
        raw = torch.stack(
            (
                fx,
                fy,
                yaw,
                torch.where(ign, torch.ones_like(u_thr), -torch.ones_like(u_thr)),
                2.0 * u_thr - 1.0,
                gx,
                gy,
            ),
            -1,
        )
        self.diag["steps"] += 1.0
        return raw.clamp(-1.0, 1.0).to(torch.float32)
