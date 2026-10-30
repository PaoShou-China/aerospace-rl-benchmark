"""Classical baseline VARIANTS for the missile-intercept and quadx-track tasks.

Four non-learning A/B arms against the shipped task defaults
(``MissileAPNBaseline`` / ``QuadXTrackKFBaseline`` in baselines.py, which is
intentionally untouched -- the orchestrator wires these classes into
``BASELINE_VARIANTS`` itself):

  missile
    ppn        pure pursuit: lateral acceleration along the instantaneous LOS
               direction, point-mass gravity compensation toward -g, mapped
               through the shared skid-to-turn kinematic inverse
    tpn        true proportional navigation ``N' Vc (lam_dot x lam_hat)``
               WITHOUT the APN ``(N/2) g`` gravity augmentation, replaced by
               a small standalone gravity-bias feedforward

  quadx_track
    pursuit    estimator-free pursuit of the raw noisy relative measurement
               (KF replaced by the measurement itself)
    pid_ema    first-order EMA target-position filter replacing the Kalman
               filter, with a finite-difference EMA velocity feedforward
               (the simpler-filter family alternative to ``cv_kf_pid``)

Every controller reads only obs['policy'], runs batched torch on env.device,
returns actions in the env's native [-1, 1] convention, and keeps all its
stateful parts clearable through reset_idx.  All constants are hard-coded
class attributes selected a priori: as A/B arms these are expected to sit at
or below the tuned defaults, and that gap is the measurement being taken.
"""
import torch

from algorithms.baselines.baselines import (Baseline, QuadXSE3Baseline,
                                            _cross_cols, _quat_to_matrix, _vec)


# --------------------------------------------------------------------------- #
# missile: shared guidance plumbing
# --------------------------------------------------------------------------- #
class _MissileRateBaseline(Baseline):
    """Shared plumbing for the missile guidance variants.

    Owns the per-axis rate-command limits imported from the env
    (roll OMEGA_MAX, pitch/yaw OMEGA_LAT), the first-order LOS-rate low-pass
    filter with its terminal Vc < 50 m/s freeze, and the skid-to-turn mapping
    of a world-frame specific-acceleration command onto the env's
    (roll, pitch, yaw) body-rate interface:
        a_lat_b = body-frame command with the thrust-axis (b_x) part dropped
        omega   = cross(b_x, a_lat_b) / V
    i.e. rotating the thrust axis toward the commanded lateral-acceleration
    direction at the turn rate a/V that curves the velocity vector
    accordingly, then normalizing per axis into [-1, 1].
    """

    LOS_FILTER = 0.8   # first-order LOS-rate low-pass coefficient (APN family)

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        from environments.missile.env import OMEGA_MAX, OMEGA_LAT
        self.omega_scale = torch.tensor((OMEGA_MAX, OMEGA_LAT, OMEGA_LAT),
                                        device=d, dtype=torch.float64)
        self.lam_filt = torch.zeros((env.num_envs, 3), device=d, dtype=torch.float64)

    def reset_idx(self, mask):
        self.lam_filt.masked_fill_(mask[:, None], 0.)

    def _filter_los(self, lam, vc):
        """First-order LOS-rate low-pass, frozen below Vc < 50 m/s."""
        alpha = torch.where(vc < 50., torch.zeros_like(vc),
                            torch.full_like(vc, self.LOS_FILTER))
        self.lam_filt += alpha[:, None] * (lam - self.lam_filt)
        return self.lam_filt

    def _rate_command(self, a_cmd_w, R, V):
        """World specific-acceleration command -> normalized rate action."""
        a_lat_b = torch.einsum('nji,nj->ni', R, a_cmd_w)
        a_lat_b = a_lat_b - a_lat_b[:, 0:1] * torch.tensor(
            (1., 0., 0.), device=self.device, dtype=torch.float64)
        b_x = torch.tensor((1., 0., 0.), device=self.device,
                           dtype=torch.float64).expand_as(a_lat_b)
        w_cmd = _cross_cols(b_x, a_lat_b) / V[:, None]
        return (w_cmd / self.omega_scale).clamp(-1., 1.).to(torch.float32)


class MissilePPNBaseline(_MissileRateBaseline):
    """Pure-pursuit guidance on the env's body-rate command interface.

    Method
        Pure pursuit commands a lateral acceleration along the INSTANTANEOUS
        LOS direction r_hat -- the classic no-lead "chase the bearing" law
        that aligns the thrust axis with the target direction:
            a_pp = K_PURSUIT * V_m * r_hat_w ,
        so the implied turn rate a/V collapses to K_PURSUIT * sin(heading
        error).  A point-mass gravity compensation toward -g is added so the
        specific acceleration the airframe must realize is a_pp - g; the
        axial component is then removed by the b_x projection in the shared
        skid-to-turn inverse, exactly as in the APN default.

        The LOS-rate low-pass filter is maintained with the identical form,
        terminal freeze (alpha = 0 for Vc < 50 m/s) and reset_idx clearing as
        the APN/TPN family for interface parity; the pure-pursuit law itself
        is LOS-direction-only and does not consume the filtered rate.

    Known limitation (why this is an A/B arm, not the default): pure pursuit
    carries no lead, so it lags the 7.5 g weaving target and formally needs
    unbounded lateral acceleration for a perfect terminal intercept against
    any crossing geometry; expect a lower success rate than APN with larger
    miss distances.
    """

    name = 'ppn'
    K_PURSUIT = 1.2   # commanded turn-rate scale [1/s]: a_pp = K * V * r_hat
    GRAV_COMP = 1.0   # point-mass gravity compensation fraction of -g

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        q, vb = o[:, 3:7], o[:, 7:10]
        r_b, lam, vc = o[:, 23:26], o[:, 29:32], o[:, 32].clamp_min(0.)
        R = _quat_to_matrix(q)
        self._filter_los(lam, vc)            # parity state; unused by the PP law
        rng = r_b.norm(dim=-1).clamp_min(1e-3)
        r_hat_w = torch.einsum('nij,nj->ni', R, r_b / rng[:, None])
        V = vb.norm(dim=-1).clamp_min(50.)
        a_pp = self.K_PURSUIT * V[:, None] * r_hat_w
        g = torch.tensor((0., 0., -9.81), device=self.device,
                         dtype=torch.float64).expand_as(r_hat_w)
        a_cmd = a_pp - self.GRAV_COMP * g    # point-mass compensation toward -g
        return self._rate_command(a_cmd, R, V)


class MissileTPNBaseline(_MissileRateBaseline):
    """True proportional navigation (TPN) on the env's body-rate interface.

    Method
        a_cmd = N' * Vc * (lam_dot x lam_hat) ,
        the closing-speed-scaled PN law, WITHOUT the APN default's
        ``(N/2) g_perp`` gravity-augmentation term -- that augmentation is
        precisely the APN/TPN distinction under test here.  A small
        STANDALONE gravity-bias feedforward ``GRAV_BIAS * g_perp`` (once per
        g, versus the APN's N/2 = 7.5 x g_perp) is retained so the missile
        does not sag below the LOS over the 4 km engagement; dropping the
        bias entirely was rejected because a pure TPN command has no
        gravity-aware term at all and loses to gravity well before the
        endgame.  lam_dot is the filtered body LOS rate rotated to world
        axes, lam_hat the unit LOS; the result passes through the shared
        skid-to-turn inverse with per-axis rate normalization.

    Known limitation: without the (N/2) g augmentation the guidance loop must
    also fight the gravity-induced LOS rotation, so persistent miss against
    the maneuvering target is expected to exceed the APN default's.
    """

    name = 'tpn'
    N_PRIME = 15.0    # navigation constant (same family as the APN default)
    GRAV_BIAS = 1.0   # standalone gravity-bias feedforward, fraction of g_perp

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        q, vb = o[:, 3:7], o[:, 7:10]
        r_b, lam, vc = o[:, 23:26], o[:, 29:32], o[:, 32].clamp_min(0.)
        R = _quat_to_matrix(q)
        lam_w = torch.einsum('nij,nj->ni', R, self._filter_los(lam, vc))
        rng = r_b.norm(dim=-1).clamp_min(1e-3)
        r_hat_w = torch.einsum('nij,nj->ni', R, r_b / rng[:, None])
        g = torch.tensor((0., 0., -9.81), device=self.device,
                         dtype=torch.float64).expand_as(r_hat_w)
        g_perp = g - (g * r_hat_w).sum(-1, keepdim=True) * r_hat_w
        a_cmd = self.N_PRIME * vc[:, None] * _cross_cols(lam_w, r_hat_w) \
            + self.GRAV_BIAS * g_perp
        V = vb.norm(dim=-1).clamp_min(50.)
        return self._rate_command(a_cmd, R, V)


# --------------------------------------------------------------------------- #
# quadx_track: estimator-free pursuit and the EMA-filter family
# --------------------------------------------------------------------------- #
class QuadXTrackPursuitBaseline(QuadXSE3Baseline):
    """Estimator-free pursuit of the raw relative measurement (no filter).

    Method
        The SE(3) tracking stack of the KF baseline is kept, but the target
        estimate IS the measurement: the reference point is the vehicle's own
        position plus the raw noisy relative measurement, remapped onto the
        task's fixed reference geometry,
            ref = [ (p_xy + rel_xy) + REF_OFFSET_XY , ALT_REF ] .
        The outer cascade produces the speed-capped velocity setpoint
        v_d = -(KX/KV) e_x (capped at V_MAX); there is NO target-velocity
        feedforward (the pursuit law has no velocity estimate), so v_ff = 0.
        Measurement noise therefore passes straight into the reference --
        that is the point of this arm: it quantifies what the KF/EMA filters
        buy over raw pursuit.

    Timeout hold (simple form, mirrors the KF baseline)
        The reference freezes where the target was last seen.  With no motion
        model the frozen reference and the last-seen measurement coincide, so
        detections simply refresh the held point; the timeout only formalizes
        the freeze, exactly as in the KF baseline's fixed hold.

    All constants are hard-coded class attributes (no constructor knobs).
    """

    name = 'pursuit'
    HOLD_TIMEOUT = 3.0       # s without detection before the fixed hold
    V_MAX = 3.0              # m/s, tracking cruise cap
    KX, KV = 4.0, 2.0        # position PD (same cascade gains as the KF arm)

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        self.offset = torch.tensor(env.REF_OFFSET_XY, device=d, dtype=torch.float64)
        self.home = torch.tensor((0., 0., env.ALT_REF), device=d, dtype=torch.float64)
        self.alt = torch.tensor(env.ALT_REF, device=d, dtype=torch.float64)
        n = env.num_envs
        self.seen = torch.zeros(n, device=d, dtype=torch.bool)
        self.since_seen = torch.zeros(n, device=d, dtype=torch.float64)
        self.ref_hold = self.home.expand(n, 3).clone()

    def reset_idx(self, mask):
        super().reset_idx(mask)
        self.seen.masked_fill_(mask, False)
        self.since_seen.masked_fill_(mask, 0.)
        self.ref_hold.copy_(torch.where(mask[:, None],
                                        self.home.expand_as(self.ref_hold),
                                        self.ref_hold))

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p, rel, vis = o[:, 10:13], o[:, 21:24], o[:, 24] > .5
        # raw reference: own position + relative measurement (world axes)
        meas_ref = torch.cat(((p[:, :2] + rel[:, :2]) + self.offset,
                              self.alt.expand(p.shape[0], 1)), -1)
        self.since_seen = torch.where(vis, torch.zeros_like(self.since_seen),
                                      self.since_seen + self.dt)
        timed_out = self.seen & (self.since_seen > self.HOLD_TIMEOUT)
        # detections refresh the held reference; once timed out it stays put
        self.ref_hold = torch.where((vis & ~timed_out)[:, None], meas_ref,
                                    self.ref_hold)
        self.seen |= vis
        ref = torch.where(self.seen[:, None], self.ref_hold,
                          self.home.expand(p.shape[0], 3))
        # no target-velocity feedforward: v_ff = 0 (no estimator)
        return self._track(ref, obs)


class QuadXTrackPIDEMABaseline(QuadXSE3Baseline):
    """First-order EMA target filter + SE(3) cascade (simpler-filter family).

    Method
        The constant-velocity Kalman filter of the default arm is replaced by
        the simplest possible recursive estimator: a first-order exponential
        moving average on the target position measurement y = p + rel (world
        axes), updated only on detection steps and held (no prediction) while
        the target is blind:
            est <- est + POS_ALPHA * (y - est) .
        The velocity estimate is the finite difference of the FILTERED
        position, itself EMA-smoothed to keep the differentiation noise
        bounded:
            v_est <- v_est + VEL_ALPHA * ((est+ - est) / dt - v_est) .
        The reference and feedforward follow the KF baseline exactly:
            ref  = [ est_xy + REF_OFFSET_XY , ALT_REF ],
            v_ff = [ v_est_xy , 0 ],
        driving the same SE(3) position/attitude cascade with the same
        V_MAX cruise cap.

    Fixed timeout hold
        Identical to the KF baseline: after HOLD_TIMEOUT seconds without a
        detection the reference freezes at the estimate held at that instant
        (and v_ff drops to zero) until a new detection arrives.

    This is the simpler-filter family alternative to ``cv_kf_pid``: one
    scalar alpha per channel versus the KF's process/measurement model.
    """

    name = 'pid_ema'
    POS_ALPHA = 0.15         # EMA coefficient on the target position estimate
    VEL_ALPHA = 0.10         # EMA coefficient on the velocity finite difference
    HOLD_TIMEOUT = 3.0       # s without detection before the fixed hold
    V_MAX = 3.0              # m/s, tracking cruise cap
    KX, KV = 4.0, 2.0        # position PD (same cascade gains as the KF arm)

    def __init__(self, env):
        super().__init__(env)
        d = self.device
        self.offset = torch.tensor(env.REF_OFFSET_XY, device=d, dtype=torch.float64)
        self.home = torch.tensor((0., 0., env.ALT_REF), device=d, dtype=torch.float64)
        self.alt = torch.tensor(env.ALT_REF, device=d, dtype=torch.float64)
        n = env.num_envs
        self.est = torch.zeros((n, 3), device=d, dtype=torch.float64)
        self.v_est = torch.zeros((n, 3), device=d, dtype=torch.float64)
        self.initialized = torch.zeros(n, device=d, dtype=torch.bool)
        self.since_seen = torch.zeros(n, device=d, dtype=torch.float64)
        self.frozen = torch.zeros(n, device=d, dtype=torch.bool)
        self.hold = self.home.expand(n, 3).clone()

    def reset_idx(self, mask):
        super().reset_idx(mask)
        self.est.masked_fill_(mask[:, None], 0.)
        self.v_est.masked_fill_(mask[:, None], 0.)
        self.initialized.masked_fill_(mask, False)
        self.since_seen.masked_fill_(mask, 0.)
        self.frozen.masked_fill_(mask, False)
        self.hold.copy_(torch.where(mask[:, None],
                                    self.home.expand_as(self.hold), self.hold))

    def actions(self, obs):
        o = _vec(obs).to(torch.float64)
        p, rel, vis = o[:, 10:13], o[:, 21:24], o[:, 24] > .5
        dt = self.dt
        y = p + rel                                    # world-axes measurement

        # ---- EMA position filter (detections only, no blind prediction) ---
        fresh = vis & ~self.initialized
        self.est = torch.where(fresh[:, None], y, self.est)
        upd = vis & self.initialized & ~fresh
        new_est = self.est + self.POS_ALPHA * (y - self.est)
        v_fd = (new_est - self.est) / dt
        new_v = self.v_est + self.VEL_ALPHA * (v_fd - self.v_est)
        self.est = torch.where(upd[:, None], new_est, self.est)
        self.v_est = torch.where(upd[:, None], new_v, self.v_est)
        self.initialized |= vis

        # ---- fixed timeout hold (KF baseline semantics) --------------------
        self.since_seen = torch.where(vis, torch.zeros_like(self.since_seen),
                                      self.since_seen + dt)
        timed_out = self.initialized & (self.since_seen > self.HOLD_TIMEOUT)
        freeze_now = timed_out & ~self.frozen
        ref_est = torch.cat((self.est[:, :2] + self.offset,
                             self.alt.expand(self.est.shape[0], 1)), -1)
        self.hold = torch.where(freeze_now[:, None], ref_est, self.hold)
        self.frozen = timed_out

        ref = torch.where(self.frozen[:, None], self.hold, ref_est)
        ref = torch.where(self.initialized[:, None], ref,
                          self.home.expand(p.shape[0], 3))
        v_ff = torch.cat((self.v_est[:, :2],
                          torch.zeros((self.est.shape[0], 1), device=o.device,
                                      dtype=torch.float64)), -1)
        v_ff = torch.where((self.initialized & ~self.frozen)[:, None], v_ff,
                           torch.zeros_like(v_ff))
        return self._track(ref, obs, v_ff)
