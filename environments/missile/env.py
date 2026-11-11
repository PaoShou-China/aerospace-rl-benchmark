"""Missile-intercept task: 6-DOF tail-controlled missile vs kinematic target.

Dynamics: the project-authored missile URDF (environments/common/
original_pyflyt/models/vehicles/missile/, SHA-256 pinned by
consistency.verify_missile_assets) on the shared Genesis TensorPlant --
6-DOF rigid body (150 kg, nose along body +x), four cross tail fins through
the LiftingSurfaceBatch aero model, body drag, and an autonomous boost-coast
engine (20 kN along body +x for the first 4 s of each episode, then unpowered
coast). Control: 60 Hz policy / 120 Hz rate-PID / 240 Hz physics; the policy
commands BODY RATES (roll +-6 rad/s, pitch/yaw +-0.5 rad/s -- coordinated-turn
limit a/V) which the plant's rate PID converts to normalized (roll, pitch, yaw)
fin commands expanded by the skid-to-turn mixer onto the four fins.

Engagement: the target is a kinematic 3-DOF aircraft advanced analytically in
torch buffers every physics substep (straight line / sinusoidal weave /
horizontal barrel roll) at 250 m/s around 4 km altitude; a collision-free
visual entity in the Genesis scene mirrors it for rendering only. The
missile spawns 2.5 km away with fire-control
lead: it aims at the straight-line extrapolation of the target over the
nominal time-to-go t_go = TGO_SCALE * R / (V_M0 + V_T), then launches at
250 m/s along that bearing; the target heading points back at the missile
within a fixed cone.

Success: miss distance < KILL_RADIUS (15 m proximity fuse) at ANY physics
substep -> immediate termination with the one-shot kill bonus. Fixed forever.
Failure: the closest-approach point passes
(range rate turns positive after 2 s, i.e. the shot is unrecoverable),
leaving the bounds |p| > 15 km or z < 0, or the 15 s time limit.

Engagement geometry (single fixed medium operating point, 2026-10-24 user
decision):
  * initial range        2500 m (midpoint of the old 1500-3500 m axis)
  * target heading cone  +-37.5 deg around the missile bearing
  * target lateral accel  7.5 g (weave amplitude/severity, barrel radius)
  * weave frequency       0.75 Hz

Reward (per inner step, only while active):
  +0.01 * (d_prev - d)          telescoping closing-progress potential
  +0.01 * (v_hat . r_hat)       velocity/LOS alignment (dense PN-consistent
                                guidance shaping; the from-scratch signal)
  +100 on kill (once)
  -10 * min(d_min / 500 m, 1) on unrecoverable miss (once)
The guidance term is selectable via the `reward_variant` ctor kwarg (default
'graded_terminal'); see K_ZEM / ZEM_GATE for the measured alternatives.
Components are accumulated and reported as Episode/rew_* scalars plus
completed_metrics keys, so reward attribution is observable.

Observation (34 dims, float32, unnormalized -- the policy-side
EmpiricalNormalization owns scaling):
  [0:3] body omega [rad/s]   [3:7] quat xyzw   [7:10] body v [m/s]
  [10:13] world p [m]        [13:16] prev action (rate cmds, rad/s)
  [16:20] fin actuation (4)  [20:21] ignited flag  [21:22] engine burning
  [22:23] burn-time left fraction
  [23:26] target rel position (body) [m]  [26:29] rel velocity (body) [m/s]
  [29:32] LOS rate (body) [rad/s]  [32:33] closing speed Vc [m/s]
  [33:34] target speed [m/s]
"""

import math

import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import COMPUTE_DTYPE, inverse_rotate
from environments.common.atmosphere import isa
from algorithms.baselines.baselines import MissileAPNBaseline
from environments.common.residual import ResidualActionMixin
from environments.common.base_env import BaseEnv
from environments.common.domain_randomization import (
    apply_static_perturbation,
    apply_staged_domain_randomization,
    stage_domain_randomization,
    resolve_domain_randomize,
)


_X_HAT = {}  # (device, dtype) -> body +x unit vector, built once


def _aim_quat(aim, roll):
    """Batched xyzw quaternion taking body +x to `aim` plus a small extra roll
    about body x.  Local on purpose: the shared dynamics.quat_from_euler maps
    three Euler angles (the rocket/fixedwing spawn convention), not an aim
    vector, so it cannot express this rotation."""
    key = (aim.device, aim.dtype)
    if key not in _X_HAT:
        # A constant, so the per-call host-to-device copy is made once.
        _X_HAT[key] = torch.tensor([1.0, 0.0, 0.0], device=aim.device, dtype=aim.dtype)
    x_hat = _X_HAT[key].expand_as(aim)
    axis = torch.linalg.cross(x_hat, aim)
    ang = torch.atan2(axis.norm(dim=-1), aim[..., 0].clamp(-1, 1))
    axis = axis / axis.norm(dim=-1).clamp_min(1e-9)[:, None]
    half = 0.5 * ang
    qa = torch.cat((half.cos()[:, None], axis * half.sin()[:, None]), -1)  # wxyz
    hr = 0.5 * roll
    qr = torch.stack((hr.cos(), hr.sin(), torch.zeros_like(hr), torch.zeros_like(hr)), -1)
    w1, x1, y1, z1 = qa.unbind(-1)
    w2, x2, y2, z2 = qr.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        -1,
    )


def _zem(r, v_rel):
    """Zero-effort miss: the closest-approach range if thrust stopped now.

    Magnitude of the component of the relative position perpendicular to the
    relative velocity. Its terminal value is the actual miss distance, so it
    shares the kill objective's units and direction.
    """
    v2 = v_rel.square().sum(-1).clamp_min(1.0)
    along = (r * v_rel).sum(-1, keepdim=True) / v2[:, None]
    return (r - along * v_rel).norm(dim=-1)


OMEGA_MAX = 6.0  # roll-rate command limit [rad/s]
OMEGA_LAT = 1.0  # pitch/yaw rate command limit [rad/s]. Raised 0.5 -> 1.0 on
# 2026-10-24 for vanilla-PPO learnability: the steady
# coordinated-turn rate is a/V (0.2-0.5 rad/s), but terminal
# guidance needs TRANSIENT authority above it -- the rate PID
# simply saturates the fins, so a 1.0 rad/s command cap lets
# the policy pull harder during the endgame correction.
KILL_RADIUS = 15.0  # proximity-fuse kill radius [m]; FIXED.
# Raised 8 -> 15 on 2026-10-24: at 8 m the BC warm start's median
# miss (9.8 m at d=0) sat ON the fuse boundary, so any ~1e-3
# weight drift from a PPO update wiped out all kills (measured
# blocker); 15 m puts the warm start comfortably inside the gate.
R_SPAWN = 2500.0  # initial range [m] (fixed medium operating point)
CONE = 37.5  # target-heading cone half-width [deg]
A_LAT = 7.5  # target lateral accel [g]
TARGET_MAX_LOAD_G = 15.0  # structural/load limit on the kinematic target
F_WEAVE = 0.75  # weave angular rate [Hz]
SEEKER_HALF_ANGLE = math.radians(60.0)  # body-fixed seeker field of view
SEEKER_MAX_RANGE = 20000.0  # m; beyond this the seeker loses lock
# Radome aberration slope: the apparent line of sight is refracted by the
# radome, rotating with the missile angle of attack.  A typical radome slope
# is 0.01-0.05 rad of boresight error per rad of AoA; the resulting LOS-rate
# error is the classical radome-slope parasitic loop.  0.02 keeps the closed
# loop of the APN baseline stable (43.8 % -> 40.6 % measured on seed 7).
RADOME_SLOPE = 0.02
# Seeker tracking-filter time constant [s]: the measured relative-state block
# lags the true geometry by a first-order filter.  20 ms is a fast tracking
# loop; 80 ms destabilised the 60 Hz APN loop (43.8 % -> 14 %).
SEEKER_FILTER_TAU = 0.02


def radome_biased_los(r_body, los_rate_body, velocity_body, slope=RADOME_SLOPE):
    """Apparent relative position and LOS rate after radome refraction.

    The radome bends the incoming wavefront by an angle proportional to the
    missile angle of attack (pitch) and sideslip (yaw), which rotates the
    measured line of sight and its rate in the body frame.  The small-angle
    rotation is about body ``y`` (pitch, angle ``slope * alpha``) followed by
    body ``z`` (yaw, angle ``-slope * beta``).

    Args:
        r_body: ``(n, 3)`` true target relative position in body frame.
        los_rate_body: ``(n, 3)`` true LOS rate in body frame.
        velocity_body: ``(n, 3)`` missile body-frame velocity.
        slope: radome slope [rad per rad].

    Returns:
        ``(r_meas, los_rate_meas)`` in the body frame.
    """
    aoa = torch.atan2(-velocity_body[:, 2], velocity_body[:, 0].clamp_min(1e-3))
    beta = torch.atan2(velocity_body[:, 1], velocity_body[:, 0].clamp_min(1e-3))
    # Positive AoA points the nose above the velocity vector, so the radome
    # bias rotates the apparent LOS toward the nose (up).
    da = -slope * aoa
    db = -slope * beta

    def rotate(u):
        ca, sa = torch.cos(da), torch.sin(da)
        u = torch.stack(
            (u[:, 0] * ca + u[:, 2] * sa, u[:, 1], -u[:, 0] * sa + u[:, 2] * ca), -1
        )
        cb, sb = torch.cos(db), torch.sin(db)
        return torch.stack(
            (u[:, 0] * cb - u[:, 1] * sb, u[:, 0] * sb + u[:, 1] * cb, u[:, 2]), -1
        )

    return rotate(r_body), rotate(los_rate_body)


def first_order_lag(state, measured, dt, tau):
    """Backward-Euler step of ``dx/dt = (measured - x)/tau`` (stable for any dt)."""
    return state + (dt / (tau + dt)) * (measured - state)


V_T = 250.0  # target speed [m/s]
V_M0 = 250.0  # missile launch speed [m/s]
Z_MID = 4000.0  # engagement altitude [m]
K_PROG = 0.01  # closing-progress gain per metre
K_ALIGN = 0.01  # velocity/LOS alignment gain (dense PN-shaped guidance signal).
# Halved 0.02 -> 0.01 on 2026-10-28 (run 1 diagnosis): the
# lead-aim ballistic policy already maxes this term (~8.7/ep,
# constant across all training phases), so at 0.02 it inflated
# the do-nothing return and diluted the kill differential.
K_ZEM = 10.0  # zero-effort-miss shaping gain (2026-10-29, 'zem' variant).
# NOT the default: measured against APN vs the zero policy the
# instantaneous ZEM barely moves along a ballistic trajectory
# (it is nearly conserved) and is sensitive to the missile's
# own velocity direction, so the term gave a 0.41/ep
# discriminative gap at 2.40/ep std -- 3x worse SNR than the
# alignment term it was meant to replace. Kept for reference.
ZEM_REF = 500.0  # ZEM normalizer [m]; full-scale miss reduction per episode.
ZEM_GATE = 800.0  # guide-term range gate [m] for 'zem' and 'align_gated'.
# Outside it the guidance signal is dominated by geometry the
# policy cannot yet act on: measured, ZEM grows from ~45 m at
# spawn to ~580 m of real miss on a weave target. 800 m
# ~= 1.6 s time-to-go, where terminal corrections pay off.
# 'align_gated' keeps the alignment gap (0.205 vs 0.184/ep)
# while cutting the policy-insensitive plateau 4.36 -> 1.18.
KILL_BONUS = 100.0  # kill bonus. Doubled 50 -> 100 on 2026-10-28 (run 1): the
# discounted terminal bonus (~0.995^150-300) was worth only
# ~11-23 vs the ~24/ep shaping plateau, too weak to select
# terminal correction; kills decayed 11% -> 1%.
MISS_PENALTY = 10.0
MISS_REF = 500.0  # miss distance paying the full penalty [m]. 2026-10-29: the
# binary miss penalty gave near-misses and far-misses identical
# terminal advantage. Linear grading supplies a smooth signal
# across the fuse boundary without changing success criteria.
TGO_SCALE = 0.8  # fire-control t_go factor for the launch-aim lead
# (1.0 = nominal closed-form; 0.8 measured best on the fixed
# medium point: shorter lead trims overshoot misses, +0.8pp
# SR on two disjoint 1024-ep seed populations)

# Kinematic-target energy model (realistic preset).  The legacy target is a
# pure 3-DOF point that holds V_T forever; a real aircraft bleeds speed in a
# sustained high-g manoeuvre and cannot exceed its thrust limit while
# recovering.  These values describe a 9 t fighter-class aircraft at the
# 4 km engagement altitude.
TARGET_MASS = 9000.0  # kg
TARGET_WING_AREA = 28.0  # m^2
TARGET_CD0 = 0.02
TARGET_ASPECT = 3.5
TARGET_OSWALD = 0.85
TARGET_MAX_THRUST = 90.0e3  # N, sea-level static thrust
TARGET_THRUST_EXPONENT = 0.7  # installed thrust lapse T ~ rho^0.7
TARGET_SPEED_MIN = 90.0  # m/s; below this the manoeuvre is not sustainable
TARGET_SPEED_MAX = 400.0  # m/s
TARGET_SPEED_TAU = 2.0  # s, pilot speed-hold time constant
G0 = 9.81


class MissileInterceptEnv(BaseEnv, ResidualActionMixin):
    """6-DOF tail-controlled missile intercept task against a kinematic target.

    Task objective: bring a tail-controlled missile within KILL_RADIUS (15 m)
    proximity of a kinematic 3-DOF aircraft (straight/weave/barrel maneuver,
    250 m/s around 4 km altitude). The agent observes an unnormalized 34-dim
    float32 tensor: body angular velocity (3), quaternion wxyz (4), body-
    frame velocity (3), world position (3), previous rate commands (3), fin
    actuation (4), ignition/burning flags (2), burn-time-left fraction (1),
    target relative position/velocity in body frame (6), LOS rate (3), closing
    speed (1), and target speed (1). Actions are 3-D body-rate commands in
    [-OMEGA_MAX, OMEGA_MAX] rad/s (roll +-6, pitch/yaw +-1), mapped affinely
    from raw [-1, 1]. Termination on kill (dist < 15 m at any substep, +100
    bonus), unrecoverable miss (range rate positive after 2 s), |p| > 15 km,
    or z < 0; truncation at 900 policy steps (15 s, 60 Hz agent rate).
    Boost-coast engine fires for the first 4 s. Domain randomization covers
    thrust (0.97, 1.03) and mass (0.97, 1.03).
    """
    conventional_controller = MissileAPNBaseline
    OMEGA_MAX = OMEGA_MAX
    PERTURBABLE_PARAMS = {
        "thrust": 1.0,
        "mass": 1.0,
        "aero": 1.0,
        "cl_alpha": 1.0,
        "cd0": 1.0,
        "stall": 1.0,
        "oswald": 1.0,
        "actuator": 1.0,
        "actuator_deadband": 1.0,
        "actuator_backlash": 1.0,
        "actuator_asymmetry": 1.0,
        "actuator_delay": 1.0,
        "sensor": 1.0,
        "cg": 1.0,
        "cg_y": 1.0,
        "cg_z": 1.0,
        "inertia": 1.0,
        "inertia_x": 1.0,
        "inertia_y": 1.0,
        "inertia_z": 1.0,
    }
    # Literature: interceptor-class solid rocket motors (e.g. THAAD booster
    # studies) have tightly characterized propulsion.  Rocket Monte Carlo
    # dispersion analyses use thrust +/-3% (3 sigma) and Isp +/-1%.  Missile
    # structural mass is well-known from manufacturing, so mass uncertainty is
    # small (+/-3%, dominated by propellant grain tolerance).
    DEFAULT_DOMAIN_RANDOMIZE = {"thrust": (0.97, 1.03), "mass": (0.97, 1.03)}

    def __init__(
        self,
        num_envs=64,
        backend="cpu",
        seed=7,
        steady_wind=None,
        dryden=0.0,
        severity="moderate",
        reward_variant="graded_terminal",
        residual=False,
        param_mismatch=None,
        domain_randomize=None,
        warmup_iterations=None,
        aero="legacy",
    ):
        """Initialize the MissileIntercept environment.

        Construct the Genesis-backed 6-DOF missile plant at 60 Hz agent /
        120 Hz control, allocate task buffers (engagement state, kinematic
        target state, reward components, domain-randomization counters),
        and finalize with a reset().
        """
        if num_envs < 1:
            raise ValueError("n must be positive")
        if reward_variant not in ("graded_terminal", "align_gated", "terminal_gauss", "zem"):
            raise ValueError(f"unknown reward_variant {reward_variant!r}")
        self.num_envs = num_envs
        self.reward_variant = reward_variant
        self.task_cfg = {
            "vehicle": "missile",
            "num_envs": num_envs,
            "backend": backend,
            "implementation": "fully_tensorized_dynamics",
            "task_defaults": "project-authored 6-DOF intercept",
            "reward_variant": reward_variant,
        }
        self.clip_actions = 1.0
        self.plant = TensorPlant(
            "missile",
            num_envs,
            backend,
            seed,
            vel_clamp=(1200.0, 40.0),
            warmup_iterations=warmup_iterations,
            aero=aero,
        )
        self.device = self.plant.device
        self._sensor = None
        if self.plant.realistic:
            from environments.common.sensors import SensorModel

            self._sensor = SensorModel(
                num_envs,
                self.device,
                seed,
                gyro_sigma=0.002,
                attitude_sigma=0.002,
                velocity_sigma=0.2,
                position_sigma=5.0,
                bias_step=5e-5,
                seeker_position_sigma=2.0,
                seeker_velocity_sigma=1.0,
                seeker_los_sigma=0.002,
                dt=1.0 / 60.0,
            )
        self.agent_hz = 60
        self.inner_steps = int(self.plant.control_hz // self.agent_hz)
        self.max_steps = 900  # 15 s
        self.max_episode_length = self.max_steps + 2
        self.num_actions = 3
        # Per-axis rate-command limits: roll spins fast, pitch/yaw are limited
        # by the coordinated-turn relation a/V (see OMEGA_LAT above).
        omega_scale = torch.tensor(
            (OMEGA_MAX, OMEGA_LAT, OMEGA_LAT), device=self.device, dtype=COMPUTE_DTYPE
        )
        self.action_low = -omega_scale
        self.action_high = omega_scale

        d = self.device

        def zeros(*s):
            """Allocate a zeros tensor on the environment device."""
            return torch.zeros(*s, device=d, dtype=COMPUTE_DTYPE)

        self.action = zeros(num_envs, self.num_actions)
        self.reward = zeros(num_envs)
        self.count = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.sim_time = zeros(num_envs)
        self.terminated = torch.zeros(num_envs, device=d, dtype=torch.bool)
        self.truncated = self.terminated.clone()
        self.out_of_bounds = self.terminated.clone()
        self.complete = self.terminated.clone()
        self.miss_flag = self.terminated.clone()
        self._observation = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(num_envs, device=d, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=d, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.returns = zeros(num_envs)
        self.completed_returns = self.returns.clone()
        self.completed_episodes = torch.zeros_like(self.episode_length_buf)
        self.completed_successes = torch.zeros_like(self.episode_length_buf)
        # engagement state
        self.min_distance = zeros(num_envs)
        self.dist = zeros(num_envs)  # current missile-target range
        self.dist_prev = zeros(num_envs)  # range at the previous inner-step boundary
        self.zem_prev = zeros(num_envs)  # zero-effort miss at the previous boundary
        self.target_pos = zeros(num_envs, 3)
        self.target_vel = zeros(num_envs, 3)
        self.seeker_lock = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        # First-order seeker tracking filter state (realistic only): the
        # measured block (relative position/velocity, LOS rate, closing speed,
        # target speed) lags the true geometry.
        self._seeker_track = (
            torch.zeros((num_envs, 11), device=self.device, dtype=COMPUTE_DTYPE)
            if self.plant.realistic
            else None
        )
        # The tracking filter starts from the first measurement of an episode.
        self._seeker_ready = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        # Field-of-view / lock-loss modelling is part of the realistic preset;
        # the legacy benchmark keeps the always-valid seeker observation.
        self.seeker_fov = bool(self.plant.realistic)
        # kinematic target parameters (reset_idx-owned)
        self.t_p0 = zeros(num_envs, 3)  # target position at t=0
        self.t_head = zeros(num_envs, 3)  # unit heading (straight/weave)
        self.t_perp = zeros(num_envs, 3)  # horizontal left-normal of the heading
        self.t_omega = zeros(num_envs)  # weave angular rate [rad/s]
        self.t_phase = zeros(num_envs)  # maneuver phase [rad]
        self.t_a = zeros(num_envs)  # lateral accel [m/s^2]
        self.t_type = torch.zeros(
            num_envs, device=d, dtype=torch.long
        )  # 0 straight 1 weave 2 barrel
        self.t_center = zeros(num_envs, 3)  # barrel circle center
        self.t_omega_b = zeros(num_envs)  # barrel angular rate (signed) [rad/s]
        self.t_radius = zeros(num_envs)  # barrel radius [m]
        # Realistic target energy state: current speed [m/s].  Legacy keeps the
        # fixed-V_T closed-form path for bit-for-bit benchmark parity.
        self.energy_model = bool(self.plant.realistic)
        self.t_speed = torch.full((num_envs,), V_T, device=d, dtype=COMPUTE_DTYPE)
        keys = (
            ("progress", "zem", "kill", "miss")
            if reward_variant == "zem"
            else ("progress", "align", "kill", "miss")
        )
        self.reward_components = {k: zeros(num_envs) for k in keys}
        self.completed_metrics = {
            "miss_distance": zeros(num_envs),
            "kill_rate": zeros(num_envs),
            "miss_rate": zeros(num_envs),
            "timeout_rate": zeros(num_envs),
        }
        for k in keys:
            self.completed_metrics[f"rew_{k}"] = zeros(num_envs)
        self._seed = seed
        self.extras = dict()
        self.cfg = dict(self.task_cfg)
        self.cfg["aero"] = aero
        self.cfg["realism_version"] = self.plant.realism_version
        self._init_wind(num_envs, seed, steady_wind, dryden, severity)
        self.controller_design = {}
        apply_static_perturbation(self, param_mismatch)
        self._dr_counter = 0
        default_dr = self.DEFAULT_DOMAIN_RANDOMIZE
        if self.plant.realistic:
            default_dr = dict(default_dr)
            default_dr.update(
                {
                    "sensor": (0.8, 1.2),
                    "cg": (0.95, 1.05),
                    "cg_y": (0.95, 1.05),
                    "cg_z": (0.95, 1.05),
                    "inertia": (0.9, 1.1),
                    "inertia_x": (0.9, 1.1),
                    "inertia_y": (0.9, 1.1),
                    "inertia_z": (0.9, 1.1),
                    "cl_alpha": (0.92, 1.08),
                    "cd0": (0.85, 1.15),
                    "stall": (0.90, 1.10),
                    "oswald": (0.95, 1.05),
                    "actuator_deadband": (0.50, 2.0),
                    "actuator_backlash": (0.50, 2.0),
                    "actuator_asymmetry": (0.70, 1.30),
                    "actuator_delay": (0.0, 2.0),
                }
            )
        self._domain_randomize = resolve_domain_randomize(domain_randomize, default_dr)
        self.cfg["domain_randomize"] = (
            dict(self._domain_randomize) if self._domain_randomize else None
        )
        self.residual = residual
        self._init_residual()
        self.reset()

    def _init_wind(self, n, seed, steady_wind, dryden, severity):
        """Optionally wire in a steady/dryden wind disturbance; calm air by default."""
        self._wind = None
        if steady_wind is not None or dryden > 0.0:
            from environments.common.disturbances import WindDisturbance

            self._wind = WindDisturbance(
                n,
                self.device,
                seed,
                steady_wind=steady_wind,
                dryden=dryden,
                severity=severity,
                dt=1.0 / self.agent_hz,
                airspeed=500.0 if self.plant.realistic else 15.0,
                reference_altitude=4000.0 if self.plant.realistic else None,
                model="dryden" if self.plant.realistic else "first_order",
                shear_exponent=0.14 if self.plant.realistic else 0.0,
                time_varying=self.plant.realistic,
            )
            self.plant.wind = torch.zeros((n, 3), device=self.device, dtype=COMPUTE_DTYPE)
            self.cfg["wind"] = dict(
                steady_wind=list(steady_wind) if steady_wind is not None else None,
                dryden=dryden,
                severity=severity,
            )

    def _scale_actions(self, actions):
        """Affinely map [-1, 1] policy actions onto the physical rate-command bounds."""
        raw = actions.to(COMPUTE_DTYPE).clamp(-self.clip_actions, self.clip_actions)
        return self.action_low + (raw + 1) * 0.5 * (self.action_high - self.action_low)

    # ------------------------------------------------------------------ #
    # kinematic target: closed-form state at the current per-replica time
    # ------------------------------------------------------------------ #
    def _target_state(self):
        """Return the kinematic target's current (p, v) for its maneuver type."""
        t = self.sim_time  # (n,)
        m0 = self.t_type == 0
        m1 = self.t_type == 1
        p_straight = self.t_p0 + V_T * self.t_head * t[:, None]
        v_straight = (V_T * self.t_head).expand_as(p_straight)
        ph = self.t_omega * t + self.t_phase  # (n,)
        s = (self.t_a / (self.t_omega * self.t_omega)) * torch.sin(ph)
        vp = (self.t_a / self.t_omega) * torch.cos(ph)
        p_weave = p_straight + self.t_perp * s[:, None]
        v_weave = v_straight + self.t_perp * vp[:, None]
        th = self.t_omega_b * t + self.t_phase  # (n,)
        p_barrel = self.t_center + self.t_radius[:, None] * torch.stack(
            (torch.cos(th), torch.sin(th), torch.zeros_like(th)), -1
        )
        v_barrel = V_T * torch.stack((-torch.sin(th), torch.cos(th), torch.zeros_like(th)), -1)
        p = torch.where(m0[:, None], p_straight, torch.where(m1[:, None], p_weave, p_barrel))
        v = torch.where(m0[:, None], v_straight, torch.where(m1[:, None], v_weave, v_barrel))
        return p, v

    def _target_direction(self):
        """Unit velocity direction of the current manoeuvre (energy model)."""
        t = self.sim_time
        ph = self.t_omega * t + self.t_phase
        weave = V_T * self.t_head + self.t_perp * (
            (self.t_a / self.t_omega) * torch.cos(ph)
        )[:, None]
        weave = weave / weave.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        th = self.t_omega_b * t + self.t_phase
        barrel = torch.stack(
            (-torch.sin(th), torch.cos(th), torch.zeros_like(th)), -1
        )
        m0 = self.t_type == 0
        m1 = self.t_type == 1
        return torch.where(m0[:, None], self.t_head, torch.where(m1[:, None], weave, barrel))

    def _target_load_factor(self, speed):
        """Manoeuvre load factor n = |lift| / (m g)."""
        t = self.sim_time
        a_lat = (self.t_a * torch.cos(self.t_omega * t + self.t_phase)).abs()
        n_weave = (1.0 + (a_lat / G0).square()).sqrt()
        n_barrel = (
            1.0 + (speed.square() / (self.t_radius * G0).clamp_min(1.0)).square()
        ).sqrt()
        return torch.where(
            self.t_type == 0,
            torch.ones_like(speed),
            torch.where(self.t_type == 1, n_weave, n_barrel),
        )

    def _target_speed_step(self, speed, active):
        """Advance the target's speed with a point-mass thrust/drag balance.

        Drag is the parabolic polar ``q S CD0 + (n m g)^2 / (q S pi AR e)``,
        the installed thrust lapses with air density, and the pilot commands
        the drag plus a proportional speed-hold term clipped by the available
        thrust.  A sustained high-g manoeuvre therefore bleeds speed and the
        target cannot recover faster than its thrust margin allows.
        """
        rho = isa(self.target_pos[:, 2])[0]
        q_s = (0.5 * rho * speed.square() * TARGET_WING_AREA).clamp_min(1.0)
        load = self._target_load_factor(speed)
        induced = (load * TARGET_MASS * G0).square() / (
            q_s * math.pi * TARGET_ASPECT * TARGET_OSWALD
        )
        drag = q_s * TARGET_CD0 + induced
        available = TARGET_MAX_THRUST * (
            (rho / 1.225).clamp_min(0.05) ** TARGET_THRUST_EXPONENT
        )
        command = drag + TARGET_MASS * (V_T - speed) / TARGET_SPEED_TAU
        thrust = command.clamp_min(0.0).minimum(available)
        dv = (thrust - drag) / TARGET_MASS
        new_speed = (speed + dv * self.plant.physics_dt).clamp(
            TARGET_SPEED_MIN, TARGET_SPEED_MAX
        )
        return torch.where(active, new_speed, speed)

    def step(self, actions):
        """Advance one policy step through inner control steps.

        Scale actions, run the plant controller and physics substeps while
        advancing the kinematic target analytically, detect kills/misses/
        bounds violations, accumulate progress + guidance shaping reward,
        and return (observations, reward, terminated, truncated, extras).

        The tensor work is split at the host decisions (pause-restore and
        reset) into segments that update persistent buffers in place, so an
        execution backend may replay each segment as one CUDA graph.
        """
        if actions.device != torch.device(self.device):
            raise ValueError(
                "Actions must already be on env.device; implicit host/device copies are disabled"
            )
        actions = self._residual_input(actions)
        self._step_begin(actions)
        for i in range(self.inner_steps):
            # terminated/truncated do not change inside a control step, so the
            # activity mask is constant and one sync covers every substep.
            # Every replica is active when a step starts (each step ends by
            # resetting all done replicas), so the first control step needs none.
            restore = i > 0 and not bool((~(self.terminated | self.truncated)).all())
            self._control_step(restore)
        term_out, trunc_out, done, time_outs = self._step_end()
        # Fresh tensors per step, as before: callers may keep them.
        term_out, trunc_out = term_out.clone(), trunc_out.clone()
        self.reset_buf = done.clone()
        self.extras["time_outs"] = time_outs.clone()
        self.extras["terminal_observations"] = self._terminal_obs_td
        done_any = bool(self.reset_buf.any())
        if done_any:
            keys = ("success_rate", "miss_distance") + tuple(
                f"rew_{k}" for k in self.reward_components
            )
            self.extras["episode"] = dict(
                zip(keys, (x.clone() for x in self._episode_stats(self.reset_buf)))
            )
            self.reset_idx(self.reset_buf, nonempty=True)
        else:
            self.extras.pop("episode", None)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    def _step_begin(self, actions):
        """Map the action onto the plant and open the step's accumulators."""
        command = self._scale_actions(actions)
        self.action.copy_(command)
        if self._wind is not None:
            _p0, _, _v0, _ = self.plant.state()
            self.plant.wind.copy_(
                self._wind.step(altitude=_p0[:, 2], airspeed=_v0.norm(dim=-1))
            )
        self.reward.fill_(0.0)
        self.count += 1

    def _control_step(self, restore):
        """One control step: target motion, physics substeps, observation, reward."""
        active = ~(self.terminated | self.truncated)
        self.dist_prev.copy_(torch.where(active, self.dist, self.dist_prev))
        self.plant.clear_contacts()
        self.plant.controller(self.action, active)
        kill_ev = torch.zeros_like(active)
        miss_ev = torch.zeros_like(active)
        for _ in range(self.plant.physics_per_control):
            sub = ~(self.terminated | self.truncated)
            if self.energy_model:
                # Budget the speed state before moving the target, then
                # integrate its position along the current manoeuvre direction.
                tv = self._target_direction() * self.t_speed[:, None]
                tp = self.target_pos + tv * self.plant.physics_dt
                self.t_speed.copy_(self._target_speed_step(self.t_speed, sub))
            else:
                tp, tv = self._target_state()
            self.target_pos.copy_(torch.where(sub[:, None], tp, self.target_pos))
            self.target_vel.copy_(torch.where(sub[:, None], tv, self.target_vel))
            tq = _aim_quat(
                tv / tv.norm(dim=-1, keepdim=True).clamp_min(1e-6), torch.zeros_like(tv[:, 0])
            )
            self.plant.physics_step(
                self.action,
                sub,
                target_state=(tp, tq, tv, torch.zeros_like(tv)),
                restore=restore,
            )
            self.sim_time.copy_(
                torch.where(sub, self.sim_time + self.plant.physics_dt, self.sim_time)
            )
            pm, _, vm, _ = self.plant.state()  # cached post-substep state
            r = tp - pm
            dist = r.norm(dim=-1)
            self.dist.copy_(torch.where(sub, dist, self.dist))
            self.min_distance.copy_(
                torch.where(sub & (dist < self.min_distance), dist, self.min_distance)
            )
            closing = (r * (tv - vm)).sum(-1)
            kill_ev |= (dist < KILL_RADIUS) & sub
            # closest approach passed while the shot is still on: unrecoverable
            miss_ev |= (closing > 0) & (self.sim_time > 2.0) & sub & ~kill_ev
        # kill preempts miss inside the same inner step
        miss_ev &= ~kill_ev
        p, q, v, w = self.plant.state()
        wb = inverse_rotate(q, w)
        vb = inverse_rotate(q, v)
        r_w = self.target_pos - p
        v_rel = self.target_vel - v
        r2 = r_w.square().sum(-1).clamp_min(1e-6)
        lam_w = torch.linalg.cross(r_w, v_rel) / r2[:, None]
        lam_b = inverse_rotate(q, lam_w)
        vc = -(r_w * v_rel).sum(-1) / r2.clamp_min(1e-6).sqrt()
        r_b = inverse_rotate(q, r_w)
        v_rel_b = inverse_rotate(q, v_rel)
        if self._seeker_track is not None:
            # Radome refraction biases the apparent LOS with angle of attack.
            r_b, lam_b = radome_biased_los(r_b, lam_b, vb, RADOME_SLOPE)
        rng = r_w.norm(dim=-1).clamp_min(1e-6)
        boresight = torch.acos((r_b[:, 0] / rng).clamp(-1.0, 1.0))
        lock = (rng < SEEKER_MAX_RANGE) & (boresight < SEEKER_HALF_ANGLE)
        if not self.seeker_fov:
            lock = torch.ones_like(lock)
        self.seeker_lock.copy_(torch.where(active, lock, self.seeker_lock))
        rel = torch.cat(
            (
                r_b,
                v_rel_b,
                lam_b,
                vc[:, None],
                self.target_vel.norm(dim=-1)[:, None],
            ),
            -1,
        )
        if self._seeker_track is not None:
            tracked = first_order_lag(
                self._seeker_track, rel, 1.0 / self.plant.control_hz, SEEKER_FILTER_TAU
            )
            # A fresh filter starts at the measurement instead of decaying from
            # zero (a zero initial state would halve the apparent range for
            # several time constants).
            tracked = torch.where((~self._seeker_ready)[:, None], rel, tracked)
            self._seeker_ready |= active
            self._seeker_track.copy_(
                torch.where(active[:, None], tracked, self._seeker_track)
            )
            rel = tracked
        rel = rel * self.seeker_lock[:, None].to(COMPUTE_DTYPE)
        observation = torch.cat(
            (wb, q[:, 1:], q[:, :1], vb, p, self.action, self.plant.auxiliary(), rel), -1
        )
        if self._observation is None:
            self._observation = observation.clone()
        else:
            self._observation.copy_(
                torch.where(active[:, None], observation, self._observation)
            )
        # reward: telescoping closing progress plus one dense guidance term.
        # No effort term: it drove commands to zero and opposed the terminal
        # corrections that produce kills. The terminal miss cost grades
        # near-misses by their closest approach instead.
        # 'graded_terminal' shapes on velocity/LOS alignment, which the lead
        # aim already maxes out; 'zem' shapes on the zero-effort miss, whose
        # stepwise decrease telescopes to the miss-distance improvement.
        progress = K_PROG * (self.dist_prev - self.dist)
        if self.reward_variant == "zem":
            zem = _zem(r_w, v_rel)
            inside = (self.dist < ZEM_GATE).to(zem.dtype)
            guide = K_ZEM * inside * (self.zem_prev - zem) / ZEM_REF
            self.zem_prev.copy_(torch.where(active, zem, self.zem_prev))
        else:
            v_hat = v / v.norm(dim=-1, keepdim=True).clamp_min(1.0)
            r_hat = r_w / r2.clamp_min(1e-6).sqrt()[:, None]
            guide = K_ALIGN * (v_hat * r_hat).sum(-1)
            if self.reward_variant == "align_gated":
                guide = guide * (self.dist < ZEM_GATE).to(guide.dtype)
        # terminal events pay once per episode: `complete`/`miss_flag` are
        # sticky within the episode, while `active` is recomputed per inner
        # step (an episode that killed on inner step 1 is still "active" on
        # inner step 2, which used to pay the terminal term twice).
        kpay = kill_ev & active & ~self.complete
        mpay = miss_ev & active & ~self.miss_flag
        if self.reward_variant == "terminal_gauss":
            # paid once after the inner loop, against the episode's final
            # closest approach (see the block after this loop)
            kill_r = torch.zeros_like(progress)
        else:
            kill_r = torch.where(
                kpay, torch.full_like(progress, KILL_BONUS), torch.zeros_like(progress)
            )
        miss_cost = MISS_PENALTY * (self.min_distance / MISS_REF).clamp(max=1.0)
        miss_r = torch.where(mpay, -miss_cost, torch.zeros_like(progress))
        shaping = progress + guide + kill_r + miss_r
        self.reward.copy_(torch.where(active, self.reward + shaping, self.reward))
        comps = (progress, guide, kill_r, miss_r)
        for k, c in zip(self.reward_components, comps):
            self.reward_components[k] += torch.where(active, c, 0.0)
        self.complete |= kill_ev & active
        self.miss_flag |= miss_ev & active
        self.terminated |= (miss_ev | ((p[:, 2] < 0) | (p.norm(dim=-1) > 15000))) & active

    def _step_end(self):
        """Time limit, terminal reward and episode bookkeeping; returns step outputs."""
        # time limit applies after the full control step (rocket semantics)
        if self.reward_variant == "terminal_gauss":
            # Soft fuse: the same 100 at zero miss, decaying over the fuse radius
            # instead of stepping at 15 m, so every episode carries a monotone
            # terminal signal. Measured on the APN endgame-ablation family
            # (88% vs 33% kill rate): span/sd 1.40 vs 1.19 for the hard bonus and
            # std 51.6 -> 25.9, with both asymptotes preserved (0 m -> 100,
            # 500 m -> the -10 miss floor).
            fuse = KILL_BONUS * torch.exp(-(self.min_distance / KILL_RADIUS).square())
            fuse = torch.where(self.terminated | self.complete, fuse, torch.zeros_like(fuse))
            self.reward.copy_(self.reward + fuse)
            self.reward_components["kill"].add_(fuse)
        active = ~(self.terminated | self.truncated)
        self.truncated |= ((self.count >= self.max_steps) | self.complete) & active

        done = self.terminated | self.truncated
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        self.completed_returns += torch.where(done, self.returns, 0.0)
        self.completed_episodes += done.to(torch.long)
        self.completed_successes += (done & self.complete).to(torch.long)
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()
        self.completed_metrics["miss_distance"] += torch.where(done, self.min_distance, 0.0)
        self.completed_metrics["kill_rate"] += (done & self.complete).to(COMPUTE_DTYPE)
        self.completed_metrics["miss_rate"] += (done & self.miss_flag).to(COMPUTE_DTYPE)
        self.completed_metrics["timeout_rate"] += (
            done & ~self.complete & ~self.miss_flag & ~self.out_of_bounds
        ).to(COMPUTE_DTYPE)

        time_outs = self.truncated & ~self.terminated & ~self.complete
        self._cache_terminal_observations()
        return term_out, trunc_out, done, time_outs

    def _episode_stats(self, done):
        """Episode summary over the finishing replicas (``done`` non-empty)."""
        stats = [
            (done & self.complete).to(COMPUTE_DTYPE).sum() / done.sum().clamp_min(1),
            (self.min_distance * done).sum() / done.sum().clamp_min(1),
        ]
        for buf in self.reward_components.values():
            stats.append((buf * done).sum() / done.sum().clamp_min(1))
        return tuple(stats)

    def get_observations(self):
        """Return the current observation packed in a TensorDict under 'policy'."""
        return self._observations_from_buf()

    def _update_observation(self):
        """Copy the float64 internal observation into the float32 obs_buf used downstream."""
        if self.obs_buf is None:
            self.obs_buf = torch.zeros(
                self._observation.shape, device=self.device, dtype=torch.float32
            )
        self.obs_buf.copy_(self._observation)
        if self._sensor is not None:
            active = ~(self.terminated | self.truncated)
            self.obs_buf.copy_(
                self._sensor.apply(
                    self.obs_buf,
                    active,
                    seeker_slice=(23, 29),
                    los_slice=(29, 32),
                )
            )
        # A lost seeker lock must survive the sensor pipeline: the noise model
        # adds independent samples to whatever is in the slice, so re-zero the
        # whole target block (relative position/velocity, LOS rate, closing
        # speed and target speed) for unlocked replicas.  A masked multiply
        # instead of an index write keeps this graph-capture friendly.
        self.obs_buf[:, 23:34].mul_(self.seeker_lock[:, None].to(self.obs_buf.dtype))

    def _apply_perturbation(self, name, scale):
        """Scale a live plant parameter; called by the DR helper."""
        if name == "thrust":
            self.plant.perturb_thrust(scale)
        elif name == "mass":
            self.plant.perturb_mass(scale)
        elif name == "aero":
            self.plant.perturb_aero(scale)
        elif name == "actuator":
            self.plant.perturb_actuator(scale)
        elif name == "cl_alpha":
            self.plant.perturb_cl_alpha(scale)
        elif name == "cd0":
            self.plant.perturb_cd0(scale)
        elif name == "stall":
            self.plant.perturb_stall(scale)
        elif name == "oswald":
            self.plant.perturb_oswald(scale)
        elif name == "actuator_deadband":
            self.plant.perturb_actuator_deadband(scale)
        elif name == "actuator_backlash":
            self.plant.perturb_actuator_backlash(scale)
        elif name == "actuator_asymmetry":
            self.plant.perturb_actuator_asymmetry(scale)
        elif name == "actuator_delay":
            self.plant.perturb_actuator_delay(scale)
        elif name == "sensor":
            if self._sensor is None:
                raise KeyError("sensor perturbation requires the realistic preset")
            self._sensor.set_scale(scale)
        elif name == "cg":
            self.plant.perturb_cg(scale, axis=0)
        elif name == "cg_y":
            self.plant.perturb_cg(scale, axis=1)
        elif name == "cg_z":
            self.plant.perturb_cg(scale, axis=2)
        elif name == "inertia":
            self.plant.perturb_inertia(scale)
        elif name == "inertia_x":
            self.plant.perturb_inertia(scale, axis=0)
        elif name == "inertia_y":
            self.plant.perturb_inertia(scale, axis=1)
        elif name == "inertia_z":
            self.plant.perturb_inertia(scale, axis=2)
        else:
            raise KeyError(name)

    def reset_idx(self, envs_idx, nonempty=None):
        """Reset selected replicas; envs_idx is a device-resident boolean mask.

        One uniform(9) draw covers ALL random quantities. Spawn geometry is a
        single fixed medium operating point (R_SPAWN / CONE / A_LAT / F_WEAVE).
        """
        if self._domain_randomize:
            self._dr_counter += 1
            stage_domain_randomization(self, self._domain_randomize, self._seed, self._dr_counter)
        # No reset warmup here, so the host never needs to know ``nonempty``.
        self._reset_replicas(envs_idx)
        if getattr(self, "_dr_scales", None):
            self.extras["dr_scales"] = {
                name: value.detach() for name, value in self._dr_scales.items()
            }

    def _reset_replicas(self, mask):
        """Device half of reset_idx (graph-capturable)."""
        if self._domain_randomize:
            apply_staged_domain_randomization(self, mask)
        p, q, v, w = self.plant.state()
        u = self.plant.rng.uniform(9, mask)
        # target spawn: horizontal bearing around the origin, fixed range
        R = R_SPAWN
        bearing = u[:, 0] * 2.0 * torch.pi
        target_p = torch.stack(
            (R * torch.cos(bearing), R * torch.sin(bearing), Z_MID + (u[:, 2] - 0.5) * 400.0), -1
        )
        # target heading: back toward the missile inside a fixed cone
        base = torch.atan2(-target_p[:, 1], -target_p[:, 0])
        cone = CONE * torch.pi / 180.0
        heading = base + (u[:, 3] * 2.0 - 1.0) * cone
        h = torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), -1)
        perp = torch.stack((-h[:, 1], h[:, 0], torch.zeros_like(h[:, 0])), -1)
        # maneuver: lateral accel a, angular rate f, type/phase/sign.
        # A 15 g structural/load limit caps the prescribed maneuver: it shrinks
        # the weave amplitude (a/omega^2) and opens the barrel radius (V^2/a),
        # so the target cannot fly an aerodynamically impossible path.
        a = torch.full_like(u[:, 0], A_LAT * 9.81).clamp_max(TARGET_MAX_LOAD_G * 9.81)
        omega = torch.full_like(u[:, 0], 2.0 * torch.pi * F_WEAVE)
        mtype = (u[:, 4] * 3.0).clamp(0, 2).to(torch.long)
        phase = u[:, 5] * 2.0 * torch.pi
        # missile spawn near the origin at engagement altitude. Fire-control
        # lead: aim at the target's PREDICTED position at the nominal flight
        # time t_go = R/(V_M0+V_T), evaluated with the same closed-form motion
        # model the target flies (straight/weave/barrel), so the launch starts
        # on a collision course for every maneuver type and the guidance only
        # has to correct residual errors.
        t_go = torch.full_like(u[:, 0], R) / (V_M0 + V_T)
        missile_p = torch.stack(
            (torch.zeros_like(bearing), torch.zeros_like(bearing), Z_MID + (u[:, 6] - 0.5) * 100.0),
            -1,
        )
        eul_roll = u[:, 7] * 0.2 - 0.1
        # barrel geometry: circle through the spawn point whose INITIAL TANGENT
        # matches the cone heading (otherwise the barrel target starts flying
        # in an arbitrary direction and the engagement degenerates into a tail
        # chase). v = R*omega_b*(-sin th, cos th), so aligning the tangent with
        # h pins the phase to atan2(-h_x, h_y) (+pi when omega_b < 0 flips the
        # traversal direction); the center then sits one radius along the
        # inward normal from the spawn point.
        Rb = V_T * V_T / a
        omega_b = (
            torch.where(u[:, 8] > 0.5, torch.ones_like(u[:, 8]), -torch.ones_like(u[:, 8]))
            * a
            / V_T
        )
        phi_barrel = torch.atan2(-h[:, 0], h[:, 1]) + torch.where(omega_b > 0, 0.0, torch.pi)
        center = target_p - Rb[:, None] * torch.stack(
            (torch.cos(phi_barrel), torch.sin(phi_barrel), torch.zeros_like(phi_barrel)), -1
        )
        # weave keeps its random phase; barrel must start on the aligned tangent
        phase = torch.where(mtype == 2, phi_barrel, phase)

        # fire-control lead: extrapolate the target along its CURRENT heading
        # for the nominal flight time (straight-line shot solution). The
        # perpendicular component of this lead is what nulls the initial LOS
        # rate; the along-LOS component is benign. A full maneuver-model
        # prediction was tested and performed WORSE (the maneuver displacement
        # over the flight is tens of metres, while the straight-line lead
        # correctly cancels the crossing geometry).
        def predict(t):
            """Extrapolate the target straight-line position at time t seconds."""
            return target_p + V_T * t[:, None] * h

        t_go = TGO_SCALE * torch.full_like(u[:, 0], R) / (V_M0 + V_T)
        pred = predict(t_go)
        aim = pred - missile_p
        aim = aim / aim.norm(dim=-1, keepdim=True)
        v_m = V_M0 * aim
        # plant state (masked writes keep other replicas frozen)
        p_new = torch.where(mask[:, None], missile_p, p)
        q_new = torch.where(mask[:, None], _aim_quat(aim, eul_roll), q)
        v_new = torch.where(mask[:, None], v_m, v)
        w_new = torch.where(mask[:, None], torch.zeros_like(w), w)
        self.plant.set_state(p_new, q_new, v_new, w_new)

        # task buffers (masked)
        def sel(x, cur):
            """Select x where mask, keep cur otherwise (handles both 1-D and 2-D tensors)."""
            return torch.where(mask[:, None] if x.dim() > 1 else mask, x, cur)

        self.t_p0.copy_(sel(target_p, self.t_p0))
        self.t_head.copy_(sel(h, self.t_head))
        self.t_perp.copy_(sel(perp, self.t_perp))
        self.t_omega.copy_(sel(omega, self.t_omega))
        self.t_phase.copy_(sel(phase, self.t_phase))
        self.t_a.copy_(sel(a, self.t_a))
        self.t_type.copy_(torch.where(mask, mtype, self.t_type))
        self.t_center.copy_(sel(center, self.t_center))
        self.t_omega_b.copy_(sel(omega_b, self.t_omega_b))
        self.t_radius.copy_(sel(Rb, self.t_radius))
        # sim_time must be zeroed BEFORE sampling the target: _target_state()
        # is a function of sim_time, so sampling at the finished episode's
        # t~4 s put the target ~1 km off its true spawn point and poisoned the
        # initial dist/dist_prev (spurious first-step progress reward) and the
        # initial observation.
        self.sim_time.copy_(sel(torch.zeros_like(self.sim_time), self.sim_time))
        tp, tv = self._target_state()
        self.target_pos.copy_(sel(tp, self.target_pos))
        self.target_vel.copy_(sel(tv, self.target_vel))
        self.t_speed.copy_(sel(torch.full_like(self.t_speed, V_T), self.t_speed))
        dist = (self.target_pos - p_new).norm(dim=-1)
        self.dist.copy_(sel(dist, self.dist))
        self.dist_prev.copy_(sel(dist, self.dist_prev))
        if self.reward_variant == "zem":
            self.zem_prev.copy_(
                sel(_zem(self.target_pos - p_new, self.target_vel - v_new), self.zem_prev)
            )
        self.min_distance.copy_(sel(torch.full_like(dist, 1e9), self.min_distance))
        self.count.masked_fill_(mask, 0)
        self.terminated.masked_fill_(mask, False)
        self.truncated.masked_fill_(mask, False)
        self.out_of_bounds.masked_fill_(mask, False)
        self.complete.masked_fill_(mask, False)
        self.miss_flag.masked_fill_(mask, False)
        for buf in self.reward_components.values():
            buf.masked_fill_(mask, 0.0)
        self.action.masked_fill_(mask[:, None], 0)
        self.plant.zero(mask)
        # No warmup on purpose: the boost engine fires autonomously on the
        # first active physics step, so a rocket-style zero-command warmup
        # would burn thrust (and shift the spawn state) before the first
        # policy step; an air-borne missile has no contacts to settle.
        self.seeker_lock.masked_fill_(mask, True)
        if self._seeker_track is not None:
            self._seeker_track.masked_fill_(mask[:, None], 0.0)
            self._seeker_ready.masked_fill_(mask, False)
        self._observe(mask)
        if self._sensor is not None:
            self._sensor.reset(mask)
        self.episode_length_buf.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)
        if self._wind is not None:
            self._wind.reset(mask)
        self._residual_reset(mask)

    def _observe(self, mask):
        """Refresh self._observation on the masked slots (no return value)."""
        p, q, v, w = self.plant.state()
        wb = inverse_rotate(q, w)
        vb = inverse_rotate(q, v)
        r_w = self.target_pos - p
        v_rel = self.target_vel - v
        r2 = r_w.square().sum(-1).clamp_min(1e-6)
        lam_w = torch.linalg.cross(r_w, v_rel) / r2[:, None]
        lam_b = inverse_rotate(q, lam_w)
        vc = -(r_w * v_rel).sum(-1) / r2.clamp_min(1e-6).sqrt()
        r_b = inverse_rotate(q, r_w)
        rng = r_w.norm(dim=-1).clamp_min(1e-6)
        boresight = torch.acos((r_b[:, 0] / rng).clamp(-1.0, 1.0))
        lock = (rng < SEEKER_MAX_RANGE) & (boresight < SEEKER_HALF_ANGLE)
        if not self.seeker_fov:
            lock = torch.ones_like(lock)
        self.seeker_lock.copy_(torch.where(mask, lock, self.seeker_lock))
        rel = torch.cat(
            (
                r_b,
                inverse_rotate(q, v_rel),
                lam_b,
                vc[:, None],
                self.target_vel.norm(dim=-1)[:, None],
            ),
            -1,
        ) * self.seeker_lock[:, None].to(COMPUTE_DTYPE)
        observation = torch.cat(
            (wb, q[:, 1:], q[:, :1], vb, p, self.action, self.plant.auxiliary(), rel), -1
        )
        if self._observation is None:
            self._observation = observation.clone()
        else:
            self._observation.copy_(torch.where(mask[:, None], observation, self._observation))

    def reset(self):
        """Reset every replica by delegating to reset_idx on an all-True mask."""
        mask = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf = mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):
        """Release the underlying Genesis plant and any other resources."""
        self.plant.close()
