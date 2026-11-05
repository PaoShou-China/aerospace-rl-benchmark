"""Rocket-Landing task: six-spherical-pad contact model with a sustained-landing
(2 s) primary success metric; see docs/rocket_benchmark_v2_zh.md.

The instantaneous touch event below is still used for reward/diagnostics.
Contact model and scoring do not change reward, fatal thresholds, initial
distribution or interfaces.

Rocket-Landing task: upstream-faithful port of PyFlyt 0.29.0.

Task definition (reward, dynamics, episode logic, initial conditions) follows
gym_envs/rocket_envs/rocket_landing_env.py + rocket_base_env.py of the bundled
original_pyflyt reference verbatim; only the physics engine differs (Genesis
rigid-body solver vs PyBullet), per the 2026-10-20 alignment decision: "the
physics engine may differ, the reward function and dynamics model must match".

Concretely, relative to the previous staged port (all measured/documented
there), these upstream behaviours are KEPT:
  * Pad contact does NOT terminate the episode (upstream L231-263): on a
    contact inner step the rocket is fatal iff the PREVIOUS inner-step speed
    exceeds 1.0 m/s or 0.35 rad/s; it "lands" (success) iff that
    speed is <0.02 m/s and <0.02 rad/s with tilt <0.1; otherwise the episode
    CONTINUES and the rocket may settle on the pad. Success is therefore an
    on-pad settling process, not a one-shot quantum touch.
  * Initial conditions follow upstream begin_reset with the options the
    landing task passes (randomize_drop=True, accelerate_drop=True):
    xy in +/-20 m (max_displacement*0.1), z in [400, 450] (ceiling*0.8..0.9),
    euler orn in +/-0.3 rad per axis, vz = -100 m/s, zero angular velocity.
    Upstream quirk reproduced: the random linear/angular velocities in
    begin_reset never fire because the option key is misspelled
    ("randoimize_drop" vs "randomize_drop"), so only the -100 m/s drop speed
    applies. The earlier staged drop (at-rest z=150) was tied to the dropped
    v3 reward and is reverted.
  * Action box is the upstream 7-dim finlet mixer convention: dims 0-2 are
    the finlet (x, y, yaw) demands in [-1,1] and the plant mixes them onto
    the four grid-fin hinge servos; ignition (dim 3) and throttle (dim 4)
    in [0,1] -- uniform exploration fires the engine on ~50% of steps,
    exactly like upstream; gimbals (dims 5-6) in [-1,1].

Reward (2026-10-28 'v5'): dense per-POLICY-step shaping. v4 kept a pure dive
net-negative per step but ignored that a fatal crash also ends the hover-cost
stream: PPO converged to +100 returns by diving onto the pad at 5 m/s, with
zero successes in 65.5M transitions. v5 adds a feasible descent-speed corridor
and a one-shot fatal penalty; the successful-settle and contact terms are
unchanged:
  r = -HOVER_COST                                time pressure, always
      + HOLD_COEF/(||p_xy||+0.1)                 <= 0.05, cannot farm hover
      + LATERAL_COEF*(d_xy_prev - d_xy)          telescoping pad approach
      + DESCENT_COEF*(z_prev - z)                descent progress
      + DECEL_COEF*(dvz+1)/e^z*sign(vz)          upstream decel term
      - CORRIDOR_COEF*relu(-vz - v_allow(z))     v_allow = 0.5+sqrt(2*2*z),
                                                 conservative vs ~3.55 m/s^2
                                                 nominal maximum net decel
      - YAW_COEF*|w_yaw| - TILT_COEF*||rpy_xy||
      + contact: CONTACT_BASE - CONTACT_VEL_COEF*|vz| per pad-contact step
      - FATAL_COST once on a new terminal transition (500: a dive-crash return
        becomes negative instead of escaping the remaining hover cost)
      + SUCCESS_BONUS once on first settle (5000 >> gamma*c/(1-gamma)~70,
        so delaying a reachable success is never profitable)
Success still does NOT truncate; the settled contact stream keeps paying.

Port details: 40 Hz policy / 120 Hz control / 240 Hz physics (all rates read
from the plant, none hardcoded), 1200 policy steps (30 s) truncation, bounds
||p_xy||>200 or z>500, any
non-pad contact or z<0 is fatal, obs = [body omega, quat, body v, p,
action (7), aux(9), pad contact flag] (30 dims), 5% starting fuel, reignitable
engine with the per-physics-step ignition>0.5 rule.

Success: pad contact with |v|<0.02, |omega|<0.02, tilt<0.1 (upstream values).
"""

import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import COMPUTE_DTYPE, euler, inverse_rotate, quat_from_euler
from algorithms.baselines.baselines import RocketPDGBaseline
from environments.common.residual import ResidualActionMixin
from environments.common.base_env import BaseEnv
from environments.common.domain_randomization import (
    apply_static_perturbation,
    apply_domain_randomization,
    resolve_domain_randomize,
)


HOVER_COST = 0.3  # per policy step, always (time pressure)
HOLD_COEF = 0.005  # holding reward HOLD_COEF/(||p_xy||+0.1), max 0.05 -- too
# small to farm against the hover cost
LATERAL_COEF = 10.0  # telescoping lateral progress toward the pad axis
DESCENT_COEF = 0.1  # descent progress (0.2 -> 0.1 by the dive analysis: a pure
# vertical dive then stays net-negative up to 150 m/s)
DECEL_COEF = 0.1  # upstream deceleration term (dvz+1)/e^z*sign(vz), 4 -> 0.1
YAW_COEF = 0.5  # -YAW_COEF*|body yaw rate|
TILT_COEF = 0.5  # -TILT_COEF*||roll/pitch||
CONTACT_BASE = 0.5  # per policy step with any pad contact
CONTACT_VEL_COEF = 1.0  # -CONTACT_VEL_COEF*|vz| on contact steps
SUCCESS_BONUS = 5000.0  # one-shot on first settle-success (B >> gamma*c/(1-gamma))
CORRIDOR_COEF = 0.5  # per-step penalty per m/s above the feasible descent corridor
CORRIDOR_ACCEL = 2.0  # m/s^2; nominal net decel is ~3.55, leaving control margin
CORRIDOR_V0 = 0.5  # m/s terminal descent allowance, below the 1.0 fatal threshold
FATAL_COST = 500.0  # one-shot on a new terminal transition, not per contact step

REWARD_KEYS = (
    "hover",
    "hold",
    "lateral",
    "descent",
    "decel",
    "corridor",
    "attitude",
    "contact",
    "fatal",
    "success",
)


def rocket_shaping(
    cfg,
    p,
    previous_position,
    vel_z,
    previous_vel_z,
    rpy,
    body_omega,
    contact,
    contact_vz,
    newly_complete,
    newly_fatal,
    fuel=None,
):
    """Dense reward per POLICY step, returned as (total, per-component dict).

    vel_z is the WORLD-frame vertical velocity. `contact` marks any pad
    contact during this policy step, `contact_vz` the world descent speed at
    the first contact inner step; `newly_complete` and `newly_fatal` fire once
    per episode on settle success and termination respectively. All
    coefficients come from cfg (see the module constants).
    """
    dxy = p[:, :2].norm(dim=-1)
    dxy_prev = previous_position[:, :2].norm(dim=-1)
    dvz = vel_z - previous_vel_z
    decel = (dvz + 1.0) * torch.exp(-p[:, 2].clamp_min(0.0)) * torch.where(vel_z < 0.0, 1.0, -1.0)
    z = p[:, 2].clamp_min(0.0)
    corridor_accel = cfg["corridor_accel"]
    if fuel is not None:
        mass = cfg["dry_mass"] + fuel * cfg["fuel_full"]
        corridor_accel = 0.6 * (cfg["max_thrust"] / mass - 9.81).clamp_min(0.5)
    v_allow = cfg["corridor_v0"] + torch.sqrt(2.0 * corridor_accel * z)
    v_excess = ((-vel_z).clamp_min(0.0) - v_allow).clamp_min(0.0)
    comps = {
        "hover": torch.full_like(dxy, -cfg["hover"]),
        "hold": cfg["hold"] / (dxy + 0.1),
        "lateral": cfg["lateral"] * (dxy_prev - dxy),
        "descent": cfg["descent"] * (previous_position[:, 2] - p[:, 2]),
        "decel": cfg["decel"] * decel,
        "corridor": -cfg["corridor"] * v_excess,
        "attitude": -cfg["yaw"] * body_omega[:, 2].abs()
        - cfg["tilt"] * rpy[:, :2].square().sum(-1).sqrt(),
        "contact": torch.where(
            contact,
            cfg["contact_base"] - cfg["contact_vel"] * contact_vz.abs(),
            torch.zeros_like(dxy),
        ),
        "fatal": -cfg["fatal"] * newly_fatal.to(dxy.dtype),
        "success": newly_complete.to(dxy.dtype) * cfg["success"],
    }
    total = sum(comps.values())
    return total, comps


class RocketLandingEnv(BaseEnv, ResidualActionMixin):
    """Rocket vertical-landing RL environment.

    Task objective: land a six-spherical-pad booster on the ground pad with
    |v| < 0.02 m/s, |omega| < 0.02 rad/s and tilt < 0.1 rad. The agent
    observes a float32 tensor of body angular velocity (3), quaternion wxyz
    (4), body-frame velocity (3), world position (3), last physical action
    (7), per-rotor throttle and auxiliary signals (10), plus a 1-dim pad
    contact flag (total 30). Actions are 7-D: finlet x/y/yaw demands [-1, 1],
    ignition [0, 1], throttle [0, 1], and TVC gimbals (2, [-1, 1]); the plant
    mixes them onto four grid-fin servos. Termination: fatal on any non-pad
    contact with speed > 1.0 m/s or spin > 0.35 rad/s, |p_xy| > 200 m, or
    z < 0 m; truncation at max_steps = 30 * agent_hz (30 s). The plant uses
    40/60 Hz agent / 120 Hz control / 240 Hz physics. Domain randomization
    covers thrust (0.97, 1.03); mass stays nominal by default.
    """
    # Upstream ICs (unchanged): xy +/-20 m, z [400,450], euler +/-0.3 rad.
    # Success criteria (pad contact thresholds) are untouched. Wind is
    # optional via the steady_wind/dryden ctor kwargs; calm air reproduces
    # the benchmark bit-for-bit.
    ic_xy_half = 20.0  # spawn xy half-range [m]
    ic_z_min = 400.0  # spawn z lower bound [m] (span 50 fixed)
    ic_euler_half = 0.3  # spawn euler per-axis half-range [rad]
    # conventional controller: ZEM/ZEV powered-descent guidance + TVC/fin attitude
    # PD (baselines.py; reuses the env's unified 7-dim finlet mixer convention)
    conventional_controller = RocketPDGBaseline
    PERTURBABLE_PARAMS = {"thrust": 1.0, "mass": 1.0}
    # Literature: rocket Monte Carlo dispersion analyses (e.g. SpaceDelta
    # Falcon-9 landing design verification) use thrust +/-3% (3 sigma) and
    # specific impulse +/-1% for the propulsion system.  Dry mass is not
    # randomized by default; it remains available as an explicit
    # param_mismatch / domain_randomize override.
    DEFAULT_DOMAIN_RANDOMIZE = {"thrust": (0.97, 1.03)}

    def __init__(
        self,
        num_envs=64,
        backend="cpu",
        seed=7,
        steady_wind=None,
        dryden=0.0,
        severity="moderate",
        reward=None,
        variation="nominal",
        agent_hz=None,
        benchmark_version="v2",
        residual=False,
        param_mismatch=None,
        domain_randomize=None,
    ):
        """Initialize the RocketLanding environment.

        Construct the Genesis-backed rocket plant at the selected agent_hz
        (40 or 60 Hz), allocate task buffers (touch telemetry, reward
        components, landing metrics, domain-randomization counters), and
        finalize with a reset().
        """
        if benchmark_version not in ("v1", "v2"):
            raise ValueError(benchmark_version)
        from environments.common.rocket_benchmark import VARIATIONS, LandingMetrics

        if variation not in VARIATIONS:
            raise ValueError(variation)
        self.benchmark_version = benchmark_version
        self.f9_real = benchmark_version == "v1"
        # Both supported versions use a 40 Hz policy by default.
        if agent_hz is None:
            agent_hz = 40
        self.num_envs = num_envs
        self.task_cfg = {
            "vehicle": "rocket",
            "num_envs": num_envs,
            "backend": backend,
            "implementation": "fully_tensorized",
            "task_defaults": "PyFlyt 0.29.0",
        }
        self.clip_actions = 1.0
        # reward coefficients (module defaults; `reward` overrides per-key for
        # the evolution sweep arms, recorded into cfg -> run.json)
        self.reward_cfg = dict(
            hover=HOVER_COST,
            hold=HOLD_COEF,
            lateral=LATERAL_COEF,
            descent=DESCENT_COEF,
            decel=DECEL_COEF,
            yaw=YAW_COEF,
            tilt=TILT_COEF,
            contact_base=CONTACT_BASE,
            contact_vel=CONTACT_VEL_COEF,
            success=SUCCESS_BONUS,
            corridor=CORRIDOR_COEF,
            corridor_accel=CORRIDOR_ACCEL,
            corridor_v0=CORRIDOR_V0,
            fatal=FATAL_COST,
        )
        if reward:
            self.reward_cfg.update(reward)

        self.plant = TensorPlant(
            "rocket",
            num_envs,
            backend,
            seed,
            rocket_variation=variation,
            rocket_benchmark=benchmark_version,
        )
        self.device = self.plant.device
        # agent_hz validation: must divide the plant control loop. All rates
        # are read from the plant, never hardcoded here.
        self.agent_hz = agent_hz
        if agent_hz <= 0 or agent_hz != int(agent_hz) or self.plant.control_hz % agent_hz != 0:
            raise ValueError(
                f"agent_hz must be a positive divisor of the "
                f"{self.plant.control_hz} Hz control loop; 50 Hz is not "
                f"(use 40 or 60)"
            )
        self.inner_steps = int(self.plant.control_hz // agent_hz)
        self.max_steps = (60 if self.f9_real else 30) * agent_hz
        self.landing_metrics = LandingMetrics(
            num_envs,
            self.device,
            agent_hz,
            velocity_limit=0.1 if self.f9_real else 0.02,
            tilt_limit=math.radians(5.0) if self.f9_real else 0.1,
        )
        self.max_episode_length = self.max_steps + 2
        # Unified 7-dim upstream finlet mixer convention (matches external benchmark):
        #   finlet x/y/yaw [-1,1], ignition [0,1], throttle [0,1], gimbals [-1,1]
        self.num_actions = 7
        self.action_low = torch.tensor(
            [-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0], device=self.device, dtype=COMPUTE_DTYPE
        )
        self.action_high = torch.ones(self.num_actions, device=self.device, dtype=COMPUTE_DTYPE)

        # initialize buffers
        self.action = torch.zeros(
            (num_envs, self.num_actions), device=self.device, dtype=COMPUTE_DTYPE
        )
        self.reward = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self.count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.terminated = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.truncated = self.terminated.clone()
        self.collision = self.terminated.clone()
        self.out_of_bounds = self.terminated.clone()
        self.complete = self.terminated.clone()
        self.fatal_collision = self.terminated.clone()
        self.obs_pad = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self._observation = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.returns = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self.completed_returns = torch.zeros_like(self.returns)
        self.completed_episodes = torch.zeros_like(self.episode_length_buf)
        self.completed_successes = torch.zeros_like(self.episode_length_buf)
        # first-touch telemetry (monitoring only; NOT part of episode logic --
        # upstream keeps simulating after a touch so the rocket can settle)
        self.touch_flag = self.terminated.clone()
        self.touch_speed = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self.touch_ang_vel = torch.zeros_like(self.touch_speed)
        self.touch_tilt = torch.zeros_like(self.touch_speed)
        self.completed_metrics = {
            "pad_touch_rate": torch.zeros_like(self.touch_speed),
            "nonfatal_touch_rate": torch.zeros_like(self.touch_speed),
            "touch_speed": torch.zeros_like(self.touch_speed),
            "touch_ang_vel": torch.zeros_like(self.touch_speed),
            "touch_tilt": torch.zeros_like(self.touch_speed),
        }
        for key in self.landing_metrics.values(self.complete, self.terminated):
            self.completed_metrics[key] = torch.zeros_like(self.touch_speed)
        # per-component reward accounting: Episode/rew_* tensorboard scalars and
        # run.json completed_metrics keys, so reward attribution is observable
        keys = REWARD_KEYS
        self.reward_components = {
            k: torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE) for k in keys
        }
        for k in keys:
            self.completed_metrics[f"rew_{k}"] = torch.zeros_like(self.touch_speed)

        self.extras = dict()
        self.cfg = dict(self.task_cfg)
        self.cfg["reward"] = dict(self.reward_cfg)
        self._init_wind(num_envs, seed, steady_wind, dryden, severity)
        self.benchmark_env_kwargs = dict(
            benchmark_version=benchmark_version, variation=variation, agent_hz=agent_hz
        )
        from pathlib import Path
        import hashlib

        rocket_model_dir = (
            Path(__file__).parents[1] / "common/original_pyflyt/models/vehicles/rocket"
        )
        model = rocket_model_dir / {
            "v1": "rocket_falcon9_stage1.urdf",
            "v2": "rocket_v2.urdf",
        }[benchmark_version]
        primary_metric = "ever_stable_2s"
        self.cfg["benchmark"] = dict(
            version=benchmark_version,
            variation=variation,
            variation_ranges=VARIATIONS[variation],
            variation_seed_offset=104729,
            policy_hz=agent_hz,
            control_hz=self.plant.control_hz,
            physics_hz=self.plant.physics_hz,
            horizon_seconds=60 if self.f9_real else 30,
            stable_samples=self.landing_metrics.required_samples,
            primary_metric=primary_metric,
            collision_asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            physics_and_success_predicate_modified=True,
            integration="native-main-v2",
            external_wind=self.cfg.get("wind"),
        )
        self._seed = seed
        self.controller_design = {}
        apply_static_perturbation(self, param_mismatch)
        self._dr_counter = 0
        self._domain_randomize = resolve_domain_randomize(
            domain_randomize, self.DEFAULT_DOMAIN_RANDOMIZE
        )
        self.residual = residual
        self._init_residual()

        self.reset()

    def _init_wind(self, n, seed, steady_wind, dryden, severity):
        """Optional wind disturbance (environments/common/disturbances.py).

        Calm air (default) leaves the plant dynamics bit-for-bit unchanged;
        with wind configured, the gust filters advance once per control step
        and the fin/body aerodynamic terms evaluate v_air = v - wind.
        """
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
            )
            self.plant.wind = torch.zeros((n, 3), device=self.device, dtype=COMPUTE_DTYPE)
            self.cfg["wind"] = dict(
                steady_wind=list(steady_wind) if steady_wind is not None else None,
                dryden=dryden,
                severity=severity,
            )

    def _scale_actions(self, actions):
        """Affinely map [-1, 1] policy actions onto the physical action bounds."""
        raw = actions.to(COMPUTE_DTYPE).clamp(-self.clip_actions, self.clip_actions)
        return self.action_low + (raw + 1) * 0.5 * (self.action_high - self.action_low)

    def step(self, actions):
        """Advance one policy step through inner control steps.

        Scale actions, run the plant controller and physics substeps for each
        inner step, detect pad contacts, fatal impacts and bounds violations,
        accumulate per-component dense reward, and return
        (observations, reward, terminated, truncated, extras).
        """
        if actions.device != torch.device(self.device):
            raise ValueError(
                "Actions must already be on env.device; implicit host/device copies are disabled"
            )
        actions = self._residual_input(actions)
        physical = self._scale_actions(actions)
        # Upstream passes the gym action straight to set_setpoint; the plant
        # applies the ignition>0.5 rule per physics step (240 Hz, reignitable)
        # and the throttle -> [0.39, 1.0] min/max thrust remap (boosters.py
        # L231-233). No extra env-level reshaping.
        command = physical.clone()
        self.action.copy_(command)
        if self._wind is not None:
            self.plant.wind.copy_(self._wind.step())
        self.touch_flag.fill_(False)
        self.count += 1
        # policy-step snapshots for the per-policy-step reward deltas
        p_start, _, v_start, _ = self.plant.state()
        p_start = p_start.clone()
        vz_start = v_start[:, 2].clone()
        complete_start = self.complete.clone()
        stable_start = self.landing_metrics.ever_stable.clone()
        terminated_start = self.terminated.clone()
        contact_any = self.terminated.clone().fill_(False)
        contact_vz = torch.zeros(self.num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        for _ in range(self.inner_steps):
            active = ~(self.terminated | self.truncated)
            p0, q0, v0, w0 = self.plant.state()
            vb0 = inverse_rotate(q0, v0)
            wb0 = inverse_rotate(q0, w0)
            # Constant for the whole control step: one sync here lets every
            # substep skip the no-op pause-restore write-back.
            restore = not bool(active.all())
            self.plant.clear_contacts()
            self.plant.controller(command, active)
            for _ in range(self.plant.physics_per_control):
                self.plant.physics_step(command, active, restore=restore)
            p, rpy, v, wb, vb = self._observe(active)
            # Upstream compute_base_term_trunc_reward: any contact other than
            # (rocket, pad) is a fatal collision, as is dropping below ground.
            collision = self.plant.ground_contact | (p[:, 2] < 0)
            if self.f9_real:
                bounds = (p[:, :2].norm(dim=-1) > 1000) | (p[:, 2] > 3000)
            else:
                bounds = (p[:, :2].norm(dim=-1) > 200) | (p[:, 2] > 500)
            touch = self.plant.pad_contact & active
            speed0 = vb0.norm(dim=-1)
            spin0 = wb0.norm(dim=-1)
            tilt0 = rpy[:, :2].norm(dim=-1)
            # Upstream L231-263: on a pad-contact step judge the PREVIOUS
            # inner-step speed magnitude (= speed0/spin0 here, the state at the
            # start of this inner step): >1.0 m/s or >0.35 rad/s is fatal;
            # <0.02/<0.02 with tilt<0.1 is a landed success; anything in between
            # keeps simulating (on-pad settling).
            if self.f9_real:
                hard_touch = (speed0 > 6.0) | (spin0 > 0.35) | (tilt0 > math.radians(15.0))
                fatal = collision | (touch & hard_touch)
                complete = torch.zeros_like(touch)
            else:
                fatal = collision | (touch & ((speed0 > 1.0) | (spin0 > 0.35)))
                complete = touch & ~fatal & (spin0 < 0.02) & (speed0 < 0.02) & (tilt0 < 0.1)
            new_touch = touch & ~self.touch_flag
            self.touch_flag |= new_touch
            self.touch_speed.copy_(torch.where(new_touch, speed0, self.touch_speed))
            self.touch_ang_vel.copy_(torch.where(new_touch, spin0, self.touch_ang_vel))
            self.touch_tilt.copy_(torch.where(new_touch, tilt0, self.touch_tilt))
            # per-policy-step contact bookkeeping: any contact while active,
            # and the world descent speed at the FIRST contact inner step
            contact_vz.copy_(torch.where(touch & ~contact_any, v0[:, 2], contact_vz))
            contact_any |= touch
            self.complete |= complete & active
            self.fatal_collision |= fatal & active
            self.obs_pad.copy_(
                torch.where(active, self.plant.pad_contact.to(COMPUTE_DTYPE), self.obs_pad)
            )
            term = fatal | bounds
            # success does NOT truncate: the rocket keeps earning the settled
            # contact stream until the time limit below
            trunc = torch.zeros_like(fatal)
            self.terminated |= term & active
            self.truncated |= trunc & active
            self.collision |= collision & active
            self.out_of_bounds |= bounds & active
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents still
        # terminate inside the substep loop above. This is the ONLY truncation:
        # a landed rocket keeps simulating (and earning) until here.
        active = ~(self.terminated | self.truncated)
        self.truncated |= (self.count >= self.max_steps) & active
        self.landing_metrics.update(
            self.obs_pad.bool(), v, wb, rpy[:, :2].norm(dim=-1), self.terminated
        )
        if self.f9_real:
            self.complete.copy_(self.landing_metrics.ever_stable)
        newly_success = (
            self.landing_metrics.ever_stable & ~stable_start
            if self.f9_real
            else self.complete & ~complete_start
        )
        shaping_cfg = self.reward_cfg
        fuel = None
        if self.f9_real:
            shaping_cfg = dict(
                shaping_cfg,
                dry_mass=25600.0,
                fuel_full=self.plant.params["propellant_params"]["total_fuel"],
                max_thrust=self.plant.params["booster_params"]["max_thrust"],
            )
            fuel = self.plant.fuel
        shaping, comps = rocket_shaping(
            shaping_cfg,
            p,
            p_start,
            v[:, 2],
            vz_start,
            rpy,
            wb,
            contact_any,
            contact_vz,
            newly_success,
            self.terminated & ~terminated_start,
            fuel=fuel,
        )
        self.reward.copy_(shaping)
        for k in comps:
            self.reward_components[k] += comps[k]

        self.reset_buf = self.terminated | self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        self.completed_returns += torch.where(self.reset_buf, self.returns, 0.0)
        self.completed_episodes += self.reset_buf.to(torch.long)
        metrics = self.landing_metrics.values(self.complete, self.terminated)
        primary = metrics["ever_stable_2s"]
        self.completed_successes += (self.reset_buf & primary).to(torch.long)
        for key, value in metrics.items():
            self.completed_metrics[key] += (self.reset_buf & value).to(COMPUTE_DTYPE)
        landed = self.reset_buf & self.touch_flag
        nonfatal_touch = landed & ~self.fatal_collision
        self.completed_metrics["pad_touch_rate"] += landed.to(COMPUTE_DTYPE)
        self.completed_metrics["nonfatal_touch_rate"] += nonfatal_touch.to(COMPUTE_DTYPE)
        self.completed_metrics["touch_speed"] += torch.where(landed, self.touch_speed, 0.0)
        self.completed_metrics["touch_ang_vel"] += torch.where(landed, self.touch_ang_vel, 0.0)
        self.completed_metrics["touch_tilt"] += torch.where(landed, self.touch_tilt, 0.0)
        done = self.reset_buf
        # reward attribution at done: accumulate the per-episode component sums
        for k, buf in self.reward_components.items():
            self.completed_metrics[f"rew_{k}"] += torch.where(done, buf, 0.0)
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()

        self.extras["time_outs"] = self.truncated & ~self.terminated & ~self.complete
        self._cache_terminal_observations()
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any = bool(done.any())
        if done_any:
            episode = {
                "success_rate": (done & primary).to(COMPUTE_DTYPE).sum() / done.sum().clamp_min(1),
                "pad_touch_rate": landed.to(COMPUTE_DTYPE).sum() / done.sum().clamp_min(1),
                "nonfatal_touch_rate": nonfatal_touch.to(COMPUTE_DTYPE).sum()
                / done.sum().clamp_min(1),
            }
            touched = done & self.touch_flag
            n = touched.to(COMPUTE_DTYPE).sum()
            # torch.where instead of a bool(touched.any()) sync: when no done
            # replica has touched the pad the masked sums are 0, so report 0
            zero = torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE)
            episode["touch_speed"] = torch.where(
                n > 0, (self.touch_speed * touched).sum() / n, zero
            )
            episode["touch_ang_vel"] = torch.where(
                n > 0, (self.touch_ang_vel * touched).sum() / n, zero
            )
            episode["touch_tilt"] = torch.where(n > 0, (self.touch_tilt * touched).sum() / n, zero)
            for k, buf in self.reward_components.items():
                episode[f"rew_{k}"] = (buf * done).sum() / done.sum().clamp_min(1)
            self.extras["episode"] = episode
        else:
            self.extras.pop("episode", None)

        # empty-mask reset is fully safe (warmup early-exits; pause-restore is
        # unconditional); the gate reuses the already-synced done_any for cost
        if done_any:
            self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    def get_observations(self):
        """Return the current observation packed in a TensorDict under 'policy'."""
        return self._observations_from_buf()

    def _observe(self, mask):
        """Refresh the internal observation buffer from plant state and pad contact."""
        p, q, v, w = self.plant.state()
        rpy = euler(q)
        wb = inverse_rotate(q, w)
        vb = inverse_rotate(q, v)
        attitude = torch.cat(
            (wb, q[:, 1:], q[:, :1], vb, p, self.action, self.plant.auxiliary()), -1
        )
        observation = torch.cat((attitude, self.obs_pad[:, None]), -1)
        if self._observation is None:
            self._observation = observation.clone()
        else:
            self._observation.copy_(torch.where(mask[:, None], observation, self._observation))
        return p, rpy, v, wb, vb

    def _update_observation(self):
        """Copy the float64 internal observation into the float32 obs_buf used downstream."""
        if self.obs_buf is None:
            self.obs_buf = torch.zeros(
                self._observation.shape, device=self.device, dtype=torch.float32
            )
        self.obs_buf.copy_(self._observation)

    def _apply_perturbation(self, name, scale):
        if name == "thrust":
            self.plant.perturb_thrust(scale)
        elif name == "mass":
            self.plant.perturb_mass(scale)
        else:
            raise KeyError(name)

    def reset_idx(self, envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask.

        Upstream ICs (rocket_landing_env.reset -> begin_reset with
        randomize_drop/accelerate_drop): xy uniform in +/-20 m, z uniform in
        [400, 450], euler orn uniform in +/-0.3 rad/axis, vz = -100 m/s, no
        angular velocity. The upstream random linear/angular velocities are
        dead code there ("randoimize_drop" typo) and are not reproduced.
        """
        mask = envs_idx
        if self._domain_randomize:
            self._dr_counter += 1
            apply_domain_randomization(
                self, self._domain_randomize, self._seed, self._dr_counter, mask
            )
        p, q, v, w = self.plant.state()
        if self.f9_real:
            u = self.plant.rng.uniform(13, mask)
            spawn = torch.zeros_like(p)
            spawn[:, :2] = (u[:, :2] * 2.0 - 1.0) * 100.0
            spawn[:, 2] = 0.65 + 800.0 + u[:, 2] * 700.0
            rpy = torch.zeros_like(p)
            rpy[:, :2] = (u[:, 3:5] * 2.0 - 1.0) * math.radians(3.0)
            rpy[:, 2] = (u[:, 5] * 2.0 - 1.0) * math.pi
            rotation = quat_from_euler(rpy)
            velocity = torch.zeros_like(v)
            velocity[:, :2] = (u[:, 6:8] * 2.0 - 1.0) * 5.0
            velocity[:, 2] = -(60.0 + 60.0 * u[:, 8])
            omega = (u[:, 9:12] * 2.0 - 1.0) * 0.02
        else:
            u = self.plant.rng.uniform(6, mask)
            spawn = torch.zeros_like(p)
            spawn[:, 0] = (u[:, 0] * 2.0 - 1.0) * self.ic_xy_half
            spawn[:, 1] = (u[:, 1] * 2.0 - 1.0) * self.ic_xy_half
            spawn[:, 2] = self.ic_z_min + u[:, 2] * (450.0 - self.ic_z_min)
            rotation = quat_from_euler((u[:, 3:] * 2.0 - 1.0) * self.ic_euler_half)
            velocity = torch.zeros_like(v)
            velocity[:, 2] = -100.0
            omega = torch.zeros_like(w)
        self.plant.set_state(
            torch.where(mask[:, None], spawn, p),
            torch.where(mask[:, None], rotation, q),
            torch.where(mask[:, None], velocity, v),
            torch.where(mask[:, None], omega, w),
        )
        self.count.masked_fill_(mask, 0)
        self.terminated.masked_fill_(mask, False)
        self.truncated.masked_fill_(mask, False)
        self.collision.masked_fill_(mask, False)
        self.out_of_bounds.masked_fill_(mask, False)
        self.complete.masked_fill_(mask, False)
        self.fatal_collision.masked_fill_(mask, False)
        self.landing_metrics.reset(mask)
        self.touch_flag.masked_fill_(mask, False)
        self.touch_speed.masked_fill_(mask, 0)
        self.touch_ang_vel.masked_fill_(mask, 0)
        self.touch_tilt.masked_fill_(mask, 0)
        for buf in self.reward_components.values():
            buf.masked_fill_(mask, 0.0)
        self.action.masked_fill_(mask[:, None], 0)
        self.plant.zero(mask)
        if self.f9_real:
            low, high = self.plant.params["propellant_params"]["initial_range"]
            sampled_fuel = (low + (high - low) * u[:, 12]) / self.plant.params["propellant_params"]["total_fuel"]
            self.plant.fuel.copy_(torch.where(mask, sampled_fuel, self.plant.fuel))
        self.obs_pad.masked_fill_(mask, 0)
        if bool(mask.any()):
            self.plant.warmup(mask)
        self._observe(mask)
        self.episode_length_buf.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)
        if self._wind is not None:
            self._wind.reset(mask)
        self._residual_reset(mask)

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
