"""Quadplane-VTOL Waypoints — a tilting-body hybrid UAV task.

Vehicle: PyFlyt quadplane (H-layout 4-quad motors + fixedwing surfaces
+ forward pusher motor, ~2.35 kg total). The vehicle is simultaneously a
quadcopter (four propellers whose rate-PID mixing produces body torques
plus collective thrust, dominant at low airspeed where aerodynamic surfaces
produce negligible force) AND a fixed-wing aircraft (ailerons, h-tail,
v-tail, main wing + forward pusher, dominant at high airspeed where
lifting-surface forces scale as v^2). There is NO explicit mode switch or
velocity-gated blending in the plant or env — the physics itself enforces
authority: prop thrust is constant w.r.t. airspeed, surface lift/drag is
zero at v=0 and quickly grows as the vehicle accelerates. The agent
naturally discovers the transition regime (~3–12 m/s body +x velocity)
where both actuation paths are usable.

Task: reach 4 waypoints sampled in a 40–75 m radius sphere at ≥25 m altitude,
starting from near-hover (z=15 m, |v| ~ 6 m/s, level). Capture radius 15 m,
out-of-bounds at |p|>120 m, 1200 steps = 30 s at 40 Hz agent rate.

Action space (8 dims):
  a0–a2: body-rate commands (p, q, r, rad/s) → rate PID → quad motor mixer
  a3:    quad collective thrust [0, 1] → same mixer, 4th channel
  a4–a6: fixed-wing surface commands, same aileron/h-tail+v-tail/dead
         mapping as FixedwingWaypointsEnv (a4=ail diff, a5=htail+vtail,
         a6=dead main wing; index_select / sign applied in plant)
  a7:    forward pusher throttle [0, 1] — directly maps to plant throttle[:, 4]
Observation: body angular rates, quaternion (w,x,y,z), body-frame linear
velocity, world position, last action, plant auxiliary (5 surfaces + 5
throttles = 10 dims), then 4 waypoints in body frame + 4 validity flags.

Reward: potential-based telescoping (same as quadx / fixedwing):
  Phi = -(distance to current target + remaining path legs)
  reward = Phi_prev - Phi + 15 * reach_count - 10 * crash
so orbiting nets zero, reaching telescopes +15 and a positive progress
component, and crashing forfeits all unrealized potential.

Implementation parallels FixedwingWaypointsEnv; every buffer is float64
physics / float32 obs, every tensor on one device, no per-replica gym
loop. See environments/common/plant.py Quadplane branch for motor mixer,
lifting surfaces, and throttle handling.
"""

import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import COMPUTE_DTYPE, inverse_rotate
from algorithms.baselines.baselines import QuadplaneHybridBaseline
from environments.common.residual import ResidualActionMixin
from environments.common.base_env import BaseEnv
from environments.common.domain_randomization import (
    apply_static_perturbation,
    apply_domain_randomization,
    resolve_domain_randomize,
)

REACH_BONUS = 15.0
CRASH_PENALTY = 10.0
CAPTURE_RADIUS = 15.0
WAYPOINT_MIN_RADIUS = 40.0
WAYPOINT_MAX_RADIUS = 75.0
WAYPOINT_MIN_ALTITUDE = 25.0

NUM_TARGETS = 4


class QuadplaneWaypointsEnv(BaseEnv, ResidualActionMixin):
    """Quadplane VTOL waypoint-following RL environment.

    Task objective: reach a sequence of 4 waypoints sampled in a 40-75 m
    radius shell at >= 25 m altitude. The agent observes a flat float32
    tensor containing body angular velocity (3), quaternion wxyz (4), body-
    frame velocity (3), world position (3), the last physical action (8),
    plant auxiliary (10 surface + throttle dims), four waypoints in body
    frame (12), plus a 4-waypoint validity mask (4). Actions are 8-D: body
    rates roll/pitch/yaw [-pi, pi] rad/s, quad collective [0, 1], three
    fixed-wing surface commands (aileron diff, h-tail + v-tail, dead main
    wing, all [-1, 1]), and forward pusher throttle [0, 1], mapped affinely
    from raw [-1, 1]. The plant handles both multirotor (low-speed) and
    fixed-wing (high-speed) regimes naturally via physics authority.
    Termination on ground contact or |p| > 120 m; truncation at 1200 policy
    steps (30 s at 40 Hz) or when all waypoints are reached. Domain
    randomization covers thrust (0.85, 1.15) and mass (0.90, 1.10).
    """
    waypoint_min_radius = WAYPOINT_MIN_RADIUS
    bounds_radius = 120.0
    # Steady-state pitch rate-PID integral preload canceling the airframe's
    # inherent nose-up torque bias (3.2 mm CG offset behind the rotor
    # centroid).  Plant property applied on every reset; value validated
    # open-loop by the hybrid baseline tuner.
    RATE_PID_INTEGRAL_PRELOAD = (0.0, 0.0213, 0.0)
    # conventional controller: hybrid mode-switching cascade autopilot
    # (baselines.py hybrid_cascade; quad collective + tilt at low speed,
    # surfaces + pusher in wing-borne flight)
    conventional_controller = QuadplaneHybridBaseline
    PERTURBABLE_PARAMS = {"thrust": 1.0, "mass": 1.0}
    # Literature: hybrid VTOL UAVs combine multirotor hover propulsion with a
    # forward pusher/prop.  Uncertainty sits between multirotor (+/-20% thrust)
    # and fixed-wing (+/-15% thrust); mass variation from fuel/payload is
    # +/-10%.  We use +/-15% thrust and +/-10% mass.
    DEFAULT_DOMAIN_RANDOMIZE = {"thrust": (0.85, 1.15), "mass": (0.90, 1.10)}

    def __init__(
        self,
        num_envs=64,
        backend="cpu",
        seed=7,
        steady_wind=None,
        dryden=0.0,
        severity="moderate",
        residual=False,
        param_mismatch=None,
        domain_randomize=None,
        warmup_iterations=None,
    ):
        """Initialize the Quadplane VTOL waypoints environment.

        Construct the Genesis-backed hybrid VTOL plant at 40 Hz agent / 120 Hz
        control, allocate task buffers (waypoints, reward, reset, domain-
        randomization counters), and optionally wire in a wind disturbance.
        Finalizes with a reset().
        """
        self.num_envs = num_envs
        self.task_cfg = {
            "vehicle": "quadplane",
            "num_envs": num_envs,
            "backend": backend,
            "implementation": "fully_tensorized",
            "task": "quadplane-waypoints-vtol",
        }
        self.clip_actions = 1.0
        self.plant = TensorPlant(
            "quadplane", num_envs, backend, seed, warmup_iterations=warmup_iterations
        )
        self.device = self.plant.device
        self.agent_hz = 40
        self.inner_steps = int(self.plant.control_hz // self.agent_hz)  # 3
        self.max_steps = 1200
        self.max_episode_length = self.max_steps + 2
        self.num_actions = 8
        # Actions map to physical quantities via _scale_actions.
        # a0-a2: body-rate commands — same convention as quadx/fixedwing.
        # a3:    quad collective thrust [0, 1].
        # a4-a6: surface commands — same aileron/h-tail+v-tail/dead mapping as fixedwing.
        # a7:    forward pusher throttle [0, 1].
        self.action_low = torch.tensor(
            [
                -math.pi,
                -math.pi,
                -math.pi,
                0.0,  # a0..a3
                -1.0,
                -1.0,
                -1.0,
                0.0,
            ],  # a4..a7
            device=self.device,
            dtype=COMPUTE_DTYPE,
        )
        self.action_high = torch.tensor(
            [
                math.pi,
                math.pi,
                math.pi,
                1.0,  # a0..a3
                1.0,
                1.0,
                1.0,
                1.0,
            ],  # a4..a7
            device=self.device,
            dtype=COMPUTE_DTYPE,
        )

        # Buffers
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
        self.targets = torch.zeros(
            (num_envs, NUM_TARGETS, 3), device=self.device, dtype=COMPUTE_DTYPE
        )
        self.target_index = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.leg_suffix = torch.zeros(
            (num_envs, NUM_TARGETS + 1), device=self.device, dtype=COMPUTE_DTYPE
        )
        self.old_distance = torch.full(
            (num_envs,), torch.inf, device=self.device, dtype=COMPUTE_DTYPE
        )
        self.new_distance = torch.full_like(self.old_distance, torch.inf)
        self._observation = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.returns = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self.completed_returns = torch.zeros_like(self.returns)
        self.completed_episodes = torch.zeros_like(self.episode_length_buf)
        self.completed_successes = torch.zeros_like(self.episode_length_buf)
        self.completed_metrics = {
            "waypoints_reached": torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE),
            "crashes": torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE),
            "out_of_bounds": torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE),
            "timeouts": torch.zeros((), device=self.device, dtype=COMPUTE_DTYPE),
        }
        self._seed = seed
        self.extras = dict()
        self.cfg = dict(self.task_cfg)
        self._init_wind(num_envs, seed, steady_wind, dryden, severity)
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

    def _potential(self):
        p = self.plant.state()[0]  # cached: no extra Genesis getter per step
        slots = torch.arange(NUM_TARGETS, device=self.device)[None] + self.target_index[:, None]
        valid = slots < NUM_TARGETS
        indices = slots.clamp_max(NUM_TARGETS - 1)
        targets = self.targets.gather(1, indices[:, :, None].expand(-1, -1, 3))
        d = (targets[:, 0] - p).norm(dim=-1) * valid[:, 0].to(COMPUTE_DTYPE)
        return -(d + self.leg_suffix.gather(1, self.target_index[:, None]).squeeze(1))

    def step(self, actions):
        """Advance one policy step through inner control steps.

        Scale actions, run the plant controller and physics substeps for each
        inner step, detect collisions, waypoint captures and bounds violations,
        accumulate the telescoping potential-based reward, and return
        (observations, reward, terminated, truncated, extras).
        """
        if actions.device != torch.device(self.device):
            raise ValueError(
                "Actions must already be on env.device; implicit host/device copies are disabled"
            )
        actions = self._residual_input(actions)
        physical = self._scale_actions(actions)
        self.action.copy_(physical)
        if self._wind is not None:
            self.plant.wind.copy_(self._wind.step())
        old_potential = self._potential()
        self.count += 1
        crash = torch.zeros_like(self.terminated)
        reach_count = torch.zeros(self.num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        for _ in range(self.inner_steps):
            active = ~(self.terminated | self.truncated)
            # Constant for the whole control step: one sync here lets every
            # substep skip the no-op pause-restore write-back.
            restore = not bool(active.all())
            self.plant.clear_contacts()
            self.plant.controller(physical, active)
            for _ in range(self.plant.physics_per_control):
                self.plant.physics_step(physical, active, restore=restore)
            p, rpy, v, wb, vb = self._observe(active)
            collision = self.plant.ground_contact
            bounds = p.norm(dim=-1) > self.bounds_radius
            reached = self.new_distance < CAPTURE_RADIUS
            self.target_index += (reached & active).to(torch.long)
            complete = self.target_index >= NUM_TARGETS
            term = collision | bounds
            trunc = complete
            crash |= term & active
            reach_count += (reached & active).to(COMPUTE_DTYPE)
            self.terminated |= term & active
            self.truncated |= trunc & active
            self.collision |= collision & active
            self.out_of_bounds |= bounds & active
            self.complete |= complete & active
        active = ~(self.terminated | self.truncated)
        self.truncated |= (self.count >= self.max_steps) & active
        self.reward.copy_(
            self._potential()
            - old_potential
            + REACH_BONUS * reach_count
            - CRASH_PENALTY * crash.to(COMPUTE_DTYPE)
        )
        self.reset_buf = self.terminated | self.truncated
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        self.completed_returns += torch.where(self.reset_buf, self.returns, 0.0)
        self.completed_episodes += self.reset_buf.to(torch.long)
        self.completed_successes += (self.reset_buf & self.complete).to(torch.long)
        done = self.reset_buf
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()

        self.extras["time_outs"] = self.truncated & ~self.terminated & ~self.complete
        self._cache_terminal_observations()
        done_any = bool(done.any())
        if done_any:
            n = done.sum().clamp_min(1)
            crashes = done & self.collision
            oob = done & self.out_of_bounds
            timeouts = done & self.truncated & ~self.complete
            self.extras["episode"] = {
                "success_rate": (done & self.complete).to(COMPUTE_DTYPE).sum() / n,
                "waypoints_reached": self.target_index[done].to(COMPUTE_DTYPE).mean(),
                "crash_rate": crashes.to(COMPUTE_DTYPE).sum() / n,
                "out_of_bounds_rate": oob.to(COMPUTE_DTYPE).sum() / n,
                "timeout_rate": timeouts.to(COMPUTE_DTYPE).sum() / n,
            }
            self.completed_metrics["waypoints_reached"] += (
                self.target_index[done].to(COMPUTE_DTYPE).sum()
            )
            self.completed_metrics["crashes"] += crashes.to(COMPUTE_DTYPE).sum()
            self.completed_metrics["out_of_bounds"] += oob.to(COMPUTE_DTYPE).sum()
            self.completed_metrics["timeouts"] += timeouts.to(COMPUTE_DTYPE).sum()
        else:
            self.extras.pop("episode", None)

        if done_any:
            self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    def get_observations(self):
        """Return the current observation packed in a TensorDict under 'policy'."""
        return self._observations_from_buf()

    def _observe(self, mask):
        """Refresh the internal observation buffer from plant state and waypoints."""
        p, q, v, w = self.plant.state()
        vwb = inverse_rotate(q[:, None, :], torch.stack((v, w), dim=1))
        vb, wb = vwb[:, 0], vwb[:, 1]
        slots = torch.arange(NUM_TARGETS, device=self.device)[None] + self.target_index[:, None]
        valid = slots < NUM_TARGETS
        indices = slots.clamp_max(NUM_TARGETS - 1)
        targets = self.targets.gather(1, indices[:, :, None].expand(-1, -1, 3))
        delta = (
            inverse_rotate(q[:, None, :].expand(-1, NUM_TARGETS, -1), targets - p[:, None, :])
            * valid[:, :, None]
        )
        distance = delta[:, 0].norm(dim=-1)
        self.old_distance.copy_(torch.where(mask, self.new_distance, self.old_distance))
        self.new_distance.copy_(torch.where(mask, distance, self.new_distance))
        attitude = torch.cat(
            (wb, q[:, 1:], q[:, :1], vb, p, self.action, self.plant.auxiliary()), -1
        )
        observation = torch.cat((attitude, delta.flatten(1), valid.to(COMPUTE_DTYPE)), -1)
        if self._observation is None:
            self._observation = observation.clone()
        else:
            self._observation.copy_(torch.where(mask[:, None], observation, self._observation))
        return p, None, v, wb, vb  # rpy unused

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
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask = envs_idx
        if self._domain_randomize:
            self._dr_counter += 1
            apply_domain_randomization(
                self, self._domain_randomize, self._seed, self._dr_counter, mask
            )
        p, q, v, w = self.plant.state()
        # Spawn: near-hover, z=15 m, level flight, ~6 m/s forward speed
        spawn = torch.zeros_like(p)
        spawn[:, 2] = 15.0
        rotation = torch.zeros_like(q)
        rotation[:, 0] = 1
        velocity = torch.zeros_like(v)
        velocity[:, 0] = 6.0  # moderate forward speed (transition regime)
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
        self.action.masked_fill_(mask[:, None], 0)
        self.plant.zero(mask)
        # The airframe has a small inherent nose-up pitch torque bias from the
        # 3.2 mm CG offset behind the rotor centroid.  Pre-load the plant's
        # rate-PID pitch integral with the steady-state compensating value so
        # the bias is canceled from the first step for every controller
        # (traditional and RL alike).  This is a plant property, not a
        # controller decision, so it lives in the env reset, not the baseline.
        preload = torch.tensor(
            self.RATE_PID_INTEGRAL_PRELOAD, device=self.device, dtype=COMPUTE_DTYPE
        )
        self.plant.pid.integral.copy_(
            torch.where(mask[:, None], preload[None, :], self.plant.pid.integral)
        )
        # Waypoint polar sampling, same convention as fixedwing/quadx
        theta = self.plant.rng.uniform(NUM_TARGETS, mask) * 2 * math.pi
        phi = self.plant.rng.uniform(NUM_TARGETS, mask) * 2 * math.pi
        radius = WAYPOINT_MIN_RADIUS + self.plant.rng.uniform(NUM_TARGETS, mask) * (
            WAYPOINT_MAX_RADIUS - WAYPOINT_MIN_RADIUS
        )
        targets = torch.stack(
            (
                radius * torch.sin(phi) * torch.cos(theta),
                radius * torch.sin(phi) * torch.sin(theta),
                (radius * torch.cos(phi)).abs().clamp_min(WAYPOINT_MIN_ALTITUDE),
            ),
            -1,
        )
        self.targets.copy_(torch.where(mask[:, None, None], targets, self.targets))
        legs = (self.targets[:, 1:] - self.targets[:, :-1]).norm(dim=-1)
        ext = torch.cat(
            (legs, torch.zeros((self.num_envs, 2), device=self.device, dtype=COMPUTE_DTYPE)), -1
        )
        suffix = ext.flip(-1).cumsum(-1).flip(-1)
        self.leg_suffix.copy_(torch.where(mask[:, None], suffix, self.leg_suffix))
        self.target_index.masked_fill_(mask, 0)
        self.old_distance.masked_fill_(mask, torch.inf)
        self.new_distance.masked_fill_(mask, torch.inf)
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
