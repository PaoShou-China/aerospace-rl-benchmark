"""QuadX-Waypoints original task in the official Genesis RL environment format.

Task: reach a sequence of 4 waypoints sampled in a 5 m flight dome (spawn at
(0, 0, 1)). Failure (termination): ground contact or |p| > 5 m; success:
reaching all waypoints before the 800-step (20 s) limit. Waypoint geometry
follows the upstream PyFlyt QuadXWaypointsEnv (WaypointHandler): polar
sampling with distance in [1, 0.9*dome] and a floor on target altitude;
capture radius is the upstream goal_reach_distance = 0.2 m.

Reward redesign (mirrors FixedwingWaypointsEnv, 2026-10-23): the upstream
shaped reward (-0.1/step + clamp_min(0) progress + 0.1/d + 100 on reach /
-100 on crash) pays infinite positive return for orbiting a waypoint without
reaching it, and the terminal +-100 are unreachable by credit assignment in a
400-step episode. It is replaced by potential-based telescoping shaping

    Phi(s) = -(distance_to_current_target + sum of remaining path legs)

which telescopes to the total path length flown successfully: progress pays
its true value, orbiting nets exactly zero, and crashing/timeouts forfeit all
unrealized potential. An explicit +15 per waypoint reached keeps a crisp
sparse event; an explicit -10 sharpens crash/OOB.

cfg dicts and buffers in __init__, step() returning
(get_observations(), rew_buf, terminated, truncated, extras), get_observations() wrapping
self.obs_buf in a TensorDict under the 'policy' group, and reset_idx() for
episodic bookkeeping. Rewards, terminations, observations and spawn
distribution are task-local; physics/actuation live in common.plant.
"""

import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import COMPUTE_DTYPE, euler, inverse_rotate, quat_from_euler
from algorithms.baselines.baselines import QuadXSE3Baseline
from environments.common.residual import ResidualActionMixin
from environments.common.base_env import BaseEnv
from environments.common.domain_randomization import (
    apply_static_perturbation,
    apply_staged_domain_randomization,
    resolve_domain_randomize,
    stage_domain_randomization,
)

NUM_TARGETS = 4
REACH_BONUS = 15.0
CRASH_PENALTY = 10.0
# Upstream PyFlyt goal_reach_distance default: the SE(3) baseline converges to
# ~0.02-0.05 m steady error, so a 0.2 m capture sphere is comfortably
# reachable for both scripted and learned policies.
CAPTURE_RADIUS = 0.2
# Waypoint shell: upstream WaypointHandler samples distance uniform in
# [1, 0.9*dome] with dome = 5 m. 2026-10-24: the benchmark shell was widened
# to [1.5, 4.8] m (the SE(3) baseline sat at 100% success, so the task was
# made slightly harder: no trivially close targets, and far targets push the
# capture sphere to 0.2 m from the out-of-bounds wall). Altitude floor:
# upstream uses 0.1 m, raised to 0.4 m so a capture pass cannot clip the
# ground-contact termination.
WAYPOINT_MIN_RADIUS = 1.5
WAYPOINT_MAX_RADIUS = 4.8
WAYPOINT_MIN_HEIGHT = 0.4
EPISODE_KEYS = (
    "success_rate",
    "waypoints_reached",
    "crash_rate",
    "out_of_bounds_rate",
    "timeout_rate",
)


class QuadXWaypointsEnv(BaseEnv, ResidualActionMixin):
    """QuadX multirotor waypoint-following RL environment.

    Task objective: reach a sequence of 4 waypoints sampled in a 5 m flight dome.
    The agent observes a flat float32 tensor containing body angular velocity (3),
    quaternion wxyz (4), body-frame velocity (3), world position (3), the last
    physical action (4), per-rotor throttle (4), and four waypoints expressed
    in body frame (12) plus a 4-waypoint validity mask (4). Actions are 4-D:
    roll, pitch and yaw rate commands in [-pi, pi] rad/s and collective throttle
    [0, 1], mapped affinely from the raw [-1, 1] policy output. Termination
    occurs on ground contact or out-of-bounds (|p| > 5 m); truncation at 800
    policy steps (20 s at 40 Hz) or when all waypoints are reached. The vehicle
    is a 27 g Crazyflie-class quadrotor; plant physics and rate-PID control
    live in environments.common.plant. Domain randomization covers thrust
    (0.80, 1.20) and mass (0.85, 1.15).
    """

    # Spawn-noise attrs are manual domain-randomization knobs (all zero in the
    # benchmark, where the only reset randomness is the waypoint draw). Wind is
    # optional via the steady_wind/dryden ctor kwargs; calm air with a fixed
    # seed reproduces a given waypoint layout bit-for-bit.
    spawn_xy_noise = 0.0  # spawn xy-offset radius [m] (manual DR knob)
    spawn_z_noise = 0.0  # spawn z-offset half-range [m] (manual DR knob)
    spawn_vel_noise = 0.0  # spawn |v0| per-axis half-range [m/s] (manual DR knob)
    spawn_att_noise = 0.0  # spawn per-axis euler half-range [rad] (manual DR knob)
    bounds_radius = 5.0  # out-of-bounds radius [m] (= flight dome)
    # conventional controller: SE(3) geometric waypoint tracking, Lee et al.
    # 2010 (baselines.py; reuses the env's rate+collective action convention)
    conventional_controller = QuadXSE3Baseline
    # plant parameters that can be mismatched / randomized. The controller
    # keeps its hardcoded nominal values (MASS=0.027, TOTAL_THRUST=2.0, ...),
    # so perturbing the plant below creates a real model mismatch.
    PERTURBABLE_PARAMS = {"thrust": 1.0, "mass": 1.0}
    # Default per-episode domain-randomization ranges (scale factors applied to
    # the plant parameter; controller keeps nominal).  Each replica samples its
    # own scale independently on every reset.
    # Literature: Kaufmann et al. "Demonstrating Agile Flight from Pixels"
    # (2024) randomizes thrust +/-20%, inertia +/-20%, mass +/-5% for agile
    # quadrotors; Romero et al. "ProxFly" (2024) uses +/-20% mass for small
    # quadcopters; "Sim-to-(Multi)-Real" (Tobin-style) randomizes around
    # Crazyflie 2.0 nominal values.  We take +/-15% mass (payload/battery)
    # and +/-20% thrust (motor efficiency, battery sag, prop variation).
    DEFAULT_DOMAIN_RANDOMIZE = {"thrust": (0.80, 1.20), "mass": (0.85, 1.15)}

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
        """Initialize the QuadX waypoints environment.

        Construct the Genesis-backed quadrotor plant at 40 Hz agent / 120 Hz
        control, allocate task buffers (waypoints, reward, reset, domain-
        randomization counters), and optionally wire in a wind disturbance.
        Finalizes with a reset().
        """
        self.num_envs = num_envs
        self.task_cfg = {
            "vehicle": "quadx",
            "num_envs": num_envs,
            "backend": backend,
            "implementation": "fully_tensorized",
            "task_defaults": "PyFlyt 0.29.0",
        }
        # Raw policy actions live in [-1, 1] and are mapped onto the original
        # physical action bounds by _scale_actions, mirroring the official
        # clip_actions convention.
        self.clip_actions = 1.0

        # vel_clamp guards the rigid contact solver: exploration bursts (32-step
        # correlated max-rate commands) can spin the quad to |w| in the hundreds
        # of rad/s and NaN the Genesis solver on propeller-ground impact
        # ("invalid accelerations"). Caps sit far above nominal trajectories
        # (waypoint flight |v|<2 m/s, |w|<10 rad/s), so only already-unphysical
        # crash states are affected. Disabled (None) unless this task opts in.
        self.plant = TensorPlant(
            "quadx", num_envs, backend, seed, vel_clamp=(15.0, 30.0),
            warmup_iterations=warmup_iterations,
        )
        self.device = self.plant.device
        self.agent_hz = 40
        self.inner_steps = int(self.plant.control_hz // self.agent_hz)
        self.max_steps = 800
        self.max_episode_length = self.max_steps + 2
        self.num_actions = 4
        self.action_low = torch.tensor(
            [-math.pi, -math.pi, -math.pi, 0.0], device=self.device, dtype=COMPUTE_DTYPE
        )
        self.action_high = torch.tensor(
            [math.pi, math.pi, math.pi, 0.8], device=self.device, dtype=COMPUTE_DTYPE
        )

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
        # per-step accumulators (persistent so a step segment updates them in place)
        self._old_potential = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self._crash = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._reach_count = torch.zeros(num_envs, device=self.device, dtype=COMPUTE_DTYPE)
        self._observation = None
        self.obs_buf = None
        self._slots_arange = None
        self._terminal_obs = None
        self._terminal_obs_td = None
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
        """Optional wind disturbance (environments/common/disturbances.py).

        Calm air (default) leaves the plant dynamics bit-for-bit unchanged;
        with wind configured, the gust filters advance once per control step
        and the plant aerodynamic terms evaluate v_air = v - wind.
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

    @property
    def current_target(self):
        """The active waypoint (n, 3); the last one once all are reached."""
        indices = self.target_index.clamp_max(NUM_TARGETS - 1)[:, None, None].expand(-1, -1, 3)
        return self.targets.gather(1, indices)[:, 0]

    def _potential(self):
        """Phi = -(distance to current target + remaining path legs); 0 when complete."""
        p = self.plant.state()[0]  # cached: no extra Genesis getter per step
        if self._slots_arange is None:
            self._slots_arange = torch.arange(NUM_TARGETS, device=self.device)
        slots = self._slots_arange[None] + self.target_index[:, None]
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
            # Constant for the whole control step: one sync here lets every
            # substep skip the no-op pause-restore write-back. Every replica is
            # active when a step starts (each step ends by resetting all done
            # replicas), so the first control step needs no sync.
            restore = i > 0 and not bool((~(self.terminated | self.truncated)).all())
            self._control_step(restore)
        term_out, trunc_out, done, time_outs = self._step_end()
        # Fresh tensors per step, as before: callers may keep them.
        term_out, trunc_out = term_out.clone(), trunc_out.clone()
        self.reset_buf = done.clone()
        self.extras["time_outs"] = time_outs.clone()
        self.extras["terminal_observations"] = self._terminal_obs_td
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any = bool(self.reset_buf.any())
        if done_any:
            self.extras["episode"] = dict(
                zip(EPISODE_KEYS, (x.clone() for x in self._episode_stats(self.reset_buf)))
            )
            # reset_idx on an empty mask is a no-op for every buffer and
            # plant.warmup early-exits on empty masks (see plant.warmup), so
            # gating it with the already-synced done_any is purely a cost guard.
            self.reset_idx(self.reset_buf, nonempty=True)
        else:
            self.extras.pop("episode", None)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    def _step_begin(self, actions):
        """Map the action onto the plant and open the step's accumulators."""
        physical = self._scale_actions(actions)
        self.action.copy_(physical)
        if self._wind is not None:
            self.plant.wind.copy_(self._wind.step())
        self._old_potential.copy_(self._potential())
        self.count += 1
        self._crash.zero_()
        self._reach_count.zero_()

    def _control_step(self, restore):
        """One control step: mixer, physics substeps, observation, events."""
        active = ~(self.terminated | self.truncated)
        self.plant.clear_contacts()
        self.plant.controller(self.action, active)
        for _ in range(self.plant.physics_per_control):
            self.plant.physics_step(self.action, active, restore=restore)
        p, _, v, wb, vb = self._observe(active)
        collision = self.plant.ground_contact
        bounds = p.norm(dim=-1) > self.bounds_radius
        reached = self.new_distance < CAPTURE_RADIUS
        self.target_index += (reached & active).to(torch.long)
        complete = self.target_index >= NUM_TARGETS
        term = collision | bounds
        trunc = complete
        self._crash |= term & active
        self._reach_count += (reached & active).to(COMPUTE_DTYPE)
        self.terminated |= term & active
        self.truncated |= trunc & active
        self.collision |= collision & active
        self.out_of_bounds |= bounds & active
        self.complete |= complete & active

    def _step_end(self):
        """Time limit, reward and episode bookkeeping; returns step outputs."""
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents and
        # waypoint completion still terminate inside the substep loop above
        active = ~(self.terminated | self.truncated)
        self.truncated |= (self.count >= self.max_steps) & active
        # potential-based telescoping reward: progress + reach bonus - crash penalty
        self.reward.copy_(
            self._potential()
            - self._old_potential
            + REACH_BONUS * self._reach_count
            - CRASH_PENALTY * self._crash.to(COMPUTE_DTYPE)
        )
        done = self.terminated | self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        self.completed_returns += torch.where(done, self.returns, 0.0)
        self.completed_episodes += done.to(torch.long)
        self.completed_successes += (done & self.complete).to(torch.long)
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()
        # 'complete' marks eval success (all waypoints reached) -- it is NOT an
        # MDP termination in the bootstrap sense; reaching the last waypoint
        # ends the episode as a truncation so off-policy critics bootstrap from
        # it, only true terminations (crash / out-of-bounds) cut the bootstrap.
        time_outs = self.truncated & ~self.terminated & ~self.complete
        self._cache_terminal_observations()
        return term_out, trunc_out, done, time_outs

    def _episode_stats(self, done):
        """Episode summary rates over the finishing replicas (``done`` non-empty).

        Masked sums equal the boolean-indexed sums exactly (integer counts in
        float32), without the data-dependent shapes of indexing.
        """
        n = done.sum().clamp_min(1)
        crashes = done & self.collision
        oob = done & self.out_of_bounds
        timeouts = done & self.truncated & ~self.complete
        reached = torch.where(done, self.target_index, 0).to(COMPUTE_DTYPE).sum()
        self.completed_metrics["waypoints_reached"] += reached
        self.completed_metrics["crashes"] += crashes.to(COMPUTE_DTYPE).sum()
        self.completed_metrics["out_of_bounds"] += oob.to(COMPUTE_DTYPE).sum()
        self.completed_metrics["timeouts"] += timeouts.to(COMPUTE_DTYPE).sum()
        return (
            (done & self.complete).to(COMPUTE_DTYPE).sum() / n,
            reached / n,
            crashes.to(COMPUTE_DTYPE).sum() / n,
            oob.to(COMPUTE_DTYPE).sum() / n,
            timeouts.to(COMPUTE_DTYPE).sum() / n,
        )

    def get_observations(self):
        """Return the current observation packed in a TensorDict under 'policy'."""
        return self._observations_from_buf()

    def _observe(self, mask):
        """Refresh the internal observation buffer from plant state and waypoints."""
        p, q, v, w = self.plant.state()
        # euler(q) is intentionally not computed: no consumer of this env's
        # _observe reads rpy, and it cost an atan2/asin chain per inner step.
        vwb = inverse_rotate(q[:, None, :], torch.stack((v, w), dim=1))
        vb, wb = vwb[:, 0], vwb[:, 1]
        if self._slots_arange is None:
            self._slots_arange = torch.arange(NUM_TARGETS, device=self.device)
        slots = self._slots_arange[None] + self.target_index[:, None]
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
        return p, None, v, wb, vb

    def _update_observation(self):
        """Copy the float64 internal observation into the float32 obs_buf used downstream."""
        if self.obs_buf is None:
            self.obs_buf = torch.zeros(
                self._observation.shape, device=self.device, dtype=torch.float32
            )
        self.obs_buf.copy_(self._observation)

    def _apply_perturbation(self, name, scale):
        """Scale a live plant parameter; called by the DR helper."""
        if name == "thrust":
            self.plant.perturb_thrust(scale)
        elif name == "mass":
            self.plant.perturb_mass(scale)
        else:
            raise KeyError(name)

    def reset_idx(self, envs_idx, nonempty=None):
        """Reset selected replicas; envs_idx is a device-resident boolean mask.

        ``nonempty`` (optional) states whether the mask selects any replica,
        sparing the reset warmup its host read when the caller already knows.
        """
        # domain randomization: resample plant params for the resetting replicas
        # (host-side draw here; applied to the masked replicas on device below)
        if self._domain_randomize:
            self._dr_counter += 1
            stage_domain_randomization(self, self._domain_randomize, self._seed, self._dr_counter)
        if nonempty is None:
            nonempty = bool(envs_idx.any())
        self._reset_replicas(envs_idx, nonempty)

    def _reset_replicas(self, mask, nonempty):
        """Device half of reset_idx (graph-capturable for a fixed ``nonempty``)."""
        if self._domain_randomize:
            apply_staged_domain_randomization(self, mask)
        p, q, v, w = self.plant.state()
        spawn = torch.zeros_like(p)
        spawn[:, 2] = 1.0
        rotation = torch.zeros_like(q)
        rotation[:, 0] = 1
        velocity = torch.zeros_like(v)
        omega = torch.zeros_like(w)
        # spawn noise attrs are manual domain-randomization knobs (all zero in
        # the benchmark); the waypoint draw below is the only reset randomness
        if self.spawn_xy_noise > 0.0 or self.spawn_z_noise > 0.0:
            u = self.plant.rng.uniform(3, mask)
            rad = self.spawn_xy_noise * u[:, 0]
            ang = 2 * math.pi * u[:, 1]
            off = torch.stack(
                (rad * ang.cos(), rad * ang.sin(), self.spawn_z_noise * (2 * u[:, 2] - 1.0)), -1
            )
            spawn = spawn + torch.where(mask[:, None], off, 0.0)
        if self.spawn_vel_noise > 0.0:
            u = self.plant.rng.uniform(3, mask)
            velocity = velocity + torch.where(
                mask[:, None], self.spawn_vel_noise * (2 * u - 1.0), 0.0
            )
        if self.spawn_att_noise > 0.0:
            u = self.plant.rng.uniform(3, mask)
            rot = quat_from_euler(self.spawn_att_noise * (2 * u - 1.0))
            rotation = torch.where(mask[:, None], rot, rotation)
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
        # upstream WaypointHandler polar sampling: distance in [1, 0.9*dome],
        # altitude |r cos(phi)| floored at WAYPOINT_MIN_HEIGHT
        theta = self.plant.rng.uniform(NUM_TARGETS, mask) * 2 * math.pi
        phi = self.plant.rng.uniform(NUM_TARGETS, mask) * 2 * math.pi
        radius = WAYPOINT_MIN_RADIUS + self.plant.rng.uniform(NUM_TARGETS, mask) * (
            WAYPOINT_MAX_RADIUS - WAYPOINT_MIN_RADIUS
        )
        targets = torch.stack(
            (
                radius * torch.sin(phi) * torch.cos(theta),
                radius * torch.sin(phi) * torch.sin(theta),
                (radius * torch.cos(phi)).abs().clamp_min(WAYPOINT_MIN_HEIGHT),
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
        self.plant.warmup(mask, nonempty=nonempty)
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
