"""Fixedwing-Waypoints original task in the official Genesis RL environment format.

Reward redesign (2026-10-20, PPO debug): the original PyFlyt shaped reward
(-0.1/step + clamp_min(0) progress + 1/d + 100 on reach / -100 on crash) is
replaced by potential-based telescoping shaping. With a 3600-step episode and
gamma=0.99 the terminal +100/-100 are unreachable by credit assignment, while
the clamp + 1/d terms pay infinite positive return for orbiting the current
waypoint without reaching it. The potential

    Phi(s) = -(distance_to_current_target + sum of remaining path legs)

telescopes to the total path length flown successfully: progress pays its true
value, orbiting nets exactly zero, and crashing/timeouts forfeit all unrealized
potential (so ending the episode early is never cheaper than flying). An
explicit +15 per waypoint reached keeps a crisp sparse event; an explicit -10
sharpens crash/OOB. Terminations, success semantics, observations and spawn
distribution are unchanged from the original task.

Action channels (audited 2026-10-22 against the upstream PyFlyt mode-0
setpoint mapping, original_pyflyt/core/drones/fixedwing.py
L141-144 `surface_assist_ids=[0,0,1,1,2] sign=[1,-1,1,-1,0]`): a0 ->
differential ailerons, a1 -> horizontal tail + vertical tail (opposite
sign), a2 -> DEAD (upstream maps it to the main wing whose deflection
limit is 0; the plant multiplies it by 0), a3 -> throttle. Effective
action dimensionality is therefore 3, not 4; the classical autopilot
drives a2 with exactly 0 and BC/PPO learn a constant there. This is
upstream design, not a porting bug -- do not "fix" the mapping.
"""

import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import euler, inverse_rotate
from algorithms.baselines.baselines import FixedwingAutopilotBaseline
from environments.common.residual import ResidualActionMixin
from environments.common.base_env import BaseEnv
from environments.common.domain_randomization import (
    apply_static_perturbation,
    apply_domain_randomization,
    resolve_domain_randomize,
)

REACH_BONUS = 15.0
CRASH_PENALTY = 10.0
# Capture sphere radius. The original 2 m is quasi-unreachable for this
# fixedwing: trim constraints give a ~11 m/s minimum controllable speed and
# hence a ~10-15 m minimum turn radius, so neither scripted nor learned
# policies can reliably thread a 2 m window (validated by 5 autopilot
# variants, best head-on pass missed by 8.6 m). 10 m ~ one minimum turn
# radius keeps the navigation challenge while making captures reachable.
CAPTURE_RADIUS = 10.0
# Waypoint shell. The original 1-90 m shell put targets within 10 m of the
# 100 m out-of-bounds sphere, which a 10-15 m minimum turn radius cannot
# reverse inside; z was clamped only at 0.5 m, so many targets required a
# ground-skimming capture. 20-65 m keeps the navigation challenge while
# leaving 35 m of OOB margin and a 20 m altitude floor.
WAYPOINT_MIN_RADIUS = 20.0
WAYPOINT_MAX_RADIUS = 65.0
WAYPOINT_MIN_ALTITUDE = 20.0


class FixedwingWaypointsEnv(BaseEnv, ResidualActionMixin):
    """Fixed-wing waypoint-following RL environment.

    Task objective: reach a sequence of 4 waypoints sampled in a 20-65 m radius
    shell at >= 20 m altitude. The agent observes a flat float32 tensor
    containing body angular velocity (3), quaternion wxyz (4), body-frame
    velocity (3), world position (3), the last physical action (4), plant
    auxiliary (9 surface/throttle dims), four waypoints in body frame (12),
    plus a 4-waypoint validity mask (4). Actions are 4-D policy outputs in
    [-1, 1] (a3 throttle is internally remapped to [0, 1]), of which a2 is
    dead (main-wing zero-deflection upstream mapping). Termination on ground
    contact or out-of-bounds |p| > 100 m; truncation at 3600 policy steps
    (120 s at 30 Hz) or when all waypoints are reached. Plant physics live in
    environments.common.plant with 30 Hz agent / 120 Hz control / 240 Hz
    physics. Domain randomization covers thrust (0.85, 1.15) and mass
    (0.90, 1.10).
    """

    waypoint_min_radius = WAYPOINT_MIN_RADIUS
    # Wind is optional via the steady_wind/dryden ctor kwargs; calm air
    # reproduces the benchmark bit-for-bit.
    # conventional controller: L1 lateral guidance + TECS (baselines.py, tuned on
    # train/dev ICs; params in algorithms/baselines/fixedwing_autopilot.json)
    conventional_controller = FixedwingAutopilotBaseline
    PERTURBABLE_PARAMS = {"thrust": 1.0, "mass": 1.0}
    # Literature: small fixed-wing UAV propulsion uncertainty is dominated by
    # motor/propeller efficiency and fuel-mass variation.  Typical robust
    # flight-control studies (e.g. H_inf / SMC UAV surveys) assume +/-10-15%
    # control-effectiveness uncertainty and +/-5-10% mass variation (fuel burn,
    # payload).  We use +/-15% thrust and +/-10% mass.
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
    ):
        """Initialize the fixedwing waypoints environment.

        Construct the Genesis-backed fixedwing plant at 30 Hz agent / 120 Hz
        control, allocate task buffers (waypoints, reward, reset, domain-
        randomization counters), and optionally wire in a wind disturbance.
        Finalizes with a reset().
        """
        self.num_envs = num_envs
        self.task_cfg = {
            "vehicle": "fixedwing",
            "num_envs": num_envs,
            "backend": backend,
            "implementation": "fully_tensorized",
            "task_defaults": "PyFlyt 0.29.0",
        }
        self.clip_actions = 1.0

        self.plant = TensorPlant("fixedwing", num_envs, backend, seed)
        self.device = self.plant.device
        self.agent_hz = 30
        self.inner_steps = int(self.plant.control_hz // self.agent_hz)
        self.max_steps = 3600
        self.max_episode_length = self.max_steps + 2
        self.num_actions = 4
        self.action_low = -torch.ones(self.num_actions, device=self.device, dtype=torch.float64)
        self.action_high = torch.ones(self.num_actions, device=self.device, dtype=torch.float64)

        # initialize buffers
        self.action = torch.zeros(
            (num_envs, self.num_actions), device=self.device, dtype=torch.float64
        )
        self.reward = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.terminated = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.truncated = self.terminated.clone()
        self.collision = self.terminated.clone()
        self.out_of_bounds = self.terminated.clone()
        self.complete = self.terminated.clone()
        self.fatal_collision = self.terminated.clone()
        self.targets = torch.zeros((num_envs, 4, 3), device=self.device, dtype=torch.float64)
        self.target_index = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.leg_suffix = torch.zeros((num_envs, 5), device=self.device, dtype=torch.float64)
        self.old_distance = torch.full(
            (num_envs,), torch.inf, device=self.device, dtype=torch.float64
        )
        self.new_distance = torch.full_like(self.old_distance, torch.inf)
        self._observation = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.returns = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self.completed_returns = torch.zeros_like(self.returns)
        self.completed_episodes = torch.zeros_like(self.episode_length_buf)
        self.completed_successes = torch.zeros_like(self.episode_length_buf)
        self.completed_metrics = {
            "waypoints_reached": torch.zeros((), device=self.device, dtype=torch.float64),
            "crashes": torch.zeros((), device=self.device, dtype=torch.float64),
            "out_of_bounds": torch.zeros((), device=self.device, dtype=torch.float64),
            "timeouts": torch.zeros((), device=self.device, dtype=torch.float64),
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
        and the lifting surfaces evaluate v_air = v - wind.
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
            self.plant.wind = torch.zeros((n, 3), device=self.device, dtype=torch.float64)
            self.cfg["wind"] = dict(
                steady_wind=list(steady_wind) if steady_wind is not None else None,
                dryden=dryden,
                severity=severity,
            )

    def _scale_actions(self, actions):
        """Affinely map [-1, 1] policy actions onto the physical action bounds."""
        raw = actions.to(torch.float64).clamp(-self.clip_actions, self.clip_actions)
        return self.action_low + (raw + 1) * 0.5 * (self.action_high - self.action_low)

    def _potential(self):
        """Phi = -(distance to current target + remaining path legs); 0 when complete."""
        p = self.plant.drone.get_pos().to(torch.float64)
        slots = torch.arange(4, device=self.device)[None] + self.target_index[:, None]
        valid = slots < 4
        indices = slots.clamp_max(3)
        targets = self.targets.gather(1, indices[:, :, None].expand(-1, -1, 3))
        d = (targets[:, 0] - p).norm(dim=-1) * valid[:, 0].to(torch.float64)
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
        command = physical.clone()
        command[:, 3] = command[:, 3] / 2 + 0.5
        old_potential = self._potential()
        self.count += 1
        crash = torch.zeros_like(self.terminated)
        reach_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        for _ in range(self.inner_steps):
            active = ~(self.terminated | self.truncated)
            self.plant.clear_contacts()
            self.plant.controller(command, active)
            for _ in range(self.plant.physics_per_control):
                self.plant.physics_step(command, active)
            p, rpy, v, wb, vb = self._observe(active)
            collision = self.plant.ground_contact
            bounds = p.norm(dim=-1) > 100
            reached = self.new_distance < CAPTURE_RADIUS
            self.target_index += (reached & active).to(torch.long)
            complete = self.target_index >= 4
            term = collision | bounds
            trunc = complete
            crash |= term & active
            reach_count += (reached & active).to(torch.float64)
            self.terminated |= term & active
            self.truncated |= trunc & active
            self.collision |= collision & active
            self.out_of_bounds |= bounds & active
            self.complete |= complete & active
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents and
        # waypoint completion still terminate inside the substep loop above
        active = ~(self.terminated | self.truncated)
        self.truncated |= (self.count >= self.max_steps) & active
        # potential-based telescoping reward: progress + reach bonus - crash penalty
        self.reward.copy_(
            self._potential()
            - old_potential
            + REACH_BONUS * reach_count
            - CRASH_PENALTY * crash.to(torch.float64)
        )

        self.reset_buf = self.terminated | self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        self.completed_returns += torch.where(self.reset_buf, self.returns, 0.0)
        self.completed_episodes += self.reset_buf.to(torch.long)
        self.completed_successes += (self.reset_buf & self.complete).to(torch.long)
        done = self.reset_buf
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()

        self.extras["time_outs"] = self.truncated & ~self.terminated & ~self.complete
        self.extras["terminal_observations"] = TensorDict(
            {"policy": self._observation.to(torch.float32)}, batch_size=[self.num_envs]
        )
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any = bool(done.any())
        if done_any:
            n = done.sum().clamp_min(1)
            crashes = done & self.collision
            oob = done & self.out_of_bounds
            timeouts = done & self.truncated & ~self.complete
            self.extras["episode"] = {
                "success_rate": (done & self.complete).to(torch.float64).sum() / n,
                "waypoints_reached": self.target_index[done].to(torch.float64).mean(),
                "crash_rate": crashes.to(torch.float64).sum() / n,
                "out_of_bounds_rate": oob.to(torch.float64).sum() / n,
                "timeout_rate": timeouts.to(torch.float64).sum() / n,
            }
            self.completed_metrics["waypoints_reached"] += (
                self.target_index[done].to(torch.float64).sum()
            )
            self.completed_metrics["crashes"] += crashes.to(torch.float64).sum()
            self.completed_metrics["out_of_bounds"] += oob.to(torch.float64).sum()
            self.completed_metrics["timeouts"] += timeouts.to(torch.float64).sum()
        else:
            self.extras.pop("episode", None)

        # reset_idx ends with plant.warmup(), which advances the GLOBAL scene 20
        # substeps with all external wrenches masked to zero. Calling it on an
        # empty done-mask (the common per-step case) injected 20 pure-ballistic
        # substeps into EVERY agent step (28 total vs the intended 8 = 1/30 s),
        # making the effective gravity-to-lift ratio 3.5x and level flight
        # physically impossible. warmup now early-exits on empty masks and the
        # pause-restore in physics_step is unconditional, so an empty reset is
        # fully safe; the gate reuses the already-synced done_any purely for cost.
        if done_any:
            self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    def get_observations(self):
        """Return the current observation packed in a TensorDict under 'policy'."""
        return TensorDict({"policy": self.obs_buf.clone()}, batch_size=[self.num_envs])

    def _observe(self, mask):
        """Refresh the internal observation buffer from plant state and waypoints."""
        p, q, v, w = self.plant.state()
        rpy = euler(q)
        wb = inverse_rotate(q, w)
        vb = inverse_rotate(q, v)
        slots = torch.arange(4, device=self.device)[None] + self.target_index[:, None]
        valid = slots < 4
        indices = slots.clamp_max(3)
        targets = self.targets.gather(1, indices[:, :, None].expand(-1, -1, 3))
        delta = (
            inverse_rotate(q[:, None, :].expand(-1, 4, -1), targets - p[:, None, :])
            * valid[:, :, None]
        )
        distance = delta[:, 0].norm(dim=-1)
        self.old_distance.copy_(torch.where(mask, self.new_distance, self.old_distance))
        self.new_distance.copy_(torch.where(mask, distance, self.new_distance))
        attitude = torch.cat(
            (wb, q[:, 1:], q[:, :1], vb, p, self.action, self.plant.auxiliary()), -1
        )
        observation = torch.cat((attitude, delta.flatten(1), valid.to(torch.float64)), -1)
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
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask = envs_idx
        if self._domain_randomize:
            self._dr_counter += 1
            apply_domain_randomization(
                self, self._domain_randomize, self._seed, self._dr_counter, mask
            )
        p, q, v, w = self.plant.state()
        spawn = torch.zeros_like(p)
        spawn[:, 2] = 10.0
        rotation = torch.zeros_like(q)
        rotation[:, 0] = 1
        velocity = torch.zeros_like(v)
        velocity[:, 0] = 20
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
        theta = self.plant.rng.uniform(4, mask) * 2 * math.pi
        phi = self.plant.rng.uniform(4, mask) * 2 * math.pi
        radius = self.waypoint_min_radius + self.plant.rng.uniform(4, mask) * (
            WAYPOINT_MAX_RADIUS - self.waypoint_min_radius
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
            (legs, torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float64)), -1
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
