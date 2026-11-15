"""Residual RL around a conventional controller, outside the task environment.

Why a wrapper
-------------
The task environments are plain batched tensor environments: they own neither a
controller instance nor a ``nominal_action`` field in ``extras``.  Residual RL
is therefore implemented as an ownership layer between the five-tuple env and
``RslVecEnv``::

    policy sample u_t  ->  residual r_t = limit * tanh(u_t)
    nominal   a_t^0    =  pi_theta(s_t)          (controller, ONCE per step)
    command   a_t^cmd  =  a_t^0 + r_t            (native normalized action space)
    applied   a_t      =  clamp(a_t^cmd, -c, c)  (env's own action bound)

The policy's log probability is computed on ``u_t`` (the sampled variable), not
on ``a_t``; the composition is a deterministic action map, and the ratio used by
PPO stays ``pi_new(u)/pi_old(u)``.  No tanh Jacobian is needed for that ratio;
the tanh only bounds the *environment* perturbation.  The environment's own
magnitude/mixer/actuator handling is untouched.

Controller state ownership
--------------------------
``controller.actions(obs)`` is called exactly once per vectorized control step
(inside ``step``), on exactly the observation the policy acted on.  After the
environment step, ``observe_applied_action(applied)`` gives stateful
controllers the action that was actually delivered (e.g. docking's fuel
bookkeeping assumes applied == command), and only then is ``reset_idx(term |
trunc)`` applied.  That ordering is the documented contract:

    observe_applied_action(a_t)      # old episode, actual action
    controller.reset_idx(done_t)     # new episode for done replicas
    cache next observation

Controllers that are purely observation-driven implement the default no-op
``observe_applied_action``.  Counters are advanced exactly once per step; a
controller is never called twice for logging or observation augmentation.

Diagnostics
-----------
Every step records, on device and without host synchronisation: policy sample,
requested residual, composed command, applied action, saturation of the residual
and of the composed command, and the fraction of the residual opposing the
nominal action (cancellation).  ``stats_snapshot()`` returns a JSON-friendly
summary for the run report.

Action channels are task-specific and are recorded in the config rather than
assumed universal: docking = normalized per-axis thrust [N], satellite =
normalized gimbal rates [rad/s], rendezvous = normalized per-axis thrust [N].
"""

import math

import torch

__all__ = [
    "ResidualVecEnv",
    "PolicyBaseController",
    "zero_init_policy_output",
    "success_gated_authority_curriculum",
]


def _as_channel_vector(value, num_actions, device, dtype=torch.float32):
    if isinstance(value, (int, float)):
        return torch.full((num_actions,), float(value), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).flatten()
    if tensor.numel() != num_actions:
        raise ValueError(f"expected {num_actions} per-channel values, got {tensor.numel()}")
    return tensor


class PolicyBaseController:
    """A frozen deterministic policy exposed as the residual base controller.

    Used by the ``frozen_bc_residual_ppo`` arm: the base policy is the BC actor
    (frozen), so the residual learns corrections on top of the distilled
    controller instead of on top of the analytical one.  It implements the same
    controller protocol as ``algorithms/baselines/baselines.Baseline``.
    """

    is_frozen_policy = True

    def __init__(self, policy, device, action_bound=1.0, name="frozen_bc"):
        self.policy = policy
        self.device = device
        self.action_bound = float(action_bound)
        self.name = name
        self.num_actions = None  # resolved on the first call

    def config(self):
        """Return a JSON-serializable dict describing this base controller."""
        return {
            "name": self.name,
            "type": "frozen_deterministic_policy",
            "action_bound": self.action_bound,
        }

    @torch.no_grad()
    def actions(self, obs):
        """Return the frozen policy's deterministic action on ``obs`` (float32, clamped)."""
        if isinstance(obs, torch.Tensor):
            from tensordict import TensorDict

            n = obs.shape[0]
            obs = TensorDict({"policy": obs}, batch_size=[n])
        action = self.policy(obs)
        action = action.to(torch.float32)
        if self.num_actions is None:
            self.num_actions = int(action.shape[-1])
        bound = self.action_bound
        if not torch.isfinite(action).all():
            raise ValueError("frozen base policy produced a non-finite action")
        # A distilled BC actor can overshoot the analytical controller's range;
        # the env clips the composed command anyway, so clip the base here too
        # instead of rejecting an otherwise usable initialization.
        return action.clamp(-bound, bound)

    def reset_idx(self, mask):
        """Stateless: the frozen policy has no recurrent / integration state."""

    def observe_applied_action(self, action):
        """Stateless."""


class ResidualVecEnv:
    """Five-tuple vector env wrapper adding a conventional controller as base.

    Parameters
    ----------
    env
        The task environment (``step`` returns obs, rew, terminated, truncated,
        extras, all on ``env.device``).  ``residual`` must be left False on the
        env itself; this wrapper is the only composition path used by the
        experiments.
    controller
        A ``Baseline``-compatible object: ``actions(obs)``, ``reset_idx(mask)``
        and optionally ``observe_applied_action(applied)``.
    residual_limit / residual_rho
        Per-channel authority.  ``residual_limit`` is the bound on ``|r_i|`` in
        normalized action units; ``residual_rho`` is the fraction of the channel
        range, ``rho_i = limit_i / (upper_i - lower_i)``.  Provide exactly one.
    residual_squash
        ``'tanh'`` (bounded, zero-at-zero) or ``'identity'`` (raw sample scaled
        by the limit).  Both keep PPO's ratio on the sampled variable.
    """

    def __init__(
        self,
        env,
        controller,
        *,
        variant=None,
        residual_limit=None,
        residual_rho=None,
        residual_squash="tanh",
        task=None,
        controller_design=None,
        teacher_identity=None,
        name="residual_ppo",
    ):
        if getattr(env, "residual", False):
            raise ValueError(
                "env.residual is enabled; the experiments compose the "
                "controller outside the env (ResidualVecEnv)"
            )
        if residual_limit is not None and residual_rho is not None:
            raise ValueError("provide residual_limit OR residual_rho, not both")
        if residual_limit is None and residual_rho is None:
            raise ValueError("residual authority is required (limit or rho)")
        if residual_squash not in ("tanh", "identity"):
            raise ValueError(f"unknown residual squash {residual_squash!r}")
        # Per-channel / phase-gated authority: residual_rho may be a dict
        # {'channels': [...], 'gate': {'index': ...}} instead of a scalar/list.
        gate_cfg = None
        if isinstance(residual_rho, dict):
            rho_spec = dict(residual_rho)
            gate_cfg = dict(rho_spec.get("gate") or {}) or None
            residual_rho = rho_spec.get("channels")
            if residual_rho is None:
                raise ValueError("dict residual_rho needs a 'channels' entry")
        d = getattr(env, "device", "cpu")
        n, a = int(env.num_envs), int(env.num_actions)
        bound = float(getattr(env, "clip_actions", 1.0))
        self.__dict__["env"] = env
        self.controller = controller
        self.task = task
        self.variant = variant
        self.controller_design = dict(controller_design or {})
        self.teacher_identity = teacher_identity or getattr(
            controller, "name", type(controller).__name__
        )
        self.name = name
        self.num_envs, self.num_actions = n, a
        self.action_lower = -bound
        self.action_upper = bound
        width = 2.0 * bound
        if residual_limit is not None:
            limits = _as_channel_vector(residual_limit, a, d)
            rho = limits / width
        else:
            rho = _as_channel_vector(residual_rho, a, d)
            limits = rho * width
        if float(limits.min()) <= 0.0:
            raise ValueError("residual limits must be strictly positive")
        if torch.any(limits > 0.5 * width + 1e-12):
            raise ValueError(
                "residual limit cannot exceed half the action range "
                "(the controller itself already spans the full bounded space)"
            )
        self.residual_limit = limits.to(torch.float32)
        self.residual_rho = rho.to(torch.float32)
        self.residual_squash = residual_squash
        # optional observation-driven phase gate: effective_limit_i =
        # limit_i * authority_fraction * (1 + gain * sigmoid((near - phase)/soft))
        # where phase = ||obs[index]||.  With gain=0 the gate is inert.
        self.phase_gate = gate_cfg
        self.authority_fraction = 1.0
        self._width = width
        self.last_authority_scale = 1.0
        if gate_cfg is not None:
            for key in ("index", "near"):
                if key not in gate_cfg:
                    raise ValueError(f"phase gate needs {key!r}: {gate_cfg}")
        self.action_lower_t = torch.full((a,), self.action_lower, device=d)
        self.action_upper_t = torch.full((a,), self.action_upper, device=d)

        # controller state starts from the current (freshly constructed) env state
        controller.reset_idx(torch.ones(n, dtype=torch.bool, device=d))
        self._obs = env.get_observations()
        self._steps = 0
        self._first_check = True
        # Host-device syncs (float()/bool() on GPU tensors) stall the CPU until
        # the whole GPU queue drains; doing them every step made residual arms
        # CPU-bound.  Validate on the first step and periodically thereafter.
        self._validate_every = 100
        self._stats = self._new_stats(d)

    # ------------------------------------------------------------------ stats
    def _new_stats(self, device):
        def z():
            return torch.zeros((), device=device, dtype=torch.float32)

        def per_channel():
            return torch.zeros(self.num_actions, device=device, dtype=torch.float32)

        return {
            "steps": z(),
            "residual_abs_sum": z(),
            "residual_abs_max": z(),
            "residual_sq_sum": z(),
            "nominal_abs_sum": z(),
            "composed_abs_sum": z(),
            "applied_abs_sum": z(),
            "composed_clip_envsteps": z(),
            "residual_clip_envsteps": z(),
            "cancellation_envsteps": z(),
            "residual_active_envsteps": z(),
            "nominal_sat_envsteps": z(),
            "residual_abs_sum_channel": per_channel(),
            "applied_abs_sum_channel": per_channel(),
            "nominal_abs_sum_channel": per_channel(),
        }

    def _record(self, nominal, residual, composed, applied):
        s = self._stats
        s["steps"] += 1
        r_abs = residual.abs()
        s["residual_abs_sum"] += r_abs.sum()
        s["residual_abs_max"] = torch.maximum(s["residual_abs_max"], r_abs.max())
        s["residual_sq_sum"] += (residual**2).sum()
        s["nominal_abs_sum"] += nominal.abs().sum()
        s["composed_abs_sum"] += composed.abs().sum()
        s["applied_abs_sum"] += applied.abs().sum()
        s["residual_abs_sum_channel"] += r_abs.sum(0).to(torch.float32)
        s["applied_abs_sum_channel"] += applied.abs().sum(0).to(torch.float32)
        s["nominal_abs_sum_channel"] += nominal.abs().sum(0).to(torch.float32)
        clipped = composed.abs() > (self.action_upper + 1e-6)
        s["composed_clip_envsteps"] += clipped.any(-1).sum().to(torch.float32)
        at_limit = r_abs > (self.residual_limit.abs().to(r_abs.dtype) - 1e-4)
        s["residual_clip_envsteps"] += at_limit.any(-1).sum().to(torch.float32)
        # residual opposing the nominal action on at least one channel
        cancelling = (residual * nominal).sum(-1) < 0
        s["cancellation_envsteps"] += cancelling.sum().to(torch.float32)
        s["residual_active_envsteps"] += (r_abs > 1e-6).any(-1).sum().to(torch.float32)
        s["nominal_sat_envsteps"] += (
            (nominal.abs() >= self.action_upper - 1e-6).any(-1).sum().to(torch.float32)
        )

    def stats_snapshot(self):
        """Return a dict summarizing residual/controller activity since construction.

        All fractions are normalized by total environment steps.  Per-channel
        values list all ``num_actions`` channels in action-order.
        """
        s = self._stats
        steps = max(float(s["steps"]), 1.0)
        envsteps = steps * self.num_envs

        def per(v):
            return [float(x) for x in v.detach().cpu()]

        return {
            "steps": int(s["steps"]),
            "envsteps": int(envsteps),
            "mean_abs_residual": float(s["residual_abs_sum"]) / envsteps,
            "rms_residual": math.sqrt(float(s["residual_sq_sum"]) / envsteps),
            "max_abs_residual": float(s["residual_abs_max"]),
            "mean_abs_nominal": float(s["nominal_abs_sum"]) / envsteps,
            "mean_abs_composed": float(s["composed_abs_sum"]) / envsteps,
            "mean_abs_applied": float(s["applied_abs_sum"]) / envsteps,
            "composed_clip_fraction": float(s["composed_clip_envsteps"]) / envsteps,
            "residual_at_limit_fraction": float(s["residual_clip_envsteps"]) / envsteps,
            "cancellation_fraction": float(s["cancellation_envsteps"]) / envsteps,
            "residual_active_fraction": float(s["residual_active_envsteps"]) / envsteps,
            "nominal_saturated_fraction": float(s["nominal_sat_envsteps"]) / envsteps,
            "mean_abs_residual_channel": [x / envsteps for x in per(s["residual_abs_sum_channel"])],
            "mean_abs_applied_channel": [x / envsteps for x in per(s["applied_abs_sum_channel"])],
            "mean_abs_nominal_channel": [x / envsteps for x in per(s["nominal_abs_sum_channel"])],
            "residual_limit": [float(x) for x in self.residual_limit],
            "residual_rho": [float(x) for x in self.residual_rho],
            "residual_squash": self.residual_squash,
            "authority_fraction": float(self.authority_fraction),
            "last_authority_scale": float(self.last_authority_scale),
            "phase_gate": dict(self.phase_gate) if self.phase_gate else None,
        }

    # ------------------------------------------------------------- authority
    def set_authority_fraction(self, fraction):
        """Scale the residual authority (success-gated curriculum hook target)."""
        fraction = float(fraction)
        if not 0.0 <= fraction:
            raise ValueError("authority fraction must be non-negative")
        self.authority_fraction = fraction
        return fraction

    def _authority_scale(self, dtype):
        """Per-channel multiplicative authority (fraction x phase gate)."""
        scale = torch.full(
            (self.num_actions,), float(self.authority_fraction), device=self.env.device, dtype=dtype
        )
        gate = self.phase_gate
        if gate is not None:
            idx = list(gate["index"])
            phase = self._obs["policy"][:, idx].to(torch.float32).norm(dim=-1)
            soft = float(gate.get("soft", 0.1))
            g = torch.sigmoid((float(gate["near"]) - phase) / max(soft, 1e-6))
            gain = float(gate.get("gain", 1.0))
            scale = scale * (1.0 + gain * g).unsqueeze(-1)
        self.last_authority_scale = scale.mean()  # stay on device; no host sync
        return scale

    def physics_limits(self):
        """Per-channel residual authority in the task's physical units (if known)."""
        lim = [float(x) for x in self.residual_limit]
        task = (self.task or "").lower()
        env = self.env
        # Nominal design constants first: domain randomization rewrites the
        # live ``env.mass`` / ``env.f_max`` as per-env tensors, which cannot be
        # reported as a single scalar.
        design = getattr(env, "controller_design", None) or {}

        def _scalar(name):
            value = design.get(name, getattr(env, name, 0.0))
            if torch.is_tensor(value):
                flat = value.reshape(-1)
                if flat.numel() != 1:
                    raise ValueError(
                        f"physics_limits: {name!r} is per-env "
                        f"{tuple(value.shape)}; env.controller_design has no "
                        f"nominal value for it"
                    )
                return float(flat.item())
            return float(value)

        if task in ("docking", "rendezvous"):
            f_max = _scalar("f_max")
            return {
                "channel": "normalized per-axis thrust (action = F/f_max)",
                "residual_limit_normalized": lim,
                "residual_limit_thrust_N": [x * f_max for x in lim],
            }
        if task == "satellite":
            scale = _scalar("action_scale")
            return {
                "channel": "normalized gimbal rate (action = rate/action_scale)",
                "residual_limit_normalized": lim,
                "residual_limit_gimbal_rate_rad_s": [x * scale for x in lim],
            }
        return {"channel": "native normalized action", "residual_limit_normalized": lim}

    # --------------------------------------------------------------- env API
    def get_observations(self):
        """Pull and cache the latest observations from the wrapped env."""
        self._obs = self.env.get_observations()
        return self._obs

    def _nominal(self):
        nominal = self.controller.actions(self._obs)
        nominal = nominal.to(torch.float32)
        if nominal.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"controller action shape {tuple(nominal.shape)} != "
                f"({self.num_envs}, {self.num_actions})"
            )
        if self._first_check or self._steps % self._validate_every == 0:
            if not torch.isfinite(nominal).all():
                raise ValueError("conventional controller produced a non-finite action")
            if float(nominal.abs().max()) > self.action_upper + 1e-6:
                raise ValueError("conventional controller action violates the env bound")
        return nominal

    def step(self, policy_action):
        """Compose policy action with the controller, step the env, and record stats.

        Contract: exactly one controller call per step; ``observe_applied_action``
        runs on the OLD episode, then ``reset_idx`` on the done mask, then the
        new observation is cached.  Returns the five-tuple from the wrapped env.
        """
        policy_action = policy_action.to(self.env.device)
        if policy_action.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"policy action shape {tuple(policy_action.shape)} != "
                f"({self.num_envs}, {self.num_actions})"
            )
        if (self._first_check or self._steps % self._validate_every == 0) and not torch.isfinite(
            policy_action
        ).all():
            raise ValueError("policy produced a non-finite action (context saved before step)")
        nominal = self._nominal()  # exactly once per step
        limit = self.residual_limit.to(policy_action.dtype) * self._authority_scale(
            policy_action.dtype
        ).to(policy_action.dtype)
        limit = limit.clamp_max(0.5 * self._width)
        if self.residual_squash == "tanh":
            residual = torch.tanh(policy_action) * limit
        else:
            residual = policy_action.clamp(-1.0, 1.0) * limit
        composed = nominal + residual
        applied = composed.clamp(self.action_lower, self.action_upper)
        self._record(nominal, residual, composed, applied)
        self.last_policy_action = policy_action
        self.last_nominal = nominal
        self.last_residual = residual
        self.last_composed = composed
        self.last_applied = applied
        obs, rew, term, trunc, extras = self.env.step(applied)
        done = term | trunc
        # actual-action bookkeeping for the OLD episode, then reset the new one
        observe = getattr(self.controller, "observe_applied_action", None)
        if callable(observe):
            observe(applied)
        self.controller.reset_idx(done)
        self._obs = obs
        self._steps += 1
        self._first_check = False
        return obs, rew, term, trunc, extras

    def reset(self):
        """Reset the wrapped env and reinitialize the controller on all replicas."""
        self._obs = self.env.reset()
        self.controller.reset_idx(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
        )
        return self._obs

    def close(self):
        """Close the wrapped environment."""
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def _find_residual(env, _seen=None):
    """Return the ResidualVecEnv inside a wrapper chain (or None)."""
    _seen = set() if _seen is None else _seen
    if isinstance(env, ResidualVecEnv):
        return env
    if id(env) in _seen:
        return None
    _seen.add(id(env))
    inner = getattr(env, "env", None)
    return _find_residual(inner, _seen) if inner is not None else None


def success_gated_authority_curriculum(
    *, start=0.01, end=1.0, success_threshold=0.9, step=0.1, min_episodes=8, log=print
):
    """Segment hook: raise residual authority only after the arm succeeds.

    Starts at ``start`` (a small fraction of the configured authority) and
    increases by ``step`` per call while the measured training success rate
    over ``completed_episodes`` is at least ``success_threshold``, capped at
    ``end``.  This is an independent ablation of the authority schedule --
    the reward, success condition and termination are untouched.
    """
    state = {"fraction": float(start)}

    def hook(runner, env, done, updates):
        res = _find_residual(env)
        if res is None:
            return {}
        raw = res.env
        episodes = float(raw.completed_episodes.detach().sum())
        successes = float(raw.completed_successes.detach().sum())
        rate = successes / episodes if episodes > 0 else 0.0
        gate = rate >= float(success_threshold) and episodes >= int(min_episodes)
        if gate:
            state["fraction"] = min(float(end), state["fraction"] + float(step))
        res.set_authority_fraction(state["fraction"])
        info = {
            "authority_fraction": state["fraction"],
            "train_success_rate": rate,
            "train_episodes": episodes,
            "gate_met": bool(gate),
        }
        log(f"[authority] update {done}/{updates}: {info}")
        return info

    return hook


def zero_init_policy_output(runner):
    """Zero the actor's final linear layer so the deterministic residual is zero.

    Structural to residual RL: the untrained deterministic policy must reproduce
    the base controller trajectory before any learning, and stochastic
    exploration must be the *only* source of action perturbation.  The
    Gaussian std is untouched.
    """
    actor = runner.alg.actor
    last = None
    for module in actor.mlp.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is None:
        raise RuntimeError("actor has no linear output layer to zero-initialise")
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    dist = actor.distribution
    with torch.no_grad():
        if hasattr(dist, "log_std_param"):
            std = float(dist.log_std_param.exp().mean())
        elif hasattr(dist, "std_param"):
            std = float(dist.std_param.mean())
        else:
            std = None
    return {"zero_init_actor_output": True, "initial_output_std": std}
