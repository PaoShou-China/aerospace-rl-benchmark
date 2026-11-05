"""Online gain-scheduling reinforcement learning around a conventional controller.

The wrapper sits between the task environment and ``RslVecEnv`` exactly like
``algorithms.residual.ResidualVecEnv``, but instead of adding a residual to the
*action*, the policy adjusts the controller's *gains* each step::

    policy sample u_t  (one dim per tunable gain, per env, in [-1, 1])
        -> g_i = map(u_i) in [low_i, high_i]   (centered at nominal_i for u_i=0)
        -> controller.set_params({name_i: g_i})   # per-env tensor gains
        -> a_t^0   = controller.actions(obs)
        -> env.step(a_t^0)

The policy's action space therefore has dimension ``len(controller.TUNABLE_PARAMS)``
(not the environment's action space).  Gains are **per-env tensors** -- every
parallel replica carries its own gain vector, just like a regular action, so the
reward signal can credit each replica's schedule independently.

Action -> gain mapping
----------------------
Each action dimension lives in ``[-1, 1]`` and maps onto the declared
``[low, high]`` interval, with the nominal (hand-tuned) value sitting at
``u = 0``.  The mapping is *asymmetric* so that ``u = -1`` reaches ``low`` and
``u = +1`` reaches ``high`` (scaled by ``gain_scale_frac``).  Parameters whose
``TUNABLE_PARAMS`` entry carries ``log=True`` are mapped in log space, so a
unit action step produces a uniform *relative* gain change -- appropriate for
stiffness, damping, filter and cost-weight gains that span orders of magnitude.

The current normalized gains are appended to the task observation (under the
``policy`` key) so the policy can close the loop on what it previously set.

Controller contract
-------------------
The controller must declare ``TUNABLE_PARAMS: dict[str, dict(low, high, default,
log=?)]`` and implement ``set_params(dict)`` / ``get_params()`` (provided by the
``Baseline`` base class).  Gains that feed derived tensors (filter matrices,
LQR gains) must override ``_rebuild_derived`` so ``set_params`` refreshes them.
``set_params`` must accept ``(num_envs,)`` tensors.
"""

import torch

__all__ = ["GainSchedulingVecEnv"]


class GainSchedulingVecEnv:
    """Five-tuple vector env wrapper: the policy schedules the base-controller gains.

    Parameters
    ----------
    env
        The task environment (``step`` returns obs, rew, terminated, truncated,
        extras, all on ``env.device``).  Must not be in residual mode.
    controller
        A ``Baseline`` with a non-empty ``TUNABLE_PARAMS`` table.
    gain_scale_frac
        Global authority fraction: how far from the nominal toward each bound
        the action ``u = +/-1`` reaches.  ``1.0`` (default) gives full authority
        over the declared ``[low, high]`` range; smaller values keep the
        schedule closer to the hand-tuned point.  Applied independently on the
        low and high sides of the nominal (and in log space for log params).
        Must be in (0, 1].  A ``TUNABLE_PARAMS`` entry may override the fraction
        for that gain with ``scale`` (also in (0, 1]).
    smoothing
        First-order EMA coefficient on the *requested* gain (0.0 = no filter).
        ``g_applied = (1-smoothing)*g_req + smoothing*g_prev``.  Reduces gain
        chatter at the cost of lag.
    observe_gains
        If True (default), append the normalized current gains
        ``(g - nominal) / authority in [-1, 1]`` to the policy observation so
        the agent can close the loop on its own schedule.

    Gains reset to nominal on the explicit ``reset()`` call and on per-replica
    termination (each replica starts its next episode from the hand-tuned point).
    """

    def __init__(
        self,
        env,
        controller,
        *,
        gain_scale_frac=1.0,
        smoothing=0.0,
        observe_gains=True,
        task=None,
        name="gain_sched",
    ):
        if getattr(env, "residual", False):
            raise ValueError(
                "env.residual is enabled; compose the controller "
                "outside the env (GainSchedulingVecEnv)"
            )
        if not getattr(controller, "TUNABLE_PARAMS", None):
            raise ValueError(
                f"{type(controller).__name__} has no TUNABLE_PARAMS; cannot schedule gains"
            )
        f = float(gain_scale_frac)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"gain_scale_frac must be in (0, 1], got {f}")
        s = float(smoothing)
        if not 0.0 <= s < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {s}")

        self.__dict__["env"] = env
        self.controller = controller
        self.task = task
        self.name = name
        self.gain_scale_frac = f
        self.smoothing = s
        self.observe_gains = observe_gains

        d = env.device
        n = int(env.num_envs)
        self.num_envs = n
        self.param_names = list(controller.TUNABLE_PARAMS.keys())
        self.num_actions = len(self.param_names)
        lows = torch.tensor(
            [controller.TUNABLE_PARAMS[k]["low"] for k in self.param_names],
            device=d,
            dtype=torch.float32,
        )
        highs = torch.tensor(
            [controller.TUNABLE_PARAMS[k]["high"] for k in self.param_names],
            device=d,
            dtype=torch.float32,
        )
        nominals = torch.tensor(
            [controller.TUNABLE_PARAMS[k]["default"] for k in self.param_names],
            device=d,
            dtype=torch.float32,
        )
        self.low = lows
        self.high = highs
        self.nominal = nominals

        # per-gain log-space flag: log params are mapped in log(low/high/nominal)
        # so action steps produce uniform *relative* gain changes.
        self.use_log = torch.tensor(
            [bool(controller.TUNABLE_PARAMS[k].get("log", False)) for k in self.param_names],
            device=d,
            dtype=torch.bool,
        )
        if (self.use_log & (lows <= 0)).any():
            bad = [
                self.param_names[i]
                for i in range(len(self.param_names))
                if self.use_log[i] and lows[i] <= 0
            ]
            raise ValueError(f"log-space params must have low > 0, got {bad}")

        # precompute log-space bounds for log params (linear bounds stay as-is).
        # clamp_min keeps the (unused) log branch finite for linear params whose
        # lower bound is exactly 0 (e.g. an integral gain), avoiding -inf.
        self.log_low = torch.log(lows.clamp_min(1e-300))
        self.log_high = torch.log(highs.clamp_min(1e-300))
        self.log_nominal = torch.log(nominals.clamp_min(1e-300))

        # authority distance from nominal to each bound.  ``gain_scale_frac``
        # is the global default; a TUNABLE_PARAMS entry may override it per
        # gain with ``scale`` (both must lie in (0, 1]).  Distances are stored
        # in the *mapped* space (linear for linear params, log for log params).
        # hi_dist[i] = scale_i * (bound[i] - nominal[i])   (positive)
        # lo_dist[i] = scale_i * (nominal[i] - bound[i])   (positive)
        scales = []
        for k in self.param_names:
            s = float(controller.TUNABLE_PARAMS[k].get("scale", f))
            if not 0.0 < s <= 1.0:
                raise ValueError(
                    f"{type(controller).__name__}.TUNABLE_PARAMS[{k!r}]['scale'] "
                    f"must be in (0, 1], got {s}"
                )
            scales.append(s)
        self.scale = torch.tensor(scales, device=d, dtype=torch.float32)
        hi_lin = self.scale * (highs - nominals)
        lo_lin = self.scale * (nominals - lows)
        hi_log = self.scale * (self.log_high - self.log_nominal)
        lo_log = self.scale * (self.log_nominal - self.log_low)
        self.hi_dist = torch.where(self.use_log, hi_log, hi_lin)
        self.lo_dist = torch.where(self.use_log, lo_log, lo_lin)

        # per-env gain matrix, start at nominal for every replica
        self.gains = nominals.unsqueeze(0).expand(n, -1).contiguous().clone()
        controller.reset_idx(torch.ones(n, dtype=torch.bool, device=d))
        self._apply_gains(self.gains)
        self._obs = env.get_observations()
        self._steps = 0

    # ------------------------------------------------------------- gain math
    def _apply_gains(self, gains):
        """Push a [num_envs, num_params] gain matrix into the controller.

        Each tunable gain is set as a ``(num_envs,)`` tensor; the controller's
        arithmetic broadcasts over the env dimension automatically.
        """
        params = {k: gains[:, i] for i, k in enumerate(self.param_names)}
        self.controller.set_params(params)

    def _request_to_gains(self, policy_action):
        """Map per-env normalized policy action [-1,1] -> physical gains.

        ``policy_action`` is ``(num_envs, num_params)``; each row maps
        independently to that replica's gain vector.

        For each parameter the action ``u`` is clamped to ``[-1, 1]`` and mapped
        asymmetrically around the nominal:

        * linear:  ``g = nominal + u * hi_dist`` (u>=0)  or  ``nominal + u * lo_dist`` (u<0)
        * log:     ``g = exp(log_nominal + u * hi_log)`` / ``... + u * lo_log)``

        ``u = 0`` always yields the nominal value; ``u = +/-1`` reaches the
        scaled bound.
        """
        a = policy_action.to(torch.float32).clamp(-1.0, 1.0)
        # signed authority: positive side uses hi_dist, negative side uses lo_dist
        signed = torch.where(a >= 0.0, a * self.hi_dist.unsqueeze(0), a * self.lo_dist.unsqueeze(0))
        # linear params: g = nominal + signed ; log params: g = exp(log_nom + signed)
        g_lin = self.nominal.unsqueeze(0) + signed
        g_log = torch.exp(self.log_nominal.unsqueeze(0) + signed)
        req = torch.where(self.use_log.unsqueeze(0), g_log, g_lin)
        # safety clamp to physical bounds (should already hold, but guard numerics)
        req = torch.minimum(torch.maximum(req, self.low.unsqueeze(0)), self.high.unsqueeze(0))
        if self.smoothing > 0.0:
            req = (1.0 - self.smoothing) * req + self.smoothing * self.gains
        return req

    def _normalized_gains(self):
        """Current gains mapped back to [-1, 1] action space (for observation).

        For each gain the normalized value is the action ``u`` that would
        reproduce it, i.e. ``(g - nominal) / dist`` on the appropriate side,
        computed in log space for log params.
        """
        g = self.gains
        g_mapped = torch.where(self.use_log.unsqueeze(0), torch.log(g.clamp_min(1e-300)), g)
        nom = torch.where(
            self.use_log.unsqueeze(0), self.log_nominal.unsqueeze(0), self.nominal.unsqueeze(0)
        )
        hi = self.hi_dist.unsqueeze(0)
        lo = self.lo_dist.unsqueeze(0)
        above = g_mapped >= nom
        ng = torch.where(
            above, (g_mapped - nom) / hi.clamp_min(1e-12), (g_mapped - nom) / lo.clamp_min(1e-12)
        )
        return ng.clamp(-1.0, 1.0)

    def _augment_obs(self, obs):
        """Return a copy of ``obs`` with the normalized gains appended.

        The input is never mutated: the stored ``self._obs`` stays raw (the
        controller consumes it directly), and only the observation handed to
        the policy carries the extra gain columns.
        """
        if not self.observe_gains:
            return obs
        if "policy" not in obs.keys():
            return obs
        ng = self._normalized_gains().to(torch.float32)
        out = obs.clone() if hasattr(obs, "clone") else dict(obs)
        out["policy"] = torch.cat((obs["policy"], ng), dim=-1)
        return out

    # ---------------------------------------------------------------- env API
    def get_observations(self):
        self._obs = self.env.get_observations()
        return self._augment_obs(self._obs)

    def step(self, policy_action):
        policy_action = policy_action.to(self.env.device)
        if policy_action.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"policy action shape {tuple(policy_action.shape)} != "
                f"({self.num_envs}, {self.num_actions})"
            )
        if not torch.isfinite(policy_action).all():
            raise ValueError("policy produced a non-finite gain action")

        self.gains = self._request_to_gains(policy_action)
        self._apply_gains(self.gains)

        nominal = self.controller.actions(self._obs)
        nominal = nominal.to(torch.float32)
        if not torch.isfinite(nominal).all():
            raise ValueError("controller produced a non-finite action after gain scheduling")
        obs, rew, term, trunc, extras = self.env.step(nominal)
        done = term | trunc
        # The env's ``terminal_observations`` hold the PRE-reset observation;
        # off-policy consumers bootstrap the critic from it, so it must carry
        # the same gain columns as every other observation.  Use the gains
        # applied during this step -- i.e. before the done-snap below.
        if self.observe_gains and isinstance(extras, dict):
            terms = extras.get("terminal_observations")
            if terms is not None and "policy" in terms.keys():
                ng = self._normalized_gains().to(terms["policy"].dtype)
                out = terms.clone() if hasattr(terms, "clone") else dict(terms)
                out["policy"] = torch.cat((terms["policy"], ng), dim=-1)
                extras["terminal_observations"] = out
        # actual-action bookkeeping for the OLD episode, then reset the new one
        observe = getattr(self.controller, "observe_applied_action", None)
        if callable(observe):
            observe(nominal)
        self.controller.reset_idx(done)
        # each finished replica snaps its gains back to nominal for the new
        # episode; still-running replicas keep their schedule.
        if done.any():
            self.gains = torch.where(done.unsqueeze(-1), self.nominal.unsqueeze(0), self.gains)
            self._apply_gains(self.gains)
        self._obs = obs
        self._steps += 1
        return self._augment_obs(obs), rew, term, trunc, extras

    def reset(self):
        self._obs = self.env.reset()
        self.controller.reset_idx(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
        )
        self.gains = self.nominal.unsqueeze(0).expand(self.num_envs, -1).contiguous().clone()
        self._apply_gains(self.gains)
        return self._augment_obs(self._obs)

    def close(self):
        return self.env.close()

    # ---- delegate everything else (device, clip_actions, etc.) to the env
    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------- diagnostics
    def stats_snapshot(self):
        """JSON-friendly summary of gain-scheduling behaviour for the run report."""
        g = self.gains.mean(0)  # mean over envs
        g_mapped = torch.where(self.use_log, torch.log(g.clamp_min(1e-300)), g)
        lo = torch.where(self.use_log, self.log_low, self.low)
        hi = torch.where(self.use_log, self.log_high, self.high)
        # position in [0, 1] measured in the *mapped* space (log for log params)
        center = ((g_mapped - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        return {
            "steps": int(self._steps),
            "param_names": list(self.param_names),
            "nominal": [float(x) for x in self.nominal.cpu()],
            "low": [float(x) for x in self.low.cpu()],
            "high": [float(x) for x in self.high.cpu()],
            "log": [bool(x) for x in self.use_log.cpu()],
            "gain_scale_frac": self.gain_scale_frac,
            "scale": [float(x) for x in self.scale.cpu()],
            "smoothing": self.smoothing,
            "gain_pos": [float(x) for x in center.cpu()],
        }
