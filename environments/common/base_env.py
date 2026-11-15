"""Unified environment contract for the batched aerospace RL tasks.

This is the repo's single source of truth for the env API.  Every task
environment inherits from :class:`BaseEnv` (and may additionally pull in
:class:`~environments.common.residual.ResidualActionMixin` for the
optional residual-action path).  The contract mirrors the parts of the
gymnasium vector-env API that the training / eval / baseline code actually
uses, without importing gymnasium:

* ``step(actions) -> (obs, reward, terminated, truncated, extras)``
* ``reset() -> obs``
* ``get_observations() -> obs``
* ``close()``

``obs`` is a :class:`tensordict.TensorDict` whose ``'policy'`` entry holds
the float32 observation batch of shape ``(num_envs, obs_dim)``.  Actions
passed to ``step`` live in the policy space ``[-1, 1]`` (``clip_actions``
is 1 everywhere); the task's own ``_scale_actions`` maps those onto the
physical actuator bounds ``action_low`` / ``action_high``.

Derived metadata
----------------
``action_space`` and ``observation_space`` are :class:`~.spaces.Box`
descriptors computed from the env's own buffers, so they stay consistent
with the actual step contract instead of being hand-maintained copies.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from .spaces import Box


class BaseEnv:
    """Contract base for batched tensor environments.

    Subclasses set ``num_envs``, ``device``, ``num_actions``, ``obs_buf``
    (after the first ``reset``) and the optional physical-bounds tensors
    ``action_low`` / ``action_high`` in their own ``__init__``; the
    properties here derive the public space descriptors from them.
    """

    # -- the step contract (implemented by each task) -----------------------
    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_observations(self):
        raise NotImplementedError

    def _observations_from_buf(self):
        """Zero-copy ``'policy'`` TensorDict over ``obs_buf``.

        The TensorDict is cached and aliases the live buffer (updated in
        place via copy_): one less full-batch device copy and one less
        TensorDict construction per step.  Consumers must not retain the
        observation expecting a snapshot; rsl_rl copies it into its rollout
        storage, and the baselines/residual wrappers only read it.
        """
        td = getattr(self, "_obs_td", None)
        if td is None:
            td = TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])
            self._obs_td = td
        return td

    def _cache_terminal_observations(self):
        """Update ``extras['terminal_observations']`` through cached buffers.

        Same content as building a fresh float32 TensorDict every step, but
        without the per-step allocation; consumed by rsl_rl before the next
        step overwrites the buffer.
        """
        if getattr(self, "_terminal_obs", None) is None:
            self._terminal_obs = torch.zeros(
                self._observation.shape, device=self.device, dtype=torch.float32
            )
            self._terminal_obs_td = TensorDict(
                {"policy": self._terminal_obs}, batch_size=[self.num_envs]
            )
        # Keep the bootstrap observation on the same sensor pipeline as the
        # policy observation; using the clean internal state here would leak
        # privileged information under ``aero="realistic"``.
        source = (
            self.obs_buf
            if getattr(self, "_sensor", None) is not None
            else self._observation
        )
        self._terminal_obs.copy_(source)
        self.extras["terminal_observations"] = self._terminal_obs_td

    # -- space descriptors (policy-facing action space is uniform) ----------
    @property
    def action_space(self) -> Box:
        """Policy input space: ``[-1, 1]`` per channel (the values ``step``
        accepts).  Uniform across every task because ``clip_actions == 1``."""
        return Box(
            low=-1.0,
            high=1.0,
            shape=(int(self.num_actions),),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def physical_action_space(self) -> Box:
        """Physical actuator bounds ``[action_low, action_high]`` when the
        task declares them; falls back to the policy box otherwise."""
        low = getattr(self, "action_low", None)
        high = getattr(self, "action_high", None)
        if low is None or high is None:
            return self.action_space
        return Box(
            low=low, high=high, shape=(int(self.num_actions),), dtype=low.dtype, device=self.device
        )

    @property
    def observation_space(self) -> Box:
        """Flat float32 box over the ``'policy'`` observation.  Bounds are
        unbounded unless the task sets ``obs_low`` / ``obs_high``; most tasks
        return raw state and rely on the policy-side normalizer."""
        obs_buf = self.obs_buf
        if obs_buf is None:
            raise RuntimeError("observation_space is available after the first reset()")
        n = int(obs_buf.shape[-1])
        low = getattr(self, "obs_low", None)
        high = getattr(self, "obs_high", None)
        if low is None or high is None:
            low = -math.inf
            high = math.inf
        return Box(low=low, high=high, shape=(n,), dtype=torch.float32, device=self.device)

    # -- design-model bookkeeping -------------------------------------------
    def _nominal_param(self, name):
        """Physical nominal (SI) value of a perturbable parameter.

        ``PERTURBABLE_PARAMS`` values are SCALE placeholders (1.0), not
        physical quantities.  Envs backed by a ``TensorPlant`` therefore
        resolve the design-model nominal from the plant (thrust in N, mass in
        kg); analytic satellite plants keep their own nominal values in
        ``controller_design`` and fall back to the live attribute here.
        """
        plant = getattr(self, "plant", None)
        if plant is not None:
            nominal = getattr(plant, "nominal_param", None)
            if callable(nominal):
                return nominal(name)
        return getattr(self, name, self.PERTURBABLE_PARAMS[name])

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        """Release engine / scene resources.  Subclasses with a plant override."""

    # -- introspection helpers ---------------------------------------------
    def __repr__(self):
        try:
            na = int(self.num_actions)
            ne = int(self.num_envs)
        except Exception:
            na = ne = "?"
        try:
            od = int(self.obs_buf.shape[-1]) if self.obs_buf is not None else "?"
        except Exception:
            od = "?"
        return (
            f"<{type(self).__name__} num_envs={ne} num_actions={na} "
            f"obs_dim={od} device={getattr(self, 'device', '?')}>"
        )
