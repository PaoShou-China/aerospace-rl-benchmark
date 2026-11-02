"""Residual RL support (opt-in, per env instance).

The policy's action is interpreted as a *residual* on top of the task's
conventional controller (the env's ``conventional_controller`` class
attribute).  Both action spaces are [-1, 1] BY DESIGN — the policy output
follows the env's ``clip_actions=1`` convention and the conventional
controllers emit actions in the same normalized convention (contractually
bounded; saturated at ±1 by their internal clamps).  The mixin therefore
just adds the two; the composite is bounded back into [-1, 1] by the env's
existing action clamping::

    u_applied = clamp(u_conventional + u_policy, -1, 1)

Usage (inside an env __init__, BEFORE ``self.reset()``)::

    def __init__(self, ..., residual=False):
        ...
        self.residual = residual
        self._init_residual()
        self.reset()

and in the class body:

    def step(self, actions):
        actions = self._residual_input(actions)
        ... existing body ...

    def reset_idx(self, envs_idx):
        ... existing body ...
        self._residual_reset(envs_idx)

With ``residual=False`` (the default) every hook is a pass-through and the
benchmark path is bit-for-bit unchanged.  When enabled, the conventional
controller is instantiated once, its stateful parts are reset on the same
mask as the env (mirroring the BC warm-start contract in
algorithms/warmstart/bc_warmstart.py), and the base action is computed from
the current policy observation at the top of each step.
"""

import torch


class ResidualActionMixin:
    """Hook methods for env classes; zero overhead when ``residual=False``."""

    def _init_residual(self):
        """Instantiate the conventional controller.  Must run before reset()."""
        if not getattr(self, "residual", False):
            return
        if getattr(self, "conventional_controller", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} declares no conventional_controller; "
                "cannot enable residual actions"
            )
        design = getattr(self, "controller_design", None)
        self._residual_ctrl = self.conventional_controller(self, design=design)
        self._residual_ctrl.reset_idx(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        self.cfg["residual"] = dict(enabled=self.residual)

    def _residual_input(self, actions):
        """Call at the top of step(): add the conventional base action.

        Both inputs are [-1, 1] by design (see module docstring); the env's
        own action clamping bounds the composite back into [-1, 1].
        """
        if not getattr(self, "residual", False):
            return actions
        with torch.no_grad():
            base = self._residual_ctrl.actions(self.get_observations())
            return base + actions.to(base.dtype)

    def _residual_reset(self, envs_idx):
        """Call at the end of reset_idx(): mirror controller state."""
        if not getattr(self, "residual", False):
            return
        self._residual_ctrl.reset_idx(envs_idx)
