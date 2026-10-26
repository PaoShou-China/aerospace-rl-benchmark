"""Opt-in, forward-pure Flash residual models for direct-action rocket PPO.

The upstream architecture is preserved: empirical observation normalization,
UnitLinear residual trunk, UnitRMSNorm, a linear head, and RSL's Gaussian
action distribution. Parameter projection is an explicit optimizer operation;
it must never depend on whether a forward pass has gradients enabled.

Use ``ProjectedFlashPPO`` to attach the PPO optimizer hook automatically.
Behavior cloning creates separate optimizers, so attach a hook to EACH one:
``attach_projection_hook(actor_optimizer, actor)`` and likewise for critic.
Observation statistics are updated only by ``update_normalization``; the
inherited RSL update/BC caller remains responsible for when that is called.
"""

from __future__ import annotations

import torch
from torch import nn
from rsl_rl.algorithms import PPO

from .flash_sac import UnitLinear, UnitRMSNorm
from .ppo_flashnet import FlashResMLPModel, FlashResTrunk

__all__ = [
    "StableFlashResTrunk", "StableFlashResMLPModel", "ProjectedFlashPPO",
    "project_parameters", "attach_projection_hook",
]


class StableFlashResTrunk(FlashResTrunk):
    """Original BN-free trunk with no parameter/buffer writes in forward."""

    def __init__(self, input_dim, hidden_dim, num_blocks=2):
        # Keep upstream module names/shapes for strict state_dict loading.
        super().__init__(input_dim, hidden_dim, num_blocks, renorm=False, use_bn=False)

    def forward(self, x):
        x = self.embedder(x, False)
        for block in self.blocks:
            x = block(x, False)
        return self.post_norm(x)

    @torch.no_grad()
    def project_parameters(self):
        """Apply the upstream unit-norm constraints, outside forward/autograd."""
        for layer in self.modules():
            if isinstance(layer, (UnitLinear, UnitRMSNorm)):
                layer.normalize_parameters()


class StableFlashResMLPModel(FlashResMLPModel):
    """State-dict-compatible FlashResMLP with explicit norm projection.

    ``use_bn=False`` is the upstream build_cfg choice for BOTH actor and
    critic. Batch-stat BN is rejected because a sample's log probability
    would then change when PPO shuffles/splits the rollout into minibatches.

    ``zero_init_output=True`` reproduces train_rocket.py's zero-head patch
    without a global monkeypatch. Its default is False, matching the reusable
    upstream model constructor rather than that task-specific training entry.

    load_state_dict deliberately does not project: an evaluation load must
    preserve the saved policy exactly. Loading an unprojected old checkpoint
    for training is a documented warm start; explicitly project it once before
    collecting ANY rollout, and record that the policy has changed.
    """

    def __init__(
        self, obs, obs_groups, obs_set, output_dim,
        trunk_hidden=128, trunk_blocks=2, use_bn=False,
        zero_init_output=False, **kwargs,
    ):
        if use_bn not in (False, None):
            raise ValueError("StableFlashResMLPModel requires use_bn=False for PPO likelihood consistency")
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            trunk_hidden=trunk_hidden, trunk_blocks=trunk_blocks,
            use_bn=False, **kwargs,
        )
        if zero_init_output:
            nn.init.zeros_(self.mlp.weight)
            nn.init.zeros_(self.mlp.bias)
        self.project_parameters()

    def _get_latent_dim(self):
        if not hasattr(self, "trunk"):
            self.trunk = StableFlashResTrunk(self.obs_dim, self._trunk_hidden, self._trunk_blocks)
        return self.trunk.output_dim

    @torch.no_grad()
    def project_parameters(self):
        """Project trunk weights only; leave the output head/std untouched."""
        self.trunk.project_parameters()


def project_parameters(*models: StableFlashResMLPModel) -> None:
    """Explicit projection after initialization, an optimizer step, or migration.

    This function does not update empirical observation statistics, clear
    gradients, or alter optimizer moments. It is projected Adam, matching the
    original weight-normalization intent rather than a new optimizer rule.
    """
    for model in dict.fromkeys(models):
        if not isinstance(model, StableFlashResMLPModel):
            raise TypeError("projection requires StableFlashResMLPModel instances")
        model.project_parameters()


def attach_projection_hook(
    optimizer: torch.optim.Optimizer, *models: StableFlashResMLPModel,
) -> torch.utils.hooks.RemovableHandle:
    """Project the supplied models immediately AFTER each optimizer.step().

    Returns PyTorch's removable hook handle. Register once per optimizer;
    optimizer hooks are not serialized, so reattach when constructing a new
    optimizer after checkpoint loading. Existing hooks survive load_state_dict
    on that optimizer. Attaching itself does not mutate the loaded policy.

    Pass both actor and critic for PPO's shared optimizer, one model for each
    independent BC optimizer. All trainable parameters of each supplied model
    must belong to the optimizer, preventing a mismatched hook from silently
    changing another policy. No process-global optimizer monkeypatch is used.
    """
    unique_models = tuple(dict.fromkeys(models))
    if not unique_models:
        raise ValueError("at least one model is required for a projection hook")
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    for model in unique_models:
        if not isinstance(model, StableFlashResMLPModel):
            raise TypeError("projection requires StableFlashResMLPModel instances")
        if any(parameter.requires_grad and id(parameter) not in owned for parameter in model.parameters()):
            raise ValueError("optimizer must own every trainable parameter of each projected model")

    def after_step(_optimizer, _args, _kwargs):
        project_parameters(*unique_models)

    return optimizer.register_step_post_hook(after_step)


class ProjectedFlashPPO(PPO):
    """Original RSL PPO with post-step Flash parameter projection.

    All PPO losses, clipping, learning rates, and normalizer update timing are
    inherited unchanged, including upstream timeout bootstrapping. Keep a
    terminal-observation bootstrap fix as a separate experimental factor.
    This opt-in class does not train residual actions or call the nominal
    controller; the actor still outputs the seven actions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._projection_hook = attach_projection_hook(self.optimizer, self._raw_actor, self._raw_critic)
