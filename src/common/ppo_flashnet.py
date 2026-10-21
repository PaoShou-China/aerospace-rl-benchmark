"""FlashSAC-trunk models for rsl-rl PPO (architecture-matched A/B arm).

Gives PPO the FlashSAC network structure (src/common/flash_sac.py): input
UnitBatchNorm embedder -> `trunk_blocks` residual blocks (UnitLinear
hidden->4*hidden, UnitBatchNorm, ReLU, UnitLinear 4*hidden->hidden,
UnitBatchNorm, ReLU, +residual; expansion 4) -> UnitRMSNorm, followed by a
plain linear head. Actor trunk 2x128, critic trunk 2x256 -- exactly
mirroring FlashSACActor / FlashSACDoubleCritic.

Norm-control semantics are replicated on the PPO update path: grad-enabled
forwards (rsl_rl PPO.update) first re-normalize every unit layer to its
target norm (UnitLinear rows to unit L2, BN scale/bias and RMSNorm scales
to sqrt(d)). BatchNorm runs with batch statistics everywhere -- see
FlashResTrunk for why FlashSAC's act-eval/update-train split cannot be
kept under PPO's importance ratio.

Deviations from FlashSAC (algorithmic, not structural): the actor output
distribution stays rsl-rl's state-independent Gaussian (PPO's entropy/KL
machinery depends on it) instead of FlashSAC's state-dependent tanh-bounded
log-std head; the critic head is a scalar linear (rsl-rl's value loss is
MSE on a scalar) instead of the 101-bin categorical head; weight
renormalization happens at the first forward after each optimizer step
(PPO steps twice per minibatch) rather than strictly after every step.

Selected via build_cfg(arch='flash_residual') -> class_name
'common.ppo_flashnet:FlashResMLPModel' (rsl_rl resolve_callable supports
qualified names; eval.py re-resolves them from config.json).
"""
import torch
import torch.nn as nn

from rsl_rl.models import MLPModel

from .flash_sac import FlashSACBlock, FlashSACEmbedder, UnitRMSNorm


class FlashResTrunk(nn.Module):
    """FlashSAC trunk: BN embedder -> residual blocks -> RMSNorm.

    BatchNorm always runs with batch statistics (training=True) in BOTH the
    rollout and the update path. FlashSAC itself acts with eval-mode running
    stats and trains with batch stats, but that split is only free in
    off-policy learning: PPO's importance ratio assumes the collect-time and
    update-time networks are the SAME function, and a running-vs-batch-stats
    mismatch between them silently corrupts the ratio (verified 2026-10-20:
    with the FlashSAC split this PPO arm flat-lined at the random-policy
    return). Batch stats at rollout (1024 replicas) and update (8k-sample
    minibatches of the same data) keep the two consistent."""

    def __init__(self, input_dim, hidden_dim, num_blocks=2, renorm=True):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(FlashSACBlock(hidden_dim) for _ in range(num_blocks))
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.output_dim = hidden_dim
        self.renorm = renorm

    def forward(self, x):
        if torch.is_grad_enabled() and self.renorm:
            with torch.no_grad():
                for m in self.modules():
                    if hasattr(m, 'normalize_parameters'):
                        m.normalize_parameters()
        x = self.embedder(x, True)
        for block in self.blocks:
            x = block(x, True)
        return self.post_norm(x)


class FlashResMLPModel(MLPModel):
    """MLPModel whose latent is the FlashSAC residual trunk instead of raw obs."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, trunk_hidden=128, trunk_blocks=2, **kwargs):
        self._trunk_hidden, self._trunk_blocks = trunk_hidden, trunk_blocks
        # obs_normalization stays off (the trunk's input BatchNorm handles
        # scaling, as in FlashSAC) and the activation is unused (head is linear)
        kwargs.pop('obs_normalization', None)
        kwargs.pop('activation', None)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=(trunk_hidden,), **kwargs)
        # replace the Sequential MLP (Linear+ReLU+...+Linear) with the
        # FlashSAC-style single linear head on the trunk latent
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = nn.Linear(self.trunk.output_dim, mlp_output_dim)
        if obs_set == 'critic':
            # The trunk's UnitRMSNorm freezes feature scale at RMS 1, so the raw
            # return magnitude (~1e3 for pendulum) can only come from this head.
            # FlashSAC never faces this: its RewardNormalizer bounds TD targets to
            # [-5, 5]. Warm-start the critic head at a return-scale gain so the
            # value surface can express state contrast within PPO's update budget
            # (default-init head flat-lined the value surface: up/down contrast
            # 1.16x vs the MLP arm's 30x -> advantages were pure noise).
            nn.init.orthogonal_(self.mlp.weight, gain=100.)

    def _get_latent_dim(self):
        if not hasattr(self, 'trunk'):  # built inside super().__init__ via this hook
            self.trunk = FlashResTrunk(self.obs_dim, self._trunk_hidden, self._trunk_blocks)
        return self.trunk.output_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)  # concat + identity "normalizer"
        return self.trunk(latent)
