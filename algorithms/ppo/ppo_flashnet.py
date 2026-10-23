"""FlashSAC-trunk models for rsl-rl PPO (architecture-matched A/B arm).

Gives PPO the FlashSAC network structure (algorithms/flashsac/flash_sac.py): input
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
import torch.nn.functional as F

from rsl_rl.models import MLPModel

from algorithms.flashsac.flash_sac import FlashSACBlock, FlashSACEmbedder, UnitBatchNorm, UnitLinear, UnitRMSNorm


class _NoBN(nn.Module):
    """UnitBatchNorm replacement that passes features through untouched."""

    def __init__(self, input_dim):
        super().__init__()

    def forward(self, x, training=False):
        return x

    def normalize_parameters(self):
        pass


class _Embedder(nn.Module):
    """FlashSACEmbedder with an injectable norm class."""

    def __init__(self, input_dim, hidden_dim, norm):
        super().__init__()
        self.norm = norm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x, training):
        return self.w(self.norm(x, training))


class _Block(nn.Module):
    """FlashSACBlock with an injectable norm class."""

    def __init__(self, hidden_dim, norm, expansion=4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.norm1 = norm(hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm2 = norm(hidden_dim)

    def forward(self, x, training):
        residual = x
        x = F.relu(self.norm1(self.w1(x), training))
        x = F.relu(self.norm2(self.w2(x), training))
        return x + residual


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
    minibatches of the same data) keep the two consistent.

    use_bn=False strips the BatchNorms (identity) -- for the CRITIC this is
    required on on-policy tasks: batch-stat BN makes the critic's input
    normalization non-stationary while GAE targets bootstrap through the
    critic itself, and the feedback loop flattens the value surface (verified
    2026-10-21: with BN the trunk critic fit only the mean return, up/down
    contrast 1.16x; OFFLINE on stationary data the same trunk fits returns
    3x BETTER than the MLP -- the architecture is fine, the batch-stat TD
    coupling is the killer). FlashSAC escapes this via its large replay
    buffer (stationary batches) + reward-normalized [-5,5] targets."""

    def __init__(self, input_dim, hidden_dim, num_blocks=2, renorm=True, use_bn=True):
        super().__init__()
        norm = UnitBatchNorm if use_bn else _NoBN
        self.embedder = _Embedder(input_dim, hidden_dim, norm)
        self.blocks = nn.ModuleList(_Block(hidden_dim, norm) for _ in range(num_blocks))
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

    def __init__(self, obs, obs_groups, obs_set, output_dim, trunk_hidden=128, trunk_blocks=2, use_bn=None, **kwargs):
        self._trunk_hidden, self._trunk_blocks = trunk_hidden, trunk_blocks
        # use_bn=None auto-selects: actor/critic both default to the role-independent
        # choice made in build_cfg (both False for PPO -- see FlashResTrunk).
        # BN whitens inputs when present; otherwise use rsl-rl's running-stats
        # EmpiricalNormalization (same as the MLP arm)
        if use_bn is None:
            use_bn = obs_set == 'actor'
        kwargs['obs_normalization'] = not use_bn
        kwargs.pop('activation', None)  # head is a plain linear
        self._use_bn = use_bn
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=(trunk_hidden,), **kwargs)
        # replace the Sequential MLP (Linear+ReLU+...+Linear) with the
        # FlashSAC-style single linear head on the trunk latent
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = nn.Linear(self.trunk.output_dim, mlp_output_dim)

    def _get_latent_dim(self):
        if not hasattr(self, 'trunk'):  # built inside super().__init__ via this hook
            self.trunk = FlashResTrunk(self.obs_dim, self._trunk_hidden, self._trunk_blocks,
                                       use_bn=self._use_bn)
        return self.trunk.output_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)  # concat + identity "normalizer"
        return self.trunk(latent)
