"""FlashSAC (Kim et al., 2026, arXiv:2604.04539, RSS'26) ported to the batched
GPU environment protocol of this suite -- a fast and stable off-policy SAC that
sharply reduces gradient updates while compensating with larger models, and
keeps bootstrapped critic learning stable by explicitly bounding weight,
feature and gradient norms.

Faithful to the official implementation (github.com/holiday-robot/flashsac):

- **Weight norm control**: every linear layer is a ``UnitLinear`` whose weight
  rows are re-normalized to unit L2 norm after initialization and after every
  optimizer step; batch-norm scale/bias pairs are constrained to sqrt(d).
- **Feature norm control**: ``UnitBatchNorm`` (batch statistics while training,
  running statistics at eval) after each linear layer, ``UnitRMSNorm`` before
  the prediction heads, residual MLP blocks with expansion 4 and ReLU.
- **Distributional ensemble critic**: two critics fused into one batched
  module (einsum over the ensemble dim) with a 101-bin categorical value head
  and two-hot TD projection; targets use the min-Q critic's log-probs.
- **tanh-Gaussian actor** with bounded log-std (tanh-mapped into [-10, 2]).
- **Temperature auto-tuning** toward ``0.5*A*log(2*pi*e*sigma^2)`` (sigma=0.15).
- **Zeta noise-repeat exploration**: per-replica Gaussian exploration noise is
  held for n consecutive steps, n ~ truncated zeta(mu=2, n<=16), so batched
  replicas produce temporally coherent exploration instead of white noise.
- **Reward normalization** by the running RMS of the discounted return, with
  the critic support [-G_max, G_max] matched to the normalized-return scale.
- **Fast-training regime**: low update-to-data ratio (1 gradient update per
  interaction step at batch 2048), actor updated every 2nd gradient step,
  warmup-cosine-decay learning rate, EMA target critic (tau=0.01).

Differences from the official repo (documented deviations, none
algorithmic): torch.compile and fp16 AMP are wired but default OFF (the
2-block 128/256-wide MLPs here are small; AMP made no difference in smoke
runs and complicates debugging), and the replay buffer is a device-resident
ring buffer in our own env protocol -- no gym, no CPU round-trips anywhere in
the loop. Episode boundaries follow this suite's convention: resets happen
inside ``step()``, so the buffer stores the pre-reset ``next_observation``
from ``extras['terminal_observations']``; true terminations cut the TD
bootstrap while time-limit truncations keep bootstrapping.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ======================================================================================
# Norm-controlled layers (official: flash_rl/agents/flashSAC/layer.py)
# ======================================================================================
class UnitLinear(nn.Module):
    """Linear layer (no bias) whose weight rows are unit-normalized after every step."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.orthogonal_(self.weight, gain=1.)

    def forward(self, x):
        return F.linear(x, self.weight)

    def normalize_parameters(self):
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm whose scale/bias pair is constrained to sqrt(d) L2 norm."""

    def __init__(self, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            training=training, momentum=self.momentum, eps=self.eps)

    def normalize_parameters(self):
        with torch.no_grad():
            scale, bias = self.weight, self.bias
            d = scale.shape[-1]
            factor = math.sqrt(d) * torch.rsqrt((scale * scale + bias * bias).sum(-1, keepdim=True) + 1e-8)
            scale.copy_(scale * factor)
            bias.copy_(bias * factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

    def normalize_parameters(self):
        with torch.no_grad():
            scale = self.weight
            d = scale.shape[-1]
            factor = math.sqrt(d) * torch.rsqrt((scale * scale).sum(-1, keepdim=True) + 1e-8)
            scale.copy_(scale * factor)


class FlashSACBlock(nn.Module):
    """Residual MLP block: unit-linear -> BN -> ReLU -> unit-linear -> BN -> ReLU -> +residual."""

    def __init__(self, hidden_dim, expansion=4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x, training):
        residual = x
        x = F.relu(self.norm1(self.w1(x), training))
        x = F.relu(self.norm2(self.w2(x), training))
        return x + residual


class FlashSACEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x, training):
        return self.w(self.norm(x, training))


class NormalTanhPolicy(nn.Module):
    """tanh-Gaussian policy head with tanh-bounded log-std in [log_std_min, log_std_max]."""

    def __init__(self, hidden_dim, action_dim, log_std_min=-10., log_std_max=2.):
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def get_mean_and_std(self, x, training):
        mean = F.linear(x, self.mean_w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.weight, self.std_bias)
        # bounded log-std via tanh reparameterization (official design)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * .5 * (1 + torch.tanh(raw_log_std))
        return mean, torch.exp(log_std)

    def forward(self, x, training):
        mean, std = self.get_mean_and_std(x, training)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # tanh Jacobian correction for log-prob
        log_prob = dist.log_prob(raw) - 2. * (math.log(2.) - raw - F.softplus(-2. * raw))
        return action, {'log_prob': log_prob.sum(-1)}


class FlashSACActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_blocks=2):
        super().__init__()
        self.embedder = FlashSACEmbedder(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(FlashSACBlock(hidden_dim) for _ in range(num_blocks))
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim, action_dim)

    def trunk(self, obs, training):
        x = self.embedder(obs, training)
        for block in self.blocks:
            x = block(x, training)
        return self.post_norm(x)

    def get_mean_and_std(self, obs, training):
        return self.predictor.get_mean_and_std(self.trunk(obs, training), training)

    def forward(self, obs, training):
        return self.predictor(self.trunk(obs, training), training)


# ------------------------------- ensemble (double) critic -----------------------------
class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1.)

    def forward(self, x):  # x: [N, B, in]
        return torch.einsum('nbi,noi->nbo', x, self.weight)

    def normalize_parameters(self):
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    def __init__(self, num_ensemble, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer('running_mean', torch.zeros(num_ensemble, input_dim))
        self.register_buffer('running_var', torch.ones(num_ensemble, input_dim))
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training):  # x: [N, B, dim]
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                b = x.shape[1]
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (b / (b - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self):
        with torch.no_grad():
            scale, bias = self.weight, self.bias
            d = scale.shape[-1]
            factor = math.sqrt(d) * torch.rsqrt((scale * scale + bias * bias).sum(-1, keepdim=True) + 1e-8)
            scale.copy_(scale * factor)
            bias.copy_(bias * factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble, input_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self):
        with torch.no_grad():
            scale = self.weight
            d = scale.shape[-1]
            factor = math.sqrt(d) * torch.rsqrt((scale * scale).sum(-1, keepdim=True) + 1e-8)
            scale.copy_(scale * factor)


class EnsembleCategoricalValue(nn.Module):
    """Categorical (101-bin) value head over [min_v, max_v]; returns expectation + log-probs."""

    def __init__(self, num_ensemble, hidden_dim, num_bins, min_v, max_v):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer('bin_values', torch.linspace(min_v, max_v, num_bins).reshape(1, 1, -1))

    def forward(self, x, training):  # x: [N, B, hidden]
        logits = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=-1)
        value = (log_prob.exp() * self.bin_values).sum(-1)
        return value, {'log_prob': log_prob}


class FlashSACDoubleCritic(nn.Module):
    """Two critics fused into batched ensemble modules (num_qs=2, clipped double Q)."""

    def __init__(self, input_dim, hidden_dim, num_bins, min_v, max_v, num_qs=2, num_blocks=2):
        super().__init__()
        self.num_qs = num_qs
        self.embedder = nn.ModuleList([EnsembleUnitBatchNorm(num_qs, input_dim),
                                       EnsembleUnitLinear(num_qs, input_dim, hidden_dim)])
        self.blocks = nn.ModuleList(nn.ModuleList([EnsembleUnitLinear(num_qs, hidden_dim, hidden_dim * 4),
                                                   EnsembleUnitBatchNorm(num_qs, hidden_dim * 4),
                                                   EnsembleUnitLinear(num_qs, hidden_dim * 4, hidden_dim),
                                                   EnsembleUnitBatchNorm(num_qs, hidden_dim)])
                                    for _ in range(num_blocks))
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(num_qs, hidden_dim, num_bins, min_v, max_v)

    def forward(self, obs, actions, training):
        x = torch.cat((obs, actions), dim=-1).unsqueeze(0).expand(self.num_qs, -1, -1)
        x = self.embedder[1](self.embedder[0](x, training))
        for w1, n1, w2, n2 in self.blocks:
            residual = x
            x = F.relu(n1(w1(x), training))
            x = F.relu(n2(w2(x), training))
            x = x + residual
        return self.predictor(self.post_norm(x), training)


# ======================================================================================
# Reward normalization (official: flash_rl/agents/utils/reward_normalization.py)
# ======================================================================================
class RunningMeanStd:
    def __init__(self, device):
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = torch.tensor(0., device=device)
        self.epsilon = 1e-4

    def update(self, x):
        sample_mean = x.mean()
        sample_var = x.var(correction=0)
        sample_count = float(x.shape[0])
        delta = sample_mean - self.mean
        total = self.count + sample_count
        ratio = sample_count / total
        new_mean = self.mean + delta * ratio
        m2 = self.var * (self.count + self.epsilon) + sample_var * sample_count + delta.pow(2) * self.count * ratio
        self.mean, self.var, self.count = new_mean, m2 / total, total


class RewardNormalizer:
    """Scales rewards so discounted returns have ~unit RMS, floored so |G| <= G_max."""

    def __init__(self, gamma, g_max, device):
        self.gamma, self.g_max = gamma, g_max
        self.g_r = torch.zeros(1, device=device)        # running discounted return per replica
        self.g_r_max = torch.zeros(1, device=device)    # running max |G|
        self.g_rms = RunningMeanStd(device)

    @torch.no_grad()
    def update_reward_stats(self, reward, terminated, truncated):
        done = (terminated | truncated).float()
        self.g_r = self.gamma * (1. - done) * self.g_r + reward
        self.g_r_max = torch.maximum(self.g_r_max, self.g_r.abs().max().reshape(1))
        self.g_rms.update(self.g_r)

    def normalize_rewards(self, rewards):
        denominator = torch.maximum(torch.sqrt(self.g_rms.var + 1e-8), self.g_r_max / self.g_max)
        return rewards / denominator


# ======================================================================================
# Device-resident replay buffer with n-step returns
# ======================================================================================
class ReplayBuffer:
    """Fixed-capacity ring buffer on `device`; uniform sampling; n-step return folding.

    Stores (obs, action, reward, next_obs, terminated, truncated). The n-step
    chain is cut at any episode boundary (terminated OR truncated) and
    bootstraps from the chain-end next_observation with factor
    (1 - terminated_in_chain), matching the official buffer's semantics.
    """

    def __init__(self, capacity, obs_dim, act_dim, n_step, gamma, min_length, batch_size, device):
        self.capacity, self.n_step, self.gamma = capacity, n_step, gamma
        self.min_length, self.batch_size, self.device = min_length, batch_size, device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros((capacity, act_dim), device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.term = torch.zeros(capacity, device=device)
        self.trunc = torch.zeros(capacity, device=device)
        self._idx = 0
        self._size = 0
        # per-replica rolling history for n-step folding: [n_step, N, ...]
        self._hist = None

    @torch.no_grad()
    def add(self, obs, act, rew, next_obs, term, trunc):
        """Add one vectorized step (all replicas at once)."""
        n = obs.shape[0]
        if self._hist is None:  # lazy init per-replica history
            self._hist = {k: torch.zeros((0, n) + v.shape[1:], device=self.device)
                          for k, v in (('o', obs), ('a', act), ('r', rew), ('no', next_obs), ('t', term), ('c', trunc))}
        for key, val in (('o', obs), ('a', act), ('r', rew), ('no', next_obs), ('t', term), ('c', trunc)):
            self._hist[key] = torch.cat((self._hist[key], val.unsqueeze(0)), dim=0)[-self.n_step:]
        if self._hist['o'].shape[0] < self.n_step:
            return
        # fold n-step: R = sum_i gamma^i r_i * prod_{j<i}(1 - done_j); bootstrap from chain end
        done_any = (self._hist['t'] + self._hist['c']).clamp_max(1.)     # [n_step, N]
        gamma_pow = self.gamma ** torch.arange(self.n_step, device=self.device, dtype=torch.float32)
        keep = torch.cumprod(1. - done_any, dim=0)                       # [n_step, N]
        r_n = (keep * gamma_pow.unsqueeze(1) * self._hist['r']).sum(0)
        term_n = ((self._hist['t'] * keep).sum(0) > 0).float()           # terminated inside the kept chain
        chain_end = (keep.sum(0) - 1.).clamp_min(0.).long()              # index of chain end per replica
        arange = torch.arange(n, device=self.device)
        next_n = self._hist['no'][chain_end, arange]
        slot = (self._idx + arange) % self.capacity
        self.obs[slot] = self._hist['o'][0]
        self.act[slot] = self._hist['a'][0]
        self.rew[slot] = r_n
        self.next_obs[slot] = next_n
        self.term[slot] = term_n
        self.trunc[slot] = done_any[0]
        self._idx = int((self._idx + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def can_sample(self):
        return self._size >= self.min_length

    @torch.no_grad()
    def sample(self):
        idx = torch.randint(self._size, (self.batch_size,), device=self.device)
        return {'observation': self.obs[idx], 'action': self.act[idx], 'reward': self.rew[idx],
                'next_observation': self.next_obs[idx], 'terminated': self.term[idx], 'truncated': self.trunc[idx]}


# ======================================================================================
# FlashSAC agent
# ======================================================================================
def flash_sac_cfg(*, buffer_capacity=1_000_000, buffer_min_length=10_000, batch_size=2048,
                  normalize_reward=True, g_max=5., learning_rate=3e-4, learning_rate_end=1.5e-4,
                  lr_warmup_rate=1e-6, actor_hidden_dim=128, actor_num_blocks=2,
                  critic_hidden_dim=256, critic_num_blocks=2, critic_num_bins=101,
                  critic_target_tau=0.01, temp_initial_value=0.01, temp_target_sigma=0.15,
                  actor_update_period=2, noise_zeta_mu=2., noise_zeta_max=16,
                  gamma=.99, n_step=1, updates_per_step=1, use_amp=False):
    """FlashSAC configuration with the paper's state-based defaults (flashSAC.yaml).

    Keyword overrides let a task vary exactly the variables under test while
    everything else stays at the shared fair-comparison values."""
    return dict(buffer_capacity=buffer_capacity, buffer_min_length=buffer_min_length,
                batch_size=batch_size, normalize_reward=normalize_reward, g_max=g_max,
                learning_rate=learning_rate, learning_rate_end=learning_rate_end,
                lr_warmup_rate=lr_warmup_rate, actor_hidden_dim=actor_hidden_dim,
                actor_num_blocks=actor_num_blocks, critic_hidden_dim=critic_hidden_dim,
                critic_num_blocks=critic_num_blocks, critic_num_bins=critic_num_bins,
                critic_target_tau=critic_target_tau, temp_initial_value=temp_initial_value,
                temp_target_sigma=temp_target_sigma, actor_update_period=actor_update_period,
                noise_zeta_mu=noise_zeta_mu, noise_zeta_max=noise_zeta_max,
                gamma=gamma, n_step=n_step, updates_per_step=updates_per_step, use_amp=use_amp)


def _warmup_cosine_decay(step, init, peak, end, warmup_steps, decay_steps):
    if step < warmup_steps:
        return init + (peak - init) * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.)
    return end + .5 * (peak - end) * (1 + math.cos(math.pi * progress))


class FlashSAC:
    """Actor/critic/temperature networks, zeta noise-repeat exploration, update math."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        self.cfg, self.device, self.act_dim = cfg, device, act_dim
        self.actor = FlashSACActor(obs_dim, cfg['actor_hidden_dim'], act_dim, cfg['actor_num_blocks']).to(device)
        critic_in = obs_dim + act_dim
        self.critic = FlashSACDoubleCritic(critic_in, cfg['critic_hidden_dim'], cfg['critic_num_bins'],
                                           -cfg['g_max'], cfg['g_max'],
                                           num_blocks=cfg['critic_num_blocks']).to(device)
        self.target_critic = FlashSACDoubleCritic(critic_in, cfg['critic_hidden_dim'], cfg['critic_num_bins'],
                                                  -cfg['g_max'], cfg['g_max'],
                                                  num_blocks=cfg['critic_num_blocks']).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.log_temperature = nn.Parameter(torch.tensor([math.log(cfg['temp_initial_value'])], device=device))
        self._temp_opt = torch.optim.Adam([self.log_temperature], lr=cfg['learning_rate'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['learning_rate'], fused=device.type == 'cuda')
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg['learning_rate'], fused=device.type == 'cuda')
        # normalize all norm-controlled parameters after initialization
        for net in (self.actor, self.critic, self.target_critic):
            for m in net.modules():
                if hasattr(m, 'normalize_parameters'):
                    m.normalize_parameters()
        self.target_entropy = .5 * act_dim * math.log(2 * math.pi * math.e * cfg['temp_target_sigma'] ** 2)
        # truncated zeta CDF over repeat lengths n in [1, noise_zeta_max]
        ns = torch.arange(1, cfg['noise_zeta_max'] + 1, device=device, dtype=torch.float32)
        pmf = ns ** (-cfg['noise_zeta_mu'])
        self._zeta_cdf = torch.cumsum(pmf / pmf.sum(), dim=0)
        self._noise = torch.zeros(act_dim, device=device)
        self._noise_count = torch.zeros((), device=device, dtype=torch.int64)
        self._noise_n = torch.ones((), device=device, dtype=torch.int64)
        self.normalizer = RewardNormalizer(cfg['gamma'], cfg['g_max'], device) if cfg['normalize_reward'] else None
        self.replay = ReplayBuffer(cfg['buffer_capacity'], obs_dim, act_dim, cfg['n_step'], cfg['gamma'],
                                   cfg['buffer_min_length'], cfg['batch_size'], device)
        self._update_step = 0
        self._lr_state = {}
        self.grad_scaler = torch.amp.GradScaler(device.type, enabled=cfg['use_amp'])

    # -------------------------------- exploration --------------------------------
    @torch.no_grad()
    def act(self, obs, temperature=1.):
        """Sample actions; training mode holds per-replica noise for zeta-distributed n steps."""
        mean, std = self.actor.get_mean_and_std(obs, training=False)
        if temperature == 0.:
            return torch.tanh(mean)
        cdf, count, n = self._zeta_cdf, self._noise_count, self._noise_n
        if self._noise.shape != mean.shape:  # lazy per-replica noise [num_envs, act_dim]
            self._noise = torch.randn_like(mean)
        reinit = (count == 0) | (count >= n)
        new_noise = torch.randn_like(mean)
        u = torch.rand((), device=cdf.device)
        new_n = (torch.argmax((u < cdf).to(torch.int64)) + 1).reshape(())
        noise = torch.where(reinit, new_noise, self._noise)
        self._noise = noise
        self._noise_n = torch.where(reinit, new_n, n)
        self._noise_count = torch.where(reinit, torch.zeros_like(count), count + 1)
        return torch.tanh(mean + std * noise * temperature)

    # --------------------------------- updates -----------------------------------
    def _apply(self, optimizer, loss):
        optimizer.zero_grad(set_to_none=True)
        if self.cfg['use_amp']:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

    def _normalize_all(self, net):
        for m in net.modules():
            if hasattr(m, 'normalize_parameters'):
                m.normalize_parameters()

    def update(self):
        cfg = self.cfg
        batch = self.replay.sample()
        if self.normalizer is not None:
            batch['reward'] = self.normalizer.normalize_rewards(batch['reward'])
        gamma_eff = cfg['gamma'] ** cfg['n_step']
        info = {}
        # ---- critic update (every gradient step) ----
        with torch.no_grad():
            next_act, next_info = self.actor(batch['next_observation'], training=False)
            next_entropy = self.temperature * next_info['log_prob']
            obs_all = torch.cat((batch['observation'], batch['next_observation']), 0)
            act_all = torch.cat((batch['action'], next_act), 0)
            qs_all, q_infos_all = self.target_critic(obs_all, act_all, training=True)
            b = batch['reward'].shape[0]
            next_qs = qs_all[:, b:]
            next_lp = q_infos_all['log_prob'][:, b:]                      # [2, B, bins]
            min_idx = next_qs.argmin(0)                                   # [B]
            bins = next_lp.shape[-1]
            next_lp = torch.gather(next_lp, 0, min_idx[None, :, None].expand(1, -1, bins))[0]
            # two-hot projection of the categorical TD target
            bin_values = torch.linspace(-cfg['g_max'], cfg['g_max'], cfg['critic_num_bins'],
                                        device=self.device).reshape(1, -1)
            width = (bin_values[0, 1] - bin_values[0, 0]).item()
            tgt = batch['reward'].reshape(-1, 1) + gamma_eff * (bin_values - next_entropy.reshape(-1, 1)) \
                * (1. - batch['terminated'].reshape(-1, 1))
            tgt = tgt.clamp(-cfg['g_max'], cfg['g_max'])
            b_pos = (tgt - bin_values[0, 0]) / width
            lower = torch.floor(b_pos)
            upper = (lower + 1).clamp_max(cfg['critic_num_bins'] - 1)
            frac = b_pos - lower
            probs = next_lp.exp()
            target_probs = torch.zeros_like(probs)
            target_probs.scatter_add_(1, lower.long().clamp_min(0), probs * (1. - frac))
            target_probs.scatter_add_(1, upper.long(), probs * frac)
            max_entropy_bonus = next_entropy.max()
        pred_qs_all, pred_infos = self.critic(obs_all, act_all, training=True)
        pred_lp = pred_infos['log_prob'][:, :b]                            # [2, B, bins]
        critic_loss = (-(target_probs.unsqueeze(0) * pred_lp).sum(-1)).mean()
        self._apply(self.critic_opt, critic_loss)
        self._normalize_all(self.critic)
        with torch.no_grad():  # EMA target update: target <- target + tau (source - target)
            src, dst = self.critic.parameters(), self.target_critic.parameters()
            torch._foreach_lerp_(list(dst), list(src), cfg['critic_target_tau'])
        info['Loss/critic_loss'] = critic_loss.detach()
        info['Loss/max_entropy_bonus'] = max_entropy_bonus.detach()
        # ---- actor + temperature update (every actor_update_period steps) ----
        if self._update_step % cfg['actor_update_period'] == 0:
            obs_all_a = torch.cat((batch['observation'], batch['next_observation']), 0)
            act_all_a, a_info = self.actor(obs_all_a, training=True)
            act = act_all_a[:b]
            log_prob = a_info['log_prob'][:b]
            for p in self.critic.parameters():
                p.requires_grad_(False)
            qs, _ = self.critic(batch['observation'], act, training=False)
            for p in self.critic.parameters():
                p.requires_grad_(True)
            q = torch.minimum(qs[0], qs[1])
            actor_loss = (log_prob * self.temperature.detach() - q).mean()
            self._apply(self.actor_opt, actor_loss)
            self._normalize_all(self.actor)
            entropy = -log_prob.mean().detach()
            temp_loss = self.temperature * (entropy - self.target_entropy)
            self._temp_opt.zero_grad(set_to_none=True)
            temp_loss.backward()
            self._temp_opt.step()
            info['Loss/actor_loss'] = actor_loss.detach()
            info['Policy/entropy'] = entropy
            info['Loss/temperature'] = temp_loss.detach()
            info['Loss/temperature_value'] = self.temperature.detach()
        self._update_step += 1
        return info

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).reshape(())

    # ------------------------------- persistence ---------------------------------
    def state_dict(self):
        # optimizer states are persisted so a resumed run continues the
        # Adam moments instead of restarting them from zero (warm restart).
        # NOTE: the replay buffer and RNG streams are NOT persisted -- loading
        # a checkpoint is a parameter/optimizer warm start, not a bit-exact
        # mid-run resume; report it as such.
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'log_temperature': self.log_temperature.detach().cpu(),
                'actor_opt': self.actor_opt.state_dict(),
                'critic_opt': self.critic_opt.state_dict(),
                'temp_opt': self._temp_opt.state_dict(),
                'normalizer': None if self.normalizer is None else
                {'g_r': self.normalizer.g_r, 'g_r_max': self.normalizer.g_r_max,
                 'g_rms_mean': self.normalizer.g_rms.mean, 'g_rms_var': self.normalizer.g_rms.var,
                 'g_rms_count': self.normalizer.g_rms.count},
                'update_step': self._update_step}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])
        self.target_critic.load_state_dict(sd['target_critic'])
        with torch.no_grad():
            self.log_temperature.copy_(sd['log_temperature'].to(self.device))
        if 'actor_opt' in sd:  # optimizer moments (absent in pre-2026-10-22 checkpoints)
            self.actor_opt.load_state_dict(sd['actor_opt'])
            self.critic_opt.load_state_dict(sd['critic_opt'])
            self._temp_opt.load_state_dict(sd['temp_opt'])
        if self.normalizer is not None and sd.get('normalizer'):
            n = self.normalizer
            n.g_r = sd['normalizer']['g_r'].to(self.device)
            n.g_r_max = sd['normalizer']['g_r_max'].to(self.device)
            n.g_rms.mean = sd['normalizer']['g_rms_mean'].to(self.device)
            n.g_rms.var = sd['normalizer']['g_rms_var'].to(self.device)
            n.g_rms.count = sd['normalizer']['g_rms_count'].to(self.device)
        self._update_step = sd.get('update_step', 0)


# ======================================================================================
# Runner: env loop, TB logging, checkpointing -- mirrors the rsl-rl OnPolicyRunner API
# ======================================================================================
class FlashSACRunner:
    """Same surface as rsl_rl OnPolicyRunner: learn(iters) / save(path) / load(path).

    Each iteration performs ONE vectorized env step (all replicas) plus
    ``updates_per_step`` gradient updates (paper: 1) -- the low
    update-to-data regime is what makes FlashSAC wall-clock fast."""

    def __init__(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.device = env, cfg, torch.device(device)
        obs_td = env.get_observations()
        obs = obs_td['policy'] if hasattr(obs_td, 'keys') else obs_td
        obs_dim = int(obs.shape[-1])
        act_dim = int(env.num_actions)
        self.agent = FlashSAC(obs_dim, act_dim, cfg, self.device)
        self.log_dir = Path(log_dir)
        self.writer = None
        self.save_interval = 100
        self._it = 0

    def _set_lr(self, step, total_updates):
        cfg = self.cfg
        warmup = max(1, int(cfg['lr_warmup_rate'] * total_updates))
        lr = _warmup_cosine_decay(step, cfg['learning_rate'], cfg['learning_rate'],
                                  cfg['learning_rate_end'], warmup, total_updates)
        for opt in (self.agent.actor_opt, self.agent.critic_opt, self.agent._temp_opt):
            for group in opt.param_groups:
                group['lr'] = lr
        return lr

    def learn(self, num_iterations):
        env, agent, cfg = self.env, self.agent, self.cfg
        device = self.device
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        total_updates = num_iterations * cfg['updates_per_step']
        n = env.num_envs
        obs_td = env.get_observations()
        obs = obs_td['policy'] if hasattr(obs_td, 'keys') else obs_td
        cur_reward = torch.zeros(n, device=device)
        cur_length = torch.zeros(n, device=device)
        rew_hist, len_hist = [], []
        recent_rew, recent_len = [], []  # for the console line (TB consumes rew_hist/len_hist)
        update_count = agent._update_step  # continue the lr schedule across resumes
        for it in range(self._it, self._it + num_iterations):
            lr = self._set_lr(update_count, total_updates)
            with torch.no_grad():
                actions = agent.act(obs, temperature=1.)
            obs_td, rew, dones, extras = env.step(actions)
            next_obs = obs_td['policy'] if hasattr(obs_td, 'keys') else obs_td
            time_outs = extras.get('time_outs', torch.zeros_like(dones))
            terminated = dones & ~time_outs
            terminal_obs = extras.get('terminal_observations')
            if terminal_obs is not None and hasattr(terminal_obs, 'keys'):  # TensorDict obs groups
                terminal_obs = terminal_obs['policy']
            if terminal_obs is not None:  # store the pre-reset state for bootstrapping
                next_obs_store = torch.where(dones.unsqueeze(-1), terminal_obs, next_obs)
            else:
                next_obs_store = next_obs
            if agent.normalizer is not None:
                agent.normalizer.update_reward_stats(rew, terminated, time_outs)
            agent.replay.add(obs, actions, rew, next_obs_store, terminated.float(), time_outs.float())
            obs = next_obs
            cur_reward += rew
            cur_length += 1
            done_any = bool(dones.any())
            if done_any:
                rew_hist += cur_reward[dones].detach().tolist()
                len_hist += cur_length[dones].detach().tolist()
                recent_rew += rew_hist[len(rew_hist) - int(dones.sum()):]
                recent_len += len_hist[len(len_hist) - int(dones.sum()):]
                cur_reward[dones] = 0.
                cur_length[dones] = 0.
                episode = extras.pop('episode', None)
                if episode and self.writer is not None:
                    for key, val in episode.items():
                        self.writer.add_scalar(f'Episode/{key}', float(val), it)
            infos = {}
            if agent.replay.can_sample():
                for _ in range(cfg['updates_per_step']):
                    infos = agent.update()
                    update_count += 1
                    self._set_lr(update_count, total_updates)
            # ---- tensorboard logging (same scalar names as the rsl-rl pipeline) ----
            if rew_hist:
                self.writer.add_scalar('Train/mean_reward', sum(rew_hist) / len(rew_hist), it)
                self.writer.add_scalar('Train/mean_episode_length', sum(len_hist) / len(len_hist), it)
                rew_hist, len_hist = [], []
            for key, val in infos.items():
                self.writer.add_scalar(key, float(val), it)
            if it == self._it or (it + 1) % 50 == 0:
                ce = int(env.completed_episodes.sum())
                cs = int(env.completed_successes.sum())
                recent = sum(recent_rew) / len(recent_rew) if recent_rew else float('nan')
                print(f'FlashSAC it {it + 1}/{self._it + num_iterations} | lr {lr:.2e} | '
                      f'episodes {ce} | successes {cs} | recent return {recent:.2f}', flush=True)
                recent_rew, recent_len = [], []
            if (it + 1) % self.save_interval == 0 or it + 1 == self._it + num_iterations:
                # record the ACTUAL completed iteration: _it is only advanced
                # after the loop, so saving without this arg would stamp every
                # mid-training checkpoint with the pre-learn iteration count
                self.save(str(self.log_dir / f'model_{it + 1}.pt'), iteration=it + 1)
        self._it += num_iterations

    def save(self, path, iteration=None):
        payload = {'agent': self.agent.state_dict(), 'cfg': self.cfg,
                   'iteration': self._it if iteration is None else iteration}
        torch.save(payload, path)
        print(f'[FlashSAC] saved checkpoint {path} (iteration {payload["iteration"]}, '
              f'warm start: replay buffer / RNG not persisted)', flush=True)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(payload['agent'])
        self._it = payload.get('iteration', 0)
        print(f'[FlashSAC] loaded checkpoint {path} (iteration {self._it})', flush=True)

    @torch.no_grad()
    def deterministic_action(self, obs):
        """Mean tanh action, no exploration noise -- for eval.py. Accepts the
        env's TensorDict observation or a raw tensor."""
        if hasattr(obs, 'keys'):
            obs = obs['policy']
        return self.agent.act(obs, temperature=0.)
