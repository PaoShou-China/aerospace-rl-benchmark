"""A learned initial-state router over two frozen direct neural controllers.

The router commits to one controller for the whole episode.  It never reads
the environment, seed, teacher, terminal result, or future observations.
Call ``reset(done_mask)`` after natural episode termination and before the
next action.  The per-episode choices are runtime state, not checkpoint data.
"""
from __future__ import annotations

import torch
from torch import nn


def initial_features(observation: torch.Tensor) -> torch.Tensor:
    """Six initial kinematic coordinates and their quadratic interactions."""
    obs = observation.double()
    q = obs[:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q * torch.where(q[:, 3:4] < 0, -1., 1.)
    position = torch.stack((obs[:, 10] / 20., obs[:, 11] / 20.,
                            (obs[:, 12] - 425.) / 25.), dim=-1)
    base = torch.cat((position, q[:, :3] / .15), dim=-1)
    products = torch.stack([base[:, i] * base[:, j]
                            for i in range(6) for j in range(i, 6)], dim=-1)
    return torch.cat((base, products), dim=-1)


class RidgeAdvantageGate(nn.Module):
    """Predict success(Dagger) minus success(PPO) from initial observations."""
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(27, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(27, dtype=torch.float64))
        self.register_buffer("coefficients", torch.zeros(28, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, initial_observations, advantage, ridge=64.):
        features = initial_features(initial_observations)
        self.feature_mean.copy_(features.mean(0))
        self.feature_scale.copy_(features.std(0, unbiased=False).clamp_min(1e-6))
        design = self.design(initial_observations)
        penalty = torch.eye(design.shape[1], dtype=design.dtype, device=design.device) * ridge
        penalty[0, 0] = 0.
        self.coefficients.copy_(torch.linalg.solve(
            design.T @ design + penalty, design.T @ advantage.double()))

    def design(self, initial_observations):
        features = (initial_features(initial_observations) - self.feature_mean) / self.feature_scale
        return torch.cat((torch.ones((len(features), 1), dtype=features.dtype,
                                     device=features.device), features), dim=-1)

    def forward(self, initial_observations):
        return self.design(initial_observations) @ self.coefficients

    def choose(self, initial_observations):
        # 0 is DAgger; exact ties use PPO.  This tie rule is fixed pre-training.
        return (self(initial_observations) <= 0).long()


class FrozenExpertSelector(nn.Module):
    def __init__(self, expert_a, expert_b, gate):
        super().__init__()
        self.experts = nn.ModuleList((expert_a, expert_b))
        self.gate = gate
        self.register_buffer("choice", torch.empty(0, dtype=torch.long), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def reset(self, done_mask=None):
        if done_mask is None:
            self.choice.fill_(-1)
        elif self.choice.numel():
            self.choice.masked_fill_(done_mask.to(device=self.choice.device, dtype=torch.bool), -1)

    def forward(self, obs):
        raw = obs["policy"]
        if self.choice.shape != (len(raw),) or self.choice.device != raw.device:
            self.choice = torch.full((len(raw),), -1, device=raw.device, dtype=torch.long)
        undecided = self.choice < 0
        if bool(undecided.any()):
            self.choice[undecided] = self.gate.choose(raw[undecided])
        actions_a = self.experts[0](obs)
        actions_b = self.experts[1](obs)
        return torch.where((self.choice == 0)[:, None], actions_a, actions_b)

    act = forward
