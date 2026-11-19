"""Four-node neural proposals conditioned on the complete factual snapshot."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Independent, Normal


@dataclass(frozen=True)
class IntentBatch:
    raw: torch.Tensor                    # C x (four nodes x six + arrival)
    snapshot_id: str

    def __post_init__(self):
        if self.raw.ndim != 2 or self.raw.shape[1] != 25 or not torch.isfinite(self.raw).all():
            raise ValueError("intent must contain 4x6 nodes and one arrival coordinate")

    @property
    def acceleration(self):
        return self.raw[:, :24].reshape(-1, 4, 6)[..., :3].tanh() * self.raw.new_tensor([3., 3., 5.])

    @property
    def attitude(self):
        return self.raw[:, :24].reshape(-1, 4, 6)[..., 3:].tanh() * .25

    @property
    def arrival(self): return .2 + self.raw[:, 24].sigmoid()  # 0.2..1.2 seconds


class IntentHead(nn.Module):
    def __init__(self, candidates=4):
        super().__init__()
        self.candidates = candidates
        width = 30 + 7 + 21 + 67 + 87 * 87 + 5
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mean = nn.Linear(64, candidates * 25)
        nn.init.normal_(self.mean.weight, std=.005)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((candidates * 25,), -2.))

    def distribution(self, observation, task, snapshot):
        if observation.shape != (30,) or task.shape != (7,):
            raise ValueError("expected observation[30] and task[target position3, velocity3, remaining time]")
        f = torch.cat((observation.detach(), task.detach(), snapshot.features()))
        mean = self.mean(self.encoder(f))
        # Distribution is on the complete RAW proposal, before tanh and decoding.
        return Independent(Normal(mean, self.log_std.clamp(-5, 1).exp()), 1)

    def forward(self, observation, task, snapshot, stochastic=False):
        d = self.distribution(observation, task, snapshot)
        raw = d.rsample() if stochastic else d.base_dist.loc
        return IntentBatch(raw.reshape(self.candidates, 25), snapshot.fingerprint)
