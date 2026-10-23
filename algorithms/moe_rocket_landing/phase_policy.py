"""Direct neural policy with flight-phase experts and state-only features.

No nominal controller is called at inference. The raw 30 observations and
seven action channels remain unchanged. Phase routing and feature scaling
are explicit candidate architecture choices, not changes to the simulator.
"""
import torch
from torch import nn
from rsl_rl.models import MLPModel


def features(o):
    o = o.float()
    q = o[:, 3:7]
    q = q * torch.where(q[:, 3:4] < 0, -1., 1.)
    xyz, qw = q[:, :3], q[:, 3:4]
    vb = o[:, 7:10]
    t = 2 * torch.cross(xyz, vb, dim=-1)
    vw = vb + qw * t + torch.cross(xyz, t, dim=-1)
    up = torch.stack((2 * (q[:, 0] * q[:, 2] + q[:, 3] * q[:, 1]),
                      2 * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0]),
                      1 - 2 * (q[:, 0].square() + q[:, 1].square())), -1)
    h = o[:, 12:13] - 2.575
    # Exclude previous command echoes and fin states: these are not needed
    # by the teacher and can otherwise encourage an action-copy shortcut.
    return torch.cat((o[:, :3] * 10, q * 10, vb / 25,
                      o[:, 10:12] / 5, h / 100,
                      o[:, 24:30], vw / 25,
                      torch.sign(h) * torch.log1p(h.abs()), h / 2,
                      vw * 2, o[:, :3] * 50, up * 10), -1).clamp(-25, 25)


class PhaseExperts(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim - 1, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, output_dim)) for _ in range(3))

    def forward(self, x):
        latent = self.trunk(x[:, :-1])
        phase = x[:, -1].long()
        outputs = torch.stack([net(latent) for net in self.experts], 1)
        return outputs[torch.arange(len(x), device=x.device), phase]


class RocketPhaseModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, width=256, **kwargs):
        kwargs.pop('hidden_dims', None)
        kwargs.pop('obs_normalization', None)
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=[width], obs_normalization=False, **kwargs)
        dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = PhaseExperts(self._get_latent_dim(), dim, width)

    def _get_latent_dim(self):
        return 34

    def get_latent(self, obs, masks=None, hidden_state=None):
        o = obs['policy']
        h = o[:, 12] - 2.575
        phase = torch.where(h > 10, 0, torch.where(h > 2, 1, 2))
        return torch.cat((features(o), phase[:, None].float()), -1)
