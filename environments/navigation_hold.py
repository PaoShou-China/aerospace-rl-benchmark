"""A held navigation packet during periodic telemetry interruptions in three civil tasks."""
import torch
from tensordict import TensorDict

# (nominal interval between interruptions, duration), in policy steps. QuadX
# runs at 40 Hz, relative orbital approach at its native step, transfer at 300 s.
OUTAGE_WINDOWS = {"quadx": (100, 12), "rendezvous": (59, 8), "transfer": (23, 3)}


class NavigationHold:
    def __init__(self, env, task, seed):
        self.env = env
        self.task = task
        self.period, self.duration = OUTAGE_WINDOWS[task]
        self.seed = int(seed) % self.period
        self.packet = env.get_observations()["policy"].clone()
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.age = torch.zeros_like(self.episodes)
        self.lanes = torch.arange(env.num_envs, device=env.device)
        self.stale = torch.zeros_like(self.episodes, dtype=torch.bool)

    def __getattr__(self, key):
        return getattr(self.env, key)

    def _obs(self, x):
        return TensorDict({"policy": x}, batch_size=[self.num_envs])

    def _clear(self, mask, fresh):
        self.packet.copy_(torch.where(mask[:, None], fresh, self.packet))
        self.age.masked_fill_(mask, 0)
        self.stale.masked_fill_(mask, False)
        self.episodes += mask.to(torch.long)

    def reset_idx(self, mask, *args, **kwargs):
        result = self.env.reset_idx(mask, *args, **kwargs)
        self.env._update_observation()
        self._clear(mask, self.env.get_observations()["policy"])
        return result

    def reset(self):
        fresh = self.env.reset()["policy"]
        self._clear(torch.ones_like(self.stale), fresh)
        return self.get_observations()

    def get_observations(self):
        return self._obs(self.packet)

    def step(self, action):
        fresh, reward, term, trunc, extras = self.env.step(action)
        truth = extras["terminal_observations"]["policy"]
        self.age += 1
        phase = (self.lanes * 37 + self.episodes * 17 + self.seed + self.age) % self.period
        self.stale.copy_(phase < self.duration)
        outgoing = torch.where(self.stale[:, None], self.packet, truth)
        if self.task == "transfer":
            # Time and onboard propellant accounting still advance during a lost orbit solution.
            outgoing[:, 0] = truth[:, 0]
            outgoing[:, 7] = truth[:, 7]
        self.packet.copy_(outgoing)
        self._clear(term | trunc, fresh["policy"])
        extras = dict(extras, civil_truth_terminal=truth, terminal_observations=self._obs(outgoing))
        return self.get_observations(), reward, term, trunc, extras
