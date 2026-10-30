"""Complete first-episode accounting for auto-reset vector environments."""
import torch


class FirstEpisodes:
    def __init__(self, env):
        self.seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.ep0 = env.completed_episodes.clone()
        self.ok0 = env.completed_successes.clone()
        self.ret0 = env.completed_returns.clone()
        self.metric0 = {k: v.clone() for k, v in getattr(env, 'completed_metrics', {}).items()}
        self.success = torch.zeros_like(self.ret0)
        self.returns = torch.zeros_like(self.ret0)
        self.episodes_completed = torch.zeros_like(self.ep0)
        # per-replica metrics must be per-env tensors; scalar entries are
        # global counters (quadx/fixedwing), tracked as run-wide deltas
        self.metrics = {k: torch.zeros_like(v) for k, v in self.metric0.items()
                        if v.shape == self.ret0.shape}
        self.global_metrics = {k: torch.zeros_like(v) for k, v in self.metric0.items()
                               if v.shape != self.ret0.shape}

    def update(self, env):
        new = ~self.seen & (env.completed_episodes > self.ep0)
        self.success.copy_(torch.where(new, env.completed_successes - self.ok0, self.success))
        self.returns.copy_(torch.where(new, env.completed_returns - self.ret0, self.returns))
        for k, value in self.metrics.items():
            value.copy_(torch.where(new, env.completed_metrics[k] - self.metric0[k], value))
        for k, value in self.global_metrics.items():
            value.copy_(env.completed_metrics[k] - self.metric0[k])
        self.seen |= new
        self.episodes_completed = env.completed_episodes - self.ep0
        return bool(self.seen.all())

    def report(self):
        seen = self.seen.cpu().tolist()
        n = sum(seen)
        successes = int(self.success.sum().item())
        metrics = {k: v.cpu().tolist() for k, v in self.metrics.items()}
        rows = [dict(replica=i, complete=seen[i], success=bool(self.success[i].item()),
                     episode_return=float(self.returns[i].item()),
                     metrics={k: values[i] for k, values in metrics.items()})
                for i in range(len(seen))]
        return dict(protocol='first_episode_per_replica', complete=all(seen),
                    episodes_finished=n, episodes_expected=len(seen), successes=successes,
                    success_rate=successes / n if n else None,
                    mean_episode_return=float(self.returns.sum().item()) / n if n else None,
                    metric_counts={k: sum(v) for k, v in metrics.items()},
                    global_metric_counts={k: float(v.cpu())
                                          for k, v in self.global_metrics.items()},
                    episodes_completed_total=int(self.episodes_completed.sum().item()),
                    episodes=rows)
