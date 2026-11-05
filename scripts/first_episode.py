"""Complete first-episode accounting for auto-reset vector environments."""

import torch


class FirstEpisodes:
    def __init__(self, env):
        self.seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.ep0 = env.completed_episodes.clone()
        self.ok0 = env.completed_successes.clone()
        self.ret0 = env.completed_returns.clone()
        self.metric0 = {k: v.clone() for k, v in getattr(env, "completed_metrics", {}).items()}
        self.success = torch.zeros_like(self.ret0)
        self.returns = torch.zeros_like(self.ret0)
        self.episodes_completed = torch.zeros_like(self.ep0)
        # per-replica metrics must be per-env tensors; scalar entries are
        # global counters (quadx/fixedwing), tracked as run-wide deltas
        self.metrics = {
            k: torch.zeros_like(v) for k, v in self.metric0.items() if v.shape == self.ret0.shape
        }
        self.global_metrics = {
            k: torch.zeros_like(v) for k, v in self.metric0.items() if v.shape != self.ret0.shape
        }

    def update(self, env):
        """Refresh the per-replica first-episode stats (device ops only).

        Deliberately returns nothing: ``bool(self.seen.all())`` is a host
        synchronisation that stalls the GPU queue on every step.  Call
        ``finished()`` at a coarse cadence instead.
        """
        new = ~self.seen & (env.completed_episodes > self.ep0)
        self.success.copy_(torch.where(new, env.completed_successes - self.ok0, self.success))
        self.returns.copy_(torch.where(new, env.completed_returns - self.ret0, self.returns))
        for k, value in self.metrics.items():
            value.copy_(torch.where(new, env.completed_metrics[k] - self.metric0[k], value))
        for k, value in self.global_metrics.items():
            value.copy_(env.completed_metrics[k] - self.metric0[k])
        self.seen |= new
        self.episodes_completed = env.completed_episodes - self.ep0

    def finished(self):
        """True once every replica has completed a full episode (one host sync)."""
        return bool(self.seen.all())

    def report(self):
        seen = self.seen.cpu().tolist()
        n = sum(seen)
        successes = int(self.success.sum().item())
        metrics = {k: v.cpu().tolist() for k, v in self.metrics.items()}
        # Per-replica rows are kept lean: only success flag and episode return.
        # The full per-metric breakdown is reported as the aggregate metric_counts
        # below; storing all ~21 metric values per replica (x num_envs) bloats
        # the saved report without adding information beyond the aggregate sums.
        rows = [
            dict(
                replica=i,
                complete=seen[i],
                success=bool(self.success[i].item()),
                episode_return=float(self.returns[i].item()),
            )
            for i in range(len(seen))
        ]
        return dict(
            protocol="first_episode_per_replica",
            complete=all(seen),
            episodes_finished=n,
            episodes_expected=len(seen),
            successes=successes,
            success_rate=successes / n if n else None,
            mean_episode_return=float(self.returns.sum().item()) / n if n else None,
            metric_counts={k: sum(v) for k, v in metrics.items()},
            global_metric_counts={k: float(v.cpu()) for k, v in self.global_metrics.items()},
            episodes_completed_total=int(self.episodes_completed.sum().item()),
            episodes=rows,
        )
