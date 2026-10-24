"""Opt-in training adapters; original environment and evaluation stay intact.

DirectRewardAdapter changes only the reward returned to the learner. Its
reset observer records terminal outcomes before the upstream auto-reset.
All physical state, RNG calls, actions, thresholds and episode counters are
owned by the unchanged upstream environment.
"""
import time

import torch

from .bc_warmstart import returns_to_go
from .rocket_flash_ppo import attach_projection_hook
from .rocket_reward_candidates import compute_reward_candidate


class DirectRewardAdapter:
    def __init__(self, env, mode="original", gamma=.995):
        if mode not in ("original", "legacy_scaled", "bounded_events"):
            raise ValueError(f"unknown reward mode: {mode}")
        self.env, self.mode, self.gamma = env, mode, gamma
        self.seen_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._terminal = {name: self.seen_success.clone() for name in
                          ("success_ever", "fatal_collision", "out_of_bounds")}
        self.reward_sums = {}
        self.reward_steps = 0
        reset = env.reset_idx

        def record_then_reset(mask):
            for name, field in self._fields().items():
                self._terminal[name].copy_(torch.where(mask, field, self._terminal[name]))
            self.seen_success.masked_fill_(mask, False)
            reset(mask)

        env.reset_idx = record_then_reset

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        # RSL optionally randomizes this buffer at the start of learning.
        self.env.episode_length_buf = value

    def _fields(self):
        return {"success_ever": self.env.complete,
                "fatal_collision": self.env.fatal_collision,
                "out_of_bounds": self.env.out_of_bounds}

    def reset(self):
        self.seen_success.zero_()
        return self.env.reset()

    def reset_reward_stats(self):
        self.reward_sums.clear()
        self.reward_steps = 0

    def step(self, actions):
        previous = self.env.get_observations()["policy"]
        seen = self.seen_success.clone()
        for value in self._terminal.values():
            value.zero_()
        obs, original, done, extras = self.env.step(actions)
        outcome = {name: torch.where(done, self._terminal[name], field)
                   for name, field in self._fields().items()}
        if self.mode == "original":
            reward, components = original, {"original": original}
        else:
            # Upstream already exposes the pre-autoreset observation at every
            # transition. Do not substitute returned reset observations.
            terminal = extras["terminal_observations"]["policy"]
            next_obs = torch.where(done[:, None], terminal, obs["policy"])
            reward, components = compute_reward_candidate(
                previous, next_obs, original, outcome, gamma=self.gamma,
                done=done, time_outs=extras["time_outs"], seen_success=seen,
                mode=self.mode,
            )
        self.seen_success.copy_((seen | outcome["success_ever"]) & ~done)
        for key, value in {"legacy_raw": original, "learner_reward": reward,
                           **components}.items():
            total = value.detach().double().sum()
            self.reward_sums[key] = self.reward_sums.get(key, torch.zeros_like(total)) + total
        self.reward_steps += self.env.num_envs
        return obs, reward, done, extras


def fit_projected_behavior_cloning(runner, obs, acts, rews, dones, *, epochs,
                                  batch_size, lr, gamma, fit_critic=True, log=print):
    """Original BC loop with post-Adam projection attached to both optimizers.

    Dataset, shuffles, target construction, normalizer updates, losses and
    exploration std follow the upstream loop. Reward variants supply their
    own collection rewards, so critic targets match each arm's reward units.
    The upstream zero-tail empirical return target is deliberately retained.
    """
    actor, critic = runner.alg.actor, runner.alg.critic
    T, N = obs.shape[:2]
    flat_obs, flat_act = obs.reshape(T * N, -1), acts.reshape(T * N, -1)
    M = T * N
    actor.train()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    actor_hook = attach_projection_hook(actor_opt, actor)
    critic_hook = None
    if fit_critic:
        flat_ret = returns_to_go(rews, dones, gamma).reshape(M)
        critic.train()
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        critic_hook = attach_projection_hook(critic_opt, critic)
    start = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(M, device=obs.device)
        a_losses, c_losses = [], []
        for i in range(0, M, batch_size):
            idx = perm[i:i + batch_size]
            td = {"policy": flat_obs[idx]}
            actor.update_normalization(td)
            a_loss = torch.nn.functional.mse_loss(actor(td), flat_act[idx])
            actor_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            actor_opt.step()
            a_losses.append(float(a_loss.detach()))
            if fit_critic:
                critic.update_normalization(td)
                c_loss = torch.nn.functional.mse_loss(critic(td).squeeze(-1), flat_ret[idx])
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                critic_opt.step()
                c_losses.append(float(c_loss.detach()))
        if (epoch + 1) % max(1, epochs // 10) == 0:
            msg = f"[bc] epoch {epoch + 1}/{epochs} actor_mse {sum(a_losses)/len(a_losses):.5f}"
            if fit_critic:
                msg += f" critic_mse {sum(c_losses)/len(c_losses):.5f}"
            log(msg)
    actor_hook.remove()
    if critic_hook is not None:
        critic_hook.remove()
    report = {"transitions": M, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "fit_critic": bool(fit_critic),
              "wall_seconds": round(time.perf_counter() - start, 1),
              "final_actor_mse": a_losses[-1], "projection_after_optimizer": True}
    if fit_critic:
        report["final_critic_mse"] = c_losses[-1]
    report["imitation_diagnostics"] = imitation_diagnostics(actor, flat_obs, flat_act, batch_size)
    log(f"[bc] projected warm start done: {report}")
    return report


@torch.no_grad()
def imitation_diagnostics(actor, obs, target, batch_size=8192):
    """Read-only in-sample fit by flight stage; not a closed-loop evaluation."""
    totals = {}
    for begin in range(0, obs.shape[0], batch_size):
        x, y = obs[begin:begin + batch_size], target[begin:begin + batch_size]
        prediction = actor({"policy": x})
        errors = (prediction - y).square()
        ignition_wrong = (prediction[:, 3] > 0) != (y[:, 3] > 0)
        masks = {"all": torch.ones(x.shape[0], device=x.device, dtype=torch.bool),
                 "above_10m": x[:, 12] >= 10,
                 "below_10m": x[:, 12] < 10,
                 "pad_contact": x[:, 29] > .5}
        for name, mask in masks.items():
            count = int(mask.sum())
            if count:
                row = totals.setdefault(name, {"count": 0, "squared_error": torch.zeros(7, device=x.device),
                                               "ignition_disagreements": 0})
                row["count"] += count
                row["squared_error"] += errors[mask].sum(0)
                row["ignition_disagreements"] += int(ignition_wrong[mask].sum())
    return {name: {"count": row["count"],
                   "action_mse": (row["squared_error"] / row["count"]).cpu().tolist(),
                   "ignition_disagreement_rate": row["ignition_disagreements"] / row["count"]}
            for name, row in totals.items()}
