"""Behavior-cloning warm start from the classical nominal controllers.

Gives PPO a warm-started actor (and critic) instead of a random network
initialization: roll the batched env with the classical baseline exposed
through ``extras['nominal_action']`` (the NominalActionMixin contract --
the env-owned controller advances exactly once per step inside ``step()``;
this module only READS the info field), regress the actor's mean head onto
the collected (obs, expert action) pairs, optionally regress the critic
onto the empirical gamma-discounted returns-to-go of the same rollouts,
and then let PPO refine from there.

Data-collection semantics ("lag-1" expert rollout): ``step()`` computes the
nominal action for the PRE-step observation, so the executed action for
step t is the expert action computed at step t-1.  The STORED pairs
(s_t, pi*(s_t)) are exact supervised targets; the visited state
distribution is a one-step-delayed expert trajectory -- slightly off the
expert's own trajectory (a mild DAgger-like state coverage), which is
harmless-to-beneficial for BC robustness and keeps the env contract intact
(no external driving of the stateful controllers, so no double-stepping of
the quadx trim / fixedwing integrals / rendezvous ratchet).

PPO-only by design: flash_sac's tanh-bounded actor head is a different
regression target and its runner is not touched here.
"""
import time

import torch

__all__ = ['collect_expert', 'returns_to_go', 'fit_behavior_cloning', 'BC_DEFAULTS']

# shared warm-start defaults; per-task entries pass a dict of overrides on
# top of these so every task's BC recipe stays directly comparable
BC_DEFAULTS = {'steps': 512, 'epochs': 100, 'batch_size': 8192, 'lr': 1e-3,
               'gamma': None, 'fit_critic': True}


def collect_expert(env, steps):
    """Roll the env for `steps` vectorized steps; expert actions come from
    ``extras['nominal_action']`` (written by step() from the pre-step obs).

    Returns device-resident tensors: obs (T, N, obs_dim), act (T, N, A),
    rew (T, N), done (T, N) where done marks the env auto-reset at that
    step.  The first step executes a zero action (no expert action exists
    before the first step); its (obs, expert action) pair is still valid.
    """
    obs0 = env.get_observations()['policy']
    num_actions = env.num_actions
    T, N = steps, env.num_envs
    obss = torch.empty(T, N, obs0.shape[-1], device=env.device, dtype=torch.float32)
    acts = torch.empty(T, N, num_actions, device=env.device, dtype=torch.float32)
    rews = torch.empty(T, N, device=env.device)
    dones = torch.empty(T, N, device=env.device, dtype=torch.bool)
    obs = obs0
    action = torch.zeros(N, num_actions, device=env.device)
    for t in range(T):
        _, rew, done, extras = env.step(action)
        expert = extras['nominal_action']  # expert action for `obs` (the pre-step state)
        obss[t], acts[t], rews[t], dones[t] = obs, expert, rew, done
        obs = env.get_observations()['policy']
        action = expert
    return obss, acts, rews, dones


def returns_to_go(rews, dones, gamma):
    """Empirical per-step discounted return-to-go of a rollout.

    rews/dones: (T, N); done at t means the episode ended after step t
    (auto-reset), so the accumulator is cut at that boundary.
    """
    T, N = rews.shape
    out = torch.empty_like(rews)
    acc = torch.zeros(N, device=rews.device)
    for t in range(T - 1, -1, -1):
        acc = rews[t] + gamma * acc * (~dones[t])
        out[t] = acc
    return out


def fit_behavior_cloning(runner, obs, acts, rews, dones, *, epochs, batch_size, lr,
                         gamma, fit_critic=True, log=print):
    """Supervised warm start of the runner's actor (and critic).

    obs/acts: (T, N, d) pairs from collect_expert; rews/dones feed the
    critic's return-to-go targets.  The actor loss is MSE between the
    model's deterministic output (the Gaussian mean) and the expert action;
    the critic loss is MSE against the discounted returns-to-go.  Both
    normalizers are fitted with update_normalization() per batch, mirroring
    what PPO's own update path does.  The exploration std / distribution is
    untouched (it is PPO's exploration knob, not a BC target).
    """
    actor, critic = runner.alg.actor, runner.alg.critic
    device = obs.device
    T, N = obs.shape[:2]
    flat_obs = obs.reshape(T * N, -1)
    flat_act = acts.reshape(T * N, -1)
    M = flat_obs.shape[0]

    actor.train()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    if fit_critic:
        targets = returns_to_go(rews, dones, gamma).reshape(T * N)
        critic.train()
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        flat_ret = targets

    start = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(M, device=device)
        a_losses, c_losses = [], []
        for i in range(0, M, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = flat_obs[idx]
            td = {'policy': b_obs}  # model.obs_groups select this key
            # fit the empirical input normalizers on the BC data first
            actor.update_normalization({'policy': b_obs})
            pred = actor(td)  # deterministic forward -> Gaussian mean
            a_loss = torch.nn.functional.mse_loss(pred, flat_act[idx])
            actor_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            actor_opt.step()
            a_losses.append(float(a_loss.detach()))
            if fit_critic:
                critic.update_normalization({'policy': b_obs})
                v = critic(td).squeeze(-1)
                c_loss = torch.nn.functional.mse_loss(v, flat_ret[idx])
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                critic_opt.step()
                c_losses.append(float(c_loss.detach()))
        if (epoch + 1) % max(1, epochs // 10) == 0:
            msg = (f'[bc] epoch {epoch + 1}/{epochs} '
                   f'actor_mse {sum(a_losses) / len(a_losses):.5f}')
            if fit_critic:
                msg += f' critic_mse {sum(c_losses) / len(c_losses):.5f}'
            log(msg)
    wall = time.perf_counter() - start
    summary = {'transitions': int(M), 'epochs': epochs, 'batch_size': batch_size,
               'lr': lr, 'fit_critic': bool(fit_critic), 'wall_seconds': round(wall, 1),
               'final_actor_mse': a_losses[-1] if a_losses else None}
    if fit_critic:
        summary['final_critic_mse'] = c_losses[-1] if c_losses else None
    log(f'[bc] warm start done in {wall:.1f}s: {summary}')
    return summary
