"""Behavior-cloning warm start from the conventional (classical) controllers.

Gives PPO a warm-started actor (and critic) instead of a random network
initialization: roll the batched env with the task's classical baseline
controller (the env's ``conventional_controller`` class attribute), regress the
actor's mean head onto the collected (obs, expert action) pairs, optionally
regress the critic onto the empirical gamma-discounted returns-to-go of the
same rollouts, and then let PPO refine from there.

Data-collection semantics: the collector owns a dedicated controller
instance and drives it directly -- compute the expert action from the
current observation, execute exactly that action, then mirror env resets
onto the controller via ``reset_idx``.  The env itself stays a plain tensor
environment (no controller state, no info field).  Stored pairs
(s_t, pi*(s_t)) are exact supervised targets on the expert's own closed-loop
trajectory.  Because the executed action IS the controller's command, its
internal self-integrations (docking fuel bookkeeping, quadx trim, fixedwing
integrals, rendezvous ratchet) stay consistent with the visited states.

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
    """Roll the env for `steps` vectorized steps under the task's classical
    controller (``env.conventional_controller``), instantiated here and driven
    directly; the env never sees it.

    Returns device-resident tensors: obs (T, N, obs_dim), act (T, N, A),
    rew (T, N), done (T, N) where done marks the env auto-reset at that
    step.  The controller's stateful parts are reset on the same mask the
    env resets on, so per-env controller state always matches the env.
    """
    if env.conventional_controller is None:
        raise RuntimeError(f'{type(env).__name__} declares no conventional_controller')
    ctrl = env.conventional_controller(env)
    obs = env.get_observations()
    N, A = env.num_envs, env.num_actions
    T = steps
    obss = torch.empty(T, N, obs['policy'].shape[-1], device=env.device, dtype=torch.float32)
    acts = torch.empty(T, N, A, device=env.device, dtype=torch.float32)
    rews = torch.empty(T, N, device=env.device)
    dones = torch.empty(T, N, device=env.device, dtype=torch.bool)
    ctrl.reset_idx(torch.ones(N, dtype=torch.bool, device=env.device))
    checked = False
    for t in range(T):
        with torch.no_grad():
            expert = ctrl.actions(obs)
        if expert.shape != (N, A) or expert.device != torch.device(env.device):
            raise ValueError(f'expert action mismatch: shape {tuple(expert.shape)} '
                             f'on {expert.device} vs ({N}, {A}) on {env.device}')
        if not checked:
            bound = float(getattr(env, 'clip_actions', 1.))
            if not bool(torch.isfinite(expert).all()) or float(expert.abs().max()) > bound + 1e-6:
                raise ValueError('expert action violates finiteness/action range')
            checked = True
        obss[t], acts[t] = obs['policy'], expert
        _, rew, term, trunc, _ = env.step(expert)
        rews[t], dones[t] = rew, term | trunc
        obs = env.get_observations()
        ctrl.reset_idx(term | trunc)
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
