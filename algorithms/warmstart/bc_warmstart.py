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

PPO-only by design: the tanh-bounded actor head of off-policy methods is a
different regression target and their runner is not touched here.
"""
import time

import torch

__all__ = ['collect_expert', 'returns_to_go', 'fit_behavior_cloning', 'BC_DEFAULTS',
           'phase_weights', 'delta_action_targets']


def delta_action_targets(acts, t_in_episode, scale):
    """Convert absolute expert actions to bounded delta targets.

    The delta parameterization executes ``a_t = clamp(a_{t-1} + scale*d_t)``
    with ``a_{-1} = 0`` at every episode start, so the supervised target is
    ``d_t = (a_t - a_{t-1}) / scale`` clipped to [-1, 1] -- the exact inverse
    of the wrapper's deterministic map on the expert's own trajectory.  Without
    this transform the actor would be fitted to absolute commands but
    interpreted as increments, which is not a valid warm start.
    """
    acts = torch.as_tensor(acts, dtype=torch.float32)
    t_in_episode = torch.as_tensor(t_in_episode).reshape(-1)
    if t_in_episode.shape[0] != acts.shape[0]:
        raise ValueError('t_in_episode must have one entry per action row')
    prev = torch.zeros_like(acts)
    if acts.shape[0] > 1:
        prev[1:] = acts[:-1]
    # an episode's first step starts from zero previous command
    prev[t_in_episode == 0] = 0.0
    return ((acts - prev) / float(scale)).clamp(-1.0, 1.0)


def phase_weights(obs, spec):
    """Per-pair sample weights for terminal-phase-weighted BC.

    ``spec`` is a JSON-friendly dict:
      ``index``      obs indices whose norm defines the phase (default [0, 1, 2],
                     i.e. the position norm of docking / rendezvous; satellite
                     uses [4, 5, 6], the attitude-error vector);
      ``near``       phase threshold (norm <= near counts as the terminal phase);
      ``weight``     weight of terminal-phase pairs (default 3.0);
      ``far_weight`` weight of the other pairs (default 1.0);
      ``normalize``  rescale weights to mean 1 (default True) so the actor loss
                     scale is unchanged and only the *relative* weighting differs.

    The phase signal is derived from the observation only -- no environment
    state is read -- and the weights never touch reward, success or termination.
    """
    spec = dict(spec or {})
    idx = list(spec.get('index') or [0, 1, 2])
    near = float(spec.get('near', 0.5))
    near_w = float(spec.get('weight', 3.0))
    far_w = float(spec.get('far_weight', 1.0))
    flat = obs.reshape(-1, obs.shape[-1]) if obs.dim() > 1 else obs.reshape(-1, 1)
    phase = flat[:, idx].norm(dim=-1)
    w = torch.where(phase <= near, flat.new_full(phase.shape, near_w),
                    flat.new_full(phase.shape, far_w))
    if bool(spec.get('normalize', True)) and float(w.mean()) > 0:
        w = w / w.mean()
    return w

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
    ctrl = env.conventional_controller(env, design=getattr(env, "controller_design", None))
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


def returns_to_go(rews, dones, gamma=None, *, bootstrap_values=None, terminated=None,
                  truncated=None, cut=None):
    """Empirical per-step discounted return-to-go of a rollout.

    ``gamma`` is REQUIRED (not optional): every return-to-go needs a
    discount factor to combine current reward with future value.  ``None``
    would silently treat future as always zero; the caller must be explicit.

    rews/dones: (T, N); done at t means the episode ended after step t
    (auto-reset), so the accumulator is cut at that boundary.

    Boundary semantics (all optional, backward compatible when omitted):
    - ``terminated``/``truncated``: (T,N) bool.  A *true* termination carries
      zero terminal value; a timeout or a collection cut does NOT.
    - ``bootstrap_values``: (T,N) value of the state AFTER step t.  When given,
      a truncated/collection-cut boundary uses ``gamma * V(s_{t+1})`` instead of
      the implicit zero terminal value.  Without it, the returned target is
      still computed but the boundary is reported as invalid (see return).
    - ``cut``: (T,N) bool marking a trajectory tail that is not a real episode
      end (fixed-length collection ended mid-episode).

    Returns ``(targets, valid)`` when ``terminated`` is provided (``valid`` is
    (T,N) bool: the target does not rely on an unfinished tail), otherwise the
    legacy single tensor with zero terminal value at every boundary.
    """
    if gamma is None:
        raise ValueError("returns_to_go: gamma is required (use 1.0 for undiscounted)")
    T, N = rews.shape
    out = torch.empty_like(rews)
    acc = torch.zeros(N, device=rews.device)
    valid = torch.ones(T, N, device=rews.device, dtype=torch.bool)
    for t in range(T - 1, -1, -1):
        d = dones[t]
        if terminated is not None:
            term = terminated[t]
            trunc = truncated[t] if truncated is not None else (d & ~term)
            cut_t = cut[t] if cut is not None else torch.zeros_like(d)
            if t == T - 1:
                # a fixed-length rollout ending mid-episode is a collection cut
                cut_t = cut_t | ~d
            if bootstrap_values is not None:
                boot = gamma * bootstrap_values[t]
                tail = rews[t] + boot * (trunc | cut_t)
            else:
                tail = rews[t].clone()
            acc = torch.where(d, tail, rews[t] + gamma * acc)
            # a target is invalid if it inherited value from an unfinished tail
            # without a bootstrap; propagate backward inside the episode
            invalid_tail = (trunc | cut_t) & (bootstrap_values is None)
            if t + 1 < T:
                valid[t] = torch.where(d, ~invalid_tail, valid[t + 1])
            else:
                valid[t] = ~invalid_tail
        else:
            acc = rews[t] + gamma * acc * (~d)
        out[t] = acc
    if terminated is not None:
        return out, valid
    return out


def fit_behavior_cloning(runner, obs, acts, rews, dones, *, epochs, batch_size, lr,
                         gamma, fit_critic=True, log=print, val=None,
                         normalizer_mode='per_batch', freeze_normalizer=False,
                         terminated=None, truncated=None, cut=None,
                         bootstrap_values=None, sample_weights=None):
    """Supervised warm start of the runner's actor (and critic).

    obs/acts: (T, N, d) pairs from collect_expert; rews/dones feed the
    critic's return-to-go targets.  The actor loss is MSE between the
    model's deterministic output (the policy mean; Beta mean under the default
    bounded policy, Gaussian mean under ``distribution='gaussian'``) and the
    expert action;
    the critic loss is MSE against the discounted returns-to-go.  Both
    normalizers are fitted with update_normalization() per batch, mirroring
    what PPO's own update path does.  The exploration std / distribution is
    untouched (it is PPO's exploration knob, not a BC target).

    Experiment extensions (all opt-in, legacy defaults unchanged):
    - ``val``: dict with ``obs``/``acts`` (same layout) evaluated every
      ``max(1, epochs//10)`` epochs WITHOUT touching the normalizer statistics.
    - ``normalizer_mode``: ``'per_batch'`` (legacy) refits the input normalizer
      from every mini-batch in every epoch; ``'fit_once'`` fits it once on the
      full training set before the epochs and keeps it frozen.  Validation and
      test data never update any statistic.
    - ``freeze_normalizer``: switch both empirical normalizers to eval mode
      after the fit so later rolls cannot drift the BC input mapping.
    - ``terminated``/``truncated``/``cut``/``bootstrap_values``: boundary-aware
      critic targets (see ``returns_to_go``).  With ``fit_critic=True`` and no
      bootstrap, targets that depend on an unfinished tail are EXCLUDED rather
      than silently treated as zero-terminal-value.  The returned summary
      records the target construction and the number of excluded pairs.
    - ``sample_weights``: optional per-pair weights (shape (M,) or the
      obs layout) applied to the actor regression only, for terminal-phase
      weighting (see ``phase_weights``).  The critic targets and PPO are
      unaffected.
    """
    actor, critic = runner.alg.actor, runner.alg.critic
    device = obs.device
    T, N = obs.shape[:2]
    flat_obs = obs.reshape(T * N, -1)
    flat_act = acts.reshape(T * N, -1)
    M = flat_obs.shape[0]
    flat_w = (sample_weights.reshape(T * N).to(device=device, dtype=flat_obs.dtype)
              if sample_weights is not None else None)

    def _val_loss():
        if val is None:
            return None
        v_obs = val['obs'].reshape(-1, flat_obs.shape[-1])
        v_act = val['acts'].reshape(-1, flat_act.shape[-1])
        was_training = actor.training
        actor.eval()
        with torch.no_grad():
            pred = actor({'policy': v_obs})
            loss = torch.nn.functional.mse_loss(pred, v_act)
        actor.train(was_training)
        return float(loss)

    actor.train()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    if normalizer_mode not in ('per_batch', 'fit_once'):
        raise ValueError(f'unknown normalizer_mode {normalizer_mode!r}')
    if normalizer_mode == 'fit_once':
        actor.update_normalization({'policy': flat_obs})

    critic_opt, flat_ret, valid = None, None, None
    flat_obs_c, flat_ret_c, m_critic = None, None, 0
    if fit_critic:
        if terminated is not None:
            targets, valid = returns_to_go(rews, dones, gamma,
                                           bootstrap_values=bootstrap_values,
                                           terminated=terminated,
                                           truncated=truncated, cut=cut)
        else:
            targets = returns_to_go(rews, dones, gamma,
                                    bootstrap_values=bootstrap_values)
            valid = torch.ones_like(targets, dtype=torch.bool)
        flat_ret = targets.reshape(T * N)
        flat_valid = valid.reshape(T * N)
        # Filter the valid pairs ONCE: the critic never sees an excluded pair,
        # so no per-minibatch ``bool(keep.any())`` host sync or re-masking.
        flat_obs_c = flat_obs[flat_valid]
        flat_ret_c = flat_ret[flat_valid]
        m_critic = int(flat_obs_c.shape[0])
        critic.train()
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        if normalizer_mode == 'fit_once':
            critic.update_normalization({'policy': flat_obs_c})

    start = time.perf_counter()
    a_losses, c_losses = [], []
    last_val = None
    for epoch in range(epochs):
        perm = torch.randperm(M, device=device)
        a_sum = torch.zeros((), device=device)
        a_batches = 0
        for i in range(0, M, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = flat_obs[idx]
            td = {'policy': b_obs}  # model.obs_groups select this key
            if normalizer_mode == 'per_batch':
                actor.update_normalization({'policy': b_obs})
            pred = actor(td)  # deterministic forward -> policy mean
            if flat_w is None:
                a_loss = torch.nn.functional.mse_loss(pred, flat_act[idx])
            else:
                # weighted per-pair MSE; the weights are normalized to mean 1
                w = flat_w[idx].unsqueeze(-1)
                per_pair = (pred - flat_act[idx]).pow(2).mean(dim=-1)
                a_loss = (w.squeeze(-1) * per_pair).sum() / w.sum().clamp_min(1e-12)
            actor_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            actor_opt.step()
            # Accumulate on-device; one host read per epoch instead of per batch.
            a_sum += a_loss.detach()
            a_batches += 1
        a_losses.append(float(a_sum / max(a_batches, 1)))
        c_epoch_batches, c_sum = 0, torch.zeros((), device=device)
        if fit_critic and m_critic:
            c_perm = torch.randperm(m_critic, device=device)
            for i in range(0, m_critic, batch_size):
                idx = c_perm[i:i + batch_size]
                b_obs_c = flat_obs_c[idx]
                td_c = {'policy': b_obs_c}
                if normalizer_mode == 'per_batch':
                    critic.update_normalization({'policy': b_obs_c})
                v = critic(td_c).squeeze(-1)
                c_loss = torch.nn.functional.mse_loss(v, flat_ret_c[idx])
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                critic_opt.step()
                c_sum += c_loss.detach()
                c_epoch_batches += 1
        if c_epoch_batches:
            c_losses.append(float(c_sum / c_epoch_batches))
        if (epoch + 1) % max(1, epochs // 10) == 0:
            msg = (f'[bc] epoch {epoch + 1}/{epochs} '
                   f'actor_mse {a_losses[-1]:.5f}')
            if fit_critic:
                msg += f' critic_mse {c_losses[-1]:.5f}' if c_losses else ' critic_mse n/a'
            last_val = _val_loss()
            if last_val is not None:
                msg += f' val_actor_mse {last_val:.5f}'
            log(msg)
    if freeze_normalizer:
        for m in (actor, critic):
            if hasattr(m, 'obs_normalizer'):
                m.obs_normalizer.eval()
    wall = time.perf_counter() - start
    summary = {'transitions': int(M), 'epochs': epochs, 'batch_size': batch_size,
               'lr': lr, 'fit_critic': bool(fit_critic), 'wall_seconds': round(wall, 1),
               'final_actor_mse': a_losses[-1] if a_losses else None,
               'final_train_actor_mse': a_losses[-1] if a_losses else None,
               'final_val_actor_mse': last_val,
               'normalizer_mode': normalizer_mode,
               'freeze_normalizer': bool(freeze_normalizer),
               'critic_target': ('boundary-aware returns_to_go; unfinished tails excluded '
                                 'unless bootstrap provided' if terminated is not None
                                 else 'legacy returns_to_go (zero terminal value)'),
               'critic_valid_pairs': int(valid.sum()) if valid is not None else None,
               'critic_excluded_pairs': int((~valid).sum()) if valid is not None else None,
               'sample_weighting': bool(flat_w is not None),
               'sample_weight_mean': (float(flat_w.mean()) if flat_w is not None else None)}
    if fit_critic:
        summary['final_critic_mse'] = c_losses[-1] if c_losses else None
    log(f'[bc] warm start done in {wall:.1f}s: {summary}')
    return summary
