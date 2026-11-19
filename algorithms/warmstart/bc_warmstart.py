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
import hashlib
import json
import time
from contextlib import nullcontext

import torch

__all__ = ['collect_expert', 'returns_to_go', 'fit_behavior_cloning', 'BC_DEFAULTS',
           'phase_weights', 'delta_action_targets', 'select_informative_pairs',
           'select_informative_indices_flat', 'select_informative_indices_rollout',
           'returns_to_go_flat', 'resolve_bc_select', 'bc_recipe_fingerprint',
           'save_bc_recipe', 'bc_recipe_matches', 'BC_SELECT_DEFAULTS',
           'BC_RECIPE_VERSION', 'BC_TASK_SELECT_METHOD']


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


def _selection_indices(obs, acts, dones, *, budget=512, segments=8, method='tss',
                       act_change=1.0, neighbor=2, terminal=2.0,
                       terminal_window=5, terminal_mask=None):
    """Return (N, B) time indices of the informative subset, or None if no-op."""
    T, N = dones.shape
    if obs.shape[:2] != dones.shape or acts.shape[:2] != dones.shape:
        raise ValueError(
            f'selection inputs disagree: obs {tuple(obs.shape)}, acts '
            f'{tuple(acts.shape)}, dones {tuple(dones.shape)}'
        )
    if terminal_mask is not None and terminal_mask.shape != dones.shape:
        raise ValueError(
            f'terminal_mask shape {tuple(terminal_mask.shape)} != dones {tuple(dones.shape)}'
        )
    if T <= 1 or N == 0:
        return None
    budget = T if budget is None else int(budget)
    if budget >= T:
        return None
    segments = max(1, int(segments))
    if method not in ('tss', 'uniform'):
        raise ValueError(f'unknown selection method {method!r}')
    seg_len = max(1, (T + segments - 1) // segments)
    bounds = [(s, min(T, s + seg_len)) for s in range(0, T, seg_len)]
    base, rem = divmod(budget, len(bounds))
    picks = []
    if method == 'uniform':
        # stratified-uniform ablation: equally spaced steps in every phase
        for s, (start, stop) in enumerate(bounds):
            k = min(stop - start, base + (1 if s < rem else 0))
            if k <= 0:
                continue
            offsets = torch.arange(k, device=dones.device) * (stop - start) // k
            picks.append((start + offsets).view(1, -1).expand(N, -1))
    else:
        raw = torch.zeros(T, N, device=acts.device, dtype=acts.dtype)
        raw[1:] = act_change * (acts[1:] - acts[:-1]).norm(dim=-1)
        score = raw.clone()
        for k in range(1, int(neighbor) + 1):
            score[k:].clamp_min_(raw[:-k])
            score[:-k].clamp_min_(raw[k:])
        if terminal and terminal_window > 0:
            boundary = dones if terminal_mask is None else terminal_mask
            term = boundary.to(score.dtype)
            window = term.clone()
            for k in range(1, int(terminal_window)):
                window[:-k].clamp_min_(term[k:])
            score = score + float(terminal) * window
        for s, (start, stop) in enumerate(bounds):
            k = min(stop - start, base + (1 if s < rem else 0))
            if k <= 0:
                continue
            top = torch.topk(score[start:stop].transpose(0, 1), k, dim=1).indices
            picks.append(torch.sort(top + start, dim=1).values)
    if not picks:  # budget too small to cover one step per phase
        return None
    return torch.cat(picks, dim=1)


def select_informative_pairs(obs, acts, rews, dones, returns=None, *, budget=512,
                             segments=8, method='tss',
                             act_change=1.0, neighbor=2,
                             terminal=2.0, terminal_window=5, terminal_mask=None):
    """Keep a fixed per-replica budget of the most informative expert steps.

    A full-episode rollout of a long-horizon task is mostly low-information
    steady flight; storing and fitting every step wastes memory and BC time.
    This follows the demonstration-curation literature:

    * **TSS (Temporal Surprise Score)** -- rank steps by the magnitude of the
      expert's action change (``||a_t - a_{t-1}||``); action changepoints
      (ignition, throttle pulses, reversals, capture) are where behaviour is
      decided, while consecutive transit frames are nearly redundant
      (``LESS IS MORE: On Data Redundancy in VLA Training``).
    * **Keyframe clusters** -- the score of a changepoint is propagated to its
      ``neighbor`` temporal neighbours (max-pooling), so the selection keeps
      dense *clusters* around keyframes instead of isolated frames, mirroring
      the keyframe focus of ``Keyframe-Focused Visual Imitation Learning``.
    * **Stratified coverage** -- the horizon is split into ``segments`` equal
      phases and the budget is divided across them, so every phase is
      represented.  This is deliberate: the redundancy study above finds broad
      coverage matters more than difficulty alone.  ``method='uniform'`` gives
      the matching stratified-random ablation.
    * **Terminal phase** -- a bonus on the ``terminal_window`` steps before
      every episode end keeps the precision-critical final approach dense
      (cf. semantic-critical-phase preservation in ``ESPADA``).

    The score uses only data the BC target already consumes (policy
    observations, expert actions and done flags); no privileged environment
    state is read.  ``returns`` (T, N), when provided, is gathered with the
    same indices so the critic still regresses the exact full-trajectory
    discounted return-to-go instead of a return recomputed on the decimated
    sequence.  Selected steps are returned in per-replica time order.

    Returns the same tuple shape as the inputs, with the (optional) returns
    gathered last.  A no-op when ``budget`` already covers the horizon.
    """
    T, N = dones.shape
    if terminal_mask is not None and terminal_mask.shape != dones.shape:
        raise ValueError(
            f'terminal_mask shape {tuple(terminal_mask.shape)} != dones {tuple(dones.shape)}'
        )
    idx = _selection_indices(
        obs, acts, dones, budget=budget, segments=segments, method=method,
        act_change=act_change, neighbor=neighbor,
        terminal=terminal, terminal_window=terminal_window,
        terminal_mask=terminal_mask,
    )
    if idx is None:
        return obs, acts, rews, dones, returns
    B = idx.shape[1]

    def _take(x):
        if x is None:
            return None
        shape = x.shape[2:]
        index = idx.transpose(0, 1).reshape(B, N, *((1,) * len(shape)))
        index = index.expand(B, N, *shape)
        return torch.gather(x, 0, index)

    return _take(obs), _take(acts), _take(rews), _take(dones), _take(returns)


def select_informative_indices_flat(obs, acts, dones, episode_id, *,
                                    terminated=None, success=None, drop_failed=True,
                                    **select_cfg):
    """Episode-aware variant of the TSS selection for flat dataset rows.

    ``flatten_split`` returns episode-contiguous rows ``(M, d)`` plus an
    ``episode_id`` column.  This returns the global row indices of the same
    TSS/keyframe + stratified selection applied inside every episode (with a
    per-episode budget), so a dataset of many long demonstrations is reduced
    without mixing or truncating episodes.  ``obs``/``acts``/``dones`` are used
    only for scoring, so callers remain free to gather whichever arrays
    (delta-transformed actions, targets, masks, ...) accompany the rows.

    ``terminated``/``success`` (per row) refine the terminal bonus to true
    episode ends instead of every auto-reset, and ``drop_failed`` removes
    expert episodes that did not succeed (falling back to keeping everything
    when the split contains no successful episode at all).

    Returns ``None`` when nothing was filtered or truncated.
    """
    if episode_id.shape != dones.shape:
        raise ValueError(
            f'episode_id shape {tuple(episode_id.shape)} != dones {tuple(dones.shape)}'
        )
    if terminated is not None and terminated.shape != dones.shape:
        raise ValueError('terminated must match the dones (M,) shape')
    if success is not None and success.shape != dones.shape:
        raise ValueError('success must match the dones (M,) shape')
    counts = torch.unique_consecutive(episode_id, return_counts=True)[1].tolist()
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + int(c))
    n_done = int(dones.sum())
    filter_failures = (success is not None and n_done > 0
                       and float(success.sum()) / n_done >= BC_DROP_FAILED_MIN_FRACTION)
    idx_parts, reduced = [], False
    for s, c in zip(starts, counts):
        e = s + int(c)
        if drop_failed and filter_failures and not bool(success[e - 1]):
            reduced = True
            continue
        term_mask = None
        if terminated is not None or success is not None:
            term_mask = torch.zeros(e - s, dtype=torch.bool, device=dones.device)
            if terminated is not None:
                term_mask |= terminated[s:e]
            if success is not None:
                term_mask |= success[s:e]
        idx = _selection_indices(
            obs[s:e].unsqueeze(1), acts[s:e].unsqueeze(1), dones[s:e].unsqueeze(1),
            terminal_mask=None if term_mask is None else term_mask.unsqueeze(1),
            **select_cfg,
        )
        if idx is None:
            idx_parts.append(torch.arange(s, e, device=obs.device))
        else:
            reduced = True
            idx_parts.append(s + idx[0])
    if not reduced:
        return None
    return torch.cat(idx_parts)


# shared warm-start defaults; per-task entries pass a dict of overrides on
# top of these so every task's BC recipe stays directly comparable.
# ``steps=None`` rolls one full episode horizon (``env.max_steps``) so the
# expert's terminal behaviour is recorded; ``select`` then keeps only the most
# informative transitions (a fixed per-replica budget) instead of training on
# and storing every step of a long episode.
#
# Curation defaults live in ONE place so the two collection paths (rollout and
# dataset) and the orchestrators cannot drift apart.  The method is resolved
# per task: terminal-type tasks (rocket / missile / docking / rendezvous) use
# the TSS keyframe ranking, tracking/cruise tasks keep broad uniform coverage
# (``LESS IS MORE`` finds coverage matters more than difficulty there).
BC_SELECT_DEFAULTS = {'budget': 512, 'segments': 8}
BC_TASK_SELECT_METHOD = {
    'rocket': 'tss', 'missile': 'tss', 'docking': 'tss', 'rendezvous': 'tss',
    'quadx': 'uniform', 'quadx_track': 'uniform', 'fixedwing': 'uniform',
    'quadplane': 'uniform', 'satellite': 'uniform', 'transfer': 'uniform',
}
# Bump whenever the curation rule or a BC default changes: cached BC
# checkpoints (orchestrators) carry this version and are refitted on mismatch.
BC_RECIPE_VERSION = 2
# Safety floor for failure filtering: a near-zero expert success rate must not
# silently throw away almost all demonstrations.  Filtering is applied only
# when at least this fraction of episode ends succeeded; otherwise the split
# is used as-is (the audited baselines are all well above this floor).
BC_DROP_FAILED_MIN_FRACTION = 0.1
BC_DEFAULTS = {'steps': 512, 'epochs': 100, 'batch_size': 8192, 'lr': 1e-3,
               'gamma': None, 'fit_critic': True,
               'select': dict(BC_SELECT_DEFAULTS), 'bc_val_fraction': 0.0}


def resolve_bc_select(task, select=None):
    """Merge the shared curation defaults and pick the task's score method.

    ``select=None`` disables the informative-subset reduction (train on every
    collected step).  An explicit ``method`` always wins.
    """
    if select is None:
        return None
    cfg = {**BC_SELECT_DEFAULTS, **dict(select)}
    cfg.setdefault('method', BC_TASK_SELECT_METHOD.get(task, 'tss'))
    return cfg


def bc_recipe_fingerprint(task, bc_cfg):
    """Stable hash of the parts of a BC recipe that determine the fitted data.

    The orchestrators store this next to a shared BC checkpoint; a mismatch
    (old 512-step data, changed selector, different budget, ...) forces a
    refit instead of silently reusing stale initialization.
    """
    payload = {
        'version': BC_RECIPE_VERSION,
        'task': task,
        'steps': bc_cfg.get('steps'),
        'collection': bc_cfg.get('collection', 'fixed_steps'),
        'select': resolve_bc_select(task, bc_cfg.get('select')),
        'epochs': bc_cfg.get('epochs'),
        'batch_size': bc_cfg.get('batch_size'),
        'lr': bc_cfg.get('lr'),
        'fit_critic': bool(bc_cfg.get('fit_critic', True)),
        'bc_val_fraction': float(bc_cfg.get('bc_val_fraction') or 0.0),
        'delta_action_scale': bc_cfg.get('delta_action_scale'),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()


def save_bc_recipe(checkpoint, task, bc_cfg):
    """Write the recipe sidecar next to a BC checkpoint."""
    payload = {
        'version': BC_RECIPE_VERSION,
        'task': task,
        'fingerprint': bc_recipe_fingerprint(task, bc_cfg),
        'recipe': {k: bc_cfg.get(k) for k in
                   ('steps', 'epochs', 'batch_size', 'lr', 'fit_critic',
                    'delta_action_scale')} | {
            'select': resolve_bc_select(task, bc_cfg.get('select')),
        },
    }
    with open(str(checkpoint) + '.recipe.json', 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def bc_recipe_matches(checkpoint, task, bc_cfg):
    """True only if ``checkpoint`` was fitted with the identical recipe."""
    path = str(checkpoint) + '.recipe.json'
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (saved.get('fingerprint') == bc_recipe_fingerprint(task, bc_cfg)
            and saved.get('task') == task)


def collect_expert(env, steps=None, *, performance=False):
    """Roll the env for `steps` vectorized steps under the task's classical
    controller (``env.conventional_controller``), instantiated here and driven
    directly; the env never sees it.

    ``steps=None`` means one full episode: the horizon is read from the env's
    ``max_steps`` (falling back to ``max_episode_length``).  Fixed step counts
    that are shorter than the episode silently truncate the expert trajectory
    (e.g. rocket's 512 steps only reach 12.8 s of a 60 s descent), so long
    horizon tasks must not rely on the 512-step shared default.

    Returns device-resident tensors: obs (T, N, obs_dim), act (T, N, A),
    rew (T, N), done (T, N) where done marks the env auto-reset at that
    step, plus ``terminated`` (T, N) and ``success`` (T, N) masks so the
    curation step can reward true terminal ends and drop failed expert
    episodes.  The controller's stateful parts are reset on the same mask the
    env resets on, so per-env controller state always matches the env.
    """
    if env.conventional_controller is None:
        raise RuntimeError(f'{type(env).__name__} declares no conventional_controller')
    if steps is None:
        steps = getattr(env, 'max_steps', None) or getattr(env, 'max_episode_length', None)
        if not steps:
            raise ValueError(
                f'{type(env).__name__} has no max_steps/max_episode_length; '
                'pass an explicit BC collection horizon'
            )
    steps = int(steps)
    if performance:
        from algorithms.warmstart.performance import prepare_collection
        prepare_collection(env)
    ctrl = env.conventional_controller(env, design=getattr(env, "controller_design", None))
    obs = env.get_observations()
    N, A = env.num_envs, env.num_actions
    T = steps
    obss = torch.empty(T, N, obs['policy'].shape[-1], device=env.device, dtype=torch.float32)
    acts = torch.empty(T, N, A, device=env.device, dtype=torch.float32)
    rews = torch.empty(T, N, device=env.device)
    dones = torch.empty(T, N, device=env.device, dtype=torch.bool)
    terms = torch.zeros(T, N, device=env.device, dtype=torch.bool)
    succs = torch.zeros(T, N, device=env.device, dtype=torch.bool)
    completed = getattr(env, 'completed_successes', None)
    succ_prev = completed.clone() if completed is not None else None
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
        # Expert rollouts can occasionally drive a replica into a state where
        # the Genesis rigid solver diverges (NaN accelerations).  The plant's
        # ``_step_scene`` recovers up to 10 times; if it still raises, or if
        # the observation silently contains NaN, reset the affected replicas
        # and continue collecting from a clean spawn state.
        try:
            with torch.no_grad() if performance else nullcontext():
                _, rew, term, trunc, _ = env.step(expert)
        except Exception:
            # The solver diverged; force a full reset of every replica to
            # clear corrupted internal solver state, then continue.
            env.reset_idx(torch.ones(N, dtype=torch.bool, device=env.device))
            ctrl.reset_idx(torch.ones(N, dtype=torch.bool, device=env.device))
            rew = torch.zeros(N, device=env.device)
            term = torch.ones(N, dtype=torch.bool, device=env.device)
            trunc = torch.zeros(N, dtype=torch.bool, device=env.device)
        if completed is not None:
            succ = completed > succ_prev
            succ_prev = completed.clone()
        else:
            succ = torch.zeros(N, dtype=torch.bool, device=env.device)
        rews[t], dones[t], terms[t], succs[t] = rew, term | trunc, term, succ
        obs = env.get_observations()
        nan_mask = ~torch.isfinite(obs['policy']).all(-1)
        if nan_mask.any():
            env.reset_idx(nan_mask)
            ctrl.reset_idx(nan_mask)
            obs = env.get_observations()
            # The artificial recovery reset is an episode boundary: close the
            # episode as a truncated (invalid-tail) end for those replicas so
            # curation never stitches pre- and post-reset states into one
            # episode or treats the recovery as a success.
            dones[t] |= nan_mask
            terms[t] &= ~nan_mask
            succs[t] &= ~nan_mask
        ctrl.reset_idx(term | trunc)
    return obss, acts, rews, dones, terms, succs


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


def returns_to_go_flat(rews, dones, terminated, episode_id, gamma):
    """Episode-vectorised return-to-go for flat dataset rows.

    Equivalent to calling ``returns_to_go`` on each episode separately with
    boundary-aware masks (success/violation = true terminal, timeout tails
    invalid), but implemented with one reverse discounted cumulative sum per
    episode instead of a Python loop over every row.  ``episode_id`` must be
    episode-contiguous (as produced by ``flatten_split``).

    Returns ``(targets, valid)`` shaped like ``rews``/``dones``.
    """
    if gamma is None:
        raise ValueError('returns_to_go_flat: gamma is required')
    if episode_id.shape != rews.shape or dones.shape != rews.shape:
        raise ValueError(
            f'returns_to_go_flat inputs disagree: rews {tuple(rews.shape)}, '
            f'dones {tuple(dones.shape)}, episode_id {tuple(episode_id.shape)}'
        )
    if terminated.shape != rews.shape:
        raise ValueError('returns_to_go_flat: terminated must match rews shape')
    counts = torch.unique_consecutive(episode_id, return_counts=True)[1].tolist()
    out = torch.empty_like(rews)
    valid = torch.zeros(dones.shape, dtype=torch.bool, device=rews.device)
    s = 0
    for c in counts:
        e = s + int(c)
        r = rews[s:e].to(torch.float32)
        disc = gamma ** torch.arange(e - s, dtype=torch.float32, device=rews.device)
        suffix = torch.flip(torch.cumsum(torch.flip(r * disc, [0]), 0), [0])
        out[s:e] = (suffix / disc.clamp_min(1e-30)).to(rews.dtype)
        # With success folded into ``terminated``, a complete episode ending in
        # a true terminal has every target valid; a timeout tail is invalid
        # from the start (returns_to_go propagates that backward).
        valid[s:e] = bool(terminated[e - 1])
        s = e
    return out, valid


def select_informative_indices_rollout(obs, acts, dones, *, terminated=None,
                                       success=None, budget=512, segments=8,
                                       method='tss', act_change=1.0, neighbor=2,
                                       terminal=2.0, terminal_window=5,
                                       drop_failed=True):
    """Episode-aware informative selection for the (T, N) rollout path.

    Unlike the rectangular phase split, episodes are located per replica from
    the ``done`` mask, so a collection horizon that contains several auto-reset
    episodes (e.g. missile kills at ~600/900 steps, quadx successes at ~670)
    is stratified inside each episode instead of across the whole window.
    The per-replica budget is distributed across that replica's episodes in
    proportion to their lengths.

    ``terminated``/``success`` refine the terminal bonus (true ends only) and
    ``drop_failed`` removes episodes that did not succeed.  Returns 1-D
    ``(time_idx, replica_idx)`` tensors, or ``None`` when nothing changed.
    """
    T, N = dones.shape
    if T <= 1 or N == 0:
        return None
    if terminated is None:
        terminated = dones
    if success is None:
        success = torch.zeros_like(dones)
    if terminated.shape != dones.shape or success.shape != dones.shape:
        raise ValueError('terminated/success must match the dones (T, N) shape')
    budget_total = T if budget is None else int(budget)
    n_done = int(dones.sum())
    filter_failures = (drop_failed and n_done > 0
                       and float(success.sum()) / n_done >= BC_DROP_FAILED_MIN_FRACTION)
    if budget_total >= T and not filter_failures:
        return None
    term_mask = terminated | success
    device = dones.device
    t_parts, n_parts = [], []
    for n in range(N):
        ends = torch.nonzero(dones[:, n], as_tuple=False).flatten() + 1
        if ends.numel() == 0 or int(ends[-1]) < T:
            ends = torch.cat([ends, torch.tensor([T], device=device)])
        starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=device), ends[:-1]])
        episodes = list(zip(starts.tolist(), ends.tolist()))
        keep = [i for i, (_, e) in enumerate(episodes)
                if (not drop_failed) or (not filter_failures) or bool(success[e - 1, n])]
        if not keep:
            continue
        lengths = [episodes[i][1] - episodes[i][0] for i in keep]
        total = sum(lengths)
        alloc = [min(L, max(1, int(round(budget_total * L / total))))
                 for L in lengths]
        while sum(alloc) > budget_total:
            j = max(range(len(alloc)), key=lambda i: alloc[i])
            if alloc[j] <= 1:
                break
            alloc[j] -= 1
        for i, k in zip(keep, alloc):
            if k <= 0:
                continue
            s, e = episodes[i]
            if k >= e - s:
                ts = torch.arange(s, e, device=device)
            else:
                idx = _selection_indices(
                    obs[s:e, n:n + 1], acts[s:e, n:n + 1], dones[s:e, n:n + 1],
                    budget=k, segments=segments, method=method,
                    act_change=act_change, neighbor=neighbor, terminal=terminal,
                    terminal_window=terminal_window,
                    terminal_mask=term_mask[s:e, n:n + 1],
                )
                ts = (torch.arange(s, e, device=device) if idx is None
                      else s + idx[0])
            t_parts.append(ts)
            n_parts.append(torch.full_like(ts, n))
    if not t_parts:
        return None
    return torch.cat(t_parts), torch.cat(n_parts)


def fit_behavior_cloning(runner, obs, acts, rews, dones, *, epochs, batch_size, lr,
                         gamma, fit_critic=True, log=print, val=None,
                         normalizer_mode='per_batch', freeze_normalizer=False,
                         terminated=None, truncated=None, cut=None,
                         bootstrap_values=None, sample_weights=None, performance=False,
                         returns=None, returns_valid=None, pair_mask=None):
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
    - ``pair_mask``: optional boolean [time, replica] selection applied after
      full-trajectory returns, excluding post-reset actor and critic samples.
    - ``returns``: optional (T, N) precomputed return-to-go targets.  Used by
      the informative-subset path (``select_informative_pairs``), where the
      exact full-trajectory returns are gathered with the selected pairs so
      the critic never sees a target recomputed on the decimated sequence.
      When given, ``terminated``/``truncated``/``cut``/``bootstrap_values``
      are ignored.
    - ``returns_valid``: optional (T, N) bool mask aligned with ``returns``;
      rows excluded by the boundary-aware target builder stay excluded after
      selection (defaults to all valid).
    """
    # Retain the time-by-replica layout until discounted returns are computed.
    # Masking or concatenating episodes first would leak returns across replicas.
    actor, critic = runner.alg.actor, runner.alg.critic
    device = obs.device
    T, N = obs.shape[:2]
    if returns is not None and tuple(returns.shape) != (T, N):
        raise ValueError(
            f'returns shape {tuple(returns.shape)} != obs (T, N) {(T, N)}'
        )
    if returns_valid is not None and tuple(returns_valid.shape) != (T, N):
        raise ValueError(
            f'returns_valid shape {tuple(returns_valid.shape)} != obs (T, N) {(T, N)}'
        )
    flat_obs = obs.reshape(T * N, -1)
    flat_act = acts.reshape(T * N, -1)
    flat_w = (sample_weights.reshape(T * N).to(device=device, dtype=flat_obs.dtype)
              if sample_weights is not None else None)

    if pair_mask is not None:
        if pair_mask.shape != (T, N) or pair_mask.dtype != torch.bool:
            raise ValueError('pair_mask must be a boolean [time, replica] tensor')
        selected = pair_mask.flatten()
        flat_obs, flat_act = flat_obs[selected], flat_act[selected]
        if flat_w is not None:
            flat_w = flat_w[selected]
    M = flat_obs.shape[0]
    if M == 0:
        raise ValueError('expert collection contains no selected pairs')

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
        if returns is not None:
            targets = returns.reshape(T, N)
            valid = (returns_valid.reshape(T, N) if returns_valid is not None
                     else torch.ones_like(targets, dtype=torch.bool))
        elif terminated is not None:
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
        if pair_mask is not None:
            flat_ret, flat_valid = flat_ret[selected], flat_valid[selected]
        # Filter the valid pairs ONCE: the critic never sees an excluded pair,
        # so no per-minibatch ``bool(keep.any())`` host sync or re-masking.
        flat_obs_c = flat_obs[flat_valid]
        flat_ret_c = flat_ret[flat_valid]
        m_critic = int(flat_obs_c.shape[0])
        critic.train()
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        if m_critic and normalizer_mode == 'fit_once':
            critic.update_normalization({'policy': flat_obs_c})

    actor_step, critic_step = None, None
    if performance and device.type == 'cuda':
        from algorithms.warmstart.performance import make_bc_step

        actor_step = make_bc_step(actor, actor_opt, normalize=normalizer_mode == 'per_batch',
                                  weighted=flat_w is not None)
        if fit_critic and m_critic:
            # Main filters invalid targets once before the independent critic
            # pass. Preserve its permutation, batch boundaries and RNG order.
            critic_step = make_bc_step(critic, critic_opt,
                                       normalize=normalizer_mode == 'per_batch', critic=True)

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
            if actor_step is not None:
                weights = () if flat_w is None else (flat_w[idx],)
                a_loss = actor_step(b_obs, flat_act[idx], *weights)
            else:
                td = {'policy': b_obs}
                if normalizer_mode == 'per_batch':
                    actor.update_normalization({'policy': b_obs})
                pred = actor(td)
                if flat_w is None:
                    a_loss = torch.nn.functional.mse_loss(pred, flat_act[idx])
                else:
                    w = flat_w[idx].unsqueeze(-1)
                    per_pair = (pred - flat_act[idx]).pow(2).mean(dim=-1)
                    a_loss = (w.squeeze(-1) * per_pair).sum() / w.sum().clamp_min(1e-12)
                actor_opt.zero_grad(set_to_none=True)
                a_loss.backward()
                actor_opt.step()
            # Consume borrowed graph outputs before the next replay overwrites
            # them; retain main's accumulation dtype and reduction order.
            a_sum += a_loss.detach()
            a_batches += 1
        a_losses.append(float(a_sum / max(a_batches, 1)))
        c_epoch_batches, c_sum = 0, torch.zeros((), device=device)
        if fit_critic and m_critic:
            c_perm = torch.randperm(m_critic, device=device)
            for i in range(0, m_critic, batch_size):
                idx = c_perm[i:i + batch_size]
                b_obs_c = flat_obs_c[idx]
                if critic_step is not None:
                    c_loss = critic_step(b_obs_c, flat_ret_c[idx])
                else:
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
               'critic_target': ('precomputed full-trajectory returns-to-go (selected pairs)'
                                 if returns is not None else
                                 ('boundary-aware returns_to_go; unfinished tails excluded '
                                  'unless bootstrap provided' if terminated is not None
                                  else 'legacy returns_to_go (zero terminal value)')),
               'critic_valid_pairs': m_critic if fit_critic else None,
               'critic_excluded_pairs': M - m_critic if fit_critic else None,
               'sample_weighting': bool(flat_w is not None),
               'sample_weight_mean': (float(flat_w.mean()) if flat_w is not None else None)}
    if fit_critic:
        summary['final_critic_mse'] = c_losses[-1] if c_losses else None
    if performance:
        summary['performance_paths'] = (["graphed_bc_actor"] if actor_step is not None else [])
        if critic_step is not None:
            summary['performance_paths'].append("graphed_bc_critic")
        if fit_critic:
            summary['performance_paths'].append("prefiltered_critic")
    log(f'[bc] warm start done in {wall:.1f}s: {summary}')
    return summary
