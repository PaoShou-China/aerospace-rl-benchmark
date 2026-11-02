"""Episode-level expert dataset for the controller-prior experiments.

This is the *independent* data interface requested by the controller-prior
study.  It does NOT replace ``collect_expert`` (the legacy BC path used by
``run_training`` stays bit-for-bit compatible); it adds the bookkeeping needed
for a defensible train/validation/test split:

- raw observations and expert actions;
- episode ID and replica ID per step, plus in-episode time;
- ``terminated`` / ``truncated`` / ``success`` / ``done`` separately (a
  success is marked ``truncated`` by the task envs, so it is recorded as a
  *terminal* event for return targeting, not as a timeout);
- terminal observation captured from ``extras['terminal_observations']``
  BEFORE the automatic reset, kept ONCE per episode in the episode table
  (the per-step duplicate is dropped: it costs as much as the observation
  tensor and is never consumed by the BC targets);
- initial condition (initial policy observation) and available task/plant
  parameters per episode;
- controller identity/configuration, random seed and collection wall time.

Splitting is by COMPLETE EPISODE only: adjacent time steps of one episode never
end up in different splits.  A fixed-length collection ends mid-episode; those
unfinished tails are marked ``cut`` and are dropped from the supervised splits
(recorded, never silently treated as true terminations).  A complete episode
that ends by timeout keeps ``truncated=True`` so the critic-target builder can
exclude its unfinished tail unless a bootstrap value is supplied.

All collection is batched on the env device (no Python physics loop); only the
one-shot episode-table construction at the end iterates over boundary indices
on the host.
"""
import json
import math
import pathlib
import time

import torch

from algorithms.warmstart.bc_warmstart import returns_to_go

__all__ = ['collect_expert_dataset', 'save_expert_dataset', 'load_expert_dataset',
           'split_dataset', 'flatten_split', 'dataset_summary', 'merge_datasets',
           'select_representative_episodes']


def _controller_config(ctrl):
    if hasattr(ctrl, 'config'):
        return ctrl.config()
    if hasattr(ctrl, 'model_config'):
        return ctrl.model_config()
    return {'name': getattr(ctrl, 'name', type(ctrl).__name__)}


@torch.no_grad()
def collect_expert_dataset(env, *, steps=None, episodes_per_replica=2,
                           max_steps=20000, controller=None, seed=None,
                           policy=None, beta=0.0, select_episodes=None, log=print):
    """Roll the task env under its conventional controller and record episodes.

    Provide either ``steps`` (fixed-length collection, unfinished tails are
    marked and dropped) or ``episodes_per_replica`` (roll until every replica
    has that many COMPLETE episodes, safety-capped by ``max_steps``).

    DAgger mode (``policy`` given): the controller still labels every visited
    state (``acts`` stores its command), but the EXECUTED action is the
    beta-mixture ``beta * expert + (1 - beta) * policy(obs)``.  With beta=0
    this is pure student rollouts relabelled by the controller (the DAgger
    aggregation step); beta=1 recovers plain expert collection.  The student
    is a plain callable obs -> action (no internal state is read from the env).
    """
    if env.conventional_controller is None and controller is None:
        raise RuntimeError(f'{type(env).__name__} declares no conventional_controller')
    ctrl = controller or env.conventional_controller(env)
    if steps is not None and max_steps is not None:
        max_steps = int(steps)
    n_env, num_actions = int(env.num_envs), int(env.num_actions)
    device = env.device
    if steps is None:
        # allocate for the requested complete episodes, hard-capped by max_steps
        T_cap = min(int(max_steps), int(episodes_per_replica) *
                    int(getattr(env, 'max_episode_length', max_steps)) + 1)
    else:
        T_cap = int(steps)
    if T_cap < 1:
        raise ValueError('collection horizon must be positive')
    obs_dim = int(env.get_observations()['policy'].shape[-1])
    obss = torch.empty(T_cap, n_env, obs_dim, device=device, dtype=torch.float32)
    acts = torch.empty(T_cap, n_env, num_actions, device=device, dtype=torch.float32)
    rews = torch.empty(T_cap, n_env, device=device, dtype=torch.float64)
    term = torch.zeros(T_cap, n_env, device=device, dtype=torch.bool)
    trunc = torch.zeros_like(term)
    done = torch.zeros_like(term)
    success = torch.zeros_like(term)
    ep_id = torch.zeros(T_cap, n_env, device=device, dtype=torch.long)
    t_in_ep = torch.zeros_like(ep_id)
    terminal_obs = torch.zeros(T_cap, n_env, obs_dim, device=device, dtype=torch.float32)

    obs = env.get_observations()
    ctrl.reset_idx(torch.ones(n_env, dtype=torch.bool, device=device))
    succ_prev = env.completed_successes.clone()
    ep_count = torch.zeros(n_env, device=device, dtype=torch.long)
    current_ep = torch.zeros_like(ep_count)
    t_ep = torch.zeros_like(ep_count)
    episodes_done = 0
    t_used = 0
    start_wall = time.perf_counter()
    checked = False
    for t in range(T_cap):
        expert = ctrl.actions(obs)
        expert = expert.to(torch.float32)
        if expert.shape != (n_env, num_actions):
            raise ValueError(f'expert action shape {tuple(expert.shape)} != ({n_env}, {num_actions})')
        if not torch.isfinite(expert).all():
            raise ValueError(f'non-finite expert action at collection step {t}')
        if not checked:
            bound = float(getattr(env, 'clip_actions', 1.))
            if float(expert.abs().max()) > bound + 1e-6:
                raise ValueError('expert action violates the env action bound')
            checked = True
        executed = expert
        if policy is not None:
            student = policy(obs)
            if not isinstance(student, torch.Tensor) or student.shape != expert.shape:
                raise ValueError('DAgger student must return an action tensor shaped '
                                 f'{tuple(expert.shape)}')
            if not torch.isfinite(student).all():
                raise ValueError(f'non-finite DAgger student action at step {t}')
            student = student.to(device=expert.device, dtype=expert.dtype)
            executed = (float(beta) * expert + (1.0 - float(beta)) * student).clamp(
                -bound, bound)
        obss[t], acts[t] = obs['policy'], expert
        term_out, trunc_out = None, None
        obs, rew, term_out, trunc_out, extras = env.step(executed)
        episode_done = term_out | trunc_out
        succ = env.completed_successes > succ_prev
        succ_prev = env.completed_successes.clone()
        rews[t], term[t], trunc[t] = rew, term_out, trunc_out
        done[t], success[t] = episode_done, succ
        ep_id[t], t_in_ep[t] = current_ep, t_ep
        if 'terminal_observations' in extras:
            terminal_obs[t] = extras['terminal_observations']['policy']
        else:
            terminal_obs[t] = obs['policy']
        # advance bookkeeping for the next step (vectorised, on device)
        t_ep = torch.where(episode_done, torch.zeros_like(t_ep), t_ep + 1)
        current_ep = current_ep + episode_done.to(torch.long)
        ep_count = ep_count + episode_done.to(torch.long)
        episodes_done += int(episode_done.sum())
        ctrl.reset_idx(episode_done)
        t_used = t + 1
        if steps is None and bool((ep_count >= episodes_per_replica).all()):
            break
    wall = time.perf_counter() - start_wall

    T = t_used
    data = {k: v[:T].contiguous() for k, v in dict(
        obs=obss, act=acts, rew=rews, terminated=term, truncated=trunc, done=done,
        success=success, episode_id=ep_id, t_in_episode=t_in_ep,
        terminal_obs=terminal_obs).items()}

    # ---------------- episode table (one host pass over boundary indices) ----
    starts = {(int(i), int(data['episode_id'][t, i])): int(t)
              for t, i in torch.nonzero((data['t_in_episode'] == 0), as_tuple=False).cpu().tolist()}
    episodes = []
    for t, i in torch.nonzero(data['done'], as_tuple=False).cpu().tolist():
        e = int(data['episode_id'][t, i])
        start = starts.get((i, e))
        if start is None:
            raise RuntimeError(f'missing episode start for replica {i} episode {e}')
        segment = slice(start, t + 1)
        ep = dict(replica=i, episode_id=e, start=start, end=t,
                  complete=True,
                  terminated=bool(data['terminated'][t, i]),
                  truncated=bool(data['truncated'][t, i]),
                  success=bool(data['success'][t, i]),
                  cut=False, steps=t - start + 1,
                  episode_return=float(data['rew'][segment, i].sum()),
                  initial_obs=[float(x) for x in data['obs'][start, i]],
                  terminal_obs=[float(x) for x in data['terminal_obs'][t, i]])
        episodes.append(ep)
    # unfinished tails (fixed-length collection only)
    unfinished = 0
    complete_keys = {(ep['replica'], ep['episode_id']) for ep in episodes}
    for t, i in torch.nonzero((data['t_in_episode'] == 0), as_tuple=False).cpu().tolist():
        e = int(data['episode_id'][t, i])
        if (i, e) not in complete_keys:
            end = int(data['episode_id'][:, i].eq(e).nonzero()[-1])
            episodes.append(dict(replica=i, episode_id=e, start=t, end=end,
                                 complete=False, terminated=False, truncated=False,
                                 success=False, cut=True, steps=end - t + 1,
                                 episode_return=float(data['rew'][t:end + 1, i].sum()),
                                 initial_obs=[float(x) for x in data['obs'][t, i]],
                                 terminal_obs=[float(x) for x in data['terminal_obs'][end, i]]))
            unfinished += 1

    # The per-step terminal observation doubles the observation tensor on disk
    # and is never used by the BC targets (the episode table above already
    # stores every episode's terminal observation), so drop it here.
    data.pop('terminal_obs', None)

    meta = dict(task=env.cfg.get('vehicle'), controller=_controller_config(ctrl),
                controller_name=getattr(ctrl, 'name', type(ctrl).__name__),
                teacher_identity=getattr(ctrl, 'name', type(ctrl).__name__),
                env_cfg=dict(env.cfg), seed=seed,
                collection=dict(protocol='episodes_per_replica' if steps is None else 'fixed_steps',
                                steps_requested=steps, episodes_requested=episodes_per_replica,
                                steps_used=int(T), envs=int(n_env), transitions=int(T * n_env),
                                complete_episodes=sum(1 for ep in episodes if ep['complete']),
                                unfinished_episodes=unfinished,
                                dropped_unfinished_tails=True,
                                episodes_done=int(episodes_done),
                                collection_wall_seconds=round(wall, 2)),
                boundary_semantics=('terminated = task violation; truncated = timeout OR success '
                                    '(success recorded separately and treated as terminal for '
                                    'return targets); collection cut = fixed-length tail'),
                dagger=(None if policy is None else {'beta': float(beta),
                                                      'student': 'policy given (labels = controller)'}))
    out = dict(data=data, episodes=episodes, meta=meta)
    if select_episodes is not None:
        # representative subset: coverage of the initial-condition space, not
        # thousands of near-duplicate rollouts
        out = select_representative_episodes(out, int(select_episodes))
    return out


@torch.no_grad()
def _farthest_point_indices(features, n):
    """Greedy max-min (farthest-point) pick on z-scored features.

    Starts from the point farthest from the feature centroid (an extreme of
    the sampled set) and repeatedly adds the point farthest from everything
    already chosen, so the selection spans the feature space instead of
    repeating its densest region.
    """
    x = torch.as_tensor(features, dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError('features must be a non-empty (M, d) matrix')
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - x.mean(dim=0)) / std
    n = int(min(int(n), x.shape[0]))
    centroid = x.mean(dim=0, keepdim=True)
    first = int(torch.cdist(x, centroid).squeeze(1).argmax())
    chosen = [first]
    remaining = torch.ones(x.shape[0], dtype=torch.bool)
    remaining[first] = False
    dist = torch.cdist(x, x[first].unsqueeze(0)).squeeze(1)
    while len(chosen) < n:
        masked = torch.where(remaining, dist, torch.full_like(dist, -1.0))
        j = int(masked.argmax())
        if not bool(remaining[j]):
            break  # every candidate has been picked (duplicate-heavy pool)
        chosen.append(j)
        remaining[j] = False
        dist = torch.minimum(dist, torch.cdist(x, x[j].unsqueeze(0)).squeeze(1))
    return chosen


def _repack_episodes(dataset, keep):
    """Re-index the (time, replica) grid so column ``j`` holds kept episode j.

    Episode segments become ``start=0`` and are padded to the longest kept
    episode (the episode table is authoritative, so padding rows are never fed
    to the BC targets).
    """
    data = dataset['data']
    n_steps, n_env = int(data['obs'].shape[0]), int(data['obs'].shape[1])
    lens = [int(ep['end']) - int(ep['start']) + 1 for ep in keep]
    horizon, n_new = max(lens), len(keep)
    out = {}
    for k, v in data.items():
        if v.dim() < 2 or v.shape[0] != n_steps or v.shape[1] != n_env:
            out[k] = v
            continue
        new = torch.zeros((horizon, n_new) + tuple(v.shape[2:]), dtype=v.dtype, device=v.device)
        t_local = (torch.arange(horizon, dtype=v.dtype, device=v.device)
                   if k == 't_in_episode' else None)
        for j, ep in enumerate(keep):
            s, e = int(ep['start']), int(ep['end']) + 1
            new[:e - s, j] = v[s:e, int(ep['replica'])]
            if k == 'episode_id':
                new[:e - s, j] = j
            elif k == 't_in_episode':
                new[:e - s, j] = t_local[:e - s]
        out[k] = new
    episodes = []
    for j, ep in enumerate(keep):
        e = dict(ep)
        e.update(replica=j, episode_id=j, start=0, end=lens[j] - 1, steps=lens[j])
        episodes.append(e)
    meta = dict(dataset['meta'])
    col = dict(meta.get('collection') or {})
    col.update(steps_used=horizon, transitions=int(sum(lens)), envs=n_new,
               complete_episodes=n_new, episodes_done=n_new)
    meta['collection'] = col
    return dict(data=out, episodes=episodes, meta=meta)


def select_representative_episodes(dataset, n_episodes, method='farthest_point'):
    """Keep a REPRESENTATIVE subset of complete episodes instead of every rollout.

    Selection runs greedy farthest-point on the z-scored initial policy
    observation of each episode, so the kept episodes cover the initial-
    condition space (extremes included) rather than duplicating its densest
    region.  Episodes are kept whole and the grid is repacked
    (``_repack_episodes``), so the episode-disjoint split still applies.
    """
    complete = [ep for ep in dataset['episodes'] if ep['complete']]
    n_want = int(n_episodes)
    if n_want <= 0 or len(complete) <= n_want:
        return dataset
    if method != 'farthest_point':
        raise ValueError(f'unknown episode selection method {method!r}')
    feats = torch.tensor([ep['initial_obs'] for ep in complete], dtype=torch.float64)
    keep = [complete[i] for i in _farthest_point_indices(feats, n_want)]
    out = _repack_episodes(dataset, keep)
    out['meta']['selection'] = dict(method=method, requested=n_want,
                                    candidates=len(complete), kept=len(keep),
                                    feature='initial policy observation (z-scored)')
    return out


def _compact_data(data):
    """On-disk encoding: half-precision states/actions, float32 rewards and the
    narrowest integer width that fits the id columns.

    ``load_expert_dataset`` restores the canonical training dtypes
    (float32 states/actions, float64 rewards, int64 ids), so nothing downstream
    of the file format sees the compact encoding."""
    out = {}
    for k, v in data.items():
        if k in ('obs', 'act', 'terminal_obs'):
            out[k] = v.to(torch.float16)
        elif k == 'rew':
            out[k] = v.to(torch.float32)
        elif k in ('episode_id', 't_in_episode'):
            id_max = int(v.max()) if v.numel() else 0
            out[k] = v.to(torch.int16 if id_max < 32767 else torch.int32)
        else:
            out[k] = v
    return out


def _restore_data(data):
    """Inverse of ``_compact_data`` (canonical dtypes for training code)."""
    out = {}
    for k, v in data.items():
        if k in ('obs', 'act', 'terminal_obs'):
            out[k] = v.to(torch.float32)
        elif k == 'rew':
            out[k] = v.to(torch.float64)
        elif k in ('episode_id', 't_in_episode'):
            out[k] = v.to(torch.long)
        else:
            out[k] = v
    return out


def _subset_replicas(dataset, keep):
    """Keep the first ``keep`` env columns -- whole replicas, hence whole
    episodes, so the episode-disjoint split keeps working unchanged."""
    data = dataset['data']
    n_env = int(data['obs'].shape[1])
    keep = max(1, min(int(keep), n_env))
    if keep >= n_env:
        return dataset
    out = {k: (v[:, :keep].contiguous() if v.dim() >= 2 and v.shape[1] == n_env else v)
           for k, v in data.items()}
    episodes = [dict(ep) for ep in dataset['episodes'] if int(ep['replica']) < keep]
    meta = dict(dataset['meta'])
    col = dict(meta.get('collection') or {})
    col['envs'] = keep
    col['transitions'] = int(out['obs'].shape[0]) * keep
    col['complete_episodes'] = sum(1 for ep in episodes if ep['complete'])
    col['replicas_dropped'] = n_env - keep
    meta['collection'] = col
    return dict(data=out, episodes=episodes, meta=meta)


def save_expert_dataset(dataset, path, *, max_mb=None):
    """Write the dataset compactly, optionally capped at ``max_mb`` MiB.

    The cap is enforced by keeping a prefix of the env replicas (whole
    replicas = whole episodes are dropped, never a slice of an episode), so a
    size-capped dataset is still episode-disjoint splittable.  Returns the
    dataset exactly as it was written: compact dtypes and, if the cap kicked
    in, fewer replicas.  Callers that need trained-dtype tensors must go
    through ``load_expert_dataset``.
    """
    path = str(path)
    ds = dict(data=_compact_data(dataset['data']), episodes=dataset['episodes'],
              meta=dict(dataset['meta']))
    limit = None if max_mb is None else float(max_mb)
    n_collected = int(ds['data']['obs'].shape[1])
    size_mb = 0.0
    for _ in range(6):
        torch.save(dict(data=ds['data'], episodes=ds['episodes'], meta=ds['meta']), path)
        size_mb = pathlib.Path(path).stat().st_size / 2**20
        if limit is None or size_mb <= limit:
            break
        n_env = int(ds['data']['obs'].shape[1])
        n_complete = sum(1 for ep in ds['episodes'] if ep['complete'])
        # never shrink below three complete episodes (the smallest splittable set)
        min_keep = max(1, math.ceil(3 * n_env / max(1, n_complete)))
        keep = max(min_keep, int(math.floor(n_env * limit / size_mb)) - 1)
        if keep >= n_env:
            break  # cannot shrink further without losing the split
        ds = _subset_replicas(ds, keep)
    ds['meta']['storage'] = dict(
        encoding='float16 obs/act, float32 reward, int16/int32 ids, '
                 'no per-step terminal observation',
        max_mb=limit,
        file_mb=round(size_mb, 2),
        within_budget=bool(limit is None or size_mb <= limit),
        replicas_written=int(ds['data']['obs'].shape[1]),
        replicas_collected=n_collected,
    )
    torch.save(dict(data=ds['data'], episodes=ds['episodes'], meta=ds['meta']), path)
    return ds


def load_expert_dataset(path):
    obj = torch.load(str(path), weights_only=False)
    obj.setdefault('episodes', [])
    obj.setdefault('meta', {})
    obj['data'] = _restore_data(obj.get('data') or {})
    return obj


def _split_keys(episodes, seed, fractions=(0.7, 0.15, 0.15)):
    complete = sorted((ep for ep in episodes if ep['complete']),
                      key=lambda ep: (ep['replica'], ep['episode_id']))
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(complete), generator=g).tolist()
    n = len(complete)
    n_val = max(1, int(round(fractions[1] * n))) if n >= 3 else 0
    n_test = max(1, int(round(fractions[2] * n))) if n >= 3 else 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f'not enough complete episodes for a train/val/test split: {n}')
    splits = {'train': order[:n_train],
              'val': order[n_train:n_train + n_val],
              'test': order[n_train + n_val:n_train + n_val + n_test]}
    return complete, splits


def split_dataset(dataset, *, seed, fractions=(0.7, 0.15, 0.15)):
    """Episode-disjoint split; returns dict split -> list of episode segments."""
    complete, index = _split_keys(dataset['episodes'], seed, fractions)
    out = {}
    for name, ids in index.items():
        out[name] = [complete[i] for i in ids]
    return out


def flatten_split(dataset, episodes):
    """Flatten a list of complete episode segments into supervised pair arrays.

    Returns dict with obs (M,d), act (M,A), rew, done, terminated, truncated,
    success, cut, episode_id, replica_id, t_in_episode.  The final pair of
    every episode is its true boundary (no cross-episode leakage).
    """
    data = dataset['data']
    obs_l, act_l, rew_l = [], [], []
    term_l, trunc_l, succ_l, done_l, cut_l = [], [], [], [], []
    ep_l, rep_l, t_l = [], [], []
    for ep in episodes:
        i, s, e = ep['replica'], ep['start'], ep['end'] + 1
        obs_l.append(data['obs'][s:e, i])
        act_l.append(data['act'][s:e, i])
        rew_l.append(data['rew'][s:e, i])
        term_l.append(data['terminated'][s:e, i])
        trunc_l.append(data['truncated'][s:e, i])
        succ_l.append(data['success'][s:e, i])
        done_l.append(data['done'][s:e, i])
        cut_l.append(torch.zeros(e - s, device=data['obs'].device, dtype=torch.bool))
        ep_l.append(torch.full((e - s,), int(ep['episode_id']), device=data['obs'].device,
                               dtype=torch.long))
        rep_l.append(torch.full((e - s,), int(i), device=data['obs'].device, dtype=torch.long))
        t_l.append(torch.arange(e - s, device=data['obs'].device, dtype=torch.long))
    cat = lambda xs: torch.cat(xs) if xs else torch.empty(0, device=data['obs'].device)
    return dict(obs=torch.cat(obs_l) if obs_l else torch.empty(0, data['obs'].shape[-1], device=data['obs'].device),
                act=torch.cat(act_l) if act_l else torch.empty(0, data['act'].shape[-1], device=data['obs'].device),
                rew=cat(rew_l), terminated=cat(term_l), truncated=cat(trunc_l),
                success=cat(succ_l), done=cat(done_l), cut=cat(cut_l),
                episode_id=cat(ep_l), replica_id=cat(rep_l), t_in_episode=cat(t_l))


def merge_datasets(datasets):
    """Concatenate datasets (same env count / obs / action dims) along time.

    Used by DAgger aggregation: the original demonstration dataset and the
    student-visited relabelled rollouts are pooled into one supervised set.
    Episode segments keep their LOCAL replica index (that is the env column
    they were recorded from) and are shifted by each source's time offset;
    episode ids are renumbered so ``(replica, episode_id)`` stays unique.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError('merge_datasets needs at least one dataset')
    if len(datasets) == 1:
        return datasets[0]
    keys = ('obs', 'act', 'rew', 'terminated', 'truncated', 'done', 'success',
            'episode_id', 't_in_episode')
    obs_shapes = {tuple(ds['data']['obs'].shape[1:]) for ds in datasets}
    act_shapes = {tuple(ds['data']['act'].shape[1:]) for ds in datasets}
    if len(obs_shapes) != 1 or len(act_shapes) != 1:
        raise ValueError(f'datasets have different env/obs/action dims: '
                         f'{sorted(obs_shapes)} / {sorted(act_shapes)} '
                         '(collect with the same num_envs and observation contract)')
    t_off, id_off = 0, 0
    episodes, pieces = [], {k: [] for k in keys}
    for ds in datasets:
        data = ds['data']
        T = int(data['obs'].shape[0])
        for ep in ds['episodes']:
            e = dict(ep)
            e['start'] = int(ep['start']) + t_off
            e['end'] = int(ep['end']) + t_off
            e['episode_id'] = int(ep['episode_id']) + id_off
            episodes.append(e)
        for k in keys:
            # the per-step episode id must follow the renumbered episode table,
            # otherwise (replica, episode_id) lookups split the wrong segments
            pieces[k].append(data[k] + id_off if k == 'episode_id' else data[k])
        t_off += T
        id_off += (max((int(ep['episode_id']) for ep in ds['episodes']), default=-1) + 1)
    merged = {k: torch.cat(v, dim=0) if v else None for k, v in pieces.items()}
    src_col = [dict(ds['meta'].get('collection') or {}) for ds in datasets]
    meta = dict(datasets[0]['meta'])
    meta['merged'] = True
    meta['sources'] = [{'controller': ds['meta'].get('controller_name'),
                        'collection': dict(ds['meta'].get('collection') or {})}
                       for ds in datasets]
    meta['collection'] = {
        'protocol': 'merged',
        'steps_used': sum(int(c.get('steps_used', 0)) for c in src_col),
        'transitions': sum(int(c.get('transitions', 0)) for c in src_col),
        'episodes_done': sum(int(c.get('episodes_done', 0)) for c in src_col),
        'complete_episodes': sum(int(c.get('complete_episodes', 0)) for c in src_col),
        'collection_wall_seconds': round(sum(float(c.get('collection_wall_seconds', 0))
                                             for c in src_col), 2),
        'sources': len(datasets)}
    meta['dagger'] = [ds['meta'].get('dagger') for ds in datasets
                      if ds['meta'].get('dagger')]
    return dict(data=merged, episodes=episodes, meta=meta)


def dataset_summary(dataset):
    meta = dict(dataset['meta'])
    col = dict(meta.get('collection') or {})
    for k in ('steps_used', 'transitions', 'collection_wall_seconds', 'episodes_done',
              'unfinished_episodes', 'steps_requested', 'episodes_requested'):
        if k in col:
            meta[k] = col[k]
    meta['collection'] = col
    episodes = dataset['episodes']
    meta['episodes_total'] = len(episodes)
    meta['episodes_complete'] = sum(1 for ep in episodes if ep['complete'])
    meta['episodes_cut'] = sum(1 for ep in episodes if ep.get('cut'))
    meta['success_episodes'] = sum(1 for ep in episodes if ep['success'])
    meta['timeout_episodes'] = sum(1 for ep in episodes if ep['truncated'] and not ep['success'])
    meta['terminated_episodes'] = sum(1 for ep in episodes if ep['terminated'] and not ep['success'])
    return meta
