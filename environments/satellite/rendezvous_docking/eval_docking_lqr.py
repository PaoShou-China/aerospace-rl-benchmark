"""Deep evaluation of the docking CWH-LQR classical baseline.

Self-contained (no CLI): edit the constants and run
`~/.conda/envs/genesislab/bin/python eval_docking_lqr.py`.

Protocol
    1. Natural-distribution arms on the batched GPU env (the env's own
       spawn distribution via the sanctioned `difficulty` knob; success
       criteria untouched): difficulty in {0, 0.5, 1} and extra seeds at
       the benchmark point, plus ablation arms (legacy incumbent gains,
       aggressive Q for the saturation study, shield-off).
    2. Deterministic scripted-IC grid: one episode per scenario; initial
       distance {15, 30, 50, 75, 100} m x direction {radial, along-track,
       cross-track} x initial velocity {0, 0.5 m/s outward, 0.5 m/s
       inward, 0.5 m/s tangent} inside the spawn envelope, plus
       out-of-envelope stress cases that exercise the 200 m range shield.
       Runs with the full design and with the shield disabled.

Per-episode metrics: outcome (success / range-fail / fuel-fail / timeout),
steps, terminal range/speed (for successes the terminal state IS the first
goal-sphere entry, since the env ends the episode at first capture),
in-episode min & max range (=> min margin to the 200 m bound), 160 m
corridor breaches, quadratic control cost, fuel, saturation fraction, and
shield/capture/fuel-derate activation counts.  Aggregates are printed and
written to results/docking/lqr_report.json.
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import math
import torch

from environments.satellite.rendezvous_docking import SpaceDockingEnv
from algorithms.baselines.baselines import DockingCWHLQRBaseline

NUM_ENVS = 1024
DIST_STEPS = 2000          # env steps per natural-distribution arm
SEEDS_MAIN = 7
DIFFICULTIES = (0., 0.5, 1.)
SEEDS_BENCH = (7, 13, 21)  # seed repeat at the benchmark difficulty
BACKEND = 'gpu'
OUT = Path(__file__).resolve().parents[2] / 'results' / 'docking' / 'lqr_report.json'

# arms on the natural distribution: (label, controller kwargs, difficulty, seed)
DIST_ARMS = (
    [('lqr_full', {}, d, SEEDS_MAIN) for d in DIFFICULTIES]
    + [('lqr_full', {}, 1., s) for s in SEEDS_BENCH[1:]]
    + [('legacy_incumbent', {'legacy': True}, 1., SEEDS_MAIN),
       ('aggressive_q', {'kappa': 1., 'w_vel': 1.}, 1., SEEDS_MAIN),
       ('no_shield', {'shield': False}, 1., SEEDS_MAIN)]
)

# scripted IC grid -----------------------------------------------------------
GRID_DISTS = (15., 30., 50., 75., 100.)
STRESS = ((150., 'x', 1.0), (180., 'x', 1.5),
          (100., 'x', 1.5), (173.2, 'd', 0.5))
GRID_ARMS = (('lqr_full', {}), ('no_shield', {'shield': False}),
             ('aggressive_q', {'kappa': 1., 'w_vel': 1.}),
             ('legacy_incumbent', {'legacy': True}))


def _grid_scenarios():
    """Deterministic IC battery -> (p0, v0) tensors (n_scen, 3)."""
    scen = []
    for axis in range(3):
        for dist in GRID_DISTS:
            p_hat = torch.zeros(3); p_hat[axis] = 1.
            t_hat = torch.zeros(3); t_hat[(axis + 1) % 3] = 1.   # in-plane tangent
            for kind in ('rest', 'out', 'in', 'tan'):
                v0 = {'rest': torch.zeros(3), 'out': p_hat * .5,
                      'in': -p_hat * .5, 'tan': t_hat * .5}[kind]
                scen.append((p_hat * dist, v0))
    for dist, kind, speed in STRESS:
        p_hat = torch.zeros(3)
        if kind == 'd':
            p_hat[:] = 1. / math.sqrt(3)
        else:
            p_hat[int({'x': 0, 'y': 1, 'z': 2}[kind])] = 1.
        scen.append((p_hat * dist, p_hat * speed))
    p0 = torch.stack([s[0] for s in scen])
    v0 = torch.stack([s[1] for s in scen])
    labels = [f'axis{a}|d={d:g}|{k}' for a in range(3) for d in GRID_DISTS
              for k in ('rest', 'out', 'in', 'tan')] \
        + [f'stress|{kind}{dist:g}|v={s:g}' for dist, kind, s in STRESS]
    return p0, v0, labels


class _EpAcc:
    """Per-env episode accumulators (device) + CPU record sink."""

    FIELDS = ('min_d', 'max_d', 'max_v', 'min_margin', 'cost', 'steps')

    def __init__(self, n, device, range_fail, corridor):
        self.range_fail, self.corridor = range_fail, corridor
        inf = float('inf')
        self.t = {f: torch.zeros(n, device=device, dtype=torch.float64) for f in self.FIELDS}
        self.t['min_d'] = torch.full((n,), inf, device=device, dtype=torch.float64)
        self.t['min_margin'] = torch.full((n,), inf, device=device, dtype=torch.float64)
        self.records = []

    def observe(self, d, v_norm):
        self.t['min_d'] = torch.minimum(self.t['min_d'], d)
        self.t['max_d'] = torch.maximum(self.t['max_d'], d)
        self.t['max_v'] = torch.maximum(self.t['max_v'], v_norm)
        self.t['min_margin'] = torch.minimum(self.t['min_margin'], self.range_fail - d)

    def step(self, actions, dt, a_max):
        u = actions.to(torch.float64) * a_max
        self.t['cost'] += (u * u).sum(-1) * dt
        self.t['steps'] += 1.

    def flush(self, done, d_T, v_T, ctrl):
        """Snapshot done envs into records (outcome refined later)."""
        idx = done.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        success = (d_T < 1.) & (v_T < .1)
        # fold the terminal state into the in-episode extrema before snapshot
        self.t['min_d'] = torch.minimum(self.t['min_d'], d_T)
        self.t['max_d'] = torch.maximum(self.t['max_d'], d_T)
        self.t['max_v'] = torch.maximum(self.t['max_v'], v_T)
        self.t['min_margin'] = torch.minimum(self.t['min_margin'], self.range_fail - d_T)
        rec = {
            'idx': idx.cpu().tolist(),
            'd_T': d_T[idx].cpu().tolist(), 'v_T': v_T[idx].cpu().tolist(),
            'success': success[idx].cpu().tolist(),
        }
        for f in self.FIELDS:
            rec[f] = self.t[f][idx].cpu().tolist()
        rec['fuel'] = ctrl.fuel_est[idx].cpu().tolist()
        rec['sat'] = ctrl.stat_sat[idx].cpu().tolist()
        rec['shield_hard'] = ctrl.stat_shield_hard[idx].cpu().tolist()
        rec['shield_soft'] = ctrl.stat_shield_soft[idx].cpu().tolist()
        rec['capture'] = ctrl.stat_capture[idx].cpu().tolist()
        rec['fuel_cap'] = ctrl.stat_fuel_cap[idx].cpu().tolist()
        # zero for the fresh episode (min-accumulators restart at +inf)
        for f in self.FIELDS:
            if f in ('min_d', 'min_margin'):
                self.t[f][idx] = float('inf')
            else:
                self.t[f][idx] = 0.
        self.records.append(rec)
        return rec


def _roll(env, ctrl, steps, obs=None, grid_mode=False, labels=None):
    """Shared rollout; returns list of per-episode dicts (+ grid scenario map)."""
    dev = env.device
    acc = _EpAcc(env.num_envs, dev, env.range_fail, .8 * env.range_fail)
    if obs is None:
        obs = env.get_observations()
    first_done = torch.zeros(env.num_envs, device=dev, dtype=torch.bool)
    n_steps = steps
    for t in range(steps):
        o = obs['policy'].to(torch.float64)
        d = o[:, :3].norm(dim=-1)
        v_norm = o[:, 3:].norm(dim=-1)
        acc.observe(d, v_norm)
        actions = ctrl.actions(obs)
        acc.step(actions, env.dt, env.f_max / env.mass)
        obs, _, reset_buf, extras = env.step(actions)
        done = reset_buf & ~first_done if grid_mode else reset_buf
        first_done |= reset_buf
        if bool(done.any()):
            term = extras['terminal_observations']['policy'].to(torch.float64)
            d_T = term[:, :3].norm(dim=-1)
            v_T = term[:, 3:].norm(dim=-1)
            acc.flush(done, d_T, v_T, ctrl)
            ctrl.reset_idx(done)
        if grid_mode and bool(first_done.all()):
            n_steps = t + 1
            break
    # attach outcomes/outlier flags onto flat records
    flat = []
    for chunk in acc.records:
        for i, idx in enumerate(chunk['idx']):
            r = {'scenario': labels[idx] if labels else idx,
                 'steps': chunk['steps'][i], 'd_T': chunk['d_T'][i], 'v_T': chunk['v_T'][i],
                 'success': chunk['success'][i], 'min_d': chunk['min_d'][i],
                 'max_d': chunk['max_d'][i], 'max_v': chunk['max_v'][i],
                 'min_margin': chunk['min_margin'][i], 'cost': chunk['cost'][i],
                 'fuel': chunk['fuel'][i], 'sat': chunk['sat'][i],
                 'shield_hard': chunk['shield_hard'][i], 'shield_soft': chunk['shield_soft'][i],
                 'capture': chunk['capture'][i], 'fuel_cap': chunk['fuel_cap'][i],
                 'corridor_breach': chunk['max_d'][i] > .8 * env.range_fail}
            flat.append(r)
    return flat, n_steps


def _summarize(flat, env_episodes, env_successes):
    """Aggregate per-episode records into reportable stats."""
    n = len(flat)
    suc = [r for r in flat if r['success']]
    fail = [r for r in flat if not r['success']]
    rng_fail = [r for r in fail if r['d_T'] > 200.]
    fuel_fail = [r for r in fail if r['d_T'] <= 200. and r['fuel'] >= 2.]
    timeouts = [r for r in fail if r not in rng_fail and r not in fuel_fail]

    def q(vals, p):
        if not vals:
            return None
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    s = {'episodes': n, 'success_rate': len(suc) / max(n, 1),
         'env_episodes_reported': env_episodes, 'env_successes_reported': env_successes,
         'range_fails': len(rng_fail), 'fuel_fails': len(fuel_fail),
         'timeouts': len(timeouts),
         'corridor_breaches': sum(r['corridor_breach'] for r in flat),
         'mean_control_cost': sum(r['cost'] for r in flat) / max(n, 1),
         'fuel_mean': sum(r['fuel'] for r in flat) / max(n, 1),
         'fuel_max': max((r['fuel'] for r in flat), default=0.),
         'sat_frac_mean': sum(r['sat'] for r in flat) / max(sum(r['steps'] for r in flat), 1.) / 3.,
         'shield_hard_mean': sum(r['shield_hard'] for r in flat) / max(n, 1),
         'capture_mean': sum(r['capture'] for r in flat) / max(n, 1),
         'fuel_cap_mean': sum(r['fuel_cap'] for r in flat) / max(n, 1)}
    if suc:
        s['time_to_success_p50'] = q([r['steps'] for r in suc], .5)
        s['time_to_success_p90'] = q([r['steps'] for r in suc], .9)
        s['time_to_success_max'] = max(r['steps'] for r in suc)
        s['terminal_d_success_p50'] = q([r['d_T'] for r in suc], .5)
        s['terminal_v_success_p50'] = q([r['v_T'] for r in suc], .5)
        s['terminal_v_success_max'] = max(r['v_T'] for r in suc)
    if timeouts:
        s['terminal_d_timeout_p50'] = q([r['d_T'] for r in timeouts], .5)
        s['terminal_v_timeout_p50'] = q([r['v_T'] for r in timeouts], .5)
    s['min_margin_worst'] = min((r['min_margin'] for r in flat), default=None)
    return s


def run_dist_arm(label, kwargs, difficulty, seed, backend=BACKEND):
    try:
        env = SpaceDockingEnv(num_envs=NUM_ENVS, backend=backend, seed=seed,
                              difficulty=difficulty)
    except RuntimeError:
        env = SpaceDockingEnv(num_envs=NUM_ENVS, backend='cpu', seed=seed,
                              difficulty=difficulty)
    torch.manual_seed(seed)
    ctrl = DockingCWHLQRBaseline(env, **kwargs)
    ep0 = env.completed_episodes.clone()
    ok0 = env.completed_successes.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        flat, _ = _roll(env, ctrl, DIST_STEPS)
    wall = time.perf_counter() - t0
    episodes = float((env.completed_episodes - ep0).sum())
    successes = float((env.completed_successes - ok0).sum())
    s = _summarize(flat, episodes, successes)
    s.update({'arm': label, 'kind': 'distribution', 'difficulty': difficulty,
              'seed': seed, 'wall_s': round(wall, 1),
              'env_consistent': abs(successes - sum(r['success'] for r in flat)) < .5
                                and abs(episodes - len(flat)) < 1.5})
    env.close()
    return s, flat


def run_grid_arm(label, kwargs, p0, v0, labels, backend=BACKEND):
    try:
        env = SpaceDockingEnv(num_envs=p0.shape[0], backend=backend, seed=SEEDS_MAIN)
    except RuntimeError:
        env = SpaceDockingEnv(num_envs=p0.shape[0], backend='cpu', seed=SEEDS_MAIN)
    torch.manual_seed(SEEDS_MAIN)
    dev = env.device
    p0d = p0.to(dev, torch.float64)
    v0d = v0.to(dev, torch.float64)
    with torch.no_grad():
        env.p.copy_(p0d)
        env.v.copy_(v0d)
        env.d0.copy_(p0d.norm(dim=-1))
        obs64 = torch.cat((p0d, v0d), -1)
        env._observation.copy_(obs64)
        env.obs_buf.copy_(obs64.to(torch.float32))
    ctrl = DockingCWHLQRBaseline(env, **kwargs)
    with torch.no_grad():
        flat, used = _roll(env, ctrl, env.max_steps, grid_mode=True, labels=labels)
    env.close()
    return {'arm': label, 'kind': 'scripted_grid', 'scenarios': len(flat),
            'steps_used': used,
            'rows': [{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in flat]}


def _fmt(v, spec='.1f'):
    return f'{v:{spec}}' if isinstance(v, float) else ('-' if v is None else str(v))


def main():
    p0, v0, labels = _grid_scenarios()
    report = {'design': {
        'Q': 'diag(1/pos_goal^2 * I3, 3/vel_goal^2 * I3) = diag(1,1,1,300,300,300)',
        'R': '1/(0.35 * f_max/mass)^2 * I3 = 20408.16 * I3',
        'gain': 'K = (R + Bd\'PBd)^-1 Bd\'PAd, DARE on the exact symplectic-Euler '
                '(Ad, Bd) of the CWH plant, scipy solve_discrete_are',
        'layers': 'per-axis saturation [-1,1]; 200 m range shield (hard brake '
                  'beyond 0.95*range_fail predicted stop, outward-thrust '
                  'projection inside 0.8*range_fail corridor); terminal capture '
                  'net within 2 m; fuel derate past 75% of the 2 kg budget'}}
    dist_out = []
    for label, kwargs, difficulty, seed in DIST_ARMS:
        s, _ = run_dist_arm(label, kwargs, difficulty, seed)
        dist_out.append(s)
        print(f"[dist] {label:18s} d={difficulty:<3} seed={seed:<2} "
              f"success={s['success_rate']:.3f} ({s['episodes']:.0f} ep) "
              f"range_fails={s['range_fails']} fuel_fails={s['fuel_fails']} "
              f"timeouts={s['timeouts']} t50={_fmt(s.get('time_to_success_p50'), '.0f')} "
              f"fuel_max={_fmt(s['fuel_max'], '.3f')} "
              f"margin_min={_fmt(s['min_margin_worst'])} consistent={s['env_consistent']}",
              flush=True)
    grid_out = []
    for label, kwargs in GRID_ARMS:
        g = run_grid_arm(label, kwargs, p0, v0, labels)
        grid_out.append(g)
        rows = g['rows']
        suc = sum(r['success'] for r in rows)
        print(f"[grid] {label:18s} success={suc}/{len(rows)} "
              f"range_fails={sum(r['d_T'] > 200 for r in rows)} "
              f"margin_min={min(r['min_margin'] for r in rows):.1f} "
              f"shield={sum(r['shield_hard'] for r in rows):.0f}",
              flush=True)
    report['grid_notes'] = {
        'stress_feasibility': {
            f'stress|{k}{d:g}|v={s:g}':
                'feasible' if d + s * s / (2 * 10. / 500.) <= 200. else 'infeasible-at-authority'
            for d, k, s in STRESS},
        'note': 'feasible = a full-authority (0.02 m/s^2) stop from the initial '
                'state stays inside the 200 m bound; infeasible cases cannot be '
                'saved by any controller and only validate graceful failure.'}
    report['distribution'] = dist_out
    report['grid'] = grid_out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
