"""Evaluation & test harness for the fixedwing classical autopilot
(L1 guidance + TECS + attitude inner loops, src/common/baselines.py).

Self-contained: every parameter is hard-coded below (no CLI arguments).
Protocol:

1. Statistical runs on the training ICs (seed 7 = the training seed) and
   development ICs (seeds 1000-1003), then on held-out test ICs (seeds
   2000-2003).  Per-waypoint pass rates come from the pre-reset terminal
   observations (valid-flag decoding, exact); the whole-episode success rate
   and the terminal-reason breakdown come from the env's own per-step
   episode rates / completed counters (exact up to float rounding).
2. Functional segment tests with harness-injected waypoint routes (a straight
   route and a 90-degree-corner turning route; the env itself and its success
   criteria are untouched): per-waypoint capture times, cross-track /
   altitude / airspeed tracking errors, max bank, and action-saturation
   statistics.

Writes results/fixedwing/autopilot_report.json and prints a summary.
"""
import json
import sys
import types
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / 'src'
_pkg = types.ModuleType('pyflyt_genesis')
_pkg.__path__ = [str(_src)]
sys.modules['pyflyt_genesis'] = _pkg
sys.path.insert(0, str(_src))

import torch

from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv
from common.baselines import build_baseline, _quat_to_R

# ---- protocol constants (train/dev/test initial-condition split) ----
TRAIN_SEEDS = [7]
DEV_SEEDS = [1000, 1001, 1002, 1003]
TEST_SEEDS = [2000, 2001, 2002, 2003]
NUM_ENVS = 1024
STAT_STEPS = 3700                    # > 3600-step episode horizon
SEGMENT_REPEATS = 3
SEG_STEPS = 3600
OUT_PATH = Path(__file__).resolve().parent.parent / 'results' / 'fixedwing' / 'autopilot_report.json'

# Harness-injected routes (waypoints the env would never sample itself; used
# only to exercise straight and turning flight, success criteria unchanged).
STRAIGHT_ROUTE = [(30., 0., 22.), (50., 0., 22.), (70., 0., 22.), (88., 0., 22.)]
TURN_ROUTE = [(45., 0., 25.), (0., 45., 25.), (-45., 0., 25.), (0., -45., 25.)]


def stat_run(seed, num_envs=NUM_ENVS, steps=STAT_STEPS):
    """Roll the autopilot on one seed; return per-episode statistics."""
    torch.manual_seed(seed)
    env = FixedwingWaypointsEnv(num_envs=num_envs, backend='gpu', seed=seed)
    ctl = build_baseline('fixedwing', env)
    obs = env.get_observations()
    ti_hist = torch.zeros(5, device=env.device, dtype=torch.long)
    sat_steps = torch.zeros((), device=env.device)
    reason = {'success': 0, 'crash': 0, 'oob': 0, 'timeout': 0, 'unattributed': 0}
    for _ in range(steps):
        prev_suc = env.completed_successes.clone()
        with torch.no_grad():
            actions = ctl.actions(obs)
            sat_steps += (actions.abs() >= 0.999).any(-1).float().sum()
            obs, _, reset_buf, extras = env.step(actions)
        ctl.reset_idx(reset_buf)
        if not bool(reset_buf.any()):
            continue
        n = int(reset_buf.sum())
        # Exact per-env success mask: completed_successes is a per-env counter
        # incremented by (reset_buf & complete) inside step.  The pre-reset
        # terminal observation lags target_index by one inner substep, so
        # successful episodes decode as ti=3 there -- route them to bucket 4
        # via the counter mask instead of the decoded value.
        suc_mask = reset_buf & (env.completed_successes > prev_suc)
        n_success = int(suc_mask.sum())
        ti_hist[4] += n_success
        # non-success terminals decode exactly: valid[k] flags encode
        # target_index = 4 - sum(valid)
        term = extras['terminal_observations']['policy'].to(torch.float64)
        valid = (term[:, 35:39] > .5).sum(-1)
        done_ti = (4 - valid)[reset_buf & ~suc_mask]
        ti_hist += torch.bincount(done_ti, minlength=5)
        ep = extras.get('episode')
        if ep is None:
            reason['unattributed'] += n
            continue
        reason['success'] += n_success
        remaining = n - n_success
        budget = {
            'crash': round(float(ep['crash_rate']) * n),
            'oob': round(float(ep['out_of_bounds_rate']) * n),
            'timeout': round(float(ep['timeout_rate']) * n),
        }
        for _ in range(remaining):
            for key in ('crash', 'oob', 'timeout'):
                if budget[key] > 0:
                    reason[key] += 1
                    budget[key] -= 1
                    break
            else:
                reason['unattributed'] += 1
    episodes = int(ti_hist.sum())
    total = max(episodes, 1)
    report = {
        'seed': seed, 'num_envs': num_envs, 'steps': steps,
        'episodes': episodes,
        'waypoint_histogram': [int(c) for c in ti_hist.tolist()],  # final count reached
        # pass rate for waypoint k = fraction of episodes that captured >= k
        'per_waypoint_pass_rate': [float(ti_hist[k:].sum()) / total for k in (1, 2, 3, 4)],
        'success_rate': float(ti_hist[4]) / total,
        'mean_waypoints': float((ti_hist.to(torch.float64) * torch.arange(5, device=env.device, dtype=torch.float64)).sum()) / total,
        'terminal_reasons': reason,
        'action_saturation_fraction': float(sat_steps) / (steps * num_envs),
    }
    env.close()
    return report


def segment_run(route, repeats=SEGMENT_REPEATS, steps=SEG_STEPS):
    """Fly a harness-injected route with a single replica; track quality."""
    torch.manual_seed(7)
    env = FixedwingWaypointsEnv(num_envs=1, backend='gpu', seed=7)
    ctl = build_baseline('fixedwing', env)
    route_t = torch.tensor(route, device=env.device, dtype=torch.float64)

    def inject():
        env.targets[0] = route_t

    inject()
    obs = env.get_observations()
    episodes = []
    cur = None
    last_ti = 0
    suc0 = int(env.completed_successes[0])
    for t in range(steps):
        o = obs['policy'].to(torch.float64)
        if cur is None:
            cur = {'captures': [], 'xtrack': [], 'h_err': [], 'v_err': [], 'bank': [],
                   'sat': 0, 'n': 0}
            last_ti = 0
        with torch.no_grad():
            a = ctl.actions(obs)
        cur['sat'] += int((a[0].abs() >= .999).any())
        cur['n'] += 1
        p, vb = o[0, 10:13], o[0, 7:10]
        R = _quat_to_R(o[0, 3:7][None])[0]
        roll = float(torch.atan2(R[2, 1], R[2, 2]))
        ti = int(env.target_index[0])
        if ti != last_ti:
            cur['captures'].append((ti, t))
            last_ti = ti
        if ti > 0:                      # cross-track vs the leg just captured
            a0, b0 = route_t[ti - 1], route_t[min(ti, 3)]
            seg = b0 - a0
            tt = float((((p - a0) * seg).sum() / seg.dot(seg)).clamp(0., 1.))
            cur['xtrack'].append(float((p - a0 - tt * seg).norm()))
        if ti < 4:
            cur['h_err'].append(abs(float(p[2]) - float(route_t[ti][2])))
        cur['v_err'].append(abs(float(vb.norm()) - 20.))
        cur['bank'].append(abs(roll))
        obs, _, reset_buf, _ = env.step(a)
        ctl.reset_idx(reset_buf)
        if bool(reset_buf[0]):
            suc_now = int(env.completed_successes[0])
            success = suc_now > suc0
            suc0 = suc_now
            cur['waypoints'] = 4 if success else 4 - int((obs['policy'][0, 35:39] > .5).sum())
            if success:
                cur['captures'].append((4, t))
            cur['reason'] = ('success' if success else
                             'crash' if bool(env.collision[0]) else
                             'oob' if bool(env.out_of_bounds[0]) else 'timeout')
            episodes.append(cur)
            cur = None
            inject()
    env.close()

    def rms(vals):
        return (sum(x * x for x in vals) / max(len(vals), 1)) ** .5

    return {'route': [list(map(float, wp)) for wp in route],
            'episodes': [{'reason': ep['reason'], 'waypoints': ep['waypoints'],
                          'captures': [[ti, t] for ti, t in ep['captures']],
                          'xtrack_max': max(ep['xtrack']) if ep['xtrack'] else 0.,
                          'xtrack_rms': rms(ep['xtrack']),
                          'alt_err_max': max(ep['h_err']) if ep['h_err'] else 0.,
                          'alt_err_rms': rms(ep['h_err']),
                          'v_err_rms': rms(ep['v_err']),
                          'bank_max_deg': max(ep['bank']) * 57.2958,
                          'saturation_fraction': ep['sat'] / max(ep['n'], 1)}
                         for ep in episodes]}


def main():
    results = {'stat_runs': [], 'segment_tests': {}}
    for label, seeds in (('train', TRAIN_SEEDS), ('dev', DEV_SEEDS), ('test', TEST_SEEDS)):
        for seed in seeds:
            rep = stat_run(seed)
            rep['split'] = label
            results['stat_runs'].append(rep)
            print(f"[{label} seed {seed}] episodes={rep['episodes']} "
                  f"sr={rep['success_rate']:.3f} wp/ep={rep['mean_waypoints']:.2f} "
                  f"pass={['%.3f' % r for r in rep['per_waypoint_pass_rate']]} "
                  f"reasons={rep['terminal_reasons']} sat={rep['action_saturation_fraction']:.4f}",
                  flush=True)
        group = [r for r in results['stat_runs'] if r['split'] == label]
        n = sum(r['episodes'] for r in group)
        results[label] = {
            'seeds': seeds, 'episodes': n,
            'success_rate': sum(r['success_rate'] * r['episodes'] for r in group) / max(n, 1),
            'per_waypoint_pass_rate': [sum(r['per_waypoint_pass_rate'][k] * r['episodes'] for r in group) / max(n, 1) for k in range(4)],
            'mean_waypoints': sum(r['mean_waypoints'] * r['episodes'] for r in group) / max(n, 1),
            'terminal_reasons': {k: sum(r['terminal_reasons'][k] for r in group) for k in reason_keys()},
        }
    for name, route in (('straight', STRAIGHT_ROUTE), ('turn', TURN_ROUTE)):
        seg = segment_run(route)
        results['segment_tests'][name] = seg
        for i, ep in enumerate(seg['episodes']):
            print(f"[{name} #{i}] reason={ep['reason']} wp={ep['waypoints']} "
                  f"cap={ep['captures']} xtrack_max={ep['xtrack_max']:.1f} "
                  f"alt_rms={ep['alt_err_rms']:.1f} v_rms={ep['v_err_rms']:.1f} "
                  f"bank_max={ep['bank_max_deg']:.0f} sat={ep['saturation_fraction']:.3f}", flush=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"report -> {OUT_PATH}")


def reason_keys():
    return ('success', 'crash', 'oob', 'timeout', 'unattributed')


if __name__ == '__main__':
    main()
