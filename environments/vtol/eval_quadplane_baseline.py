"""Evaluation harness for the quadplane cascaded PID baseline (hybrid of
position PD + attitude PD + rate PID + gentle forward pusher, implemented
in algorithms/baselines/baselines.py as ``QuadplaneHybridBaseline``).

Self-contained: every parameter is hard-coded below (no CLI arguments).
Protocol mirrors environments/fixedwing/eval_fixedwing_autopilot.py:

1. Statistical runs on the training ICs (seed 7), development ICs
   (seeds 1000-1003), and held-out test ICs (seeds 2000-2003).
   Per-waypoint pass rates come from the env's own completed_successes
   counter; terminal-reason breakdown uses episode-level stats.
2. Single-environment functional run: capture times, waypoints reached,
   attitude envelope, action saturation.

Writes results/quadplane/cascade_report.json and prints a summary.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch

from environments.vtol.env import QuadplaneWaypointsEnv
from algorithms.baselines.baselines import build_baseline, _quat_to_R, _rot_to_rpy

TRAIN_SEEDS = [7]
DEV_SEEDS = [1000, 1001, 1002, 1003]
TEST_SEEDS = [2000, 2001, 2002, 2003]
NUM_ENVS = 64
STAT_STEPS = 5000                      # > quadplane episode horizon
OUT_PATH = _ROOT / 'results' / 'quadplane' / 'cascade_report.json'


def stat_run(seed, num_envs=NUM_ENVS, steps=STAT_STEPS):
    """Roll the baseline on one seed; return per-episode statistics."""
    torch.manual_seed(seed)
    env = QuadplaneWaypointsEnv(num_envs=num_envs, backend='cpu', seed=seed)
    ctl = build_baseline('quadplane', env)
    obs = env.get_observations()
    ti_hist = torch.zeros(5, dtype=torch.long)
    sat_steps = torch.tensor(0.)
    max_roll = torch.tensor(0.)
    max_pitch = torch.tensor(0.)
    crash_count = 0
    timeout_count = 0
    success_count = 0
    for _ in range(steps):
        prev_suc = env.completed_successes.clone()
        with torch.no_grad():
            actions = ctl.actions(obs)
            sat_steps += (actions.abs() >= 0.999).any(-1).float().sum()
            obs, _, term, trunc, _ = env.step(actions)
            reset_buf = term | trunc
        ctl.reset_idx(reset_buf)

        o = obs['policy'].to(torch.float64)
        R = _quat_to_R(o[:, 3:7])
        roll, pitch, _ = _rot_to_rpy(R)
        max_roll = torch.maximum(max_roll, roll.abs().max().detach())
        max_pitch = torch.maximum(max_pitch, pitch.abs().max().detach())

        if not bool(reset_buf.any()):
            continue
        suc_mask = reset_buf & (env.completed_successes > prev_suc)
        n_success = int(suc_mask.sum())
        success_count += n_success
        crash_count += int(term.sum())
        timeout_count += int(trunc.sum())
        ti_hist[4] += n_success
        # Non-success terminals: target_index from policy observation is
        # the one the env was heading for when reset fired.
        done_idx = reset_buf & ~suc_mask
        if done_idx.any():
            done_ti = obs['policy'][done_idx, 13].to(torch.long)
            ti_hist += torch.bincount(done_ti, minlength=5)
    episodes = int(ti_hist.sum())
    total = max(episodes, 1)
    report = {
        'seed': seed, 'num_envs': num_envs, 'steps': steps,
        'episodes': episodes,
        'waypoint_histogram': [int(c) for c in ti_hist.tolist()],
        'per_waypoint_pass_rate': [float(ti_hist[k:].sum()) / total for k in (1, 2, 3, 4)],
        'success_rate': float(ti_hist[4]) / total,
        'mean_waypoints': float((ti_hist.to(torch.float64) * torch.arange(5)).sum()) / total,
        'terminal_reasons': {
            'success': success_count,
            'crash': crash_count,
            'timeout': timeout_count,
        },
        'action_saturation_fraction': float(sat_steps) / (steps * num_envs),
        'max_roll_deg': float(max_roll) * 57.2958,
        'max_pitch_deg': float(max_pitch) * 57.2958,
    }
    env.close()
    return report


def single_run(seed=7, steps=STAT_STEPS):
    """Single-env functional run; track capture times and attitude envelope."""
    torch.manual_seed(seed)
    env = QuadplaneWaypointsEnv(num_envs=1, backend='cpu', seed=seed)
    ctl = build_baseline('quadplane', env)
    obs = env.get_observations()
    captures = []
    last_ti = int(env.target_index[0])
    for t in range(steps):
        prev_suc = int(env.completed_successes[0])
        with torch.no_grad():
            actions = ctl.actions(obs)
        obs, _, term, trunc, _ = env.step(actions)
        ctl.reset_idx(term | trunc)
        ti = int(env.target_index[0])
        if ti != last_ti:
            captures.append((ti, t))
            last_ti = ti
        if term[0] or trunc[0]:
            break
    success = int(env.completed_successes[0]) > prev_suc
    env.close()
    return {
        'seed': seed, 'steps': t + 1,
        'captures': [[ti, t] for ti, t in captures],
        'targets_reached': last_ti if success else last_ti,
        'success': success,
        'crashed': bool(term[0]),
    }


def main():
    results = {'stat_runs': [], 'single_runs': []}
    for label, seeds in (('train', TRAIN_SEEDS), ('dev', DEV_SEEDS), ('test', TEST_SEEDS)):
        for seed in seeds:
            rep = stat_run(seed)
            rep['split'] = label
            results['stat_runs'].append(rep)
            print("[%s seed %d] episodes=%d sr=%.3f wp/ep=%.2f "
                  "pass=[%s] crash=%d timeout=%d sat=%.4f "
                  "max_roll=%.0f° max_pitch=%.0f°" % (
                      label, seed, rep['episodes'], rep['success_rate'],
                      rep['mean_waypoints'],
                      ', '.join('%.3f' % r for r in rep['per_waypoint_pass_rate']),
                      rep['terminal_reasons']['crash'],
                      rep['terminal_reasons']['timeout'],
                      rep['action_saturation_fraction'],
                      rep['max_roll_deg'], rep['max_pitch_deg']))
    for name in ('train', 'dev', 'test'):
        group = [r for r in results['stat_runs'] if r['split'] == name]
        n = sum(r['episodes'] for r in group)
        results[name] = {
            'seeds': [r['seed'] for r in group], 'episodes': n,
            'success_rate': sum(r['success_rate'] * r['episodes'] for r in group) / max(n, 1),
            'per_waypoint_pass_rate': [sum(r['per_waypoint_pass_rate'][k] * r['episodes'] for r in group) / max(n, 1) for k in range(4)],
            'mean_waypoints': sum(r['mean_waypoints'] * r['episodes'] for r in group) / max(n, 1),
            'crash_rate': sum(r['terminal_reasons']['crash'] for r in group) / max(n, 1),
            'timeout_rate': sum(r['terminal_reasons']['timeout'] for r in group) / max(n, 1),
            'max_roll_deg_max': max(r['max_roll_deg'] for r in group),
            'max_pitch_deg_max': max(r['max_pitch_deg'] for r in group),
        }
    for seed in TRAIN_SEEDS + DEV_SEEDS + TEST_SEEDS:
        sr = single_run(seed)
        results['single_runs'].append(sr)
        print("[single seed %d] steps=%d captures=%s reached=%d %s" % (
            seed, sr['steps'], sr['captures'], sr['targets_reached'],
            'SUCCESS' if sr['success'] else 'CRASH' if sr['crashed'] else 'TIMEOUT'))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print("report -> %s" % OUT_PATH)


if __name__ == '__main__':
    main()
