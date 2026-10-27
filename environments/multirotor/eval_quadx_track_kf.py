"""Tuning-set selection for the QuadX-Track classical baseline.

The constant-velocity Kalman filter noise parameter (SIGMA_ACCEL) and the
outer-loop position gains (KX, KV) are selected on an INDEPENDENT tuning seed
(101) that is never used for the reported evaluation (seed 7).  Everything
else -- sensor noise model, hold timeout, speed cap, altitude and horizontal
offset -- stays fixed.  The grid is deliberately small; the point is a
reproducible, documented selection, not an exhaustive sweep.

Usage: python eval_quadx_track_kf.py [--num-envs 256] [--seed 101]
Writes results/quadx_track/kf_tuning.json and prints the table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
from environments.common.plant import TensorRandom
from environments.multirotor import QuadXTrackEnv
from algorithms.baselines.baselines import QuadXTrackKFBaseline

GRID = [
    # (sigma_accel, kx, kv)
    (0.5, 4.0, 2.0),
    (0.5, 4.0, 4.0),
    (1.5, 4.0, 2.0),
    (1.5, 4.0, 4.0),
    (4.0, 4.0, 2.0),
    (4.0, 4.0, 4.0),
]


def evaluate(env, sigma_accel, kx, kv, seed, num_envs, max_steps=2000):
    # fresh scenario population for this combo: re-seed both streams, reset all
    env.plant.rng = TensorRandom(num_envs, env.device, seed)
    env.sensor_rng = TensorRandom(num_envs, env.device, seed + 7919)
    env.reset()
    ctrl = QuadXTrackKFBaseline(env, sigma_accel=sigma_accel, kx=kx, kv=kv)
    mask = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    ctrl.reset_idx(mask)
    obs = env.get_observations()
    # cumulative counters survive reset_idx/reset, so score this combo on deltas
    ep0 = float(env.completed_episodes.sum())
    suc0 = float(env.completed_successes.sum())
    m0 = {k: v.clone() for k, v in env.completed_metrics.items()}
    steps = 0
    with torch.no_grad():
        while steps < max_steps:
            actions = ctrl.actions(obs)
            obs, _, term, trunc, _ = env.step(actions)
            ctrl.reset_idx(term | trunc)
            steps += 1
            if float(env.completed_episodes.sum()) - ep0 >= num_envs:
                break
    episodes = float(env.completed_episodes.sum()) - ep0
    successes = float(env.completed_successes.sum()) - suc0
    m = {k: float(v - m0[k]) / max(episodes, 1.) for k, v in env.completed_metrics.items()}
    score = successes / max(episodes, 1.) + .5 * m['visible_fraction'] - .2 * m['track_error']
    row = {'sigma_accel': sigma_accel, 'kx': kx, 'kv': kv,
           'success_rate': successes / max(episodes, 1.), 'score': score,
           'episodes': episodes, 'env_steps': steps, **m}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--out', type=Path,
                        default=_ROOT / 'results' / 'quadx_track' / 'kf_tuning.json')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = QuadXTrackEnv(num_envs=args.num_envs, backend=args.backend, seed=args.seed)
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for sigma, kx, kv in GRID:
            row = evaluate(env, sigma, kx, kv, args.seed, args.num_envs)
            rows.append(row)
            print(json.dumps(row), flush=True)
    best = max(rows, key=lambda r: (r['score'], r['success_rate'], -r['track_error']))
    report = {'task': 'quadx_track', 'tuning_seed': args.seed, 'num_envs': args.num_envs,
              'grid': [{'sigma_accel': s, 'kx': kx, 'kv': kv} for s, kx, kv in GRID],
              'fixed': {'sigma_meas': QuadXTrackKFBaseline.SIGMA_MEAS,
                        'hold_timeout': QuadXTrackKFBaseline.HOLD_TIMEOUT,
                        'v_max': QuadXTrackKFBaseline.V_MAX},
              'selection_rule': 'max(success_rate + 0.5*visible_fraction - 0.2*track_error); '
                                'ties by success_rate then track_error',
              'selected': {k: best[k] for k in ('sigma_accel', 'kx', 'kv')},
              'rows': rows, 'wall_seconds': time.perf_counter() - start}
    args.out.write_text(json.dumps(report, indent=2))
    print('\nselected:', report['selected'])
    env.close()


if __name__ == '__main__':
    main()
