"""Evaluation of the classical (non-learned) baselines on the batched GPU envs.

Usage: python baseline_eval.py quadx|fixedwing|rocket|missile|satellite|transfer|docking|rendezvous
Defaults mirror eval.py: 1024 envs, seed 7, same env construction (if
results/<task>/config.json exists, its env_kwargs are restored),
same success accounting from the env's completed-episode counters.  Each
baseline reads only the 'policy' observation group.

Unbiased episode accounting: the loop runs until EVERY replica has completed
at least one full episode (safety-capped by --max-steps), so no in-flight
episode is dropped (fixed-wing episodes last up to 3600 steps).
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch
from scripts.first_episode import FirstEpisodes

from environments.multirotor import QuadXWaypointsEnv
from environments.fixedwing import FixedwingWaypointsEnv
from environments.rocket import RocketLandingEnv
from environments.missile import MissileInterceptEnv
from environments.satellite.attitude_control import SatelliteAttitudeEnv
from environments.satellite.orbit_transfer import LowThrustTransferEnv
from environments.satellite.rendezvous_docking import SpaceDockingEnv, SpaceRendezvousEnv

from algorithms.baselines.baselines import build_baseline

ENVS = {'quadx': QuadXWaypointsEnv, 'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
        'missile': MissileInterceptEnv,
        'satellite': SatelliteAttitudeEnv, 'transfer': LowThrustTransferEnv,
        'docking': SpaceDockingEnv, 'rendezvous': SpaceRendezvousEnv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vehicle', choices=list(ENVS))
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=20000,
                        help='safety cap on env steps while every replica finishes one episode')
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=None,
                        help='results directory (default results/<vehicle>)')
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent.parent / 'results' / args.vehicle
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    # match eval.py's env construction when a training config is available
    cfg_path = out / 'config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    env_kwargs = dict(cfg.get('env_kwargs') or {})
    # Old checkpoints predate versioned geometry/metrics. Never silently
    # reinterpret their environment as v2. Fresh baselines use v2 by default.
    if args.vehicle == 'rocket' and cfg:
        env_kwargs.setdefault('benchmark_version', 'v1')
    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed, **env_kwargs)
    controller = build_baseline(args.vehicle, env)

    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    start = time.perf_counter()
    steps = 0
    with torch.no_grad():
        while steps < args.max_steps:
            actions = controller.actions(obs)
            obs, _, term, trunc, _ = env.step(actions)
            reset_buf = term | trunc
            controller.reset_idx(reset_buf)
            steps += 1
            if cohort.update(env):
                break
    wall = time.perf_counter() - start

    report = {'task': args.vehicle, 'baseline': controller.name,
              'controller_info': 'classical, policy-observation-only',
              'env_steps': steps, 'num_envs': args.num_envs, 'env_kwargs': env_kwargs,
              'eval_wall_seconds': wall}
    report.update(cohort.report())
    report['benchmark'] = env.cfg.get('benchmark')
    print(json.dumps(report, indent=2), flush=True)
    (out / 'baseline.json').write_text(json.dumps(report, indent=2))
    env.close()
    if not report['complete']:
        raise RuntimeError('Incomplete first-episode cohort; partial report saved')


if __name__ == '__main__':
    main()
