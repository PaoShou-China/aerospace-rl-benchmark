"""Evaluation of the classical (non-learned) baselines on the batched GPU envs.

Usage: python baseline_eval.py quadx|fixedwing|rocket|satellite|transfer|docking|rendezvous|pendulum
Defaults mirror eval.py: 1024 envs, 2000 env-steps, seed 7, same env
construction, same success accounting from the env's completed-episode
counters.  Each baseline reads only the 'policy' observation group.
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

_src = Path(__file__).resolve().parent / 'src'
_pkg = types.ModuleType('pyflyt_genesis')
_pkg.__path__ = [str(_src)]
sys.modules['pyflyt_genesis'] = _pkg
sys.path.insert(0, str(_src))

import torch

from pyflyt_genesis.quadx import QuadXHoverEnv
from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv
from pyflyt_genesis.rocket import RocketLandingEnv
from pyflyt_genesis.satellite import SatelliteAttitudeEnv
from pyflyt_genesis.transfer import LowThrustTransferEnv
from pyflyt_genesis.docking import SpaceDockingEnv, SpaceRendezvousEnv
from pyflyt_genesis.pendulum import PendulumEnv

from common.baselines import build_baseline

ENVS = {'quadx': QuadXHoverEnv, 'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
        'satellite': SatelliteAttitudeEnv, 'transfer': LowThrustTransferEnv,
        'docking': SpaceDockingEnv, 'rendezvous': SpaceRendezvousEnv, 'pendulum': PendulumEnv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vehicle', choices=list(ENVS))
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=None,
                        help='results directory (default results/<vehicle>)')
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent / 'results' / args.vehicle
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed)
    controller = build_baseline(args.vehicle, env)

    ep0 = env.completed_episodes.detach().clone()
    ok0 = env.completed_successes.detach().clone()
    ret0 = env.completed_returns.detach().clone()
    obs = env.get_observations()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            actions = controller.actions(obs)
            obs, _, reset_buf, _ = env.step(actions)
            controller.reset_idx(reset_buf)
    wall = time.perf_counter() - start

    def gpu_sum(t):
        return float(t.detach().sum().cpu())

    episodes = gpu_sum(env.completed_episodes - ep0)
    successes = gpu_sum(env.completed_successes - ok0)
    rets = gpu_sum(env.completed_returns - ret0)
    report = {'task': args.vehicle, 'baseline': controller.name,
              'controller_info': 'classical, policy-observation-only',
              'env_steps': args.steps, 'num_envs': args.num_envs,
              'episodes_finished': episodes, 'successes': successes,
              'success_rate': successes / max(episodes, 1.),
              'mean_episode_return': rets / max(episodes, 1.),
              'eval_wall_seconds': wall}
    print(json.dumps(report, indent=2), flush=True)
    (out / 'baseline.json').write_text(json.dumps(report, indent=2))
    env.close()


if __name__ == '__main__':
    main()
