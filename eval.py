"""Deterministic evaluation of a trained RSL-RL checkpoint on the batched GPU envs.

Usage: python eval.py quadx|fixedwing|rocket|satellite|transfer|docking|rendezvous
Defaults: 1024 envs on GPU, 2000 env-steps with the frozen deterministic policy
(mean action, no exploration noise). Success rate / mean return / episode length
are counted by the env itself over the episodes finished inside the window.
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
sys.path.insert(0, str(_src))  # exposes `common` for the FlashSAC runner

import torch
from rsl_rl.runners import OnPolicyRunner

from pyflyt_genesis.quadx import QuadXHoverEnv
from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv
from pyflyt_genesis.rocket import RocketLandingEnv
from pyflyt_genesis.satellite import SatelliteAttitudeEnv
from pyflyt_genesis.transfer import LowThrustTransferEnv
from pyflyt_genesis.docking import SpaceDockingEnv, SpaceRendezvousEnv
from pyflyt_genesis.pendulum import PendulumEnv

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
                        help='training dir holding checkpoint.pt (default results/<vehicle>)')
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent / 'results' / args.vehicle
    ckpt = out / 'checkpoint.pt'
    assert ckpt.exists(), f'missing {ckpt}'
    cfg = json.loads((out / 'config.json').read_text())

    torch.manual_seed(args.seed)
    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed)
    if cfg.get('algo') == 'flash_sac':
        from common.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        policy = runner.deterministic_action
    else:
        runner = OnPolicyRunner(env, cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        actor = runner.alg.actor.eval()
        policy = actor

    ep0 = env.completed_episodes.detach().clone()
    ok0 = env.completed_successes.detach().clone()
    ret0 = env.completed_returns.detach().clone()
    obs = env.get_observations()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    wall = time.perf_counter() - start

    def gpu_sum(t):
        return float(t.detach().sum().cpu())

    episodes = gpu_sum(env.completed_episodes - ep0)
    successes = gpu_sum(env.completed_successes - ok0)
    rets = gpu_sum(env.completed_returns - ret0)
    report = {'task': args.vehicle, 'checkpoint': str(ckpt), 'env_steps': args.steps,
              'num_envs': args.num_envs, 'episodes_finished': episodes,
              'successes': successes,
              'success_rate': successes / max(episodes, 1.),
              'mean_episode_return': rets / max(episodes, 1.),
              'eval_wall_seconds': wall}
    print(json.dumps(report, indent=2), flush=True)
    (out / 'eval.json').write_text(json.dumps(report, indent=2))
    env.close()


if __name__ == '__main__':
    main()
