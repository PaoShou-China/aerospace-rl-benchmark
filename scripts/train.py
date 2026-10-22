"""Legacy quick-sweep PPO entry for all 7 real tasks (CLI aggregator).

Tasks: the three original-task Genesis RL environments (quadx, fixedwing,
rocket) plus the fully tensorized space environments (satellite CMG attitude
control, low-thrust perigee-raising transfer, CWH docking / rendezvous).

This is the small-config sweep entry (default 64 envs, 32-step rollouts,
64x64 MLP, adaptive-KL schedule). For experiment runs use the self-contained
per-task entries `train_<task>.py` instead -- they share
src/common/train_utils.py, ship the project-standard 1024-env configuration
and expose the DIFFICULTY constant; this file passes no env kwargs (always
benchmark difficulty).

Usage: python train.py quadx|fixedwing|rocket|satellite|transfer|docking|rendezvous [--num-envs 64] [--updates 1000]
Defaults: GPU backend, seed 7, PPO config verified in the small-scale run.
"""
import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

# src/ is the pyflyt_genesis package root: env subpackages keep their relative
# imports and the frozen original_pyflyt sources keep their absolute ones.
_src = Path(__file__).resolve().parent.parent / 'src'
_pkg = types.ModuleType('pyflyt_genesis')
_pkg.__path__ = [str(_src)]
sys.modules['pyflyt_genesis'] = _pkg

import torch
from rsl_rl.runners import OnPolicyRunner

from pyflyt_genesis.quadx import QuadXHoverEnv
from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv
from pyflyt_genesis.rocket import RocketLandingEnv
from pyflyt_genesis.satellite import SatelliteAttitudeEnv
from pyflyt_genesis.transfer import LowThrustTransferEnv
from pyflyt_genesis.docking import SpaceDockingEnv, SpaceRendezvousEnv

ENVS = {'quadx': QuadXHoverEnv, 'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
        'satellite': SatelliteAttitudeEnv, 'transfer': LowThrustTransferEnv,
        'docking': SpaceDockingEnv, 'rendezvous': SpaceRendezvousEnv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vehicle', choices=['quadx', 'fixedwing', 'rocket', 'satellite',
                                            'transfer', 'docking', 'rendezvous'])
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true',
                        help='resume from the newest model_*.pt / checkpoint.pt already in --out')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = args.out or Path(__file__).resolve().parent.parent / 'results' / args.vehicle
    out.mkdir(parents=True, exist_ok=True)

    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed)
    model_cfg = {'class_name': 'MLPModel', 'hidden_dims': [64, 64], 'activation': 'tanh', 'obs_normalization': True}
    cfg = {'num_steps_per_env': 32, 'save_interval': 100, 'obs_groups': {'actor': ['policy'], 'critic': ['policy']},
           'actor': {**model_cfg, 'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': .5, 'std_type': 'log'}},
           'critic': dict(model_cfg),
           'algorithm': {'class_name': 'PPO', 'learning_rate': .0003, 'num_learning_epochs': 4, 'num_mini_batches': 4,
                         'schedule': 'adaptive', 'desired_kl': .01, 'entropy_coef': .001, 'gamma': .99, 'lam': .95},
           'logger': 'tensorboard'}
    # OnPolicyRunner mutates cfg in place (resolving/popping class_name); pass a
    # deep copy so the dumped config.json stays a faithful training config.
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=str(out), device=str(env.device))
    if args.resume:
        cands = sorted(out.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        ckpt = cands[-1] if cands else out / 'checkpoint.pt'
        if ckpt.exists():
            runner.load(str(ckpt))
            print(f'resumed from {ckpt}', flush=True)
        else:
            print('no checkpoint found, training from scratch', flush=True)
    start = time.perf_counter()
    runner.learn(args.updates)
    elapsed = time.perf_counter() - start
    runner.save(str(out / 'checkpoint.pt'))

    report = {'task': env.cfg, 'device': str(env.device), 'updates': args.updates,
              'transitions': args.updates * cfg['num_steps_per_env'] * args.num_envs,
              'train_wall_seconds': elapsed,
              'completed_episodes': env.completed_episodes.detach().cpu().tolist(),
              'completed_successes': env.completed_successes.detach().cpu().tolist(),
              'physics_step_calls': getattr(getattr(env, 'plant', None), 'physics_steps', None),
              'note': 'Original default tasks on device-resident tensors; NumPy seeded RNG bitstreams and physics engine numerics are not identical to PyFlyt.'}
    if getattr(env, 'completed_metrics', None):
        eps = max(int(env.completed_episodes.detach().sum()), 1)
        report['completed_metrics'] = {k: float(v.detach().sum()) / eps for k, v in env.completed_metrics.items()}
    (out / 'run.json').write_text(json.dumps(report, indent=2))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    main()
