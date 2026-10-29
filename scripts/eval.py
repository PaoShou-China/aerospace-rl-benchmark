"""Deterministic evaluation of a trained RSL-RL checkpoint on the batched GPU envs.

Usage: python eval.py quadx|fixedwing|rocket|missile|satellite|transfer|docking|rendezvous
Defaults: 1024 envs on GPU, seed 7, frozen deterministic policy (mean action,
no exploration noise). The env configuration used at training time
(env_kwargs) is restored from results/<task>/config.json.

Unbiased episode accounting: the loop runs until EVERY replica has completed
at least one full episode (safety-capped by --max-steps), so no in-flight
episode is dropped and a zero-success run cannot be an artifact of a too-short
window (fixed-wing episodes last up to 3600 steps).
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

from algorithms.train_utils import RslVecEnv
from rsl_rl.runners import OnPolicyRunner

from environments.multirotor import QuadXWaypointsEnv, QuadXTrackEnv
from environments.fixedwing import FixedwingWaypointsEnv
from environments.rocket import RocketLandingEnv
from environments.missile import MissileInterceptEnv
from environments.satellite.attitude_control import SatelliteAttitudeEnv
from environments.satellite.orbit_transfer import LowThrustTransferEnv
from environments.satellite.rendezvous_docking import SpaceDockingEnv, SpaceRendezvousEnv

ENVS = {'quadx': QuadXWaypointsEnv, 'quadx_track': QuadXTrackEnv,
        'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
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
                        help='training dir holding checkpoint.pt (default results/<vehicle>)')
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent.parent / 'results' / args.vehicle
    ckpt = out / 'checkpoint.pt'
    assert ckpt.exists(), f'missing {ckpt}'
    cfg = json.loads((out / 'config.json').read_text())
    # Per-dim exploration std is dumped as a JSON list (train_rocket's _JsonStd
    # keeps json.dumps working). rsl_rl builds the distribution as
    # `init_std * torch.ones(output_dim)`, which needs tensor semantics: a plain
    # list raises "only integer tensors of a single element can be converted to
    # an index" when the runner is reconstructed. Restore the tensor form so a
    # saved config is reloadable exactly as it was at training time.
    init_std = cfg.get('actor', {}).get('distribution_cfg', {}).get('init_std')
    if isinstance(init_std, list):
        cfg['actor']['distribution_cfg']['init_std'] = torch.tensor(init_std)

    torch.manual_seed(args.seed)
    # restore the exact env construction used at training time
    env_kwargs = dict(cfg.get('env_kwargs') or {})
    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed, **env_kwargs)
    if cfg.get('algo') == 'flash_sac':
        from algorithms.flashsac.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        policy = runner.deterministic_action
    else:
        runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        actor = runner.alg.actor.eval()
        policy = actor

    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    start = time.perf_counter()
    steps = 0
    with torch.no_grad():
        while steps < args.max_steps:
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            steps += 1
            if cohort.update(env):
                break
    wall = time.perf_counter() - start

    report = {'task': args.vehicle, 'checkpoint': str(ckpt), 'env_steps': steps,
              'num_envs': args.num_envs, 'env_kwargs': env_kwargs,
              'eval_wall_seconds': wall}
    report.update(cohort.report())
    report['benchmark'] = env.cfg.get('benchmark')
    # per-first-episode mean task metrics (touch quality etc.), from the cohort
    n = max(report['episodes_finished'], 1)
    report['task_metrics'] = {k: v / n for k, v in report['metric_counts'].items()}
    # global counters (quadx/fixedwing) cover every episode in the window
    m = max(report['episodes_completed_total'], 1)
    report['task_metrics'].update(
        {k: v / m for k, v in report['global_metric_counts'].items()})
    print(json.dumps(report, indent=2), flush=True)
    (out / 'eval.json').write_text(json.dumps(report, indent=2))
    env.close()
    if not report['complete']:
        raise RuntimeError('Incomplete first-episode cohort; partial report saved')


if __name__ == '__main__':
    main()
