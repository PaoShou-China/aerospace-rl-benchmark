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

from environments.multirotor import QuadXWaypointsEnv, QuadXTrackEnv
from environments.fixedwing import FixedwingWaypointsEnv
from environments.rocket import RocketLandingEnv
from environments.missile import MissileInterceptEnv
from environments.vtol.env import QuadplaneWaypointsEnv
from environments.satellite.attitude_control import SatelliteAttitudeEnv
from environments.satellite.orbit_transfer import LowThrustTransferEnv
from environments.satellite.rendezvous_docking import SpaceDockingEnv, SpaceRendezvousEnv

from algorithms.baselines.baselines import BASELINE_VARIANTS, build_baseline

ENVS = {'quadx': QuadXWaypointsEnv, 'quadx_track': QuadXTrackEnv,
        'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
        'missile': MissileInterceptEnv, 'quadplane': QuadplaneWaypointsEnv,
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
    parser.add_argument('--baseline', default=None,
                        help='controller variant from BASELINE_VARIANTS (default: '
                             'the task\'s registered baseline)')
    parser.add_argument('--num-cmgs', type=int, choices=(4, 6), default=None,
                        help='satellite only: override num_cmgs for this evaluation')
    args = parser.parse_args()
    if args.baseline is not None and args.baseline not in BASELINE_VARIANTS.get(args.vehicle, {}):
        parser.error(f'unknown baseline {args.baseline!r} for {args.vehicle}; '
                     f'known: {sorted(BASELINE_VARIANTS.get(args.vehicle, {}))}')

    out = args.out or Path(__file__).resolve().parent.parent / 'results' / args.vehicle
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    # match eval.py's env construction when a training config is available
    cfg_path = out / 'config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    env_kwargs = dict(cfg.get('env_kwargs') or {})
    if args.num_cmgs is not None:
        if args.vehicle != 'satellite':
            parser.error('--num-cmgs is only valid for the satellite task')
        env_kwargs['num_cmgs'] = args.num_cmgs
    env = ENVS[args.vehicle](num_envs=args.num_envs, backend=args.backend, seed=args.seed, **env_kwargs)
    controller = build_baseline(args.vehicle, env, args.baseline)

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
              'seed': args.seed, 'eval_wall_seconds': wall}
    report.update(cohort.report())
    report['benchmark'] = env.cfg.get('benchmark')
    # per-first-episode mean task metrics (touch quality etc.), from the cohort
    n = max(report['episodes_finished'], 1)
    report['task_metrics'] = {k: v / n for k, v in report['metric_counts'].items()}
    # global counters (quadx/fixedwing) cover every episode in the window
    m = max(report['episodes_completed_total'], 1)
    report['task_metrics'].update(
        {k: v / m for k, v in report['global_metric_counts'].items()})
    if hasattr(controller, 'config'):
        report['controller_config'] = controller.config()
    if hasattr(controller, 'metrics'):
        metrics = {}
        for k, v in controller.metrics.items():
            try:
                metrics[k] = float(v.detach().cpu())
            except (AttributeError, TypeError):
                metrics[k] = v
        steps_m = max(metrics.get('steps', 0.), 1.)
        envsteps = steps_m * max(env.num_envs, 1)  # sums accumulate over the batch
        metrics['rate_sat_envstep_fraction'] = metrics.get('clamp_envsteps', 0.) / envsteps
        metrics['gimbal_sat_fraction'] = (metrics.get('sat_gimbal_envsteps', 0.)
                                          / (envsteps * max(env.num_gimbals, 1)))
        metrics['mean_tau_err_norm'] = metrics.get('tau_err_norm_sum', 0.) / envsteps
        metrics['mean_tau_err_rel'] = metrics.get('tau_err_rel_sum', 0.) / envsteps
        metrics['mean_rate_request_norm'] = metrics.get('rate_request_norm_sum', 0.) / envsteps
        metrics['mean_m_norm'] = metrics.get('m_norm_sum', 0.) / envsteps
        metrics['mean_sigma_min_ratio'] = metrics.get('sigma_min_ratio_sum', 0.) / envsteps
        report['controller_metrics'] = metrics
    print(json.dumps(report, indent=2), flush=True)
    fname = 'baseline.json' if args.baseline is None else f'baseline_{controller.name}.json'
    (out / fname).write_text(json.dumps(report, indent=2))
    env.close()
    if not report['complete']:
        raise RuntimeError('Incomplete first-episode cohort; partial report saved')


if __name__ == '__main__':
    main()
