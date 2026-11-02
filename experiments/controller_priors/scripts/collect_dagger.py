"""DAgger aggregation: student rollouts relabelled by the conventional controller.

Iteration 0 executes the expert (beta = 1) so the initial data matches the
plain demonstration set; later iterations execute a beta-mixture of the
current student and the controller (beta_i = beta ** i), store the CONTROLLER
label at every visited state, and merge all iterations (plus an optional
base demonstration dataset) into one supervised dataset used by ``bc_ppo`` /
``bc_dagger_ppo``.

The controller is only used as an action LABELER (``actions(obs)``); it never
reads environment state and no reward/success/termination is modified.
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from rsl_rl.runners import OnPolicyRunner

from algorithms.train_utils import RslVecEnv
from algorithms.warmstart.expert_dataset import (collect_expert_dataset,
                                                 load_expert_dataset,
                                                 merge_datasets, save_expert_dataset,
                                                 dataset_summary)
from scripts.common import (run_dir, ensure_new_run_dir, env_class, make_controller,
                            make_env, write_json, run_metadata, task_config)
from scripts import conditions


def _load_student(checkpoint, env):
    cfg = json.loads((Path(checkpoint).parent / 'config.json').read_text())
    init_std = cfg.get('actor', {}).get('distribution_cfg', {}).get('init_std')
    if isinstance(init_std, list):
        cfg['actor']['distribution_cfg']['init_std'] = torch.tensor(init_std)
    runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/cp_dagger_tb',
                            device=str(env.device))
    runner.load(str(checkpoint))
    return runner.alg.actor.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--condition', default='default')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--beta', type=float, default=0.5,
                   help='per-iteration expert-mixing decay; iteration i executes '
                        'beta**i (iteration 0 = pure expert)')
    p.add_argument('--bc-checkpoint', type=Path, required=True)
    p.add_argument('--base-dataset', type=Path, default=None)
    p.add_argument('--num-envs', type=int, default=None)
    p.add_argument('--episodes-per-replica', type=int, default=None)
    p.add_argument('--max-dataset-mb', type=float, default=50.0,
                   help='hard cap on the saved dataset size (MiB); whole replicas '
                        'are dropped to meet it')
    p.add_argument('--select-episodes', type=int, default=None,
                   help='representative episodes kept per iteration; default: '
                        'task config data_selection.dagger_episodes (falls back to '
                        'data_selection.episodes)')
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--backend', default='gpu')
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--run-tag', default=None)
    args = p.parse_args()

    cfg = task_config(args.task)
    cond = conditions.resolve(args.task, args.condition)
    # dataset directory, distinct from the bc_dagger ARM's run directory
    out = run_dir(args.task, 'bc_dagger_dataset', args.condition, args.seed,
                  args.out_root, args.run_tag)
    ensure_new_run_dir(out)
    torch.manual_seed(args.seed)
    n_envs = int(args.num_envs or cfg['num_envs'])
    episodes = int(args.episodes_per_replica
                   or (cfg.get('dagger') or {}).get('episodes_per_replica', 1))
    select_episodes = args.select_episodes
    if select_episodes is None:
        sel_cfg = cfg.get('data_selection') or {}
        select_episodes = int(sel_cfg.get('dagger_episodes')
                              or sel_cfg.get('episodes') or 0) or None
    train_kw = conditions.train_env_kwargs(args.task, cond, cfg)
    env = make_env(args.task, cfg, num_envs=n_envs, seed=args.seed,
                   backend=args.backend, env_kwargs=train_kw)
    controller = make_controller(args.task, env, cond)
    actor = _load_student(args.bc_checkpoint, env)
    print(f'[dagger] task={args.task} condition={args.condition} seed={args.seed} '
          f'envs={n_envs} episodes_per_replica={episodes} '
          f'select_episodes={select_episodes}', flush=True)

    def student(obs):
        with torch.no_grad():
            return actor(obs)

    datasets = []
    if args.base_dataset is not None:
        datasets.append(load_expert_dataset(args.base_dataset))
    t0 = time.perf_counter()
    iterations = []
    for it in range(args.iterations):
        beta = float(args.beta) ** it
        collect_kw = {} if args.max_steps is None else {'max_steps': int(args.max_steps)}
        ds = collect_expert_dataset(
            env, episodes_per_replica=episodes, controller=controller,
            seed=args.seed, policy=student, beta=beta,
            select_episodes=select_episodes, **collect_kw)
        meta = ds['meta']
        meta['dagger_iteration'] = it
        datasets.append(ds)
        iterations.append({'iteration': it, 'beta': beta,
                           'transitions': meta['collection']['transitions'],
                           'collection_wall_seconds': meta['collection']['collection_wall_seconds']})
        print(f'[dagger] iteration {it}: beta={beta} '
              f'transitions={meta["collection"]["transitions"]}', flush=True)
    merged = merge_datasets(datasets)
    merged = save_expert_dataset(merged, out / 'dagger_dataset.pt',
                                 max_mb=args.max_dataset_mb)
    summary = dataset_summary(merged)
    storage = merged['meta'].get('storage') or {}
    if storage.get('replicas_written') != storage.get('replicas_collected'):
        print(f'[dagger] size cap {args.max_dataset_mb} MiB: kept '
              f'{storage["replicas_written"]}/{storage["replicas_collected"]} '
              f'replicas (whole episodes dropped)', flush=True)
    file_mb = (out / 'dagger_dataset.pt').stat().st_size / 2**20
    summary['dataset_mb'] = round(file_mb, 2)
    summary['dagger_iterations'] = iterations
    summary['base_dataset'] = str(args.base_dataset) if args.base_dataset else None
    summary['bc_checkpoint'] = str(args.bc_checkpoint)
    summary['wall_seconds'] = round(time.perf_counter() - t0, 2)
    write_json(out / 'summary.json', summary)
    write_json(out / 'meta.json', run_metadata(
        task=args.task, method='bc_dagger', condition=args.condition, seed=args.seed,
        extra={'num_envs': n_envs, 'episodes_per_replica': episodes,
               'iterations': iterations, 'train_env_kwargs': train_kw,
               'controller_design': cond.get('controller_design'),
               'baseline_variant': cond.get('baseline_variant')}))
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    main()
