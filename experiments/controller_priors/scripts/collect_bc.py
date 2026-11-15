"""Collect the episode-level expert dataset for one (task, condition, seed)."""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from algorithms.warmstart.expert_dataset import (collect_expert_dataset, save_expert_dataset,
                                                 dataset_summary, split_dataset)
from scripts.common import (ROOT, run_dir, ensure_new_run_dir, make_env, make_controller,
                            write_json, run_metadata, task_config)
from scripts import conditions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--condition', default='default')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num-envs', type=int, default=None)
    p.add_argument('--episodes-per-replica', type=int, default=None)
    p.add_argument('--max-dataset-mb', type=float, default=50.0,
                   help='hard cap on the saved dataset size (MiB); whole replicas '
                        'are dropped to meet it')
    p.add_argument('--select-episodes', type=int, default=None,
                   help='keep only this many representative (farthest-point) '
                        'episodes; default: task config data_selection.episodes')
    p.add_argument('--backend', default='gpu')
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--run-tag', default=None)
    p.add_argument('--reuse', action='store_true', help='reuse an existing complete dataset')
    args = p.parse_args()

    cfg = task_config(args.task)
    cond = conditions.resolve(args.task, args.condition)
    out = run_dir(args.task, 'bc_dataset', args.condition, args.seed, args.out_root, args.run_tag)
    dataset_path = out / 'dataset.pt'
    if dataset_path.exists() and args.reuse:
        print(f'[bc-data] reusing {dataset_path}', flush=True)
        return
    ensure_new_run_dir(out)
    torch.manual_seed(args.seed)

    n_envs = int(args.num_envs or cfg['num_envs'])
    ep = int(args.episodes_per_replica or conditions.bc_overrides(args.task, cond, cfg)['episodes_per_replica'])
    sel = args.select_episodes
    if sel is None:
        sel = int((cfg.get('data_selection') or {}).get('episodes') or 0) or None
    env_kwargs = conditions.train_env_kwargs(args.task, cond, cfg)
    env = make_env(args.task, cfg, num_envs=n_envs, seed=args.seed,
                   backend=args.backend, env_kwargs=env_kwargs)
    controller = make_controller(args.task, env, cond)
    print(f'[bc-data] task={args.task} condition={args.condition} seed={args.seed} '
          f'envs={n_envs} episodes_per_replica={ep} select_episodes={sel}', flush=True)
    dataset = collect_expert_dataset(env, episodes_per_replica=ep, controller=controller,
                                     seed=args.seed, select_episodes=sel)
    dataset['meta'].update(run_metadata(task=args.task, method='bc_dataset',
                                        condition=args.condition, seed=args.seed))
    dataset['meta']['train_env_kwargs'] = env_kwargs
    dataset['meta']['controller_design'] = cond.get('controller_design')
    dataset['meta']['residual_rho'] = cond['residual_rho']
    dataset = save_expert_dataset(dataset, dataset_path, max_mb=args.max_dataset_mb)
    summary = dataset_summary(dataset)
    splits = split_dataset(dataset, seed=args.seed)
    summary['split_episodes'] = {k: len(v) for k, v in splits.items()}
    summary['split_protocol'] = 'episode-disjoint 70/15/15 by complete episode'
    summary['dataset_path'] = str(dataset_path)
    write_json(out / 'dataset.json', summary)
    file_mb = dataset_path.stat().st_size / 2**20
    storage = dataset['meta'].get('storage') or {}
    if storage.get('replicas_written') != storage.get('replicas_collected'):
        print(f'[bc-data] size cap {args.max_dataset_mb} MiB: kept '
              f'{storage["replicas_written"]}/{storage["replicas_collected"]} '
              f'replicas (whole episodes dropped)', flush=True)
    print(json.dumps({k: summary[k] for k in
                      ('steps_used', 'transitions', 'episodes_complete', 'success_episodes',
                       'timeout_episodes', 'terminated_episodes', 'unfinished_episodes',
                       'collection_wall_seconds', 'split_episodes') if k in summary}, indent=2))
    print(f'[bc-data] success rate {summary["success_episodes"]}/{summary["episodes_complete"]} '
          f'complete episodes; saved {file_mb:.1f} MiB', flush=True)
    env.close()


if __name__ == '__main__':
    main()
