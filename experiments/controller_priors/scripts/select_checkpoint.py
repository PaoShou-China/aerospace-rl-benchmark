"""Select the best stage checkpoint by DETERMINISTIC validation success.

Motivation: the shipped arms keep the LAST checkpoint, but the pilot showed
non-monotone success (BC -> PPO collapses mid-run while earlier stages are
still good).  This script evaluates ``stage_*.pt`` and ``checkpoint.pt`` on a
held-out VALIDATION seed (different from the reporting eval seed), ranks them
by (success rate desc, cohort complete desc, terminal error asc, return desc)
and writes:

    selection/selection.json     full candidate table + the selected checkpoint
    selection/best_checkpoint.pt copy of the winner
    selection/det_<name>.json    per-candidate validation report

The ranking is a pure function (``rank_candidates``) so it is unit-tested
without a simulator.  Selection never overwrites evaluations of the official
reporting seed: candidates are re-rolled under the validation seed only.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# NOTE: the ``scripts`` name collides with the repository-level scripts
# package when this module is imported from a test process; the heavy imports
# are therefore deferred to main(), leaving ``rank_candidates`` importable as a
# pure function (file-loaded in the tests).
TERMINAL_KEY = {'docking': 'terminal_range_m',
                'rendezvous': 'terminal_range_m',
                'satellite': 'terminal_attitude_error_rad'}


def rank_candidates(rows, terminal_key=None):
    """Sort candidate checkpoint evaluations (best first).

    Primary: deterministic validation success rate.  Tie-breakers: a complete
    cohort is better; lower terminal error is better; higher mean return is
    better.  ``rows`` is a list of dicts with those keys; a missing/incomplete
    value sorts last.
    """
    def val(r, k):
        v = r.get(k)
        return v if isinstance(v, (int, float)) and v == v else None

    def key(r):
        succ = val(r, 'success_rate')
        comp = 1 if r.get('complete') else 0
        term = val(r, terminal_key) if terminal_key else None
        ret = val(r, 'mean_episode_return')
        return (succ is not None, succ if succ is not None else -1.0,
                comp,
                term is not None, -(term if term is not None else float('inf')),
                ret is not None, ret if ret is not None else float('-inf'))

    return sorted(rows, key=key, reverse=True)


def _candidates(out):
    rows = sorted(out.glob('stage_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    if (out / 'checkpoint.pt').exists():
        rows.append(out / 'checkpoint.pt')
    return rows


def main():
    from scripts import evaluate as ev
    from scripts import conditions
    from scripts.common import run_dir, write_json, task_config
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--condition', default='default')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--val-seed', type=int, default=None,
                   help='validation seed (default: seed + 1000, disjoint from the report)')
    p.add_argument('--num-envs', type=int, default=None)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--backend', default='gpu')
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--run-tag', default=None)
    args = p.parse_args()

    cfg = task_config(args.task)
    cond = conditions.resolve(args.task, args.condition)
    out = run_dir(args.task, args.method, args.condition, args.seed,
                  args.out_root, args.run_tag)
    cands = _candidates(out)
    if not cands:
        raise FileNotFoundError(f'no stage_*.pt / checkpoint.pt under {out}')
    val_seed = int(args.val_seed if args.val_seed is not None else args.seed + 1000)
    ev_cfg = ev._load_eval_config(out, args.method, args.task, cond, cfg)
    n_envs = int(args.num_envs or cfg['eval']['num_envs'])
    max_steps = int(args.max_steps or ev_cfg['max_steps'])
    sel_dir = out / 'selection'
    sel_dir.mkdir(parents=True, exist_ok=True)
    plt_rows = []
    for ckpt in cands:
        torch.manual_seed(val_seed)
        env, controller = ev._make_eval_env(args.task, cfg, ev_cfg, num_envs=n_envs,
                                            backend=args.backend, eval_seed=val_seed,
                                            method=args.method, cond=cond)
        actor, _ = ev._load_actor(env, ckpt)
        roll = ev._rollout(args.task, cfg, env, method=args.method, actor=actor,
                           controller=controller, deterministic=True,
                           eval_seed=val_seed, max_steps=max_steps,
                           dump_replicas=ev_cfg['dump_replicas'])
        row = {'checkpoint': str(ckpt), 'name': ckpt.name,
               'success_rate': roll['report']['success_rate'],
               'complete': roll['report']['complete'],
               'episodes_finished': roll['report']['episodes_finished'],
               'mean_episode_return': roll['report']['mean_episode_return'],
               'terminal_error': (roll['physical']['mean_completed'] or {}).get(
                   TERMINAL_KEY.get(args.task)),
               'init_digest': roll['init_digest']}
        plt_rows.append(row)
        write_json(sel_dir / f'det_{ckpt.stem}.json',
                   {**row, 'val_seed': val_seed,
                    'physical': roll['physical'], 'per_replica': roll['rows']})
        print(f'[select] {ckpt.name}: success={row["success_rate"]} '
              f'return={row["mean_episode_return"]}', flush=True)
        env.close()
    ranked = rank_candidates(plt_rows, TERMINAL_KEY.get(args.task))
    best = ranked[0]
    shutil.copy2(best['checkpoint'], sel_dir / 'best_checkpoint.pt')
    write_json(sel_dir / 'selection.json',
               {'task': args.task, 'method': args.method, 'condition': args.condition,
                'seed': args.seed, 'val_seed': val_seed,
                'terminal_key': TERMINAL_KEY.get(args.task),
                'ranking': ranked, 'best': best['checkpoint'],
                'best_name': best['name']})
    print(json.dumps({'best': best['name'], 'success_rate': best['success_rate'],
                      'val_seed': val_seed}, indent=2), flush=True)


if __name__ == '__main__':
    main()
