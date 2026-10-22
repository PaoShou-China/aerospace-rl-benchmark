"""Aggregate all per-task artifacts under results/ into one summary table.

Usage: python summarize.py [--out results/summary.md]

Collects, per task (whatever exists):
  run.json      training run: algo, updates, transitions, wall time, final
                training success rate (sum successes / sum episodes)
  baseline.json classical-controller evaluation (same metrics)
  eval.json     eval.py output: deterministic policy success rate / return
  config.json   env difficulty and checkpoint presence
Prints a markdown table and writes it to --out (default results/summary.md).
"""
import argparse
import json
from pathlib import Path

TASKS = ['quadx', 'fixedwing', 'rocket', 'satellite', 'transfer', 'docking', 'rendezvous']


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def collect(task):
    root = Path(__file__).resolve().parent.parent / 'results' / task
    run, base, ev = load(root / 'run.json'), load(root / 'baseline.json'), load(root / 'eval.json')
    cfg = load(root / 'config.json') or {}
    row = {'task': task,
           'difficulty': cfg.get('difficulty'),
           'trained': (root / 'checkpoint.pt').exists(),
           'algo': (run or {}).get('algo') or cfg.get('algo'),
           'updates': (run or {}).get('updates'),
           'transitions': (run or {}).get('transitions'),
           'train_h': None, 'train_sr': None, 'ksteps_s': None, 'vram_gib': None,
           'baseline': (base or {}).get('baseline'), 'base_sr': (base or {}).get('success_rate'),
           'base_ret': (base or {}).get('mean_episode_return'),
           'eval_sr': (ev or {}).get('success_rate'), 'eval_ret': (ev or {}).get('mean_episode_return'),
           'vis': (root / 'vis.mp4').exists(), 'vis_ckpt': (root / 'vis_ckpt.mp4').exists()}
    if run:
        if run.get('train_wall_seconds') is not None:
            row['train_h'] = run['train_wall_seconds'] / 3600.
        if run.get('train_throughput_steps_per_s'):
            row['ksteps_s'] = run['train_throughput_steps_per_s'] / 1e3
        row['vram_gib'] = run.get('peak_vram_gib')
        eps, ok = run.get('completed_episodes'), run.get('completed_successes')
        if eps and ok:
            row['train_sr'] = sum(ok) / max(sum(eps), 1.)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'results' / 'summary.md')
    args = parser.parse_args()

    rows = [collect(t) for t in TASKS]
    header = ('| task | algo | updates | transitions | train h | k steps/s | vram GiB | train SR | baseline | base SR | base ret |'
              ' eval SR | eval ret | vis | vis-ckpt |')
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['algo'] or '-'} | {r['updates'] if r['updates'] is not None else '-'}"
            f" | {r['transitions'] if r['transitions'] is not None else '-'}"
            f" | {fmt(r['train_h'], 2)} | {fmt(r['ksteps_s'], 1)} | {fmt(r['vram_gib'], 2)} | {fmt(r['train_sr'])} | {r['baseline'] or '-'}"
            f" | {fmt(r['base_sr'])} | {fmt(r['base_ret'], 1)}"
            f" | {fmt(r['eval_sr'])} | {fmt(r['eval_ret'], 1)}"
            f" | {'yes' if r['vis'] else '-'} | {'yes' if r['vis_ckpt'] else '-'} |")
    table = '\n'.join(lines)
    print(table)
    args.out.write_text(table + '\n')
    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
