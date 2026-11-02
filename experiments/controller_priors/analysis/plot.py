"""Figures for the controller-prior study (only existing, completed data)."""
import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE.parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ROOT = ROOT / 'results' / 'controller_priors'
METHOD_STYLE = {
    'traditional': ('#333333', 'o', 'traditional controller'),
    'bc_only': ('#1f77b4', 's', 'BC only'),
    'ppo_scratch': ('#d62728', '^', 'PPO scratch'),
    'bc_ppo': ('#2ca02c', 'D', 'BC -> PPO'),
    'residual_ppo': ('#9467bd', 'v', 'Residual PPO'),
}
TASKS = ('docking', 'satellite', 'rendezvous')
COST_KEYS = {'docking': 'propellant_kg', 'rendezvous': 'propellant_kg',
             'satellite': 'gimbal_rate_squared_integral_proxy'}


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _fig_dir(out_root):
    d = Path(out_root) / '_analysis' / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'wrote {path}')


def _filter(rows, **kw):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def plot_learning_curves(stage_summary, out_root):
    for task in TASKS:
        pts = [r for r in stage_summary if r['task'] == task and r['condition'] == 'default']
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHOD_STYLE:
            mpts = sorted([r for r in pts if r['method'] == method], key=lambda r: r['stage'])
            if not mpts:
                continue
            c, mk, label = METHOD_STYLE[method]
            x = [r['stage'] for r in mpts]
            y = [r['success_mean'] if r['success_mean'] is not None else np.nan for r in mpts]
            sd = [r['success_sd'] or 0 for r in mpts]
            ax.errorbar(x, y, yerr=sd, color=c, marker=mk, capsize=3, lw=1.6, label=label)
        ax.set_xlabel('PPO updates')
        ax.set_ylabel('deterministic first-episode success')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=.25)
        ax.set_title(f'{task}: learning curves (mean +- sd across seeds)')
        ax.legend(fontsize=8)
        _save(fig, _fig_dir(out_root) / f'learning_curves_{task}.png')


def plot_success_and_cost(summary, out_root):
    for task in TASKS:
        rows = [r for r in summary if r['task'] == task and r['condition'] == 'default'
                and r['mode'] == 'det' and r['stage'] is None]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        methods = [m for m in METHOD_STYLE if any(r['method'] == m for r in rows)]
        succ = [next(r['success_mean'] for r in rows if r['method'] == m) for m in methods]
        sd = [next((r['success_sd'] or 0) for r in rows if r['method'] == m) for m in methods]
        colors = [METHOD_STYLE[m][0] for m in methods]
        axes[0].bar(range(len(methods)), succ, yerr=sd, color=colors, capsize=4)
        axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels([METHOD_STYLE[m][2] for m in methods], rotation=20, ha='right',
                                fontsize=8)
        axes[0].set_ylabel('first-episode success')
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(axis='y', alpha=.25)
        ck = COST_KEYS.get(task)
        if ck:
            vals = [next((r['physical'] or {}).get(ck) for r in rows if r['method'] == m)
                    for m in methods]
            axes[1].bar(range(len(methods)), [v if v is not None else np.nan for v in vals],
                        color=colors)
            axes[1].set_xticks(range(len(methods)))
            axes[1].set_xticklabels([METHOD_STYLE[m][2] for m in methods], rotation=20,
                                    ha='right', fontsize=8)
            axes[1].set_ylabel(ck.replace('_', ' '))
            axes[1].grid(axis='y', alpha=.25)
        fig.suptitle(f'{task}: final deterministic success and physical cost '
                     '(cost shown next to its success fraction)')
        _save(fig, _fig_dir(out_root) / f'success_cost_{task}.png')


def _condition_panel(summary, out_root, conditions, title, fname, mode='det'):
    present = []
    for task in TASKS:
        for cond in conditions:
            rs = [r for r in summary if r['task'] == task and r['condition'] == cond
                  and r['mode'] == mode and r['stage'] is None]
            if rs:
                present.append((task, cond, rs))
    if not present:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(present) + 3), 4))
    labels, vals, errs, colors = [], [], [], []
    for task, cond, rs in present:
        for m in METHOD_STYLE:
            r = next((x for x in rs if x['method'] == m), None)
            if not r:
                continue
            labels.append(f'{task}\n{cond}\n{METHOD_STYLE[m][2]}')
            vals.append(r['success_mean'] if r['success_mean'] is not None else np.nan)
            errs.append(r['success_sd'] or 0)
            colors.append(METHOD_STYLE[m][0])
    if not labels:
        return
    ax.bar(range(len(labels)), vals, yerr=errs, color=colors, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel('deterministic first-episode success')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=.25)
    ax.set_title(title)
    _save(fig, _fig_dir(out_root) / fname)


def plot_scans(summary, out_root, runs):
    # controller quality
    q = sorted({r['condition'] for r in summary if r['condition'].startswith('quality_')})
    _condition_panel(summary, out_root, ['default'] + q,
                     'controller quality scan (true plant unchanged)',
                     'scan_controller_quality.png')
    # coverage
    cov = sorted({r['condition'] for r in summary if r['condition'].startswith('coverage_')})
    _condition_panel(summary, out_root, ['default'] + cov,
                     'demonstration coverage (same sample count)',
                     'scan_demonstration_coverage.png')
    # sample count
    smp = sorted({r['condition'] for r in summary if r['condition'].startswith('samples_')})
    _condition_panel(summary, out_root, ['default'] + smp,
                     'demonstration sample count (same coverage)',
                     'scan_demonstration_samples.png')
    # model mismatch
    mm = sorted({r['condition'] for r in summary if r['condition'].startswith('mismatch_')})
    _condition_panel(summary, out_root, ['default'] + mm,
                     'model mismatch (plant shift vs pinned design model)',
                     'scan_model_mismatch.png')
    # residual authority
    tasks_with_auth = sorted({r['task'] for r in summary
                              if r['condition'].startswith('authority_')
                              and r['mode'] == 'det' and r['stage'] is None})
    if tasks_with_auth:
        fig, axes = plt.subplots(1, len(tasks_with_auth),
                                 figsize=(4.2 * len(tasks_with_auth), 4), squeeze=False)
        for ax, task in zip(axes[0], tasks_with_auth):
            auth = sorted([r for r in summary if r['task'] == task
                           and r['condition'].startswith('authority_')
                           and r['mode'] == 'det' and r['stage'] is None],
                          key=lambda r: float(r['condition'].split('_')[1]))
            xs, ys, es = [], [], []
            for r in auth:
                xs.append(float(r['condition'].split('_')[1]))
                ys.append(r['success_mean'] if r['success_mean'] is not None else np.nan)
                es.append(r['success_sd'] or 0)
            ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, color='#9467bd')
            ax.set_xlabel('residual authority rho = limit / action range')
            ax.set_ylabel('deterministic first-episode success')
            ax.set_xscale('log')
            ax.set_ylim(-0.02, 1.05)
            ax.grid(alpha=.25)
            ax.set_title(f'{task}: residual authority')
        _save(fig, _fig_dir(out_root) / 'scan_residual_authority.png')
    # residual cancellation / composition diagnostics
    diag = [r for r in summary if r['method'] == 'residual_ppo' and r['mode'] == 'det'
            and r['stage'] is None and r.get('residual_stats')]
    if diag:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f'{r["task"]}\n{r["condition"]}' for r in diag]
        vals = [r['residual_stats'].get('cancellation_fraction', np.nan) for r in diag]
        comp = [r['residual_stats'].get('composed_clip_fraction', np.nan) for r in diag]
        ax.bar(np.arange(len(diag)) - .2, vals, width=.4, label='residual opposes nominal',
               color='#9467bd')
        ax.bar(np.arange(len(diag)) + .2, comp, width=.4, label='composed command clipped',
               color='#ff7f0e')
        ax.set_xticks(range(len(diag)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('fraction of first-episode env steps (run-wide stats)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.25)
        ax.set_title('residual composition diagnostics')
        _save(fig, _fig_dir(out_root) / 'residual_diagnostics.png')


def plot_trajectories(out_root):
    files = sorted(Path(out_root).glob('*/*/default/seed_*/eval/trajectory.npz'))
    if not files:
        return
    def pick(method_substr):
        for f in files:
            if method_substr in str(f):
                return f
        return files[0]
    choices = [('traditional', pick('traditional')), ('ppo_scratch', pick('ppo_scratch')),
               ('residual_ppo', pick('residual_ppo'))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, f) in zip(axes, choices):
        if not f.exists():
            ax.axis('off')
            continue
        z = np.load(f)
        done = np.zeros(0, dtype=bool)
        keys = sorted([k for k in z.files if k.startswith('obs_')],
                      key=lambda k: int(k.split('_')[1]))
        for k in keys[:2]:
            obs = z[k]
            done = z[k.replace('obs_', 'done_')]
            is_sat = 'satellite' in str(f)
            if is_sat and obs.shape[-1] >= 10:
                att = np.linalg.norm(obs[:, 4:7], axis=-1)
                rate = np.linalg.norm(obs[:, 7:10], axis=-1)
                ax.plot(att, label=f'replica {k.split("_")[1]} attitude error')
                ax.plot(rate, alpha=.5, label=f'replica {k.split("_")[1]} body rate')
            elif obs.shape[-1] >= 6:
                rng = np.linalg.norm(obs[:, :3], axis=-1)
                spd = np.linalg.norm(obs[:, 3:6], axis=-1)
                ax.plot(rng, label=f'replica {k.split("_")[1]} range')
                ax.plot(spd, alpha=.5, label=f'replica {k.split("_")[1]} speed')
            if k.replace('obs_', 'applied_') in z.files:
                u = z[k.replace('obs_', 'applied_')]
                ax.plot(np.linalg.norm(u, axis=-1), alpha=.35,
                        label=f'replica {k.split("_")[1]} |action|')
        ax.set_xlabel('control step')
        ax.set_ylabel('state norm / commanded norm')
        ax.set_title(f'{name}: first episode (ended={bool(done.any())})')
        ax.grid(alpha=.25)
        ax.legend(fontsize=6)
    _save(fig, _fig_dir(out_root) / 'trajectories.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--tag-map', default=None,
                   help='per-task tag selection, e.g. docking=pilot_v2,satellite=pilot_v1')
    args = p.parse_args()
    from analysis.aggregate import collect, summarise, main as agg_main
    rows, stages, datasets = collect(args.out_root, tag_map=args.tag_map)
    summary, stage_summary = summarise(rows, stages)
    d = _fig_dir(args.out_root)
    plot_learning_curves(stage_summary, args.out_root)
    plot_success_and_cost(summary, args.out_root)
    plot_scans(summary, args.out_root, rows)
    plot_trajectories(args.out_root)
    print(f'figures in {d}')


if __name__ == '__main__':
    main()
