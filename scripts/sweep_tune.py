"""Per-task hyperparameter sweep harness for PPO vs FlashSAC (tuning only).

Usage: python sweep_tune.py <task> [--only algo_arm_name ...]
       (argv exists ONLY here so parallel lanes can select one task; the
       production entries train_*.py stay hard-coded with no CLI.)

For each task this script runs a small, hard-coded candidate grid per
algorithm and writes one summary JSON per candidate to
results/sweep/<task>/<algo>_<name>/summary.json, then prints a table.
Sweep updates are reduced (see SWEEP_UPDATES / FS_UPDATES); envs, seed and
every non-candidate hyperparameter stay at the production entry's values so
the winner can be transplanted verbatim into train_*.py.

Candidate design (2026-10-22):
  PPO   -- production baseline + 2 single-knob variants chosen from the
           project's measured history (fixed 1e-3 beat 3e-4 on quadx /
           fixedwing / docking / rendezvous A/Bs; docking's 1024-env plateau
           mirrors the pre-recipe rendezvous config, so a rendezvous-style
           long-rollout arm is included there; satellite's gate-proven
           sigma bands get an arm each).
  FlashSAC -- never tuned on these tasks (paper defaults only), so the grid
           is wider: baseline (paper) + peak lr 1e-3 + batch 4096 /
           updates_per_step 2 + n-step 3. Per-task gamma matches the task's
           PPO gamma (horizon-driven), not the paper's 0.99.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from common.train_utils import run_training  # noqa: E402

NUM_ENVS = 1024
SEED = 7
BACKEND = 'gpu'
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

# --- per-task production PPO baselines (verbatim from each train_*.py) -----
PPO_BASE = {
    'quadx':      dict(lr_schedule='fixed', learning_rate=3e-4, entropy_coef=0.01,
                       init_std=1.0, num_steps_per_env=100, num_learning_epochs=5),
    'fixedwing':  dict(lr_schedule='fixed', learning_rate=3e-4, num_steps_per_env=64,
                       num_learning_epochs=10, entropy_coef=0.0, value_loss_coef=0.5),
    'rocket':     dict(lr_schedule='fixed', learning_rate=3e-4, gamma=0.995, lam=0.97,
                       entropy_coef=0.007, num_learning_epochs=10, num_steps_per_env=64,
                       init_std=None),  # filled from train_rocket.INIT_STD (_JsonStd)
    'satellite':  dict(lr_schedule='fixed', learning_rate=3e-4, gamma=0.999,
                       num_steps_per_env=150, num_mini_batches=5, num_learning_epochs=40,
                       entropy_coef=0.0, init_std=1.0, clip_param=0.2),
    'transfer':   dict(gamma=0.9999, lr_schedule='fixed', learning_rate=2.5e-4, lam=0.99,
                       entropy_coef=4.75e-8, num_learning_epochs=30, clip_param=0.3,
                       value_loss_coef=0.5),
    'docking':    dict(lr_schedule='fixed', learning_rate=3e-4, gamma=0.999,
                       entropy_coef=0.005),
    'rendezvous': dict(num_steps_per_env=250, gamma=0.999, num_mini_batches=1,
                       lr_schedule='fixed', learning_rate=3e-4, entropy_coef=0.003,
                       trunk_hidden=64, critic_trunk_hidden=64),
}

PPO_CANDIDATES = {
    'quadx': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('ent005', dict(entropy_coef=0.005)),
    ],
    'fixedwing': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('ent003', dict(entropy_coef=0.003)),
    ],
    'rocket': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('gamma999', dict(gamma=0.999)),
    ],
    'satellite': [
        ('base', {}),
        ('std05_lr1e3', dict(init_std=0.5, learning_rate=1e-3)),
        ('std02_lr3e5', dict(init_std=0.2, learning_rate=3e-5)),
    ],
    'transfer': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('ent1e4', dict(entropy_coef=1e-4)),
    ],
    'docking': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('rdv_rollout', dict(num_steps_per_env=250, num_mini_batches=1)),
    ],
    'rendezvous': [
        ('base', {}),
        ('lr1e3', dict(learning_rate=1e-3)),
        ('trunk128', dict(trunk_hidden=128, critic_trunk_hidden=128)),
    ],
}

# FlashSAC sweep updates = env steps per env (steps_per_iter == 1); sized so
# each arm sees >= ~4 episodes per env. gamma follows the task's horizon.
FS_BASE_GAMMA = {'quadx': 0.99, 'fixedwing': 0.999, 'rocket': 0.995,
                 'satellite': 0.999, 'transfer': 0.9999, 'docking': 0.999,
                 'rendezvous': 0.999}
FS_UPDATES = {'quadx': 800, 'fixedwing': 800, 'rocket': 2000,
              'satellite': 1500, 'transfer': 800, 'docking': 1000,
              'rendezvous': 1000}
FS_CANDIDATES = [
    ('base', {}),
    ('lr1e3', dict(learning_rate=1e-3)),
    ('b4096_u2', dict(batch_size=4096, updates_per_step=2)),
    ('nstep3', dict(n_step=3)),
]

# Production env classes / kwargs (mirror each train_*.py).
from pyflyt_genesis.quadx import QuadXHoverEnv            # noqa: E402
from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv  # noqa: E402
from pyflyt_genesis.rocket import RocketLandingEnv        # noqa: E402
from pyflyt_genesis.satellite import SatelliteAttitudeEnv  # noqa: E402
from pyflyt_genesis.transfer import LowThrustTransferEnv  # noqa: E402
from pyflyt_genesis.docking import SpaceDockingEnv, SpaceRendezvousEnv  # noqa: E402

ENVS = {
    'quadx': (QuadXHoverEnv, {}),
    'fixedwing': (FixedwingWaypointsEnv, {}),
    'rocket': (RocketLandingEnv, {}),
    'satellite': (SatelliteAttitudeEnv, {}),
    'transfer': (LowThrustTransferEnv, {}),
    'docking': (SpaceDockingEnv, {}),
    'rendezvous': (SpaceRendezvousEnv, {}),
}
# docking/rocket production entries zero-init the policy output head via
# module-level patches; importing the train module applies them identically.
import train_docking   # noqa: E402,F401  (applies FlashResMLPModel zero-head patch)
import train_rocket    # noqa: E402,F401  (applies MLP/FlashRes zero-head patches)
PPO_BASE['rocket']['init_std'] = train_rocket.INIT_STD

SMOKE = bool(__import__('os').environ.get('SWEEP_SMOKE'))


def _sr(report):
    """(whole-run SR, last-quarter SR) from a run_training report."""
    eps, ok = report['completed_episodes'], report['completed_successes']
    total = sum(ok) / max(sum(eps), 1.)
    n = len(eps)
    tail = slice(max(n * 3 // 4, 1), None)
    te, tok = eps[tail], ok[tail]
    tail_sr = sum(tok) / max(sum(te), 1.)
    return total, tail_sr


def run_one(task, env_cls, env_kwargs, algo, name, overrides, updates, out_root):
    out = out_root / f'{algo}_{name}'
    t0 = time.perf_counter()
    kw = dict(overrides)
    if algo == 'ppo':
        kw.update(PPO_BASE[task])
    else:
        kw.setdefault('gamma', FS_BASE_GAMMA[task])
    report = run_training(task, env_cls, num_envs=NUM_ENVS, updates=updates, seed=SEED,
                          backend=BACKEND, resume=False, clean=True, out_dir=out,
                          algo=algo, env_kwargs=dict(env_kwargs) or None,
                          bc_warmstart=BC_WARMSTART if algo == 'ppo' else None,
                          **kw)
    total, tail = _sr(report)
    summary = {'task': task, 'algo': algo, 'arm': name, 'overrides': {k: str(v) for k, v in overrides.items()},
               'updates': updates, 'wall_s': round(time.perf_counter() - t0, 1),
               'sr_total': round(total, 4), 'sr_tail25': round(tail, 4),
               'completed_episodes': int(sum(report['completed_episodes']))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[sweep] {task} {algo}_{name}: SR total {total:.3f} tail {tail:.3f} '
          f'({summary["wall_s"]}s)', flush=True)
    return summary


def main():
    task = sys.argv[1]
    only = set(sys.argv[2:]) or None
    env_cls, env_kwargs = ENVS[task]
    out_root = REPO / 'results' / 'sweep' / task
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ov in PPO_CANDIDATES[task]:
        tag = f'ppo_{name}'
        if only and tag not in only and name not in only:
            continue
        results.append(run_one(task, env_cls, env_kwargs, 'ppo', name, ov,
                               20 if SMOKE else 200, out_root))
    for name, ov in FS_CANDIDATES:
        tag = f'flash_sac_{name}'
        if only and tag not in only and name not in only:
            continue
        results.append(run_one(task, env_cls, env_kwargs, 'flash_sac', name, ov,
                               50 if SMOKE else FS_UPDATES[task], out_root))
    results.sort(key=lambda r: (r['algo'], -r['sr_tail25']))
    print(f'\n=== {task} sweep results (sorted by tail SR) ===')
    for r in results:
        print(f"{r['algo']:>10} {r['arm']:<14} tail {r['sr_tail25']:.3f}  "
              f"total {r['sr_total']:.3f}  eps {r['completed_episodes']}  "
              f"wall {r['wall_s']}s")
    (out_root / 'sweep_summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
