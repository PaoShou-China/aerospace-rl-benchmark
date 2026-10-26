"""Rocket v4 reward-coefficient evolution harness (tuning only).

Usage: python evolve_reward.py [arm ...]   (argv exists ONLY so parallel
lanes can each select one arm; production entry train_rocket.py stays
hard-coded with no CLI).

Each arm trains PPO with the production config of train_rocket.py
(1024 envs, seed 7, BC warmstart, gamma 0.995, ...) and only overrides
reward coefficients through the env's `reward` ctor kwarg. One generation
is a set of arms; the winner and its mutations are appended to ARMS as the
next generation. Per-arm output goes to results/rocket_evo/<arm>/ with a
summary.json; selection is by tail-25% success rate, tie-broken by the
whole-run pad touch rate and mean return.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from algorithms.train_utils import run_training  # noqa: E402
from environments.rocket import RocketLandingEnv  # noqa: E402


def _load_entry(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# applies the MLP/FlashRes zero-head patches and provides INIT_STD / BC config
train_rocket = _load_entry('train_rocket', 'environments/rocket/train_rocket.py')

NUM_ENVS = 1024
SEED = 7
BACKEND = 'gpu'
UPDATES = 600               # evolution budget per arm (production: 1000)
if __import__('os').environ.get('EVO_SMOKE'):  # wiring smoke test
    NUM_ENVS, UPDATES, BACKEND = 32, 3, 'cpu'
WIND = dict(steady_wind=None, dryden=.3, severity='moderate')

# --- generations -----------------------------------------------------------
# g1 (2026-10-24): descent_coef sweep around the analysis-driven 0.1, all with
# BC warmstart. RESULT: all 0% success (touch rate 0.0004/0.066/0.107 for
# 0.05/0.1/0.2) -- the BC init is broken closed-loop on rocket (the PDG
# terminal pulse servo is bang-bang and MSE-distillation averages it to death;
# measured BC closed-loop SR 0% with 0 pad touches), so g2 drops BC (arms
# suffixed _nb) and trains from the zero-init do-nothing policy, upstream's
# own setting.
ARMS = {
    'g1_base':   {},
    'g1_desc05': {'descent': 0.05},
    'g1_desc20': {'descent': 0.2},
    # g2: no BC, 600 updates; descent direction (g1 touch-rate signal) vs
    # stronger time pressure
    'g2_base_nb':   {},
    'g2_desc20_nb': {'descent': 0.2},
    'g2_hover06_nb': {'hover': 0.6},
}
NO_BC_SUFFIX = '_nb'


def _sr(report):
    eps, ok = report['completed_episodes'], report['completed_successes']
    n = len(eps)
    tail = slice(max(n * 3 // 4, 1), None)
    return (sum(ok) / max(sum(eps), 1.),
            sum(ok[tail]) / max(sum(eps[tail]), 1.))


def run_arm(name):
    out = REPO / 'results' / 'rocket_evo' / name
    env_kw = dict(WIND)
    env_kw['reward'] = dict(ARMS[name])
    t0 = time.perf_counter()
    report = run_training('rocket', RocketLandingEnv, num_envs=NUM_ENVS, updates=UPDATES,
                          seed=SEED, backend=BACKEND, resume=False, clean=True, out_dir=out,
                          algo='ppo', init_std=train_rocket.INIT_STD, num_steps_per_env=64,
                          lr_schedule='fixed', learning_rate=3e-4, gamma=0.995, lam=0.97,
                          entropy_coef=0.007, num_learning_epochs=10, clip_param=0.2,
                          value_loss_coef=1.0, desired_kl=0.01,
                          env_kwargs=env_kw,
                          bc_warmstart=None if name.endswith(NO_BC_SUFFIX)
                          else train_rocket.BC_WARMSTART)
    total, tail = _sr(report)
    cm = report.get('completed_metrics', {})
    summary = {'arm': name, 'reward_override': ARMS[name], 'updates': UPDATES,
               'wall_s': round(time.perf_counter() - t0, 1),
               'sr_total': round(total, 4), 'sr_tail25': round(tail, 4),
               'pad_touch_rate': round(cm.get('pad_touch_rate', 0.), 4),
               'touch_speed': round(cm.get('touch_speed', 0.), 4),
               # mean episode return proxy: sum of the per-component rew_* means
               'mean_return': round(sum(v for k, v in cm.items() if k.startswith('rew_')), 2),
               'completed_episodes': int(sum(report['completed_episodes']))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"[evo] {name}: SR total {total:.3f} tail {tail:.3f} "
          f"touch {summary['pad_touch_rate']:.3f} ({summary['wall_s']}s)", flush=True)
    return summary


if __name__ == '__main__':
    names = sys.argv[1:] or list(ARMS)
    results = [run_arm(n) for n in names]
    results.sort(key=lambda r: (-r['sr_tail25'], -r['pad_touch_rate']))
    print('\n=== rocket reward evolution (sorted by tail SR) ===')
    for r in results:
        print(f"{r['arm']:<14} tail {r['sr_tail25']:.3f}  total {r['sr_total']:.3f}  "
              f"touch {r['pad_touch_rate']:.3f}  ret {r['mean_return']:>10.1f}  "
              f"wall {r['wall_s']}s")
