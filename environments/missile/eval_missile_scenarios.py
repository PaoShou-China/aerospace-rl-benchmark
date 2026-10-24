"""Scenario evaluation for the missile-intercept APN baseline.

Classical-guidance counterpart for the missile RL arms (MissileAPNBaseline
in algorithms/baselines/baselines.py): augmented PN (N=4, half-N gravity
compensation) mapped to the body-rate command interface, driven at the
env's single fixed medium operating point.

Honest scope note: the originally planned grid (approach-geometry aspect
angle {0, 45, 90, 180 deg} x maneuver {straight, weave, weave at max
severity, barrel} x initial range {2000, 5000, 8000 m}) cannot be driven on
this env -- the spawn geometry is generated internally in reset_idx
(initial range, target-heading cone around the missile bearing, and
maneuver severity are fixed env constants) and exposes no injection points
for aspect angle or range; 5000 and 8000 m lie outside the env's spawn
support altogether.  The report therefore breaks the single operating
point down by the env's own maneuver type (t_type 0 straight / 1 weave /
2 barrel, drawn uniformly at reset).  The env, reward and success criteria
are never modified.

Measurement protocol: completed_metrics['miss_distance'] slots are stale at
termination (the env auto-resets internally inside step()), so a spy wraps
env.reset_idx and records (maneuver type, kill flag, true min_distance)
BEFORE the internal reset clears them -- the same ground-truth approach as
test_missile_apn.py T3.  The baseline filter state is cleared on
env.reset_buf (repo contract) after every step.

256 replicas, capped at ~256 completed episodes (one per replica).
Reported per maneuver type: success rate (proximity-fuse kill, miss < 15 m),
median miss distance over kills and over all episodes.  Results are written
to results/missile/scenarios.json.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from algorithms.baselines.baselines import MissileAPNBaseline
from environments.missile.env import MissileInterceptEnv

# ---- defaults (hard-coded; edit here, no CLI) ----
BACKEND = 'gpu'
SEED = 7
NUM_ENVS = 256
MAX_EPISODES = 256            # episodes collected (~one per replica)
MAX_STEPS = 4000              # safety cap on env steps (episodes <= 900)
TYPE_NAMES = ('straight', 'weave', 'barrel')   # env.t_type 0 / 1 / 2
OUT = Path(__file__).resolve().parents[2] / 'results' / 'missile' / 'scenarios.json'


def run_cell():
    """Drive the APN baseline at the fixed operating point; returns the summary dict."""
    torch.manual_seed(SEED)
    env = MissileInterceptEnv(num_envs=NUM_ENVS, backend=BACKEND, seed=SEED)
    ctrl = MissileAPNBaseline(env)
    obs = env.reset()
    ctrl.reset_idx(env.reset_buf)
    types, kills, mins = [], [], []
    orig_reset = env.reset_idx

    def spy(mask):
        types.append(env.t_type[mask].clone())
        kills.append(env.complete[mask].clone())
        mins.append(env.min_distance[mask].clone())
        orig_reset(mask)

    env.reset_idx = spy
    try:
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                act = ctrl.actions(obs)
            obs, _, term, trunc, _ = env.step(act)
            ctrl.reset_idx(env.reset_buf)
            if sum(int(r.numel()) for r in types) >= MAX_EPISODES:
                break
    finally:
        env.reset_idx = orig_reset
    env.close()
    types = torch.cat(types).cpu()
    kills = torch.cat(kills).cpu()
    mins = torch.cat(mins).cpu()

    def med(sel=None, only_kills=False):
        m = mins if sel is None else mins[sel]
        k = kills if sel is None else kills[sel]
        if only_kills:
            m = m[k]
        return round(float(m.to(torch.float64).median()), 1) if m.numel() else None

    cell = {'episodes': int(types.numel()),
            'success_rate': round(float(kills.to(torch.float64).mean()), 4),
            'median_miss_kills_m': med(only_kills=True),
            'median_miss_all_m': med(),
            'by_type': {}}
    for t, name in enumerate(TYPE_NAMES):
        sel = types == t
        n = int(sel.sum())
        cell['by_type'][name] = {
            'episodes': n,
            'success_rate': round(float(kills[sel].to(torch.float64).mean()), 4) if n else None,
            'median_miss_kills_m': med(sel, only_kills=True),
            'median_miss_all_m': med(sel),
        }
    return cell


def main():
    cell = run_cell()
    report = {'task': 'missile', 'baseline': MissileAPNBaseline.name,
              'protocol': 'single fixed medium operating point; the planned '
                          'aspect-angle x range grid is not injectable (spawn '
                          'geometry internal to reset_idx, 5-8 km outside the '
                          'spawn support) -- see docstring',
              'num_envs': NUM_ENVS, 'max_episodes': MAX_EPISODES,
              'seed': SEED, 'cell': cell}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
