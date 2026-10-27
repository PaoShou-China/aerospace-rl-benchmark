"""Rocket PD-landing baseline: gain-selection scenario for RocketPDLandingBaseline.

The PD law's constants are chosen HERE, on tuning seeds, and only the frozen
winner is re-evaluated on the official harness (scripts/baseline_eval.py,
256 envs, seed 7 -> results/rocket/baseline_pd_landing.json).  One candidate
per run: a 128-replica batch, one episode per replica (1900 steps), calm air,
spawn ICs = the env's own begin_reset distribution (xy +/-20 m, z 400-450,
vz -100 m/s, euler +/-0.3 rad) -- no per-initial-state hand adjustment.

Each run captures, inside a reset hook (the env self-resets inside step(),
so episode-end states must be read BEFORE reset_idx clears the buffers):
  * outcome split success / fatal / oob / timeout,
  * touch quality on the LAST recorded touch (speed, ang vel, tilt),
  * end state (|v|, height, pad offset, ground contact),
  * fuel used, ignition pulses, post-touch relights, pad-contact flickers.
Ranking: success rate first, then fewer >1 m/s touches, fewer >0.35 rad/s
spins, lower median touch speed.

The grid is one-factor-at-a-time around the shipped constants (the factors
interact weakly: terminal band / pulse-leg lateral / mid-descent lateral /
attitude are separable loops), followed by one combined-best confirmation
run.  No CLI: edit the grid and run `python rocket_pd_tune.py`.
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from environments.rocket import RocketLandingEnv
from algorithms.baselines.baselines import RocketPDLandingBaseline, _quat_to_R

NUM_ENVS = 128
STEPS = 1900                          # 30 s episode + margin, one episode/replica
SEED = 101                            # tuning seed, DISTINCT from eval seed 7
OUT = _ROOT / 'results' / 'rocket_pd_tune'

# shipped constants (the base every candidate perturbs)
BASE = dict(KP_XY=0.25, KD_XY=1.00, LAT_TILT=0.14, A_LAT_MAX=5.0,
            KP_ATT=12000.0, KD_ATT=8000.0,
            V_TERM_OFF=0.30, V_TERM_ON=0.40,
            KV_LAT=1.5, A_LAT_PULSE=1.5)

# one-factor-at-a-time grid around the base (name -> single-knob override)
GRID = [
    ('base', {}),
    # terminal speed band (settle-phase touch speed vs relight catch)
    ('term_tight', dict(V_TERM_OFF=0.20, V_TERM_ON=0.35)),
    ('term_wide', dict(V_TERM_OFF=0.30, V_TERM_ON=0.50)),
    # pulse-leg lateral velocity damping
    ('kvlat_lo', dict(KV_LAT=1.0)),
    ('kvlat_hi', dict(KV_LAT=2.5)),
    ('alat_lo', dict(A_LAT_PULSE=1.0)),
    ('alat_hi', dict(A_LAT_PULSE=2.5)),
    # mid-descent position/velocity PD and tilt budget
    ('kpxy_lo', dict(KP_XY=0.15)),
    ('kpxy_hi', dict(KP_XY=0.40)),
    ('kdxy_lo', dict(KD_XY=0.70)),
    ('kdxy_hi', dict(KD_XY=1.40)),
    ('tilt_lo', dict(LAT_TILT=0.10)),
    # attitude loop
    ('kpatt_lo', dict(KP_ATT=9000.0)),
    ('kpatt_hi', dict(KP_ATT=16000.0)),
]


def run_one(name, overrides):
    """One candidate: patched constants, 128 replicas, reset-hook metrics."""
    consts = dict(BASE)
    consts.update(overrides)
    for k, v in consts.items():
        setattr(RocketPDLandingBaseline, k, v)
    torch.manual_seed(SEED)
    env = RocketLandingEnv(num_envs=NUM_ENVS, backend='gpu', seed=SEED)
    ctrl = RocketPDLandingBaseline(env)
    dev = env.device

    # episode-end metrics captured inside the reset hook (the env self-resets
    # inside step(), so the terminated/truncated/touch buffers must be read
    # BEFORE its reset_idx clears them): mask-indexed snapshot per finished
    # replica, then the running counters for that replica are zeroed
    rec = {'outcome': [], 'ts': [], 'tw': [], 'tt': [], 'xy': [], 'v': [],
           'touched': [], 'fuel': [], 'ign': [], 'post': []}
    ign_run = torch.zeros(NUM_ENVS, device=dev, dtype=torch.float64)
    post_run = torch.zeros(NUM_ENVS, device=dev, dtype=torch.float64)
    prev_ign = torch.zeros(NUM_ENVS, device=dev, dtype=torch.bool)
    after = torch.zeros(NUM_ENVS, device=dev, dtype=torch.bool)
    orig = env.reset_idx

    def hook(mask):
        if bool(mask.any()):
            p, q, v, w = env.plant.state()
            rec['outcome'] += torch.where(env.out_of_bounds, 2,
                              torch.where(env.fatal_collision, 1,
                              torch.where(env.complete, 0, 3)))[mask].cpu().tolist()
            rec['ts'] += env.touch_speed[mask].cpu().tolist()
            rec['tw'] += env.touch_ang_vel[mask].cpu().tolist()
            rec['tt'] += env.touch_tilt[mask].cpu().tolist()
            rec['xy'] += p[:, :2].norm(dim=-1)[mask].cpu().tolist()
            rec['v'] += v.norm(dim=-1)[mask].cpu().tolist()
            rec['touched'] += env.touch_flag[mask].cpu().tolist()
            rec['fuel'] += env.plant.fuel[mask].cpu().tolist()
            rec['ign'] += ign_run[mask].cpu().tolist()
            rec['post'] += post_run[mask].cpu().tolist()
            ign_run[mask] = 0.; post_run[mask] = 0.; after[mask] = False
        orig(mask)

    env.reset_idx = hook
    obs = env.get_observations()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(STEPS):
            obs, _, term, trunc, _ = env.step(ctrl.actions(obs))
            ctrl.reset_idx(term | trunc)
            o = obs['policy'].to(torch.float64)
            active = ~(env.terminated | env.truncated)
            ign = o[:, 24] > .5
            ign_run += (ign & ~prev_ign & active).to(torch.float64)
            prev_ign.copy_(ign)
            after |= env.touch_flag & ~after
            post_run += (ign & after & active).to(torch.float64)
    wall = time.perf_counter() - t0
    oc = np.asarray(rec['outcome']); ts = np.asarray(rec['ts'])
    tw = np.asarray(rec['tw']); tt = np.asarray(rec['tt'])
    t_ = np.asarray(rec['touched'], dtype=bool)
    m = {
        'name': name, 'consts': consts, 'episodes': int(len(oc)),
        'success': float((oc == 0).mean()) if len(oc) else 0.,
        'fatal': float((oc == 1).mean()) if len(oc) else 0.,
        'oob': float((oc == 2).mean()) if len(oc) else 0.,
        'timeout': float((oc == 3).mean()) if len(oc) else 0.,
        'touch_gt1': float((ts[t_] > 1).mean()) if t_.any() else 0.,
        'spin_gt035': float((tw[t_] > .35).mean()) if t_.any() else 0.,
        'touch_speed_med': float(np.median(ts[t_])) if t_.any() else 9.9,
        'touch_speed_max': float(ts[t_].max()) if t_.any() else 9.9,
        'spin_med': float(np.median(tw[t_])) if t_.any() else 9.9,
        'tilt_med': float(np.median(tt[t_])) if t_.any() else 9.9,
        'end_v_med': float(np.median(rec['v'])) if rec['v'] else 0.,
        'end_xy_med': float(np.median(rec['xy'])) if rec['xy'] else 0.,
        'touched_frac': float(t_.mean()) if len(t_) else 0.,
        'fuel_used': float((0.05 - np.asarray(rec['fuel'])).mean() * 410.9)
                     if rec['fuel'] else 0.,
        'ign_pulses': float(np.mean(rec['ign'])) if rec['ign'] else 0.,
        'post_touch_ign': float(np.mean(np.asarray(rec['post'])[t_]))
                          if t_.any() else 0.,
        'wall_s': round(wall, 1),
    }
    env.close()
    return m


def rank_key(m):
    """success first; then touch quality tiebreakers (lower is better)."""
    return (-m['success'], m['touch_gt1'], m['spin_gt035'], m['touch_speed_med'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ov in GRID:
        m = run_one(name, ov)
        results.append(m)
        print(f"[tune] {name:<10} success {m['success']:.3f} fatal {m['fatal']:.3f} "
              f"touch>1 {m['touch_gt1']:.2f} spin>.35 {m['spin_gt035']:.2f} "
              f"ts_med {m['touch_speed_med']:.2f} post_ign {m['post_touch_ign']:.1f} "
              f"({m['wall_s']}s)", flush=True)
        (OUT / 'tune_report.json').write_text(json.dumps(results, indent=2))
    results.sort(key=rank_key)
    best = results[0]
    # combined-best confirmation: winner's single knobs stacked on the base
    combo = {}
    for m in results:
        for k, v in m['consts'].items():
            if BASE.get(k) != v:
                combo[k] = v
    print(f"\nbest single: {best['name']} success {best['success']:.3f}")
    if best['name'] != 'base' and combo:
        m = run_one('combined_best', combo)
        results.append(m)
        (OUT / 'tune_report.json').write_text(json.dumps(results, indent=2))
        print(f"[tune] combined  success {m['success']:.3f} fatal {m['fatal']:.3f} "
              f"touch>1 {m['touch_gt1']:.2f} spin>.35 {m['spin_gt035']:.2f} "
              f"({m['wall_s']}s)", flush=True)
    results.sort(key=rank_key)
    print('\n=== rocket PD tune ranking ===')
    for m in results:
        print(f"{m['name']:<14} success {m['success']:.3f}  touch>1 {m['touch_gt1']:.2f}  "
              f"spin>.35 {m['spin_gt035']:.2f}  ts_med {m['touch_speed_med']:.2f}")
    (OUT / 'tune_report.json').write_text(json.dumps(results, indent=2))
    print(f"\nreport: {OUT / 'tune_report.json'}")


if __name__ == '__main__':
    main()
