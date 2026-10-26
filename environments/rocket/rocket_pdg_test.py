"""Verification entry for the classical rocket PDG baseline (RocketPDGBaseline).

Self-contained: every run parameter is hard-coded below, no CLI arguments.
Run `python rocket_pdg_test.py` (defaults mirror the training convention:
1024 replicas, seed 7, gpu backend).  Edit SCENARIOS/STEPS to rescope.

Scenarios (hard-coded):
  A random    : the env's own upstream ICs, 1024 replicas (statistical arm)
  B corners   : xy +/-20 m corners x z {400,450} x tilt {0, +/-0.3} (64 repl)
  C nominal   : xy=0, z=425, zero tilt (8 replicas)
  D low-z     : xy=0, z=400, zero tilt (thrust-margin worst case, 8 replicas)
  E lateral   : xy=0, z=425, lateral velocity +/-3 m/s (8+8 replicas)

Reported per scenario: outcome breakdown (success / fatal collision /
out-of-bounds / timeout), touchdown speed/attitude/spin distribution,
settle behaviour among gentle touches, peak realized overload (T/(m g)),
fuel consumed and remaining, cumulative thrust impulse, engine-on /
saturation / coast fractions, ignition pulse count, and guidance
infeasibility statistics (the analytic law has no solver, so solver-failure
rate is N/A by construction; saturation/infeasible fractions are the honest
analogs).  Results are appended to results/rocket/pdg_test_report.json.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from environments.rocket import RocketLandingEnv
from algorithms.baselines.baselines import RocketPDGBaseline
from environments.common.dynamics import quat_from_euler

NUM_ENVS = 1024
STEPS = 1250                # > the 1200-step time limit: since the 2026-10-22
                            # env change a landed success does NOT truncate (the
                            # episode runs to the time limit), so a 700-step
                            # window never sees a reset on a settled rocket and
                            # would report 0 successes; 1250 lets every
                            # surviving episode reach its time-limit reset
SEED = 7
BACKEND = 'gpu'
T_MAX = RocketPDGBaseline.T_MAX
M_DRY = RocketPDGBaseline.M_DRY
FUEL_FULL = RocketPDGBaseline.FUEL_FULL
G = RocketPDGBaseline.G
DT = 1. / 40.

SCENARIOS = {
    'A_random': None,       # env's own reset distribution
    'B_corners': dict(xy=[(-20., -20.), (20., -20.), (-20., 20.), (20., 20.)],
                      z=[400., 450.], rpy=[(0., 0., 0.), (.3, 0., 0.), (-.3, 0., 0.),
                                           (0., .3, 0.), (0., -.3, 0.)]),
    'C_nominal': dict(xy=[(0., 0.)], z=[425.], rpy=[(0., 0., 0.)]),
    'D_low_z': dict(xy=[(0., 0.)], z=[400.], rpy=[(0., 0., 0.)]),
    'E_lateral_v': dict(xy=[(0., 0.)], z=[425.], rpy=[(0., 0., 0.)],
                        vxy=[(3., 0.), (-3., 0.), (0., 3.), (0., -3.)]),
}
REPEAT = 16                 # scenario replicas per IC combination (B..E)


def spawn(env, xy, z, rpy, vxy=(0., 0.)):
    """Deterministically place every replica at one IC set (no RNG use)."""
    n = env.num_envs
    dev = env.device
    p = torch.tensor((xy[0], xy[1], z), device=dev, dtype=torch.float64).expand(n, 3).contiguous()
    q = quat_from_euler(torch.tensor(rpy, device=dev, dtype=torch.float64).expand(n, 3).contiguous())
    v = torch.tensor((vxy[0], vxy[1], -100.), device=dev, dtype=torch.float64).expand(n, 3).contiguous()
    w = torch.zeros_like(v)
    env.plant.set_state(p, q, v, w)
    env._observe(torch.ones(n, device=dev, dtype=torch.bool))
    env._update_observation()


def run_scenario(name, ic, controller_cls=RocketPDGBaseline):
    torch.manual_seed(SEED)
    # Legacy telemetry uses the instantaneous v1 event. For v2 scoring use
    # scripts/rocket_benchmark_eval.py (full horizon, sustained landing).
    env = RocketLandingEnv(num_envs=NUM_ENVS, backend=BACKEND, seed=SEED, benchmark_version='v1')
    ctrl = controller_cls(env)
    dev = env.device
    n = env.num_envs

    # ---- per-env trackers (device-resident, synced once at the end) ----
    peak_load = torch.zeros(n, device=dev, dtype=torch.float64)
    impulse = torch.zeros(n, device=dev, dtype=torch.float64)
    toggles = torch.zeros(n, device=dev, dtype=torch.float64)
    prev_ign = torch.zeros(n, device=dev, dtype=torch.bool)

    # per-episode snapshots, taken inside the reset hook before state is wiped
    snaps = {'outcome': [], 'touch_v': [], 'touch_w': [], 'touch_tilt': [],
             'peak_load': [], 'fuel': [], 'impulse': [], 'toggles': [], 'touched': []}
    orig_reset_idx = env.reset_idx

    def reset_idx_hook(mask):
        snaps['touched'].append(env.touch_flag[mask])
        snaps['touch_v'].append(env.touch_speed[mask])
        snaps['touch_w'].append(env.touch_ang_vel[mask])
        snaps['touch_tilt'].append(env.touch_tilt[mask])
        snaps['outcome'].append(torch.where(env.out_of_bounds[mask], 2,
                                           torch.where(env.fatal_collision[mask], 1,
                                                       torch.where(env.complete[mask], 0, 3))))
        snaps['peak_load'].append(peak_load[mask])
        snaps['fuel'].append(env.plant.fuel[mask])
        snaps['impulse'].append(impulse[mask])
        snaps['toggles'].append(toggles[mask])
        peak_load.masked_fill_(mask, 0.)
        impulse.masked_fill_(mask, 0.)
        toggles.masked_fill_(mask, 0.)
        orig_reset_idx(mask)

    env.reset_idx = reset_idx_hook

    if ic is not None:
        combos = [(xy, z, rpy, vxy)
                  for xy in ic['xy'] for z in ic['z'] for rpy in ic['rpy']
                  for vxy in ic.get('vxy', [(0., 0.)])]
        act = torch.zeros(n, 7, device=dev)
        env.step(act)                     # clears the construction reset_buf
        P = torch.zeros(n, 3, device=dev, dtype=torch.float64)
        V = torch.zeros(n, 3, device=dev, dtype=torch.float64)
        E = torch.zeros(n, 3, device=dev, dtype=torch.float64)
        for i in range(n):
            xy, z, rpy, vxy = combos[(i // REPEAT) % len(combos)]
            P[i] = torch.tensor((xy[0], xy[1], z))
            E[i] = torch.tensor(rpy)
            V[i] = torch.tensor((vxy[0], vxy[1], -100.))
        env.plant.set_state(P, quat_from_euler(E), V, torch.zeros_like(V))
        env._observe(torch.ones(n, device=dev, dtype=torch.bool))
        env._update_observation()
        obs = env.get_observations()
    else:
        obs = env.get_observations()

    diag0 = {k: v.clone() for k, v in ctrl.diag.items()}
    for _ in range(STEPS):
        act = ctrl.actions(obs)
        obs, _, term, trunc, _ = env.step(act)
        reset_buf = term | trunc
        ctrl.reset_idx(reset_buf)
        o = obs['policy'].to(torch.float64)
        thr, fuel = o[:, 26], o[:, 25]      # aux: 4 finlet act + ignited=24, fuel=25, throttle=26
        m = M_DRY + fuel * FUEL_FULL
        load = thr * T_MAX / (m * G)
        active = ~(env.terminated | env.truncated)
        peak_load.copy_(torch.where(active, torch.maximum(peak_load, load), peak_load))
        impulse += torch.where(active, thr * T_MAX * DT, torch.zeros_like(impulse))
        ign = o[:, 24] > .5
        toggles += (ign & ~prev_ign).to(torch.float64)
        prev_ign.copy_(ign)

    diag = {k: float((v - diag0[k]).sum()) for k, v in ctrl.diag.items()}
    out = {k: torch.cat(v) for k, v in snaps.items()}
    env.reset_idx = orig_reset_idx

    # ---- reduce on host once ----
    def r(x):
        return x.detach().cpu().numpy()

    outcome = r(out['outcome'])
    n_ep = len(outcome)
    touched = r(out['touched']).astype(bool)
    tv = r(out['touch_v'])
    tw = r(out['touch_w'])
    tt = r(out['touch_tilt'])

    def q(xs, ps):
        xs = sorted(xs)
        return {f'p{int(p * 100)}': (xs[min(int(p * len(xs)), len(xs) - 1)] if xs else None) for p in ps}

    gentle = touched & (tv <= 1.0)
    report = {
        'scenario': name,
        'episodes': n_ep,
        'success_rate': float((outcome == 0).mean()) if n_ep else None,
        'fatal_collision_rate': float((outcome == 1).mean()) if n_ep else None,
        'out_of_bounds_rate': float((outcome == 2).mean()) if n_ep else None,
        'timeout_rate': float((outcome == 3).mean()) if n_ep else None,
        'touch_rate': float(touched.mean()) if n_ep else None,
        'touch_speed': {**q(tv[touched].tolist(), (.1, .5, .9)),
                        'mean': float(tv[touched].mean()) if touched.any() else None,
                        'max': float(tv[touched].max()) if touched.any() else None},
        'touch_tilt': {'mean': float(tt[touched].mean()) if touched.any() else None,
                       'max': float(tt[touched].max()) if touched.any() else None},
        'touch_spin': {'mean': float(tw[touched].mean()) if touched.any() else None,
                       'max': float(tw[touched].max()) if touched.any() else None},
        'gentle_touches': int(gentle.sum()),
        'success_among_gentle': (float((outcome[gentle] == 0).mean()) if gentle.any() else None),
        'peak_overload_g': {'mean': float(out['peak_load'].mean()),
                            'max': float(out['peak_load'].max())},
        'fuel_remaining_frac': {'mean': float(out['fuel'].mean()),
                                'min': float(out['fuel'].min())},
        'fuel_used_kg': float((0.05 - out['fuel']).mean() * FUEL_FULL),
        'impulse_Ns': {'mean': float(out['impulse'].mean()),
                       'max': float(out['impulse'].max())},
        'ignition_pulses_per_episode': float(out['toggles'].mean()),
        'fuel_out_episodes': int((out['fuel'] <= 1e-6).sum()),
        'guidance': {'solver_failure_rate': 'N/A (analytic, no optimizer)',
                     'saturation_frac': diag['sat_max'] / max(diag['steps'], 1.),
                     'coast_frac': diag['coast'] / max(diag['steps'], 1.),
                     'infeasible_demand_frac': diag['infeasible'] / max(diag['steps'], 1.),
                     'tgo_floor_frac': diag['floor'] / max(diag['steps'], 1.),
                     'ignition_toggles_total': int(diag['ignitions'])},
    }
    env.close()
    return report


def main():
    reports = []
    for name, ic in SCENARIOS.items():
        rep = run_scenario(name, ic)
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)
    out = Path(__file__).resolve().parent.parent / 'results' / 'rocket' / 'pdg_test_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f'\nreport written to {out}')


if __name__ == '__main__':
    main()
