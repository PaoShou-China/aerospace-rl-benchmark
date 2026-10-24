"""Evaluation sweep for the CWH two-impulse rendezvous baseline.

Classical-guidance counterpart for the rendezvous RL arms
(RendezvousTwoImpulseBaseline in algorithms/baselines/baselines.py): receding-horizon
CWH two-impulse terminal guidance -- both impulses solved every step from
the exact symplectic-Euler state transition matrix (pulse 1 targets the hold
point at the scheduled arrival time, pulse 2 nulls the arrival velocity) and
realized as finite thrust through the ZEM/ZEV min-energy map with per-axis
saturation.

Part 1  benchmark: env default spawns (|p0| in [1000, 1500] m since
        2026-10-24, widened from [800, 1200] m; |v0| <= 1 m/s),
        1024 replicas x 2 episodes = ~2048 samples.
Part 2  grid: injected initial states, |p0| in {150, 300, 600, 800, 1200,
        1600, 1900} m x |v0| in {0, 0.25, 0.5, 1.0, 2.0} m/s (2.0 is beyond
        the task's 1 m/s spec, probing the feasibility envelope), 256
        replicas each, first episode only.  A stress point (1900 m, 12 m/s)
        exercises the infeasibility path (degraded to velocity nulling).
        The grid only injects initial states on the test side; the env,
        reward and success criteria are never modified.

Reported per condition: success rate, keep-out / range / fuel / timeout
breakdown, terminal position & velocity error, total Delta-v, fuel, thrust
saturation rate, infeasible count, episode length.  Results are written to
results/rendezvous/baseline_sweep.json.
"""

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from algorithms.baselines.baselines import build_baseline
from environments.satellite.rendezvous_docking import SpaceRendezvousEnv

# ---- defaults (hard-coded; edit here, no CLI) ----
BACKEND = 'gpu'
SEED = 7
BENCH_NUM_ENVS = 1024
BENCH_STEPS = 2050            # ~2 episodes per replica
GRID_NUM_ENVS = 256
GRID_STEPS = 1005             # first episode ends by the 1000-step timeout
GRID_DISTANCES = (150., 300., 600., 800., 1200., 1600., 1900.)
GRID_SPEEDS = (0., .25, .5, 1., 2.)            # 2.0 beyond the 1 m/s spec
STRESS = (1900., 12.)                          # exercises the infeasibility path
OUT = Path(__file__).resolve().parents[2] / 'results' / 'rendezvous' / 'baseline_sweep.json'


def _unit_dirs(n, gen, device):
    u1 = torch.rand(n, device=device, generator=gen)
    u2 = torch.rand(n, device=device, generator=gen)
    z = 1. - 2. * u1
    r = torch.sqrt((1. - z * z).clamp_min(0.))
    th = 2. * math.pi * u2
    return torch.stack((r * torch.cos(th), r * torch.sin(th), z), -1)


def run_condition(name, num_envs, steps, d0=None, v0=None, first_only=False, seed=SEED):
    torch.manual_seed(seed)
    env = SpaceRendezvousEnv(num_envs=num_envs, backend=BACKEND, seed=seed)
    ctl = build_baseline('rendezvous', env)
    obs = env.reset()
    if d0 is not None:
        gen = torch.Generator(device=env.device).manual_seed(seed)
        p0 = d0 * _unit_dirs(num_envs, gen, env.device)
        v0v = v0 * _unit_dirs(num_envs, gen, env.device)
        env.p.copy_(p0)
        env.v.copy_(v0v)
        env.d0.copy_(p0.norm(dim=-1))
        mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        env._observe(mask)
        env._update_observation()
        obs = env.get_observations()

    dev = env.device
    eps = []
    tracked = torch.ones(num_envs, dtype=torch.bool, device=dev)   # episode being measured
    dv_acc = torch.zeros(num_envs, dtype=torch.float64, device=dev)
    fuel_acc = torch.zeros(num_envs, dtype=torch.float64, device=dev)
    sat_acc = torch.zeros(num_envs, dtype=torch.float64, device=dev)
    len_acc = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for _ in range(steps):
        a = ctl.actions(obs)
        thrust = a.double().clamp(-1., 1.) * env.f_max
        tnorm = thrust.norm(dim=-1)
        dv_acc += tnorm * env.dt / env.mass
        fuel_acc += tnorm * env.dt / (env.isp * env.g0)
        sat_acc += (a.abs() >= 1. - 1e-6).any(-1).double()
        len_acc += 1
        obs, _, term, trunc, extras = env.step(a)
        reset_buf = term | trunc
        done = reset_buf & tracked if first_only else reset_buf
        if bool(done.any()):
            term = extras['terminal_observations']['policy'].double()[done]
            d = term[:, :3].norm(dim=-1)
            sp = term[:, 3:].norm(dim=-1)
            succ = (d < env.pos_goal) & (sp < env.vel_goal)
            ko = d < env.keep_out
            rngf = d > env.range_fail
            fuelf = ~(succ | ko | rngf)
            rec = dict(
                succ=succ.tolist(), ko=ko.tolist(), rng=rngf.tolist(), fuel_fail=fuelf.tolist(),
                d=d.tolist(), v=sp.tolist(),
                dv=dv_acc[done].tolist(), fuel=fuel_acc[done].tolist(),
                sat=(sat_acc[done] / len_acc[done].double()).tolist(),
                steps=len_acc[done].tolist(),
                infeas=ctl.infeasible[done].tolist(),
            )
            n = len(rec['d'])
            for j in range(n):
                eps.append({k: v[j] for k, v in rec.items()})
        ctl.reset_idx(reset_buf)
        tracked &= ~reset_buf
        dv_acc = torch.where(done, torch.zeros_like(dv_acc), dv_acc)
        fuel_acc = torch.where(done, torch.zeros_like(fuel_acc), fuel_acc)
        sat_acc = torch.where(done, torch.zeros_like(sat_acc), sat_acc)
        len_acc = torch.where(done, torch.zeros_like(len_acc), len_acc)
    env.close()

    n = len(eps)
    s = sum(e['succ'] for e in eps)
    inf = [e for e in eps if e['infeas']]
    suc = [e for e in eps if e['succ']]
    agg = dict(
        condition=name, episodes=n,
        success=s, success_rate=s / max(n, 1),
        keepout=sum(e['ko'] for e in eps), range_fail=sum(e['rng'] for e in eps),
        fuel_fail=sum(e['fuel_fail'] for e in eps), timeout=n - s - sum(e['ko'] for e in eps)
        - sum(e['rng'] for e in eps) - sum(e['fuel_fail'] for e in eps),
        infeasible=len(inf), infeasible_success=sum(e['succ'] for e in inf),
        d_end_mean=sum(e['d'] for e in eps) / max(n, 1),
        d_end_max=max((e['d'] for e in eps), default=0.),
        v_end_mean=sum(e['v'] for e in eps) / max(n, 1),
        v_end_max=max((e['v'] for e in eps), default=0.),
        dv_mean=sum(e['dv'] for e in eps) / max(n, 1),
        dv_max=max((e['dv'] for e in eps), default=0.),
        fuel_mean=sum(e['fuel'] for e in eps) / max(n, 1),
        sat_rate_mean=sum(e['sat'] for e in eps) / max(n, 1),
        steps_mean=sum(e['steps'] for e in eps) / max(n, 1),
        d_end_success_mean=sum(e['d'] for e in suc) / max(len(suc), 1),
        v_end_success_mean=sum(e['v'] for e in suc) / max(len(suc), 1),
    )
    return agg


def main():
    results = []
    print(f'benchmark spawns (|p0| 1000-1500 m, |v0| <= 1 m/s), {BENCH_NUM_ENVS} envs x 2 eps')
    agg = run_condition('benchmark', BENCH_NUM_ENVS, BENCH_STEPS, first_only=False)
    results.append(agg)
    _print(agg)

    print(f'\ninitial-state grid ({GRID_NUM_ENVS} eps each, first episode only), '
          f'v0=2.0 is beyond the 1 m/s task spec')
    k = 0
    for d0 in GRID_DISTANCES:
        for v0 in GRID_SPEEDS:
            k += 1
            agg = run_condition(f'd{d0:g}_v{v0:g}', GRID_NUM_ENVS, GRID_STEPS,
                                d0=d0, v0=v0, first_only=True, seed=SEED + k)
            results.append(agg)
            _print(agg)

    print(f'\nstress point (beyond envelope, exercises infeasibility):')
    agg = run_condition(f'd{STRESS[0]:g}_v{STRESS[1]:g}', 64, GRID_STEPS,
                        d0=STRESS[0], v0=STRESS[1], first_only=True, seed=SEED + 1000)
    results.append(agg)
    _print(agg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nsaved -> {OUT}')


def _print(a):
    print(f"{a['condition']:>16s} | eps {a['episodes']:4d} succ {100 * a['success_rate']:5.1f}% | "
          f"ko {a['keepout']:3d} rng {a['range_fail']:3d} fuel {a['fuel_fail']:3d} t/o {a['timeout']:3d} | "
          f"infeas {a['infeasible']:3d} (succ {a['infeasible_success']:2d}) | "
          f"d_end {a['d_end_mean']:7.2f}/{a['d_end_max']:8.2f} "
          f"v_end {a['v_end_mean']:.3f}/{a['v_end_max']:.3f} | "
          f"dv {a['dv_mean']:5.2f}/{a['dv_max']:5.2f} fuel {a['fuel_mean']:5.2f} "
          f"sat {100 * a['sat_rate_mean']:5.1f}% steps {a['steps_mean']:5.0f}")


if __name__ == '__main__':
    main()
