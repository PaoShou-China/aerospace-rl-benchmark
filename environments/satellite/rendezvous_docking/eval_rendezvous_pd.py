"""Verification, tuning and formal comparison harness for the CWH-compensated
PD rendezvous baseline (RendezvousCWHPDBaseline in algorithms/baselines/baselines.py).

Part 0  small-signal verification: injected small offsets (10-100 m) where the
        per-axis thrust demand stays below the 0.02 m/s^2 authority.  Checks
        (a) zero saturation, (b) the env trajectory matches a reference
        double-integrator PD rollout (v+ = v + dt a_des, p+ = p + dt v+,
        consuming the same float32 observation each step) -- the coupling
        compensation must cancel the CWH plant terms; the residual deviation
        is only the float32 observation/action quantization (bound 1e-6 m) --
        and (c) 100% capture.
Part 1  dev-scenario gain selection: (wn, zeta) grid on an INDEPENDENT
        scenario set (natural spawn at seed 101 + injected IC grid at seeds
        201+, all disjoint from the formal seed-7 evaluation).  Ranking:
        success rate, then fuel, then episode time.  The winner is frozen
        into RendezvousCWHPDBaseline.WN/ZETA before Part 2 runs.
Part 2  formal comparison (seed 7, 1024 envs, one completed episode per
        replica, identical spawn sequence for every method):
            - cwh_pd          the frozen compensated PD (this work)
            - cwh_two_impulse the existing ZEM/ZEV (PN) guidance baseline
            - ppo             results/rendezvous/checkpoint.pt (deterministic)
            - flash_sac       results/rendezvous_flashsac/checkpoint.pt
        Metrics per episode: success, task time (steps), fuel, Delta-v,
        keep-out / range / fuel / timeout breakdown, closest approach
        (min |p| over the episode) and thrust saturation ratio (fraction of
        steps with any axis at the limit; also the per-axis fraction).

Self-contained, no CLI; edit the RUN_* switches below.  Results are written
to results/rendezvous/pd_report.json.
"""
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from algorithms.baselines.baselines import build_baseline
from environments.satellite.rendezvous_docking import SpaceRendezvousEnv

# ---- switches & constants (hard-coded, no CLI) ----
RUN_VERIFY = True
RUN_TUNE = True
RUN_FORMAL = False

BACKEND = 'gpu'
SEED = 7                       # formal seed (protocol); tuning uses 101+/201+
NUM_ENVS = 1024
OUT = _ROOT / 'results' / 'rendezvous' / 'pd_report.json'
PPO_DIR = _ROOT / 'results' / 'rendezvous'
SAC_DIR = _ROOT / 'results' / 'rendezvous_flashsac'

# Part 0: small-signal cases (distance [m], speed [m/s]); demands stay < 0.02 m/s^2
VERIFY_NUM_ENVS = 256
VERIFY_STEPS = 800             # 100 m at wn=0.01 settles into the goal ~500 s
VERIFY_DISTANCES = (10., 30., 100.)
VERIFY_SPEEDS = (0., .1)
VERIFY_TOL_M = 1e-5            # float32 obs/action quantization bound (p~100 m -> q eps ~1e-5 m)

# Part 1: dev grid (independent seeds)
TUNE_WN = (.004, .006, .008, .010, .013, .016)
TUNE_ZETA = (.8, 1.0, 1.2)
TUNE_NATURAL_ENVS = 256        # natural spawn at seed 101
TUNE_GRID_ENVS = 256           # per injected (d0, v0) condition, seeds 201+
TUNE_GRID_D0 = (1000., 1250., 1500.)
TUNE_GRID_V0 = (0., 1.)

FORMAL_MAX_STEPS = 3000        # safety cap; episodes end by the 1000-step timeout


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _unit_dirs(n, gen, device):
    u1 = torch.rand(n, device=device, generator=gen)
    u2 = torch.rand(n, device=device, generator=gen)
    z = 1. - 2. * u1
    r = torch.sqrt((1. - z * z).clamp_min(0.))
    th = 2. * math.pi * u2
    return torch.stack((r * torch.cos(th), r * torch.sin(th), z), -1)


def _inject(env, d0, v0, seed):
    """Deterministic injected ICs: |p0| = d0, |v0| = v0, independent uniform
    directions (test-side only; env dynamics/reward/criteria untouched)."""
    gen = torch.Generator(device=env.device).manual_seed(seed)
    p0 = d0 * _unit_dirs(env.num_envs, gen, env.device)
    v0v = v0 * _unit_dirs(env.num_envs, gen, env.device)
    env.p.copy_(p0)
    env.v.copy_(v0v)
    env.d0.copy_(p0.norm(dim=-1))
    mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._observe(mask)
    env._update_observation()
    return env.get_observations()


def rollout(env, policy, steps, obs=None):
    """Drive `policy` (callable obs -> action; optional reset_idx) until every
    replica has finished its FIRST episode (or the step cap).  Metrics are
    harness-side and identical for classical controllers and RL checkpoints."""
    n = env.num_envs
    dev = env.device
    fuel_acc = torch.zeros(n, dtype=torch.float64, device=dev)
    dv_acc = torch.zeros(n, dtype=torch.float64, device=dev)
    sat_any = torch.zeros(n, dtype=torch.float64, device=dev)
    sat_ax = torch.zeros(n, dtype=torch.float64, device=dev)
    len_acc = torch.zeros(n, dtype=torch.long, device=dev)
    min_dist = torch.full((n,), float('inf'), dtype=torch.float64, device=dev)
    live = torch.ones(n, dtype=torch.bool, device=dev)  # first episode unrecorded
    eps = []
    act = policy.actions if hasattr(policy, 'actions') else policy
    if obs is None:
        obs = env.get_observations()
    for _ in range(steps):
        if not bool(live.any()):
            break
        dist = obs['policy'].double()[:, :3].norm(dim=-1)
        min_dist = torch.where(live, torch.minimum(min_dist, dist), min_dist)
        a = act(obs)
        a64 = a.double().clamp(-1., 1.)
        livef = live.to(torch.float64)
        tnorm = a64.norm(dim=-1) * env.f_max
        fuel_acc += tnorm * env.dt / (env.isp * env.g0) * livef
        dv_acc += tnorm * env.dt / env.mass * livef
        sat_any += (a64.abs() >= 1. - 1e-6).any(-1).double() * livef
        sat_ax += (a64.abs() >= 1. - 1e-6).double().sum(-1) * livef
        len_acc += live.to(torch.long)
        obs, _, term, trunc, extras = env.step(a)
        done_all = term | trunc
        if hasattr(policy, 'reset_idx'):
            policy.reset_idx(done_all)
        done = done_all & live
        if bool(done.any()):
            tobs = extras['terminal_observations']['policy'].double()[done]
            d_t = tobs[:, :3].norm(dim=-1)
            v_t = tobs[:, 3:6].norm(dim=-1)
            idx = done.nonzero(as_tuple=True)[0]
            min_dist[idx] = torch.minimum(min_dist[idx], d_t)
            succ = (d_t < env.pos_goal) & (v_t < env.vel_goal)
            ko = d_t < env.keep_out
            rng = d_t > env.range_fail
            lens = len_acc[idx]
            timeout = (~succ) & (~ko) & (~rng) & (lens >= env.max_steps)
            fuel_fail = (~succ) & (~ko) & (~rng) & (~timeout)
            for j in range(idx.numel()):
                eps.append(dict(
                    succ=bool(succ[j]), ko=bool(ko[j]), rng=bool(rng[j]),
                    timeout=bool(timeout[j]), fuel_fail=bool(fuel_fail[j]),
                    d_end=float(d_t[j]), v_end=float(v_t[j]),
                    min_dist=float(min_dist[idx][j]),
                    steps=int(lens[j]), fuel=float(fuel_acc[idx][j]),
                    dv=float(dv_acc[idx][j]),
                    sat=float(sat_any[idx][j] / lens[j].clamp_min(1)),
                    sat_axis=float(sat_ax[idx][j] / (3. * lens[j].clamp_min(1))),
                ))
            live[idx] = False
    return eps


def aggregate(name, eps):
    n = len(eps)
    suc = [e for e in eps if e['succ']]
    fail = [e for e in eps if not e['succ']]
    m = lambda key, rows: sum(e[key] for e in rows) / max(len(rows), 1)
    return dict(
        method=name, episodes=n,
        successes=len(suc), success_rate=len(suc) / max(n, 1),
        keepout=sum(e['ko'] for e in eps), range_fail=sum(e['rng'] for e in eps),
        fuel_fail=sum(e['fuel_fail'] for e in eps), timeout=sum(e['timeout'] for e in eps),
        steps_mean_all=m('steps', eps),
        steps_mean_success=m('steps', suc),
        fuel_mean_all=m('fuel', eps),
        fuel_mean_success=m('fuel', suc),
        fuel_max=max((e['fuel'] for e in eps), default=0.),
        dv_mean_success=m('dv', suc),
        min_dist_mean_all=m('min_dist', eps),
        min_dist_mean_fail=m('min_dist', fail),
        min_dist_worst_fail=max((e['min_dist'] for e in fail), default=0.),
        d_end_success_mean=m('d_end', suc), v_end_success_mean=m('v_end', suc),
        v_end_success_max=max((e['v_end'] for e in suc), default=0.),
        sat_any_mean=m('sat', eps), sat_axis_mean=m('sat_axis', eps),
    )


# --------------------------------------------------------------------------- #
# Part 0: small-signal, no-saturation verification
# --------------------------------------------------------------------------- #
def verify():
    print('== Part 0: small-signal no-saturation verification ==', flush=True)
    rows = []
    for d0 in VERIFY_DISTANCES:
        for v0 in VERIFY_SPEEDS:
            torch.manual_seed(SEED)
            env = SpaceRendezvousEnv(num_envs=VERIFY_NUM_ENVS, backend=BACKEND, seed=SEED)
            ctl = build_baseline('rendezvous_cwhpd', env)
            obs = _inject(env, d0, v0, seed=SEED + 9000)
            # reference double-integrator PD rollout on the SAME float32 obs
            p_ref = env.p.clone()
            v_ref = env.v.clone()
            live = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            max_dev = torch.zeros((), device=env.device, dtype=torch.float64)
            sat_steps = 0
            succ_eps = 0
            for _ in range(VERIFY_STEPS):
                if not bool(live.any()):
                    break
                a = ctl.actions(obs)
                sat_steps += int(((a.double().abs() >= 1. - 1e-6).any(-1) & live).sum())
                o = obs['policy'].double()
                a_des = -ctl.kp * o[:, :3] - ctl.kd * o[:, 3:6]
                v_ref = v_ref + a_des * env.dt
                p_ref = p_ref + v_ref * env.dt
                obs, _, term, trunc, extras = env.step(a)
                done = (term | trunc) & live
                ctl.reset_idx(term | trunc)
                if bool(done.any()):
                    tobs = extras['terminal_observations']['policy'].double()[done]
                    succ = (tobs[:, :3].norm(dim=-1) < env.pos_goal) & \
                           (tobs[:, 3:6].norm(dim=-1) < env.vel_goal)
                    succ_eps += int(succ.sum())
                    live[done] = False
                if bool(live.any()):
                    dev_now = (env.p - p_ref).norm(dim=-1)[live].max()
                    max_dev = torch.maximum(max_dev, dev_now)
            done_eps = VERIFY_NUM_ENVS - int(live.sum())
            rows.append(dict(d0=d0, v0=v0, sat_steps=sat_steps,
                             max_pos_deviation_m=float(max_dev),
                             episodes=done_eps, successes=succ_eps))
            print(f'  d0={d0:5.0f} v0={v0:.1f} | sat_steps {sat_steps} | '
                  f'max |p_env - p_ref| {float(max_dev):.3e} m | succ {succ_eps}/{done_eps}',
                  flush=True)
            env.close()
    ok = all(r['sat_steps'] == 0 and r['max_pos_deviation_m'] < VERIFY_TOL_M
             and r['successes'] == r['episodes'] > 0 for r in rows)
    print(f'  Part 0 {"PASSED" if ok else "FAILED"}', flush=True)
    return dict(passed=bool(ok), tol_m=VERIFY_TOL_M, rows=rows)


# --------------------------------------------------------------------------- #
# Part 1: dev-scenario gain selection (independent seeds)
# --------------------------------------------------------------------------- #
def tune():
    print('== Part 1: dev-scenario gain sweep (seeds 101/201+, disjoint from formal seed 7) ==',
          flush=True)
    arms = []
    for wn in TUNE_WN:
        for zeta in TUNE_ZETA:
            eps = []
            t0 = time.perf_counter()
            # natural spawn, independent seed
            torch.manual_seed(101)
            env = SpaceRendezvousEnv(num_envs=TUNE_NATURAL_ENVS, backend=BACKEND, seed=101)
            ctl = build_baseline('rendezvous_cwhpd', env)
            ctl.wn, ctl.zeta = wn, zeta
            ctl.kp, ctl.kd = wn * wn, 2. * zeta * wn
            eps += rollout(env, ctl, FORMAL_MAX_STEPS)
            env.close()
            # injected IC grid, independent seeds
            k = 0
            for d0 in TUNE_GRID_D0:
                for v0 in TUNE_GRID_V0:
                    k += 1
                    torch.manual_seed(201 + k)
                    env = SpaceRendezvousEnv(num_envs=TUNE_GRID_ENVS, backend=BACKEND,
                                             seed=201 + k)
                    ctl = build_baseline('rendezvous_cwhpd', env)
                    ctl.wn, ctl.zeta = wn, zeta
                    ctl.kp, ctl.kd = wn * wn, 2. * zeta * wn
                    obs = _inject(env, d0, v0, seed=201 + k)
                    eps += rollout(env, ctl, FORMAL_MAX_STEPS, obs=obs)
                    env.close()
            agg = aggregate(f'wn{wn:g}_zeta{zeta:g}', eps)
            agg['wn'], agg['zeta'] = wn, zeta
            agg['wall'] = round(time.perf_counter() - t0, 1)
            arms.append(agg)
            print(f"  wn {wn:.3f} zeta {zeta:.1f} | succ {100 * agg['success_rate']:5.1f}% "
                  f"(ko {agg['keepout']} rng {agg['range_fail']} fuel {agg['fuel_fail']} "
                  f"t/o {agg['timeout']}) | fuel {agg['fuel_mean_success']:.2f} kg | "
                  f"t {agg['steps_mean_success']:.0f} | sat {100 * agg['sat_any_mean']:.1f}% | "
                  f"minD fail {agg['min_dist_mean_fail']:.1f} m", flush=True)
    ranked = sorted(arms, key=lambda a: (-a['success_rate'], a['fuel_mean_success'],
                                         a['steps_mean_success']))
    best = ranked[0]
    print(f"  winner: wn {best['wn']:g} zeta {best['zeta']:g} "
          f"(succ {100 * best['success_rate']:.2f}%)", flush=True)
    return dict(arms=arms, winner={'wn': best['wn'], 'zeta': best['zeta']})


# --------------------------------------------------------------------------- #
# Part 2: formal fixed-parameter comparison (seed 7)
# --------------------------------------------------------------------------- #
def _load_rl(out_dir, env):
    cfg = json.loads((out_dir / 'config.json').read_text())
    if cfg.get('algo') == 'flash_sac':
        from algorithms.flashsac.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(out_dir / 'checkpoint.pt'))
        return runner.deterministic_action
    from algorithms.train_utils import RslVecEnv
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/eval_tb', device=str(env.device))
    runner.load(str(out_dir / 'checkpoint.pt'))
    return runner.alg.actor.eval()


def formal():
    print('== Part 2: formal comparison (seed 7, 1024 envs, frozen gains) ==', flush=True)
    report = []
    for method in ('cwh_pd', 'cwh_two_impulse', 'ppo', 'flash_sac'):
        torch.manual_seed(SEED)
        env = SpaceRendezvousEnv(num_envs=NUM_ENVS, backend=BACKEND, seed=SEED)
        if method == 'cwh_pd':
            policy = build_baseline('rendezvous_cwhpd', env)
        elif method == 'cwh_two_impulse':
            policy = build_baseline('rendezvous', env)
        elif method == 'ppo':
            policy = _load_rl(PPO_DIR, env)
        else:
            policy = _load_rl(SAC_DIR, env)
        t0 = time.perf_counter()
        eps = rollout(env, policy, FORMAL_MAX_STEPS)
        agg = aggregate(method, eps)
        agg['eval_wall_seconds'] = round(time.perf_counter() - t0, 1)
        if hasattr(policy, 'wn'):
            agg['gains'] = {'wn': policy.wn, 'zeta': policy.zeta,
                            'kp': policy.kp, 'kd': policy.kd}
        report.append(agg)
        print(f"  {method:18s} | succ {100 * agg['success_rate']:5.1f}% "
              f"(ko {agg['keepout']} rng {agg['range_fail']} fuel {agg['fuel_fail']} "
              f"t/o {agg['timeout']}) | t {agg['steps_mean_success']:.0f} | "
              f"fuel {agg['fuel_mean_success']:.2f} | minD {agg['min_dist_mean_all']:.1f} | "
              f"sat {100 * agg['sat_any_mean']:.1f}%", flush=True)
        env.close()
    return report


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {'seed_formal': SEED, 'num_envs': NUM_ENVS,
           'note': 'Part0 small-signal verification; Part1 dev-seed gain sweep; '
                   'Part2 formal fixed-parameter seed-7 comparison'}
    if RUN_VERIFY:
        doc['verify'] = verify()
    if RUN_TUNE:
        doc['tune'] = tune()
    if RUN_FORMAL:
        doc['formal'] = formal()
    OUT.write_text(json.dumps(doc, indent=2))
    print(f'\nsaved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
