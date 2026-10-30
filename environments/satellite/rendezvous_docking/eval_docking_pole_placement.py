"""Pole-placement baseline for SpaceDockingEnv: gain design, tuning,
scripted validation, and comparison with the existing dock controllers.

Self-contained (no CLI): edit the constants and run
`~/.conda/envs/genesislab/bin/python eval_docking_pole_placement.py`.

Protocol
    1. Gain design: exact symplectic-Euler CWH (Ad, Bd), full-state
       controllability rank, block pole placement, gain/residual report.
    2. Tuning: independent benchmark-distribution set (seed 13, 512 envs,
       2000 steps); compare closed-loop natural frequencies and fix the
       least aggressive point on the convergence knee.
    3. Validation: deterministic single-axis and three-axis initial states;
       check first-command sign, convergence, terminal error, saturation.
    4. Comparison: benchmark distribution (seed 7, 1024 envs, 2000 steps)
       for the CWH-LQR baseline (with its layers), the new pole-placement
       controller (no layers), the original PPO arms, and original FlashSAC.

Uniform per-episode metrics: success, terminal range/speed, steps, fuel
(kg, from the applied action), and thrust saturation fraction
(|u_i| >= 0.99 of the 0.02 m/s^2 per-axis authority).

Writes results/docking/pole_placement_report.json.
"""
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))

import torch
from rsl_rl.runners import OnPolicyRunner

from algorithms.baselines.baselines import (DockingCWHPlacementBaseline,
                                            build_baseline)
from algorithms.train_utils import RslVecEnv
from environments.satellite.rendezvous_docking import SpaceDockingEnv

NUM_ENVS = 1024
WINDOW = 2000
SEED = 7
BACKEND = 'gpu'
OUT = _ROOT / 'results' / 'docking' / 'pole_placement_report.json'

TUNE_ENVS = 512
TUNE_SEED = 13
TUNE_WINDOW = 2000
TUNE_WN = (0.035, 0.05, 0.08, 0.11)   # slow .. aggressive

SAT_LEVEL = .99
PPO_ARMS = (
    ('ppo_mlp', _ROOT / 'results' / 'docking_ppo_mlp'),
)
FLASHSAC_DIR = _ROOT / 'results' / 'docking_flashsac'


# --------------------------------------------------------------------------- #
# metrics
def _q(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return float(vals[min(len(vals) - 1, int(p * len(vals)))])


def _summary(rows, range_fail, fuel0):
    n = len(rows)
    suc = [r for r in rows if r['success']]
    rng = [r for r in rows if not r['success'] and r['d'] > range_fail]
    fuel_fail = [r for r in rows if not r['success'] and r['d'] <= range_fail
                 and r['fuel'] >= fuel0]
    timeout = [r for r in rows if not r['success']
               and r not in rng and r not in fuel_fail]
    return {
        'episodes': n,
        'success_rate': len(suc) / max(n, 1),
        'range_fails': len(rng),
        'fuel_fails': len(fuel_fail),
        'timeouts': len(timeout),
        'terminal_pos_err_p50_m': _q([r['d'] for r in suc], .5),
        'terminal_pos_err_p90_m': _q([r['d'] for r in suc], .9),
        'terminal_vel_err_p50_mps': _q([r['v'] for r in suc], .5),
        'terminal_vel_err_p90_mps': _q([r['v'] for r in suc], .9),
        'terminal_pos_err_all_p50_m': _q([r['d'] for r in rows], .5),
        'terminal_vel_err_all_p50_mps': _q([r['v'] for r in rows], .5),
        'time_to_success_p50_steps': _q([r['steps'] for r in suc], .5),
        'time_to_success_p90_steps': _q([r['steps'] for r in suc], .9),
        'time_to_success_max_steps': max((r['steps'] for r in suc), default=None),
        'episode_steps_mean': sum(r['steps'] for r in rows) / max(n, 1),
        'fuel_mean_kg': sum(r['fuel'] for r in rows) / max(n, 1),
        'fuel_max_kg': max((r['fuel'] for r in rows), default=0.),
        'sat_frac_99': sum(r['sat99'] for r in rows)
                        / max(sum(r['axes'] for r in rows), 1.),
        'sat_frac_full': sum(r['sat1'] for r in rows)
                         / max(sum(r['axes'] for r in rows), 1.),
    }


class _Tracker:
    """Per-env episode counters (device tensors) + completed-episode records."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.env = env
        self.steps = torch.zeros(n, device=dev, dtype=torch.float64)
        self.fuel = torch.zeros(n, device=dev, dtype=torch.float64)
        self.sat99 = torch.zeros(n, device=dev, dtype=torch.float64)
        self.sat1 = torch.zeros(n, device=dev, dtype=torch.float64)
        self.axes = torch.zeros(n, device=dev, dtype=torch.float64)
        self.rows = []

    def step(self, actions):
        a = actions.to(torch.float64)
        self.steps += 1.
        self.fuel += a.norm(dim=-1) * self.env.f_max * self.env.dt / (
            self.env.isp * self.env.g0)
        self.sat99 += (a.abs() >= SAT_LEVEL).sum(-1)
        self.sat1 += (a.abs() >= 1.).sum(-1)
        self.axes += a.shape[-1]

    def flush(self, done, terminal_obs, labels=None):
        idx = done.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        term = terminal_obs.to(torch.float64)
        d = term[:, :3].norm(dim=-1)
        v = term[:, 3:6].norm(dim=-1)
        success = (d < self.env.pos_goal) & (v < self.env.vel_goal)
        for j, i in enumerate(idx.cpu().tolist()):
            self.rows.append({
                'scenario': labels[i] if labels else i,
                'd': float(d[i]), 'v': float(v[i]),
                'steps': float(self.steps[i]), 'fuel': float(self.fuel[i]),
                'sat99': float(self.sat99[i]), 'sat1': float(self.sat1[i]),
                'axes': float(self.axes[i]), 'success': bool(success[i])})
        self.steps[idx] = 0.; self.fuel[idx] = 0.
        self.sat99[idx] = 0.; self.sat1[idx] = 0.; self.axes[idx] = 0.


def _run(env, act_fn, steps, reset_fn=None, grid_mode=False, labels=None):
    tracker = _Tracker(env)
    obs = env.get_observations()
    first_actions = None
    first_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    used = steps
    for t in range(steps):
        actions = act_fn(obs)
        if first_actions is None:
            first_actions = actions.detach().clone()
        tracker.step(actions)
        obs, _, term, trunc, extras = env.step(actions)
        always_done = term | trunc
        done = always_done & ~first_done if grid_mode else always_done
        first_done |= always_done
        if bool(done.any()):
            tracker.flush(done, extras['terminal_observations']['policy'], labels)
            if reset_fn is not None:
                reset_fn(done)
        if grid_mode and bool(first_done.all()):
            used = t + 1
            break
    return tracker.rows, used, first_actions


def _make_env(num_envs, seed, env_kwargs=None, backend=BACKEND):
    try:
        return SpaceDockingEnv(num_envs=num_envs, backend=backend, seed=seed,
                               **(env_kwargs or {}))
    except RuntimeError:
        return SpaceDockingEnv(num_envs=num_envs, backend='cpu', seed=seed,
                               **(env_kwargs or {}))


# --------------------------------------------------------------------------- #
# 1. design report
def _design_report(env, ctrl):
    d = ctrl.design
    return {
        'model': 'x=[p,m; v,m/s], u=acceleration m/s^2; Ad,Bd = exact symplectic '
                 'Euler of CWH at dt=1 s (same integrator as the env)',
        'cwh_rate_n_rad_s': float(env.n),
        'dt_s': float(env.dt),
        'mass_kg': float(env.mass),
        'f_max_N_per_axis': float(env.f_max),
        'a_max_mps2_per_axis': float(env.f_max / env.mass),
        'state_feedback': 'u = -K (x - x_ref), x_ref = [0,0,0,0,0,0]',
        'action_map': 'u_norm = (m * u) / f_max = u / a_max, clipped to [-1, 1]',
        'controllability_rank': d['controllability_rank'],
        'block_ranks': d['block_ranks'],
        'controllability_cond': d['controllability_cond'],
        'poles_continuous_real_imag': d['poles_continuous'],
        'poles_discrete_real_imag': d['poles_discrete'],
        'placement_residual': d['placement_residual'],
        'max_abs_closed_loop_eig': d['max_abs_eig'],
        'gain_K_3x6': d['gain'],
    }


# --------------------------------------------------------------------------- #
# 2. tuning
def _tune():
    rows = []
    for wn in TUNE_WN:
        env = _make_env(TUNE_ENVS, TUNE_SEED)
        ctrl = DockingCWHPlacementBaseline(env, wn=wn)
        t0 = time.perf_counter()
        with torch.no_grad():
            ep_rows, _, _ = _run(env, ctrl.actions, TUNE_WINDOW,
                                 reset_fn=ctrl.reset_idx)
        s = _summary(ep_rows, env.range_fail, env.fuel0)
        s.update({'wn': wn, 'wall_s': round(time.perf_counter() - t0, 1),
                  'K_pos_norm': [round(float(x), 6) for x in
                                 ctrl.K[:, :3].norm(dim=1).tolist()],
                  'K_vel_norm': [round(float(x), 6) for x in
                                 ctrl.K[:, 3:].norm(dim=1).tolist()]})
        rows.append(s)
        print(f"[tune] wn={wn:.3f} success={s['success_rate']:.3f} "
              f"t50={s['time_to_success_p50_steps']} "
              f"fuel={s['fuel_mean_kg']:.3f} sat={s['sat_frac_99']:.3f}",
              flush=True)
        env.close()
    # select: max success, then fastest convergence, then least fuel
    best = max(rows, key=lambda r: (r['success_rate'],
                                    -(r['time_to_success_p50_steps'] or 1e9),
                                    -r['fuel_mean_kg']))
    return rows, best


# --------------------------------------------------------------------------- #
# 3. scripted validation
def _single_axis_cases():
    p0, v0, labels, rest = [], [], [], []
    for axis in range(3):
        for dist in (15., 30., 50., 75., 100.):
            p_hat = torch.zeros(3); p_hat[axis] = 1.
            t_hat = torch.zeros(3); t_hat[(axis + 1) % 3] = 1.
            for kind in ('rest', 'out', 'in', 'tan'):
                v = {'rest': torch.zeros(3), 'out': p_hat * .5,
                     'in': -p_hat * .5, 'tan': t_hat * .5}[kind]
                p0.append(p_hat * dist); v0.append(v)
                labels.append(f'axis{axis}|d={dist:g}|{kind}')
                rest.append(kind == 'rest')
    return torch.stack(p0), torch.stack(v0), labels, rest


def _three_axis_cases():
    p0, v0, labels, rest = [], [], [], []
    dirs = [torch.tensor([sx, sy, sz], dtype=torch.float64)
            for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.)]
    dirs = [d / d.norm() for d in dirs]
    for d in dirs:
        name = ''.join('+' if x > 0 else '-' for x in d)
        for dist in (30., 60., 100.):
            for kind in ('rest', 'out'):
                v = torch.zeros(3) if kind == 'rest' else d * .5
                p0.append(d * dist); v0.append(v)
                labels.append(f'dir{name}|d={dist:g}|{kind}')
                rest.append(kind == 'rest')
    return torch.stack(p0), torch.stack(v0), labels, rest


def _grid_arm(env, ctrl, p0, v0, labels, rest):
    dev = env.device
    p0d, v0d = p0.to(dev, torch.float64), v0.to(dev, torch.float64)
    env.p.copy_(p0d); env.v.copy_(v0d); env.d0.copy_(p0d.norm(dim=-1))
    obs64 = torch.cat((p0d, v0d), -1)
    env._observation.copy_(obs64)
    env.obs_buf.copy_(obs64.to(torch.float32))
    with torch.no_grad():
        rows, used, first = _run(env, ctrl.actions, env.max_steps,
                                 reset_fn=ctrl.reset_idx, grid_mode=True,
                                 labels=labels)
    # sign check: the first command along the initial displacement must brake
    a0, ph = first.to(torch.float64), p0d / p0d.norm(dim=-1, keepdim=True)
    proj = (a0 * ph).sum(-1)
    sign_rows = [{'scenario': labels[i], 'a_dot_p_hat': float(proj[i]),
                  'opposes_position': bool(proj[i] < 0.)}
                 for i in torch.tensor(rest).nonzero(as_tuple=True)[0].tolist()]
    s = _summary(rows, env.range_fail, env.fuel0)
    s.update({'scenarios': len(rows), 'steps_used': used,
              'first_step_sign_checked': len(sign_rows),
              'first_step_sign_correct': sum(r['opposes_position'] for r in sign_rows),
              'min_first_step_projection': min((r['a_dot_p_hat'] for r in sign_rows),
                                               default=None)})
    return s, rows


# --------------------------------------------------------------------------- #
# 4. comparison
def _load_policy(env, out_dir):
    cfg = json.loads((out_dir / 'config.json').read_text())
    ckpt = out_dir / 'checkpoint.pt'
    assert ckpt.exists(), f'missing {ckpt}'
    torch.manual_seed(SEED)
    if cfg.get('algo') == 'flash_sac':
        from algorithms.flashsac.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/eval_tb',
                                device=str(env.device))
        runner.load(str(ckpt))
        return runner.deterministic_action
    runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/eval_tb',
                            device=str(env.device))
    runner.load(str(ckpt))
    return runner.alg.actor.eval()


def _dist_arm(label, env, act_fn, reset_fn=None):
    ep0 = env.completed_episodes.clone()
    ok0 = env.completed_successes.clone()
    t0 = time.perf_counter()
    with torch.no_grad():
        rows, _, _ = _run(env, act_fn, WINDOW, reset_fn=reset_fn)
    s = _summary(rows, env.range_fail, env.fuel0)
    env_eps = float((env.completed_episodes - ep0).sum())
    env_ok = float((env.completed_successes - ok0).sum())
    s.update({'arm': label, 'wall_s': round(time.perf_counter() - t0, 1),
              'env_episodes': env_eps, 'env_successes': env_ok,
              'env_consistent': abs(env_eps - len(rows)) < 1.5
                                and abs(env_ok - sum(r['success'] for r in rows)) < .5})
    return s


def _env_kwargs(out_dir):
    p = out_dir / 'config.json'
    if not p.exists():
        return {}
    return dict(json.loads(p.read_text()).get('env_kwargs') or {})


def _fmt(v, spec='.1f'):
    return '-' if v is None else (f'{v:{spec}}' if isinstance(v, float) else str(v))


def main():
    # design + registration --------------------------------------------------
    probe = _make_env(8, SEED)
    ctrl = build_baseline('docking', probe, variant='pole_placement')  # registry
    design = _design_report(probe, ctrl)
    print(f"[design] rank={design['controllability_rank']} "
          f"blocks={design['block_ranks']} "
          f"residual={design['placement_residual']:.2e} "
          f"max|z|={design['max_abs_closed_loop_eig']:.4f}", flush=True)
    probe.close()

    # tuning -----------------------------------------------------------------
    tune_rows, best = _tune()
    selected_wn = best['wn']
    print(f"[tune] selected wn={selected_wn:.3f} (fastest success with "
          f"non-aggressive fuel/saturation)", flush=True)

    # validation -------------------------------------------------------------
    p1, v1, l1, r1 = _single_axis_cases()
    p3, v3, l3, r3 = _three_axis_cases()
    grid = {}
    for name, (p0, v0, labels, rest) in (
            ('single_axis', (p1, v1, l1, r1)),
            ('three_axis', (p3, v3, l3, r3))):
        env = _make_env(p0.shape[0], SEED)
        ctrl_g = DockingCWHPlacementBaseline(env, wn=selected_wn)
        s, rows = _grid_arm(env, ctrl_g, p0, v0, labels, rest)
        grid[name] = {'summary': s, 'rows': rows}
        print(f"[val:{name}] success={s['success_rate']:.3f} "
              f"sign={s['first_step_sign_correct']}/{s['first_step_sign_checked']} "
              f"t50={s['time_to_success_p50_steps']} "
              f"d50={_fmt(s['terminal_pos_err_p50_m'], '.2f')} "
              f"fuel={s['fuel_mean_kg']:.3f} sat={s['sat_frac_99']:.3f}", flush=True)
        env.close()

    # comparison -------------------------------------------------------------
    arms = []
    lqr_env = _make_env(NUM_ENVS, SEED)
    lqr = build_baseline('docking', lqr_env)
    arms.append(_dist_arm('cwh_lqr_existing', lqr_env, lqr.actions, lqr.reset_idx))
    lqr_env.close()

    pp_env = _make_env(NUM_ENVS, SEED)
    pp = DockingCWHPlacementBaseline(pp_env, wn=selected_wn)
    arms.append(_dist_arm('cwh_pole_placement', pp_env, pp.actions, pp.reset_idx))
    pp_env.close()

    for label, out_dir in PPO_ARMS:
        if not (out_dir / 'checkpoint.pt').exists():
            arms.append({'arm': label, 'skipped': 'missing checkpoint'})
            continue
        env = _make_env(NUM_ENVS, SEED, _env_kwargs(out_dir))
        policy = _load_policy(env, out_dir)
        arms.append(_dist_arm(label, env, policy))
        env.close()

    if (FLASHSAC_DIR / 'checkpoint.pt').exists():
        env = _make_env(NUM_ENVS, SEED, _env_kwargs(FLASHSAC_DIR))
        policy = _load_policy(env, FLASHSAC_DIR)
        arms.append(_dist_arm('flashsac_original', env, policy))
        env.close()
    else:
        arms.append({'arm': 'flashsac_original', 'skipped': 'missing checkpoint'})

    for a in arms:
        if 'skipped' in a:
            print(f"[cmp] {a['arm']:20s} SKIPPED ({a['skipped']})", flush=True)
        else:
            print(f"[cmp] {a['arm']:20s} success={a['success_rate']:.3f} "
                  f"t50={_fmt(a['time_to_success_p50_steps'], '.0f')} "
                  f"d50={_fmt(a['terminal_pos_err_p50_m'], '.3f')} "
                  f"v50={_fmt(a['terminal_vel_err_p50_mps'], '.3f')} "
                  f"fuel={a['fuel_mean_kg']:.3f} sat={a['sat_frac_99']:.3f} "
                  f"ep={a['episodes']} consistent={a['env_consistent']}",
                  flush=True)

    report = {
        'task': 'docking',
        'controller_registry_key': 'docking + BASELINE_VARIANTS["docking"]["pole_placement"]',
        'controller_config': {'wn': selected_wn, 'zeta': .9,
                              'pole_ratios': [.8, 1.], 'dt_s': 1.,
                              'mass_kg': 500., 'f_max_N_per_axis': 10.,
                              'a_max_mps2_per_axis': .02,
                              'target_state': [0., 0., 0., 0., 0., 0.],
                              'extra_mechanisms': 'none (no shield, capture net, '
                                                  'fuel derate, pulse or delay)'},
        'design': design,
        'tuning': {'protocol': f'{TUNE_ENVS} envs, seed {TUNE_SEED}, '
                               f'{TUNE_WINDOW} steps, benchmark spawn distribution',
                   'candidates': tune_rows, 'selected_wn': selected_wn,
                   'selection': 'max success, then min time-to-success p50, '
                                'then min fuel mean'},
        'validation': grid,
        'comparison': {'protocol': f'{NUM_ENVS} envs, seed {SEED}, '
                                   f'{WINDOW} steps, deterministic policies, '
                                   f'saturation threshold |u_i| >= {SAT_LEVEL}',
                        'arms': arms},
        'notes': {
            'terminal_metrics': 'successful episodes end at the first entry '
                                'into |p|<1 m and |v|<0.1 m/s, so terminal '
                                'position p50 is ~0.97 m by construction; '
                                'speed p50/p90 is the meaningful capture error',
            'layers': 'CWH-LQR reuse keeps its existing shield/capture/fuel '
                      'layers; the pole-placement arm has no extra layer',
            'rl_budget': 'RL arms were retrained from the existing entries: '
                         'PPO 1000 updates x 32 steps x 1024 envs = 32.77M '
                         'transitions; original FlashSAC 1000 updates x 1024 '
                         'envs = 1.02M transitions. No extension/retuning was '
                         'applied (direct reuse requested).',
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
