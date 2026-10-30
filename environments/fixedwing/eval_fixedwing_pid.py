"""Fixed-wing conventional-cascaded-PID baseline: scenario checks + formal comparison.

Two independent stages:

1. ``scenarios`` -- single-episode, injected straight / left-turn / right-turn /
   climb / descent / airspeed routes.  These are the *tuning and sanity* scenes
   used to set the PID gains in ``fixedwing_pid.json``; they are not part of
   the held-out comparison.
2. ``benchmark`` -- full random-waypoint episodes on held-out initial-condition
   seeds (default 2000-2003, which were never used for tuning).  Controllers:
   the frozen L1+TECS baseline, this cascaded PID, and the shipped PPO /
   FlashSAC checkpoints when present.  Reports success rate, waypoints per
   episode, mission time, crash/OOB/timeout split, and writes raw trajectories
   plus top-down plots for the first N replicas.

All controllers read only the public ``'policy'`` observation group.  The PID
uses ``V = ||v_b||`` (ground speed; true airspeed in calm air) and never reads
the internal wind truth.  All comparisons run in calm air, matching the frozen
evaluation protocol, even though the RL checkpoints were trained with 30%
Dryden gusts.

Usage
-----
    python environments/fixedwing/eval_fixedwing_pid.py scenarios \
        --out results/fixedwing_pid
    python environments/fixedwing/eval_fixedwing_pid.py benchmark \
        --num-envs 1024 --seeds 2000 2001 2002 2003 --out results/fixedwing_pid
    python environments/fixedwing/eval_fixedwing_pid.py all --out results/fixedwing_pid
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environments.fixedwing import FixedwingWaypointsEnv
from algorithms.baselines.baselines import build_baseline, _quat_to_R, _rot_to_rpy
from algorithms.baselines.baselines import FixedwingAutopilotBaseline, FixedwingPIDBaseline
from algorithms.train_utils import RslVecEnv
from rsl_rl.runners import OnPolicyRunner

DT = 1. / 30.
OOB_RADIUS = 100.

# --------------------------------------------------------------------------- #
# Tuning / sanity routes (not used for the formal comparison)
# --------------------------------------------------------------------------- #
ROUTES = {
    # Turn / climb / descent scenes use only waypoint slots 2-3 (start index 2),
    # so each leg is long enough (>= 45 m) for the ~22 m minimum turn radius to
    # line up.  The env completes at target_index 4, i.e. after both legs.
    'straight': [(20., 0., 20.), (40., 0., 20.), (60., 0., 20.), (80., 0., 20.)],
    'left_turn': [(45., 0., 20.), (30., 40., 20.)],
    'right_turn': [(45., 0., 20.), (30., -40., 20.)],
    'climb': [(45., 0., 15.), (80., 0., 22.)],
    'descent': [(45., 0., 28.), (80., 0., 22.)],
    'speed': [(20., 0., 20.), (40., 0., 20.), (60., 0., 20.), (80., 0., 20.)],
}
ROUTE_START_INDEX = {'left_turn': 2, 'right_turn': 2, 'climb': 2, 'descent': 2}
# Initial state per scenario: (altitude m, forward speed m/s). Cruise scenes
# start level at 16 m/s; climb/descent start offset so the vertical loop has
# room to work (the env itself always spawns at 10 m / 20 m/s).
START_STATE = {
    'straight': (20., 16.), 'left_turn': (20., 16.), 'right_turn': (20., 16.),
    'climb': (12., 16.), 'descent': (32., 16.), 'speed': (20., 16.),
}
SCENARIO_MAX_STEPS = 1200          # 40 s; scenario runs are not the benchmark


class LearnedPolicy:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def actions(self, obs):
        return self._fn(obs)


def load_policy(kind, env, out):
    cfg_path, ckpt = out / 'config.json', out / 'checkpoint.pt'
    if not (cfg_path.exists() and ckpt.exists()):
        return None
    cfg = json.loads(cfg_path.read_text())
    if kind == 'flashsac':
        from algorithms.flashsac.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        fn = runner.deterministic_action
    else:
        runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/eval_tb', device=str(env.device))
        runner.load(str(ckpt))
        actor = runner.alg.actor.eval()
        fn = lambda o: actor(o)
    return LearnedPolicy(kind, fn)


def make_env(num_envs, seed, backend):
    return FixedwingWaypointsEnv(num_envs=num_envs, backend=backend, seed=seed)


def set_route(env, route, start_index=0, idx=0):
    """Inject a fixed route into waypoint slots ``start_index .. 3``.

    The env completes once ``target_index`` reaches 4, so a route that fills
    only the last two slots tests a short, controlled two-leg mission.
    """
    i0 = start_index
    env.targets[idx, i0:i0 + len(route)] = torch.tensor(
        route, dtype=torch.float64, device=env.device)
    env.target_index[idx] = i0
    env.old_distance[idx] = torch.inf
    env.new_distance[idx] = torch.inf
    env._observe(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env._update_observation()


def set_initial_state(env, alt, speed, idx=0):
    """Place replica ``idx`` level at ``alt`` m flying ``speed`` m/s along +x."""
    p, q, v, w = env.plant.state()
    p[idx] = 0.
    q[idx] = 0.
    q[idx, 0] = 1.
    v[idx] = 0.
    v[idx, 0] = speed
    w[idx] = 0.
    p[idx, 2] = alt
    env.plant.set_state(p, q, v, w)
    env._observe(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env._update_observation()


def tracking_state(obs_policy, idx=0):
    o = obs_policy.to(torch.float64)
    wb, q, vb, p = o[idx, :3], o[idx, 3:7], o[idx, 7:10], o[idx, 10:13]
    R = _quat_to_R(q[None])
    rpy = _rot_to_rpy(R)
    vw = R[0] @ vb
    V = vb.norm()
    course = math.degrees(math.atan2(vw[1].item(), vw[0].item())) if vw[:2].norm() > 2 else math.degrees(rpy[2][0])
    target = p + R[0] @ o[idx, 23:26]
    heading = math.degrees(math.atan2((target[1] - p[1]).item(), (target[0] - p[0]).item()))
    err = (heading - course + 180.) % 360. - 180.
    return {
        'z': p[2].item(), 'target_z': target[2].item(), 'alt_err': target[2].item() - p[2].item(),
        'V': V.item(), 'roll_deg': math.degrees(rpy[0][0].item()),
        'pitch_deg': math.degrees(rpy[1][0].item()), 'course_deg': course,
        'course_err_deg': err, 'dist_xy': (target[:2] - p[:2]).norm().item(),
    }


def termination_reason(info, success):
    """Recover the dominant termination cause from the env's episode extras.

    env.step() resets finished replicas internally, so the discrete collision /
    out-of-bounds / complete flags are re-read from ``info['episode']``, which
    is snapshotted before the reset.
    """
    ep = info.get('episode')
    if ep is None:
        return 'unknown'
    for key, name in (('crash_rate', 'crash'), ('out_of_bounds_rate', 'oob'),
                      ('timeout_rate', 'timeout')):
        if float(ep[key]) > 0.:
            return name
    return 'complete' if success else 'unknown'


def run_scenario(kind, name, backend='cpu', v_ref=None):
    env = make_env(1, 7, backend)
    try:
        env.reset()
        alt, speed = START_STATE[name]
        set_initial_state(env, alt, speed)
        set_route(env, ROUTES[name], ROUTE_START_INDEX.get(name, 0))
        ctl = build_baseline('fixedwing', env, kind)
        if v_ref is not None:
            ctl.v_ref = v_ref
        ctl.reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
        obs = env.get_observations()
        rows = []
        info = {}
        for step in range(SCENARIO_MAX_STEPS):
            rows.append(tracking_state(obs['policy']))
            obs, _, term, trunc, info = env.step(ctl.actions(obs))
            ctl.reset_idx(term | trunc)
            if bool(term.any() or trunc.any()):
                break
        success = bool(trunc.any() and not info['time_outs'].any())
        after = rows[60:] if len(rows) > 60 else rows
        last = rows[-1]
        res = {
            'scenario': name, 'controller': ctl.name, 'success': success,
            'end': termination_reason(info, success),
            'steps': step + 1, 'time_s': (step + 1) * DT,
            'alt_rms_m': float(np.sqrt(np.mean([(r['alt_err']) ** 2 for r in after]))),
            'course_rms_deg': float(np.sqrt(np.mean([r['course_err_deg'] ** 2 for r in after]))),
            'V_mean_last': float(np.mean([r['V'] for r in after[-60:]])),
            'V_rms_last': float(np.sqrt(np.mean([(r['V'] - (v_ref or 16.)) ** 2
                                                  for r in after[-60:]]))),
            'roll_min_deg': float(min(r['roll_deg'] for r in rows)),
            'roll_max_deg': float(max(r['roll_deg'] for r in rows)),
            'pitch_min_deg': float(min(r['pitch_deg'] for r in rows)),
            'pitch_max_deg': float(max(r['pitch_deg'] for r in rows)),
            'min_alt': float(min(r['z'] for r in rows)),
            'alt_first': rows[0]['z'], 'alt_last': last['z'],
            'target_z_last': last['target_z'],
        }
        if v_ref is not None:
            res['v_ref'] = v_ref
        return res
    finally:
        env.close()


def scenario_verdict(res):
    """Pass/fail for the tuning scenes (separate from the formal benchmark)."""
    if not res['success']:
        return False
    name = res['scenario']
    if name == 'straight':
        return (res['alt_rms_m'] < 2.5 and res['course_rms_deg'] < 3.
                and abs(res['V_mean_last'] - 16.) < 1.5)
    if name == 'speed':
        return (res['alt_rms_m'] < 2.5 and res['course_rms_deg'] < 3.
                and abs(res['V_mean_last'] - res['v_ref']) < 1.5
                and res['V_rms_last'] < 1.5)
    if name in ('left_turn', 'right_turn'):
        sign = -1. if name == 'left_turn' else 1.
        extreme = res['roll_min_deg'] if sign < 0. else res['roll_max_deg']
        return sign * extreme >= 25. and res['min_alt'] > 10.
    if name == 'climb':
        return (res['pitch_min_deg'] < -3. and res['alt_last'] - res['alt_first'] > 3.
                and res['V_mean_last'] > 11.)
    if name == 'descent':
        return (res['pitch_max_deg'] > 2. and res['alt_first'] - res['alt_last'] > 4.
                and res['min_alt'] > 12.)
    return False


def run_speed_case(v_ref, backend='cpu'):
    res = run_scenario('pid_cascade', 'speed', backend, v_ref=v_ref)
    res['scenario'] = f'speed_{int(v_ref)}'
    res['pass'] = (res['success'] and abs(res['V_mean_last'] - v_ref) < 1.5
                   and res['V_rms_last'] < 1.5)
    return res


# --------------------------------------------------------------------------- #
# Formal held-out waypoint benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(controller, seed, num_envs, backend, max_steps, traj_envs):
    env = make_env(num_envs, seed, backend)
    try:
        obs = env.get_observations()
        if hasattr(controller, 'reset_idx'):
            controller.reset_idx(torch.ones(num_envs, dtype=torch.bool, device=env.device))
        t = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        first_success = torch.zeros_like(finished)
        first_time = torch.zeros(num_envs, dtype=torch.float64, device=env.device)
        ep_targets = env.targets[:traj_envs].clone()
        traj = [[] for _ in range(traj_envs)]
        ep_times, ep_success = [], []
        # Episode accounting follows scripts/eval.py: every episode that
        # finishes inside the window counts, not only the first one per replica.
        ep0 = float(env.completed_episodes.detach().sum().cpu())
        start = time.perf_counter()
        with torch.no_grad():
            for step in range(max_steps):
                actions = controller.actions(obs)
                obs, _, term, trunc, info = env.step(actions)
                reset = term | trunc
                t += 1
                if bool(reset.any()):
                    succ = reset & trunc & ~info['time_outs']
                    ep_times.extend((t[reset] * DT).cpu().tolist())
                    ep_success.extend(succ[reset].cpu().tolist())
                    first = reset & ~finished
                    first_success |= first & succ
                    first_time[first] = t[first].to(torch.float64) * DT
                    t[reset] = 0
                    finished |= reset
                # first-episode trajectories for a small, human-readable subset
                pos = obs['policy'][:traj_envs, 10:13].to(torch.float64).cpu().numpy()
                for j in range(traj_envs):
                    if not finished[j] and len(traj[j]) < max_steps:
                        traj[j].append(pos[j].tolist())
                if bool(finished.all()):
                    break
        wall = time.perf_counter() - start
        episodes = max(int(env.completed_episodes.detach().sum().cpu() - ep0), 1)
        successes = int(env.completed_successes.detach().sum().cpu())
        m = {k: float(v.detach().cpu()) for k, v in env.completed_metrics.items()}
        succ_t = np.array([tt for tt, ok in zip(ep_times, ep_success) if ok])
        report = {
            'controller': controller.name, 'seed': seed, 'num_envs': num_envs,
            'episodes': episodes, 'successes': successes,
            'success_rate': successes / episodes,
            'mean_waypoints': m['waypoints_reached'] / episodes,
            'crash_rate': m['crashes'] / episodes,
            'oob_rate': m['out_of_bounds'] / episodes,
            'timeout_rate': m['timeouts'] / episodes,
            'mean_success_time_s': float(succ_t.mean()) if len(succ_t) else None,
            'mean_episode_time_s': float(np.mean(ep_times)) if ep_times else None,
            'env_steps': step + 1, 'wall_seconds': wall,
        }
        trajectories = {
            'controller': controller.name, 'seed': seed,
            'episodes': [{'success': bool(first_success[j].cpu()),
                          'time_s': float(first_time[j].cpu()),
                          'targets': ep_targets[j].cpu().tolist(), 'positions': traj[j]}
                         for j in range(traj_envs)],
        }
        return report, trajectories
    finally:
        env.close()


def plot_trajectories(trajectories, out_png):
    eps = trajectories['episodes']
    n = len(eps)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for j, ep in enumerate(eps):
        ax = axes[j // cols][j % cols]
        ax.add_patch(plt.Circle((0, 0), OOB_RADIUS, fill=False, ls='--', lw=0.8, color='0.6'))
        p = np.array(ep['positions'])
        if len(p):
            sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap='viridis', s=3)
            ax.plot(p[:, 0], p[:, 1], lw=0.5, color='0.4', alpha=0.6)
        t = np.array(ep['targets'])
        ax.scatter(t[:, 0], t[:, 1], marker='*', s=90, color='crimson', zorder=3)
        ax.plot([0, t[0, 0]], [0, t[0, 1]], color='0.8', ls=':', lw=0.8)
        ax.set_xlim(-OOB_RADIUS, OOB_RADIUS)
        ax.set_ylim(-OOB_RADIUS, OOB_RADIUS)
        ax.set_aspect('equal')
        ax.set_title(f"ep{j} {'OK' if ep['success'] else 'FAIL'} {ep['time_s']:.1f}s", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{trajectories['controller']} seed {trajectories['seed']} (x-y, color=altitude)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def summarize(reports):
    by = {}
    for r in reports:
        by.setdefault(r['controller'], []).append(r)
    out = {}
    for name, rows in by.items():
        keys = ['success_rate', 'mean_waypoints', 'mean_success_time_s',
                'mean_episode_time_s', 'crash_rate', 'oob_rate', 'timeout_rate']
        vals = {}
        for k in keys:
            xs = [r[k] for r in rows if r[k] is not None]
            vals[k] = {'mean': float(np.mean(xs)) if xs else None,
                       'min': float(np.min(xs)) if xs else None,
                       'max': float(np.max(xs)) if xs else None}
        out[name] = {'seeds': [r['seed'] for r in rows], 'episodes': sum(r['episodes'] for r in rows),
                     'metrics': vals}
    return out


def markdown(summary):
    lines = ['| controller | success | waypoints/ep | mission time (s) | crash | OOB | timeout |',
             '|---|---|---|---|---|---|---|']
    for name, s in summary.items():
        m = s['metrics']
        def f(k, p=1):
            v = m[k]['mean']
            return 'n/a' if v is None else f'{v:.{p}f}'
        lines.append(f"| {name} | {m['success_rate']['mean']*100:.1f}% "
                     f"[{m['success_rate']['min']*100:.1f}, {m['success_rate']['max']*100:.1f}] | "
                     f"{f('mean_waypoints', 2)} | {f('mean_success_time_s')} | "
                     f"{m['crash_rate']['mean']*100:.1f}% | {m['oob_rate']['mean']*100:.1f}% | "
                     f"{m['timeout_rate']['mean']*100:.1f}% |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['scenarios', 'benchmark', 'all'])
    ap.add_argument('--num-envs', type=int, default=1024)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2000, 2001, 2002, 2003])
    ap.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    ap.add_argument('--out', type=Path, default=_ROOT / 'results' / 'fixedwing_pid')
    ap.add_argument('--controllers', nargs='+',
                    default=['l1_tecs', 'pid_cascade', 'ppo', 'flashsac'])
    ap.add_argument('--max-steps', type=int, default=4000)
    ap.add_argument('--traj-envs', type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode in ('scenarios', 'all'):
        rows, l1_rows = [], []
        for name in ['straight', 'left_turn', 'right_turn', 'climb', 'descent']:
            r = run_scenario('pid_cascade', name, args.backend)
            r['pass'] = scenario_verdict(r)
            rows.append(r)
            print(json.dumps(r, indent=2), flush=True)
            r1 = run_scenario(None, name, args.backend)
            r1['pass'] = scenario_verdict(r1)
            l1_rows.append(r1)
        for v in (14., 18., 22.):
            r = run_speed_case(v, args.backend)
            rows.append(r)
            print(json.dumps(r, indent=2), flush=True)
        (args.out / 'scenarios_pid.json').write_text(json.dumps(rows, indent=2))
        (args.out / 'scenarios_l1_tecs.json').write_text(json.dumps(l1_rows, indent=2))
        npass = sum(r['pass'] for r in rows)
        n1 = sum(r['pass'] for r in l1_rows)
        print(f'SCENARIOS pid: {npass}/{len(rows)} pass | l1_tecs: {n1}/{len(l1_rows)} pass',
              flush=True)

    if args.mode in ('benchmark', 'all'):
        reports, trajs = [], {}
        for seed in args.seeds:
            for kind in args.controllers:
                env = make_env(args.num_envs, seed, args.backend)
                try:
                    if kind in ('ppo', 'flashsac'):
                        ctl = load_policy(kind, env, _ROOT / 'results' / ('fixedwing_' + kind))
                        if ctl is None:
                            print(f'[skip] {kind}: results/fixedwing_{kind}/checkpoint.pt missing',
                                  flush=True)
                            continue
                    else:
                        ctl = build_baseline('fixedwing', env, None if kind == 'l1_tecs' else kind)
                    rep, tj = run_benchmark(ctl, seed, args.num_envs, args.backend,
                                            args.max_steps, args.traj_envs)
                finally:
                    env.close()
                reports.append(rep)
                print(json.dumps(rep, indent=2), flush=True)
                if kind not in trajs:
                    trajs[kind] = tj
                    (args.out / f'trajectories_{kind}_seed{seed}.json').write_text(json.dumps(tj))
                    plot_trajectories(tj, args.out / f'trajectories_{kind}_seed{seed}.png')
        summary = summarize(reports)
        payload = {'protocol': {'seeds': args.seeds, 'num_envs': args.num_envs,
                                'max_steps': args.max_steps, 'env': 'calm air',
                                'note': 'PID gains tuned on scenario routes + seed 7; '
                                        'all reported seeds are held out.'},
                   'runs': reports, 'summary': summary}
        (args.out / 'comparison.json').write_text(json.dumps(payload, indent=2))
        md = markdown(summary)
        (args.out / 'comparison.md').write_text(md + '\n')
        print(md, flush=True)


if __name__ == '__main__':
    main()
