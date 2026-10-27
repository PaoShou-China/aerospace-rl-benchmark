"""Steering-law comparison for SatelliteAttitudeEnv.

Arms
  damped    existing quaternion PD + singularity-robust damped pseudoinverse
  mp_pinv   quaternion PD + ordinary Moore-Penrose pseudoinverse (same PD gains)
  ppo       shipped PPO checkpoint (results/satellite_ppo)
  flash_sac shipped FlashSAC checkpoint (results/satellite_flashsac)

Scenarios (per CMG cluster size; the 4- and 6-CMG plants are different scales)
  random_target        official reset distribution (target uniform in SO(3))
  near_singular        gimbals parked at a momentum-neutral near-singular
                       configuration, random target drawn by the normal reset
  singular_180         exactly singular (4-CMG roof singularity, rank-1
                       momentum-neutral) / near-singular (6-CMG), target 180 deg

Instrumentation (computed identically for every arm, including RL):
  - attitude error  ||log(q^-1 q_target)|| and |omega|, initial and terminal,
    plus the mean over the episode;
  - settling time   steps/seconds until the env's success termination;
  - gimbal saturation  fraction of env-steps with any |action| >= 1, fraction
    of gimbal commands at the +-action_scale bound, and peak requested rate
    (pre-clamp for the classical arms);
  - torque error  tau_ref is the SHARED quaternion-PD command evaluated on the
    pre-step state for every arm; tau_actual = -kappa gr^T delta_applied uses
    the gimbal rates actually delivered after the env's clamp; we report the
    norm and the fraction of the cluster's peak torque authority;
  - rank diagnostics  sigma_min/sigma_max of the 3xN Jacobian and the fraction
    of steps below the standard pinv rank threshold.

Usage
  python scripts/satellite_steering_compare.py                 # all available arms
  python scripts/satellite_steering_compare.py --arms damped,mp_pinv
  python scripts/satellite_steering_compare.py --ppo-dir results/satellite_ppo \
      --flash-dir results/satellite_flashsac
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

from environments.satellite.attitude_control import SatelliteAttitudeEnv
from algorithms.baselines.baselines import build_baseline

# Momentum-neutral near-singular configurations (verified numerically).
# 4-CMG roof singularity: theta_i = pi/2 - pyramid-skew -> all four torque
# directions collapse onto -z, Sum r_i = 0 exactly (rank 1).
PYRAMID_SKEW = math.radians(54.7356)
THETA4_SINGULAR = math.pi / 2 - PYRAMID_SKEW
THETA4_NEAR = None  # filled below
THETA6_NEAR = (-2.177243665, 1.378914139, -0.192070305,
               0.273303902, -1.297523787, -0.606455308)

DEFAULT_TIMES = (1, 5, 10, 20, 50, 100, 200, 400, 800, 1200)


def _theta4_near(off=0.01):
    return (THETA4_SINGULAR + off, THETA4_SINGULAR - off,
            THETA4_SINGULAR + off, THETA4_SINGULAR - off)


THETA4_NEAR = _theta4_near()

SCENARIOS = {
    4: (('random_target', None, None),
        ('near_singular', THETA4_NEAR, None),
        ('singular_180', THETA4_SINGULAR, (0., 1., 0., 0.))),
    6: (('random_target', None, None),
        ('near_singular', THETA6_NEAR, None),
        ('singular_180', THETA6_NEAR, (0., 1., 0., 0.))),
}


def _inject(env, theta, target_quat):
    """Park every replica at a chosen gimbal configuration (w,x,y,z state)."""
    n = env.num_envs
    if theta is not None:
        th = torch.as_tensor(theta, device=env.device, dtype=torch.float64)
        env.gimbal.copy_(th[None].expand(n, -1))
        env.q.zero_()
        env.q[:, 0] = 1.
        env.omega.zero_()
        env.gimbal_rate.zero_()
        if target_quat is not None:
            env.target_quat.copy_(torch.as_tensor(target_quat, device=env.device,
                                                  dtype=torch.float64)[None].expand(n, -1))
        env.count.zero_()
        env.terminated.zero_()
        env.truncated.zero_()
        env.complete.zero_()
        env.action.zero_()
        env.episode_length_buf.zero_()
        env.returns.zero_()
        env.completed_returns.zero_()
        env.completed_episodes.zero_()
        env.completed_successes.zero_()
        env._prev_phi.copy_(env._potential(
            env._attitude_error().norm(dim=-1), env._manipulability()[0]))
        env._observe(torch.ones(n, device=env.device, dtype=torch.bool))
        env._update_observation()
    return env.get_observations()


def _load_policy(arm, env, cfg, ckpt, device):
    if arm == 'ppo':
        from algorithms.train_utils import RslVecEnv
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/sat_eval_tb',
                                device=str(device))
        runner.load(str(ckpt))
        return runner.alg.actor.eval()
    from algorithms.flashsac.flash_sac import FlashSACRunner
    runner = FlashSACRunner(env, cfg, log_dir='/tmp/sat_eval_tb', device=str(device))
    runner.load(str(ckpt))
    return runner.deterministic_action


def _stats(t):
    t = t.detach().to(torch.float64)
    if t.numel() == 0:
        return {'mean': None, 'median': None, 'p90': None, 'min': None, 'max': None}
    return {'mean': float(t.mean()), 'median': float(t.median()),
            'p90': float(torch.quantile(t, .9)), 'min': float(t.min()),
            'max': float(t.max())}


def run_arm(arm, env, policy, scenario, seed, ref, ctrl=None):
    """One full episode on every replica; returns aggregate metrics."""
    # identical targets for every arm: reset the per-replica RNG stream
    env.rng.counter.zero_()
    torch.manual_seed(seed)
    obs = env.reset()
    if scenario[1] is not None:
        obs = _inject(env, scenario[1], scenario[2])

    n = env.num_envs
    dev = env.device
    n_cmg = env.num_gimbals
    # per-replica episode accounting (exactly one episode per replica)
    recorded = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_steps = torch.zeros(n, dtype=torch.float64, device=dev)
    att_sum = torch.zeros(n, dtype=torch.float64, device=dev)
    omg_sum = torch.zeros_like(att_sum)
    tau_err_sum = torch.zeros_like(att_sum)
    tau_cmd_sum = torch.zeros_like(att_sum)
    tau_act_sum = torch.zeros_like(att_sum)
    sat_any_steps = torch.zeros_like(att_sum)
    sat_gimbal = torch.zeros_like(att_sum)
    rank_def_steps = torch.zeros_like(att_sum)
    rate_req_sum = torch.zeros_like(att_sum)
    rate_req_max = torch.zeros_like(att_sum)
    sigma_ratio_sum = torch.zeros_like(att_sum)
    sigma_ratio_min = torch.full_like(att_sum, float('inf'))
    m_min = torch.full_like(att_sum, float('inf'))
    init_att = obs['policy'][:, 4:7].norm(dim=-1).to(torch.float64).clone()
    term_att = torch.zeros_like(att_sum)
    term_omg = torch.zeros_like(att_sum)
    term_m = torch.zeros_like(att_sum)
    term_rate = torch.zeros_like(att_sum)
    settle = torch.full_like(att_sum, -1.)
    outcome = torch.full((n,), -1, dtype=torch.long, device=dev)  # 0 success 1 runaway 2 timeout
    tau_err_max = torch.zeros((), dtype=torch.float64, device=dev)
    # per-gimbal peak torque is h * |g x r| * rate_limit = h * action_scale
    # (|g x r| = 1; the rotor_speed in gr cancels kappa = h / rotor_speed)
    authority = env.rotor_h_dyn * n_cmg * env.action_scale

    steps = 0
    wall0 = time.perf_counter()
    with torch.no_grad():
        while steps < env.max_steps + 5:
            o = obs['policy'].to(torch.float64)
            tau_ref, gr, m_norm, _ = ref._pd_command(o)
            actions = policy(obs)
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions)
            actions = actions.to(dev)
            delta_applied = actions.to(torch.float64).clamp(-1., 1.) * env.action_scale
            # torque actually delivered at the pre-step gimbal configuration
            tau_act = -ref.kappa * (gr.transpose(1, 2) @ delta_applied.unsqueeze(-1)).squeeze(-1)
            tor_err = (tau_act - tau_ref).norm(dim=-1)
            s = torch.linalg.svdvals(gr)
            sigma_ratio = s[:, -1] / s[:, 0].clamp_min(1e-30)
            rank_def = sigma_ratio < ref.pinv_rtol
            sat_g = actions.abs() >= 1. - 1e-6

            active = ~recorded
            ep_steps += active.double()
            att_sum += torch.where(active, o[:, 4:7].norm(dim=-1), torch.zeros_like(att_sum))
            omg_sum += torch.where(active, o[:, 7:10].norm(dim=-1), torch.zeros_like(att_sum))
            tau_err_sum += torch.where(active, tor_err, torch.zeros_like(att_sum))
            tau_cmd_sum += torch.where(active, tau_ref.norm(dim=-1), torch.zeros_like(att_sum))
            tau_act_sum += torch.where(active, tau_act.norm(dim=-1), torch.zeros_like(att_sum))
            sat_any_steps += torch.where(active, sat_g.any(-1).double(), torch.zeros_like(att_sum))
            sat_gimbal += torch.where(active, sat_g.double().sum(-1), torch.zeros_like(att_sum))
            rank_def_steps += torch.where(active, rank_def.double(), torch.zeros_like(att_sum))
            if ctrl is not None:
                # pre-clamp request of the classical steering law (the action
                # passed to the env is already clipped)
                rate_req = ctrl.last_rate_request.to(torch.float64)
            else:
                rate_req = actions.abs().max(-1).values.to(torch.float64) * env.action_scale
            rate_req_sum += torch.where(active, rate_req, torch.zeros_like(att_sum))
            rate_req_max = torch.maximum(rate_req_max,
                                         torch.where(active, rate_req, torch.zeros_like(att_sum)))
            sigma_ratio_sum += torch.where(active, sigma_ratio, torch.zeros_like(att_sum))
            sigma_ratio_min = torch.minimum(
                sigma_ratio_min, torch.where(active, sigma_ratio,
                                             torch.full_like(att_sum, float('inf'))))
            m_min = torch.minimum(m_min, torch.where(active, m_norm,
                                                     torch.full_like(att_sum, float('inf'))))
            tau_err_max = torch.maximum(tau_err_max,
                                        torch.where(active, tor_err,
                                                    torch.zeros((), dtype=torch.float64,
                                                                device=dev)).max())

            obs, _, term, trunc, extras = env.step(actions)
            steps += 1
            done = term | trunc
            first = done & ~recorded
            if first.any():
                to = extras['terminal_observations']['policy'].to(torch.float64)
                ep_steps_safe = ep_steps.clamp_min(1.)
                term_att = torch.where(first, to[:, 4:7].norm(dim=-1), term_att)
                term_omg = torch.where(first, to[:, 7:10].norm(dim=-1), term_omg)
                term_m = torch.where(first, to[:, 10 + 3 * n_cmg], term_m)
                term_rate = torch.where(first, to[:, 10 + 2 * n_cmg:10 + 3 * n_cmg]
                                        .norm(dim=-1), term_rate)
                settle = torch.where(first, ep_steps_safe, settle)
                time_outs = extras['time_outs']
                success = trunc & ~term & ~time_outs
                timeout = trunc & ~term & time_outs
                outcome = torch.where(first & success, torch.zeros_like(outcome), outcome)
                outcome = torch.where(first & (term | (success & term)),
                                      torch.ones_like(outcome), outcome)
                outcome = torch.where(first & timeout, 2 * torch.ones_like(outcome), outcome)
            recorded |= done
            if bool(recorded.all()):
                break
    wall = time.perf_counter() - wall0

    n_ep = float(n)
    steps_total = float(ep_steps.sum())
    succ = outcome == 0
    run = outcome == 1
    tout = outcome == 2
    settle_succ = settle[succ]
    # settle is in control steps; one step is dt * substeps = 0.025 s at 40 Hz
    # (env.dt = 2.5 ms is the RK4 physics step, not the control period)
    control_period = env.dt * env.substeps
    settle_sec = settle_succ * control_period

    def mean(t):
        return float(t.double().sum() / max(n_ep, 1.))

    res = {
        'episodes': int(n_ep),
        'success_rate': float(succ.double().mean()),
        'fail_runaway': int(run.sum()),
        'fail_timeout': int(tout.sum()),
        'wall_seconds': wall,
        'attitude_error_initial': _stats(init_att),
        'attitude_error_terminal': _stats(term_att),
        'attitude_error_terminal_success': _stats(term_att[succ]),
        'attitude_error_episode_mean': mean(att_sum / ep_steps.clamp_min(1.)),
        'omega_terminal': _stats(term_omg),
        'omega_episode_mean': mean(omg_sum / ep_steps.clamp_min(1.)),
        'settling_steps_success': _stats(settle_succ),
        'settling_seconds_success': _stats(settle_sec),
        'gimbal_sat_envstep_fraction': mean(sat_any_steps / ep_steps.clamp_min(1.)),
        'gimbal_sat_fraction': float(sat_gimbal.sum() / (steps_total * n_cmg)),
        'rate_request_mean': mean(rate_req_sum / ep_steps.clamp_min(1.)),
        'rate_request_max': float(rate_req_max.max()),
        'torque_error_mean': mean(tau_err_sum / ep_steps.clamp_min(1.)),
        'torque_error_max': float(tau_err_max),
        'torque_error_fraction_of_authority': float(tau_err_sum.sum()
                                                    / (steps_total * authority)),
        'torque_command_mean': mean(tau_cmd_sum / ep_steps.clamp_min(1.)),
        'torque_actual_mean': mean(tau_act_sum / ep_steps.clamp_min(1.)),
        'sigma_min_ratio_mean': float(sigma_ratio_sum.sum() / steps_total),
        'sigma_min_ratio_min': float(sigma_ratio_min.min())
        if bool(torch.isfinite(sigma_ratio_min).any()) else None,
        'm_norm_min': float(m_min.min()) if bool(torch.isfinite(m_min).any()) else None,
        'rank_def_envstep_fraction': float(rank_def_steps.sum() / steps_total),
        'peak_torque_authority': float(authority),
        'terminal_m_norm': _stats(term_m),
        'terminal_gimbal_rate_norm': _stats(term_rate),
    }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--arms', default='damped,mp_pinv,ppo,flash_sac')
    parser.add_argument('--cmgs', default='4,6')
    parser.add_argument('--ppo-dir', type=Path,
                        default=_ROOT / 'results' / 'satellite_ppo')
    parser.add_argument('--flash-dir', type=Path,
                        default=_ROOT / 'results' / 'satellite_flashsac')
    parser.add_argument('--out', type=Path,
                        default=_ROOT / 'results' / 'satellite' / 'steering_compare.json')
    parser.add_argument('--md', type=Path,
                        default=_ROOT / 'results' / 'satellite' / 'steering_compare.md')
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    cmg_list = [int(c) for c in args.cmgs.split(',') if c.strip()]

    results = {}
    controller_cfg = {}
    env_cfg = {'num_envs': args.num_envs, 'seed': args.seed,
               'torque_reference': 'shared quaternion PD (kp=3, kd=4) on pre-step state',
               'torque_actual': '-kappa * gr^T @ delta_applied (post-clamp, pre-step angles)',
               'settling': 'env success termination (att_err<0.05 rad, |omega|<0.02 rad/s)'}

    for cmg in cmg_list:
        results[cmg] = {}
        env = SatelliteAttitudeEnv(num_envs=args.num_envs, backend=args.backend,
                                   seed=args.seed, num_cmgs=cmg)
        env_cfg.setdefault('physics_dt_s', env.dt)
        env_cfg.setdefault('substeps', env.substeps)
        env_cfg.setdefault('control_period_s', env.dt * env.substeps)
        env_cfg.setdefault('control_hz', env.agent_hz)
        env_cfg.setdefault('max_episode_steps', env.max_steps)
        ref = build_baseline('satellite', env, 'quat_pd_mp_pinv_cmg')
        controller_cfg[str(cmg)] = {'quat_pd_mp_pinv_cmg': ref.config()}
        controller_cfg[str(cmg)]['quat_pd_damped_cmg'] = build_baseline(
            'satellite', env, 'quat_pd_damped_cmg').config()

        policies = {}
        for arm in arms:
            if arm in ('damped', 'mp_pinv'):
                variant = 'quat_pd_damped_cmg' if arm == 'damped' else 'quat_pd_mp_pinv_cmg'
                ctrl = build_baseline('satellite', env, variant)
                policies[arm] = {'fn': (lambda obs, c=ctrl: c.actions(obs)), 'ctrl': ctrl}
            elif arm in ('ppo', 'flash_sac'):
                ckpt_dir = args.ppo_dir if arm == 'ppo' else args.flash_dir
                ckpt = ckpt_dir / 'checkpoint.pt'
                cfg_path = ckpt_dir / 'config.json'
                if not ckpt.exists() or not cfg_path.exists():
                    results[cmg].setdefault('_unavailable', {})[arm] = f'missing {ckpt}'
                    continue
                cfg = json.loads(cfg_path.read_text())
                trained_cmg = int((cfg.get('env_kwargs') or {}).get('num_cmgs', 4))
                if trained_cmg != cmg:
                    results[cmg].setdefault('_unavailable', {})[arm] = (
                        f'checkpoint trained with {trained_cmg}-CMG, not {cmg}-CMG')
                    continue
                controller_cfg[str(cmg)][arm] = {'checkpoint': str(ckpt),
                                                 'algo': cfg.get('algo')}
                policies[arm] = {'fn': _load_policy(arm, env, cfg, ckpt, env.device),
                                 'ctrl': None}

        for scen in SCENARIOS[cmg]:
            name, theta, tgt = scen
            results[cmg][name] = {}
            for arm in arms:
                if arm not in policies:
                    continue
                t0 = time.perf_counter()
                results[cmg][name][arm] = run_arm(
                    arm, env, policies[arm]['fn'], scen, args.seed, ref,
                    ctrl=policies[arm]['ctrl'])
                results[cmg][name][arm]['eval_wall_seconds'] = time.perf_counter() - t0
                print(f'[{cmg}-CMG {name}] {arm}: success='
                      f'{results[cmg][name][arm]["success_rate"]:.3f} '
                      f'settle={results[cmg][name][arm]["settling_steps_success"]["mean"]}',
                      flush=True)
        env.close()

    payload = {'config': env_cfg, 'controller_config': controller_cfg,
               'scenarios': {str(k): [s[0] for s in v] for k, v in SCENARIOS.items()},
               'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, args.md)
    print(f'wrote {args.out}\nwrote {args.md}', flush=True)


def write_markdown(payload, path):
    cp = payload['config'].get('control_period_s', 0.025)
    lines = ['# Satellite attitude control: steering-law comparison', '',
             'Generated by `scripts/satellite_steering_compare.py`.', '',
             '- Torque reference (all arms): shared quaternion PD command '
             '(`kp=3`, `kd=4`) evaluated on the pre-step state.',
             '- Torque actual: `tau = -kappa * gr^T @ delta_applied` after the '
             'env gimbal-rate clamp, at the pre-step gimbal angles.',
             f'- Settling = control steps until the env success termination '
             f'(`att_err < 0.05 rad`, `|omega| < 0.02 rad/s`); one control step '
             f'= {cp} s (40 Hz), episode cap 1200 steps.', '']
    for cmg in payload['results']:
        lines += [f'## {cmg}-CMG', '',
                  '| scenario | arm | success | settle steps (mean/p90) | settle s '
                  '(mean) | final att err (mean) | sat envstep | sat gimbal | '
                  'torque err (mean) | torque err (% authority) | failures '
                  '(runaway/timeout) |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        unavailable = payload['results'][cmg].get('_unavailable')
        for scen, arms in payload['results'][cmg].items():
            if scen == '_unavailable':
                continue
            for arm, r in arms.items():
                st = r['settling_steps_success']
                ss = r['settling_seconds_success']
                lines.append(
                    f"| {scen} | {arm} | {r['success_rate']:.3f} | "
                    f"{st['mean'] if st['mean'] is not None else '-'} / "
                    f"{st['p90'] if st['p90'] is not None else '-'} | "
                    f"{ss['mean'] if ss['mean'] is not None else '-'} | "
                    f"{r['attitude_error_terminal']['mean']:.4f} | "
                    f"{r['gimbal_sat_envstep_fraction']:.3f} | "
                    f"{r['gimbal_sat_fraction']:.3f} | "
                    f"{r['torque_error_mean']:.4g} | "
                    f"{100 * r['torque_error_fraction_of_authority']:.3f}% | "
                    f"{r['fail_runaway']}/{r['fail_timeout']} |")
        if unavailable:
            lines += ['', 'Unavailable arms: ' + ', '.join(
                f'`{k}` ({v})' for k, v in unavailable.items())]
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
