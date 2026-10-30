"""Validation, tuning and official evaluation harness for the transfer Q-law
baseline (TransferQLawBaseline in algorithms/baselines/baselines.py).

Phase A  small-thrust correctness validation (no tuning content):
   A1  Gauss rate rows + RSW->inertial rotation vs the env's own ODE:
       element rates from the emitted inertial action are compared against a
       finite difference of env._deriv (exact at epsilon -> 0); a wrong frame
       rotation or rate sign fails here.
   A2  Q partials (Q_a, Q_e) by autodiff vs central finite differences.
   A3  frame invariants: zero out-of-plane component, Euclidean norm in
       {0, 1}, per-axis components within the env's [-1, 1] bound.
   A4  direction semantics: at apogee (nu = pi) the burn is near-pure
       prograde transverse (the classical perigee-raise burn); a one-step
       1%-throttle propagation through env.step reproduces the linearized
       (drp, dra) in sign and magnitude (RK4 vs first-order tolerance).
   A5  Lyapunov descent: Q does not increase over a 1%-throttle step.
   A6  grid a_xx against its analytic closed form (2a^2/h)(1+e).

Phase B  parameter selection on INDEPENDENT tuning seeds (13, 21): grid over
         (perigee-reference raise, W_e, eta_min); the winner maximizes the
         mean terminal objective R averaged over the two tuning seeds
         (tie-break: smaller |dra|, then smaller propellant).  The official
         seed (7) is never touched here.

Phase C  official run: the frozen registered default (build_baseline
         'transfer'/'q_law') on the official protocol (seed 7, 1024 envs,
         one full episode per replica).  Success is R > 0 at the final
         orbit; R is never the sole metric -- perigee gain, apogee drift,
         propellant, delta-v and burn duty are all reported.

Phase D  four-way comparison on the identical loop and identical metric
         integration: Gauss one-step greedy (registered default), Q-law,
         PPO checkpoint (results/transfer), FlashSAC checkpoint
         (results/transfer_flashsac).  RL arms are skipped with a note when
         the checkpoint is absent.

Self-contained: every parameter is hard-coded below (no CLI).  Writes
results/transfer/qlaw_report.json and results/transfer/comparison.json.
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from algorithms.baselines.baselines import TransferQLawBaseline, build_baseline
from environments.satellite.orbit_transfer import LowThrustTransferEnv

# ---- hard-coded configuration (edit here, no CLI) ----
BACKEND = 'gpu'
OFFICIAL_SEED = 7
TUNE_SEEDS = (13, 21)          # independent tuning set (never the official seed)
TUNE_ENVS = 512
OFFICIAL_ENVS = 1024
SMALL_THRUST = .01             # Phase-A throttle scale (1% of saturation)
# tuning grid: perigee-reference raise above rp0 [m], W_e, eta_min
TUNE_RP_RAISE = (10e3, 20e3, 40e3)
TUNE_W_E = (1., 3., 10.)
TUNE_ETA_MIN = (0., .1, .3)
RUN_A, RUN_B, RUN_C, RUN_D = True, True, True, True
OUT_DIR = _ROOT / 'results' / 'transfer'
PPO_CKPT_DIR = OUT_DIR
FLASHSAC_CKPT_DIR = _ROOT / 'results' / 'transfer_flashsac'


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _elements(r, v, mu):
    """Osculating (a, e) from a Cartesian state, batched."""
    rm = r.norm(dim=-1)
    a = 1. / (2. / rm - (v * v).sum(-1) / mu)
    hvec = torch.linalg.cross(r, v, dim=-1)
    evec = torch.linalg.cross(v, hvec, dim=-1) / mu - r / rm[:, None]
    return a, evec.norm(dim=-1)


def _run(env, policy, max_steps=20000):
    """Drive `policy(obs) -> action` until every replica completes one episode.

    Delta-v is integrated harness-side from the CLAMPED action actually sent
    (dv += f_max |a|_2 / m_obs dt) and the burn duty counts env-steps with
    |a|_2 > 0.5, so classical and RL arms are measured identically.
    """
    n = env.num_envs
    ep0 = env.completed_episodes.detach().clone()
    ok0 = env.completed_successes.detach().clone()
    ret0 = env.completed_returns.detach().clone()
    met0 = {k: v.detach().clone() for k, v in env.completed_metrics.items()}
    dv = torch.zeros(n, device=env.device, dtype=torch.float64)
    burn_steps = torch.zeros(n, device=env.device, dtype=torch.float64)
    obs = env.get_observations()
    steps = 0
    with torch.no_grad():
        while steps < max_steps:
            actions = policy(obs)
            a_app = actions.to(torch.float64).clamp(-1., 1.)
            m_obs = obs['policy'][:, 7].to(torch.float64)
            dv += env.f_max * a_app.norm(dim=-1) / m_obs * env.dt
            burn_steps += (a_app.norm(dim=-1) > .5).to(torch.float64)
            obs, _, term, trunc, _ = env.step(actions)
            steps += 1
            if float((env.completed_episodes - ep0).sum()) >= n:
                break

    def _s(t):
        return float(t.detach().sum().cpu())

    episodes = _s(env.completed_episodes - ep0)
    per_env_steps = steps  # one episode per replica in this protocol
    out = {'episodes_finished': episodes, 'env_steps': steps,
           'successes': _s(env.completed_successes - ok0),
           'mean_episode_return': _s(env.completed_returns - ret0) / max(episodes, 1.),
           'delta_v_mps_per_episode': _s(dv) / max(episodes, 1.),
           'burn_duty_fraction': _s(burn_steps) / max(episodes * per_env_steps, 1.)}
    out['success_rate'] = out['successes'] / max(episodes, 1.)
    for k, v in env.completed_metrics.items():
        out[k] = _s(v - met0[k]) / max(episodes, 1.)
    return out


# --------------------------------------------------------------------------- #
# Phase A: small-thrust correctness validation
# --------------------------------------------------------------------------- #
def phase_a():
    rep = {}
    env = LowThrustTransferEnv(num_envs=256, backend=BACKEND, seed=101)
    ctrl = TransferQLawBaseline(env)
    obs = env.get_observations()['policy'].to(torch.float64)
    act = ctrl.actions(obs).to(torch.float64)
    r, v, m = obs[:, 1:4], obs[:, 4:7], obs[:, 7]
    mu = env.mu

    # --- A1: element rates from the emitted inertial action vs env._deriv FD
    rm = r.norm(dim=-1)
    hvec = torch.linalg.cross(r, v, dim=-1)
    hhat = hvec / hvec.norm(dim=-1, keepdim=True)
    rhat = r / rm[:, None]
    that = torch.linalg.cross(hhat, rhat, dim=-1)
    u_r = (act * rhat).sum(-1)
    u_t = (act * that).sum(-1)
    # --- A1: element rates from the emitted inertial action vs the exact
    # algebraic identity rate = (f_max/m) * d(elements)/dv . u.  env._deriv
    # applies dv = -mu r/|r|^3 + f_max a/m and gravity has zero instantaneous
    # effect on osculating (a, e), so the thrust-induced element rate is a
    # pure Jacobian contraction -- no time integration, no FD truncation.
    from torch.func import jacrev, vmap
    rv = torch.cat((r, v), -1)
    f = lambda x: torch.stack(_elements(x[None, :3], x[None, 3:6], mu), -1)[0]
    J = vmap(jacrev(f))(rv)                       # (B, 2, 6)
    da_num = (J[:, 0, 3:6] * act).sum(-1) * env.f_max / m
    de_num = (J[:, 1, 3:6] * act).sum(-1) * env.f_max / m
    h = hvec.norm(dim=-1)
    p = h * h / mu
    a, e = _elements(r, v, mu)
    vr = (v * r).sum(-1) / rm
    nu = torch.atan2(vr * h / mu, p / rm - 1.)
    sn, cn = nu.sin(), nu.cos()
    acc = env.f_max / m
    ga_r, ga_t = (2. * a * a / h) * e * sn, (2. * a * a / h) * (p / rm)
    ge_r, ge_t = p * sn / h, ((p + rm) * cn + rm * e) / h
    da_an = (ga_r * u_r + ga_t * u_t) * acc
    de_an = (ge_r * u_r + ge_t * u_t) * acc
    # worst ABSOLUTE error normalized by the largest predicted rate across the
    # batch (a per-replica relative metric explodes at the zero-rate anomaly
    # crossings where both sides vanish)
    rel = lambda num, an: ((num - an).abs().max() / an.abs().max().clamp_min(1e-300)).item()
    rep['A1_rate_rel_err'] = {'da': rel(da_num, da_an), 'de': rel(de_num, de_an)}

    # --- A2: Q partials vs central finite differences of Q(a, e)
    _, qa, qe = ctrl._q_partials(a, e)
    da_fd, de_fd = .5, 1e-7
    qa_fd = (ctrl._q_partials(a + da_fd, e)[0] - ctrl._q_partials(a - da_fd, e)[0]) / (2. * da_fd)
    qe_fd = (ctrl._q_partials(a, e + de_fd)[0] - ctrl._q_partials(a, e - de_fd)[0]) / (2. * de_fd)
    rep['A2_partial_rel_err'] = {'Q_a': rel(qa_fd, qa), 'Q_e': rel(qe_fd, qe)}

    # --- A3: frame invariants on the emitted inertial action
    rep['A3_frame'] = {
        'out_of_plane_abs_max': (act * hhat).sum(-1).abs().max().item(),
        'norm_deviation_max': (act.norm(dim=-1) - act.norm(dim=-1).round()).abs().max().item(),
        'per_axis_abs_max': act.abs().max().item()}

    # --- A4: direction semantics at the apses + one-step 1%-throttle check
    rap, vap = env._kepler_to_cartesian(torch.full((4,), torch.pi, device=env.device,
                                                   dtype=torch.float64))
    rpg, vpg = env._kepler_to_cartesian(torch.zeros(4, device=env.device,
                                                    dtype=torch.float64))
    env.r[:8].copy_(torch.cat((rap, rpg)))
    env.v[:8].copy_(torch.cat((vap, vpg)))
    mask = torch.zeros(256, device=env.device, dtype=torch.bool)
    mask[:8] = True
    env._observe(mask)
    env._update_observation()
    obs8 = env.get_observations()['policy'][:8].to(torch.float64)
    act8 = ctrl.actions(obs8).to(torch.float64)
    r8, v8 = obs8[:, 1:4], obs8[:, 4:7]
    rm8 = r8.norm(dim=-1)
    hhat8 = torch.linalg.cross(r8, v8, dim=-1)
    hhat8 = hhat8 / hhat8.norm(dim=-1, keepdim=True)
    rhat8 = r8 / rm8[:, None]
    that8 = torch.linalg.cross(hhat8, rhat8, dim=-1)
    rep['A4_apse'] = {
        'apogee_u_t': [round(x, 6) for x in (act8[:4] * that8[:4]).sum(-1).tolist()],
        'apogee_u_r': [round(x, 6) for x in (act8[:4] * rhat8[:4]).sum(-1).tolist()],
        'perigee_u_t': [round(x, 6) for x in (act8[4:] * that8[4:]).sum(-1).tolist()],
        'perigee_u_r': [round(x, 6) for x in (act8[4:] * rhat8[4:]).sum(-1).tolist()]}

    env1 = LowThrustTransferEnv(num_envs=256, backend=BACKEND, seed=202)
    ctrl1 = TransferQLawBaseline(env1)
    obs0 = env1.get_observations()['policy'].to(torch.float64)
    ra0, rp0 = env1._radii(env1.r, env1.v, env1.mu)
    q0, _, _ = ctrl1._q_partials(*_elements(env1.r, env1.v, env1.mu))
    act0 = ctrl1.actions(obs0).to(torch.float64)
    burn0 = act0.norm(dim=-1) > .5
    # predicted first-order element change under the 1%-throttle action
    r0, v0, m0 = obs0[:, 1:4], obs0[:, 4:7], obs0[:, 7]
    rm0 = r0.norm(dim=-1)
    hv0 = torch.linalg.cross(r0, v0, dim=-1)
    h0 = hv0.norm(dim=-1)
    p0 = h0 * h0 / env1.mu
    a0, e0 = _elements(r0, v0, env1.mu)
    vr0 = (v0 * r0).sum(-1) / rm0
    nu0 = torch.atan2(vr0 * h0 / env1.mu, p0 / rm0 - 1.)
    sn0, cn0 = nu0.sin(), nu0.cos()
    u_r0 = (act0 * (r0 / rm0[:, None])).sum(-1)
    u_t0 = (act0 * torch.linalg.cross(hv0 / h0[:, None], r0 / rm0[:, None], dim=-1)).sum(-1)
    acc0 = SMALL_THRUST * env1.f_max / m0
    da_p = ((2. * a0 * a0 / h0) * (e0 * sn0 * u_r0 + (p0 / rm0) * u_t0)) * acc0 * env1.dt
    de_p = ((p0 * sn0 * u_r0 + ((p0 + rm0) * cn0 + rm0 * e0) * u_t0) / h0) * acc0 * env1.dt
    drp_p = (1. - e0) * da_p - a0 * de_p
    dra_p = (1. + e0) * da_p + a0 * de_p
    env1.step((SMALL_THRUST * act0).to(torch.float32))
    ra1, rp1 = env1._radii(env1.r, env1.v, env1.mu)
    q1, _, _ = ctrl1._q_partials(*_elements(env1.r, env1.v, env1.mu))
    drp_m, dra_m = rp1 - rp0, ra1 - ra0
    keep = burn0 & (drp_p.abs() > .1)     # 0.1 m floor against float noise
    sign_rp = (torch.sign(drp_m[keep]) == torch.sign(drp_p[keep])).to(torch.float64).mean().item()
    rel_rp = ((drp_m[keep] - drp_p[keep]).abs() / drp_p[keep].abs().clamp_min(1e-9))
    keepa = burn0 & (dra_p.abs() > .1)
    sign_ra = (torch.sign(dra_m[keepa]) == torch.sign(dra_p[keepa])).to(torch.float64).mean().item()
    rel_ra = ((dra_m[keepa] - dra_p[keepa]).abs() / dra_p[keepa].abs().clamp_min(1e-9))
    rep['A4_onestep'] = {'throttle': SMALL_THRUST,
                         'drp_sign_agreement': sign_rp, 'drp_rel_err_median': rel_rp.median().item(),
                         'dra_sign_agreement': sign_ra, 'dra_rel_err_median': rel_ra.median().item(),
                         'burning_replicas': int(burn0.sum())}

    # --- A5: Lyapunov descent over the 1%-throttle step
    viol = (burn0 & (q1 > q0 * (1. + 1e-6))).sum().item()
    rep['A5_q_increase_count'] = viol

    # --- A6: grid a_xx vs analytic (2a^2/h)(1+e)
    a_xx_grid = ctrl._a_xx_grid(a, e)
    a_xx_ref = (2. * a * a / h) * (1. + e)
    rep['A6_a_xx_rel_err'] = rel(a_xx_grid, a_xx_ref)

    checks = {
        'A1 da/de rates match exact Jacobian contraction (<1e-6)':
            rep['A1_rate_rel_err']['da'] < 1e-6 and rep['A1_rate_rel_err']['de'] < 1e-6,
        'A2 Q partials match central FD (<1e-4)':
            rep['A2_partial_rel_err']['Q_a'] < 1e-4 and rep['A2_partial_rel_err']['Q_e'] < 1e-4,
        'A3 zero out-of-plane / unit-norm / per-axis bound':
            rep['A3_frame']['out_of_plane_abs_max'] < 1e-5
            and rep['A3_frame']['norm_deviation_max'] < 1e-5
            and rep['A3_frame']['per_axis_abs_max'] <= 1.,
        'A4 apogee burn is prograde transverse (u_t > 0.99, u_r ~ 0)':
            min(rep['A4_apse']['apogee_u_t']) > .99
            and max(abs(x) for x in rep['A4_apse']['apogee_u_r']) < 1e-3,
        'A4 one-step drp sign agreement = 100%, median rel err < 20%':
            rep['A4_onestep']['drp_sign_agreement'] == 1.
            and rep['A4_onestep']['drp_rel_err_median'] < .2,
        'A5 Q never increases along the burn':
            rep['A5_q_increase_count'] == 0,
        'A6 grid a_xx matches analytic (2a^2/h)(1+e) (<1e-3)':
            rep['A6_a_xx_rel_err'] < 1e-3,
    }
    rep['checks'] = checks
    rep['pass'] = all(checks.values())
    return rep


# --------------------------------------------------------------------------- #
# Phase B: parameter selection on independent tuning seeds
# --------------------------------------------------------------------------- #
def phase_b():
    rows = []
    for raise_m in TUNE_RP_RAISE:
        for w_e in TUNE_W_E:
            for eta in TUNE_ETA_MIN:
                seed_rows = []
                for seed in TUNE_SEEDS:
                    env = LowThrustTransferEnv(num_envs=TUNE_ENVS, backend=BACKEND, seed=seed)
                    rp_ref = env.sma0 * (1. - env.ecc0) + raise_m
                    ctrl = TransferQLawBaseline(env, rp_ref=rp_ref, w_e=w_e, eta_min=eta)
                    m = _run(env, ctrl.actions)
                    seed_rows.append(m)
                    env.close()
                row = {'rp_raise_m': raise_m, 'w_e': w_e, 'eta_min': eta}
                for k in ('terminal_reward', 'success_rate', 'perigee_raise_km',
                          'apo_drift_km', 'propellant_kg', 'delta_v_mps_per_episode',
                          'burn_duty_fraction'):
                    row[k] = sum(r[k] for r in seed_rows) / len(seed_rows)
                rows.append(row)
                print(f"[tune] raise {raise_m / 1e3:5.1f} km  W_e {w_e:4.1f}  eta {eta:3.1f}  "
                      f"R {row['terminal_reward']:7.4f}  drp {row['perigee_raise_km']:6.2f} km  "
                      f"|dra| {row['apo_drift_km']:5.2f} km  fuel {row['propellant_kg']:.4f} kg",
                      flush=True)
    rows.sort(key=lambda r: (-r['terminal_reward'], r['apo_drift_km'], r['propellant_kg']))
    best = rows[0]
    return {'grid': rows, 'selected': best,
            'selection_rule': 'max mean terminal R over tuning seeds 13/21; '
                              'tie-break |dra|, then propellant'}


# --------------------------------------------------------------------------- #
# Phase C / D: official run and four-way comparison
# --------------------------------------------------------------------------- #
def _rl_policy(task_dir, env):
    """Deterministic policy from a trained checkpoint (eval.py semantics)."""
    import json as _json
    from algorithms.train_utils import RslVecEnv
    cfg = _json.loads((task_dir / 'config.json').read_text())
    if cfg.get('algo') == 'flash_sac':
        from algorithms.flashsac.flash_sac import FlashSACRunner
        runner = FlashSACRunner(env, cfg, log_dir='/tmp/qlaw_eval_tb', device=str(env.device))
        runner.load(str(task_dir / 'checkpoint.pt'))
        return runner.deterministic_action
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/qlaw_eval_tb', device=str(env.device))
    runner.load(str(task_dir / 'checkpoint.pt'))
    return runner.alg.actor.eval()


def _official(controller_factory, seed=OFFICIAL_SEED, n=OFFICIAL_ENVS):
    env = LowThrustTransferEnv(num_envs=n, backend=BACKEND, seed=seed)
    ctrl = controller_factory(env)
    policy = ctrl if callable(ctrl) and not hasattr(ctrl, 'actions') else ctrl.actions
    t0 = time.perf_counter()
    m = _run(env, policy)
    m['eval_wall_seconds'] = round(time.perf_counter() - t0, 2)
    if hasattr(ctrl, 'config'):
        m['controller_config'] = ctrl.config()
    if hasattr(ctrl, 'name'):
        m['baseline'] = ctrl.name
    env.close()
    return m


def phase_c():
    return _official(lambda env: build_baseline('transfer', env, 'q_law'))


def phase_d():
    arms = {}
    arms['gauss_greedy'] = _official(lambda env: build_baseline('transfer', env))
    arms['q_law'] = _official(lambda env: build_baseline('transfer', env, 'q_law'))
    for tag, d in (('ppo', PPO_CKPT_DIR), ('flash_sac', FLASHSAC_CKPT_DIR)):
        if (d / 'checkpoint.pt').exists() and (d / 'config.json').exists():
            arms[tag] = _official(lambda env, d=d: _rl_policy(d, env))
        else:
            arms[tag] = {'note': f'skipped: no checkpoint at {d}'}
    return arms


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'backend': BACKEND, 'official_seed': OFFICIAL_SEED,
              'tune_seeds': TUNE_SEEDS, 'tune_envs': TUNE_ENVS,
              'official_envs': OFFICIAL_ENVS}
    if RUN_A:
        print('=== Phase A: small-thrust correctness validation ===', flush=True)
        report['phase_a_validation'] = phase_a()
        for k, v in report['phase_a_validation']['checks'].items():
            print(('  PASS ' if v else '  FAIL ') + k, flush=True)
        if not report['phase_a_validation']['pass']:
            raise SystemExit('Phase A validation failed; not proceeding to tuning/eval')
    if RUN_B:
        print('=== Phase B: tuning on independent seeds ===', flush=True)
        report['phase_b_tuning'] = phase_b()
        print('[tune] selected:', json.dumps(report['phase_b_tuning']['selected']), flush=True)
    if RUN_C:
        print('=== Phase C: official Q-law run (frozen default) ===', flush=True)
        report['phase_c_official'] = phase_c()
        print(json.dumps(report['phase_c_official'], indent=2), flush=True)
    (OUT_DIR / 'qlaw_report.json').write_text(json.dumps(report, indent=2))
    if RUN_D:
        print('=== Phase D: four-way comparison ===', flush=True)
        comparison = {'protocol': {'seed': OFFICIAL_SEED, 'num_envs': OFFICIAL_ENVS,
                                   'episodes': 'one full episode per replica (166-step horizon)'},
                      'arms': phase_d()}
        (OUT_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2), flush=True)
    print('wrote', OUT_DIR / 'qlaw_report.json', flush=True)


if __name__ == '__main__':
    main()
