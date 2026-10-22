"""Classic (non-learning) baseline for the toy Pendulum task: energy-shaping
swing-up + discrete-time LQR stabilization at the upright equilibrium.

Motivation: pair the learned baselines (rsl-rl PPO / FlashSAC) with a
traditional control method on the same task, same reward, same success bar.
This entry is read-only with respect to the environment: it consumes the
policy observation [cos th, sin th, thdot] and feeds 1-D actions; env,
reward and success criteria are untouched.

Verified environment conventions (src/pendulum/env.py, gym Pendulum-v1 spec)
---------------------------------------------------------------------------
- Angle zero: th=0 is the UPRIGHT (unstable) equilibrium; hanging is th=+-pi.
  Dynamics th'' = 3g/(2l) sin(th) + 3u/(m l^2) (after sin(th+pi) = -sin th).
- Target equilibrium: (th, thdot) = (0, 0).
- Parameters: g=10, m=1, l=1, NO damping term; rod inertia I = 2/3 m l^2
  (implied by the 3g/(2l) gravity coefficient). |thdot| clamped to 8 rad/s.
- Action: 1-D, torque u = clip(a, -1, 1) * max_torque (2 N m).
- Time step: dt=0.05, semi-implicit Euler -- velocity updates first, then the
  position integrates the *unclipped* new velocity; the clamp is applied last.
- Reset (benchmark difficulty 1): th ~ U[-pi, pi], thdot ~ U[-1, 1].
- Reward (pre-integration state): r = -(th_w^2 + 0.1 thdot^2 + 0.001 u^2),
  th_w = angle wrapped to [-pi, pi]; 200-step truncation, no early
  termination; success = completed-episode return > -200.

Controller
----------
Far from upright (energy mode):
    E_rel = 1/2 I thdot^2 + m g l (cos th - 1)      (energy relative to upright)
    u = clip(-k_e * (E_rel - E_t) * thdot, +-max_torque)
    (k_e = 3, target E_t = +0.5 J). dE_rel/dt = 2 u thdot, so this pumps
    energy up when E_rel < E_t and dissipates above it; per-step it
    maximizes dE/dt under |u| <= 2 whenever saturated (along-motion pumping).
    k_e > 1 only matters in the unsaturated deep-crawl phase near hanging
    (|u| = k_e |E| |thdot| < 2), where it multiplies the pumping rate;
    E_t > 0 lets the final approach cross the LQR boundary with a small
    energy margin instead of hovering just below the homoclinic.
    Stuck rescue: the pump has a degenerate fixed point at rest near the
    hanging position (thdot=0 => u=0); when |thdot| < 1e-3 and |th_w| > 2.5
    a one-sided kick u = max_torque * sign(sin th_w) displaces the state
    along gravity (the fast escape direction) and the pump takes over.
Near upright (LQR mode):
    Exact linearization of the ACTUAL discrete plant about (0,0): with
    a_g = 3g/(2l), b_u = 3/(m l^2), the env's semi-implicit Euler step is
        thdot+ = thdot + dt (a_g th + b_u u)
        th+    = th + dt thdot+
    => Ad = [[1 + a_g dt^2, dt], [a_g dt, 1]], Bd = [[b_u dt^2], [b_u dt]].
    Q = diag(1, 0.1), R = 0.001 match the per-step reward weights exactly
    (theta^2, 0.1 thdot^2, 0.001 u^2 in torque units), so the discrete LQR
    (DARE) locally maximizes the true episode return. At dt=0.05:
    K = [19.693, 5.263], u = -K [th_w, thdot], closed-loop eigenvalues
    0.854 / 0.247 (stable, well damped). The capture basin under torque
    saturation was measured at |th| <~ 0.4 rad from rest.
Switching (hysteresis against chatter):
    enter LQR when |th_w| <= 0.35 rad (inside the measured capture basin);
    demote to energy only when |th_w| > 0.55 rad. Angle wrap is handled by
    using th_w = atan2(sin th, cos th) everywhere; actions are saturated by
    construction (clip to +-max_torque, then action = u / max_torque, which
    the env re-clips identically).

Metrics reported (per suite)
----------------------------
success rate (return > -200), cumulative return, stabilization time (first
step after which |th_w| <= 0.1 and |thdot| <= 0.5 holds to episode end;
"settled" = stabilized with a >= 50-step hold), max |thdot| per episode,
switching failure ratio (episodes with >=1 LQR->energy demotion / episodes
that entered LQR) and the fraction of saturated action steps.

Self-contained: every parameter is hard-coded below, no CLI arguments.
Run `python classic_pendulum.py`.
"""
import math
import sys
import types
from pathlib import Path

import torch

_src = Path(__file__).resolve().parent / 'src'
_pkg = types.ModuleType('pyflyt_genesis')
_pkg.__path__ = [str(_src)]
sys.modules['pyflyt_genesis'] = _pkg
sys.path.insert(0, str(_src))  # exposes `common` / `pyflyt_genesis` subpackages

from pyflyt_genesis.pendulum import PendulumEnv

# ----------------------------- configuration -----------------------------
NUM_ENVS_RANDOM = 2048        # replicas for the reset-distribution suite
EPISODES_RANDOM = 5           # completed episodes per replica (auto-resets)
SEED = 7
STEPS = PendulumEnv.max_steps  # 200; episodes truncate there

K_ENERGY = 3.0                # energy-pump gain k_e in u = -k_e (E_rel - E_t) thdot;
                              # >1 speeds up the unsaturated deep-crawl phase near
                              # hanging (saturated regime is k_e-invariant)
ENERGY_TARGET = 0.5           # pump toward E_rel = +0.5 J: the last approach to the
                              # LQR boundary carries a small energy margin (apex
                              # 0.32 rad < LQR_ENTER) and captures faster than the
                              # classic homoclinic target (A/B: ~+1.5 pp success)
STUCK_THDOT = 1e-3            # |thdot| below this in energy mode => rescue kick
STUCK_TH = 2.5                # rescue zone: only near the hanging position
LQR_ENTER = 0.35              # |th_w| <= this -> switch energy -> LQR [rad]
LQR_EXIT = 0.55               # hysteresis: LQR -> energy beyond this [rad]
SETTLE_TH = 0.1               # stabilization region |th_w| <= ... [rad]
SETTLE_THDOT = 0.5            # ... and |thdot| <= ... [rad/s]
SETTLE_MIN_HOLD = 50          # steps in-region required to call it "settled"
Q_LQR = (1., 0.1)             # DARE weights: match reward (th^2, 0.1 thdot^2)
R_LQR = 0.001                 # DARE weight on torque u (reward's 0.001 u^2)
DARE_ITERS = 20000            # fallback value-iteration cap (no scipy)


def solve_dare(Ad, Bd, Q, R):
    """Discrete algebraic Riccati equation; scipy when available."""
    try:
        from scipy.linalg import solve_discrete_are
        P = torch.tensor(solve_discrete_are(Ad.numpy(), Bd.numpy(), Q.numpy(), R.numpy()),
                         dtype=Ad.dtype)
    except ImportError:
        P = Q.clone()
        I = torch.eye(Ad.shape[0], dtype=Ad.dtype)
        for _ in range(DARE_ITERS):
            BtPB = Bd.T @ P @ Bd
            Pn = Q + Ad.T @ P @ Ad \
                - (Ad.T @ P @ Bd) @ torch.linalg.solve(R + BtPB, Bd.T @ P @ Ad)
            if torch.max(torch.abs(Pn - P)) < 1e-12 * (1. + torch.max(torch.abs(Pn))):
                P = Pn
                break
            P = Pn
    return torch.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)  # u = -K x


class EnergyLQR:
    """Batched hysteresis-switched energy-shaping / LQR controller."""

    def __init__(self, num_envs, device, env_cls=PendulumEnv):
        g, m, l, dt = env_cls.g, env_cls.m, env_cls.l, env_cls.dt
        self.max_torque = env_cls.max_torque
        self.num_envs = num_envs
        self.device = device
        a_g, b_u = 3. * g / (2. * l), 3. / (m * l ** 2)
        Ad = torch.tensor([[1. + a_g * dt * dt, dt], [a_g * dt, 1.]], dtype=torch.float64)
        Bd = torch.tensor([[b_u * dt * dt], [b_u * dt]], dtype=torch.float64)
        Q = torch.diag(torch.tensor(Q_LQR, dtype=torch.float64))
        R = torch.tensor([[R_LQR]], dtype=torch.float64)
        self.k = solve_dare(Ad, Bd, Q, R).reshape(2)          # [k_th, k_thdot]
        self.mode = torch.zeros(num_envs, dtype=torch.bool, device=device)  # False=energy

    def reset(self, mask):
        self.mode.masked_fill_(mask, False)                   # fresh episodes: energy mode

    def torque(self, th_w, thdot):
        """Torque commands [N m] from the wrapped angle and angular velocity."""
        prev = self.mode.clone()                              # snapshot before in-place flips
        # hysteresis switch, evaluated on the wrapped angle (wrap-safe)
        self.mode |= ~self.mode & (th_w.abs() <= LQR_ENTER)   # promote at entry
        self.mode &= ~(self.mode & (th_w.abs() > LQR_EXIT))   # demote beyond exit
        # LQR about the upright equilibrium, saturated
        u_lqr = (self.k[0] * th_w + self.k[1] * thdot)
        u_lqr = (-u_lqr).clamp(-self.max_torque, self.max_torque)
        # energy shaping relative to the upright: E = 1/2 I thdot^2 + mgl(cos th - 1)
        energy = thdot ** 2 / 3. + PendulumEnv.g * PendulumEnv.l * (torch.cos(th_w) - 1.)
        u_energy = (-K_ENERGY * (energy - ENERGY_TARGET) * thdot).clamp(-self.max_torque, self.max_torque)
        # rescue the pump's at-rest degenerate fixed point near hanging:
        # kick along gravity (sign of sin th) to escape the stable equilibrium
        stuck = ~self.mode & (th_w.abs() > STUCK_TH) & (thdot.abs() < STUCK_THDOT)
        kick = self.max_torque * torch.where(torch.sin(th_w) >= 0., 1., -1.)
        u_energy = torch.where(stuck, kick, u_energy)
        return torch.where(self.mode, u_lqr, u_energy), prev


def _fresh_env(num_envs, seed):
    return PendulumEnv(num_envs=num_envs, backend='cpu', seed=seed)


def _inject(env, th0, thdot0):
    """Deterministic initial states for the offline suites (state injection
    only; the env implementation, reward and success bar are untouched)."""
    env.reset()
    d = env.device
    env.th.copy_(torch.as_tensor(th0, dtype=torch.float64, device=d))
    env.thdot.copy_(torch.as_tensor(thdot0, dtype=torch.float64, device=d))
    env._observe(torch.ones(env.num_envs, dtype=torch.bool, device=d))
    env._update_observation()
    return env.get_observations()


def rollout(env, ctrl, obs, steps=STEPS):
    """Unified batched rollout engine: advances every replica for `steps`,
    tracks per-episode metrics, and records one row per completed episode at
    each done event (deterministic suites complete exactly one episode at the
    horizon, the random suite auto-resets and records every episode)."""
    n, d = env.num_envs, env.device
    keys = ('ret', 'maxspd', 'entries', 'demos', 'last_bad', 'sat')
    m = dict(ret=torch.zeros(n, dtype=torch.float64, device=d),
             maxspd=torch.zeros(n, dtype=torch.float64, device=d),
             entries=torch.zeros(n, dtype=torch.long, device=d),
             demos=torch.zeros(n, dtype=torch.long, device=d),
             last_bad=torch.full((n,), -1, dtype=torch.long, device=d),
             sat=torch.zeros(n, dtype=torch.long, device=d),
             step=torch.zeros(n, dtype=torch.long, device=d))
    rows = {k: [] for k in keys}
    for _ in range(steps):
        o = obs['policy'].double()
        th_w = torch.atan2(o[:, 1], o[:, 0])
        thdot = o[:, 2]
        m['maxspd'] = torch.maximum(m['maxspd'], thdot.abs())
        torque, prev = ctrl.torque(th_w, thdot)
        m['entries'] += (~prev & ctrl.mode).long()
        m['demos'] += (prev & ~ctrl.mode).long()
        action = (torque / env.max_torque).clamp(-1., 1.).reshape(n, env.num_actions)
        m['sat'] += (action.abs() >= 1. - 1e-9).long()[:, 0]
        obs, rew, done, _ = env.step(action)
        m['ret'] += rew.double()
        m['step'] += 1
        bad = (th_w.abs() > SETTLE_TH) | (thdot.abs() > SETTLE_THDOT)
        m['last_bad'] = torch.where(bad, m['step'] - 1, m['last_bad'])
        if bool(done.any()):
            for k in keys:
                rows[k].append(m[k][done].cpu())
            for k, v in m.items():
                v.masked_fill_(done, -1 if k == 'last_bad' else 0)
            ctrl.reset(done)
    out = {k: torch.cat(v) for k, v in rows.items()}
    out['settle'] = out['last_bad'] + 1                     # steps after which in-region
    return out


def _report(name, m, extra=''):
    n = m['ret'].numel()
    succ = (m['ret'] > PendulumEnv.success_return).double()
    settled = m['settle'] <= STEPS - SETTLE_MIN_HOLD
    ent = m['entries'] > 0
    ret = m['ret']
    st = m['settle'][settled]
    print(f'\n== {name} =={extra}')
    print(f'  episodes            : {n}')
    print(f'  success rate        : {succ.mean() * 100:6.2f} %   (return > {PendulumEnv.success_return})')
    print(f'  return mean / median: {ret.mean():8.1f} / {ret.median():8.1f}   '
          f'(min {ret.min():.1f}, p10 {ret.quantile(.1):.1f}, p90 {ret.quantile(.9):.1f})')
    if settled.any():
        print(f'  settle time mean/med: {st.double().mean():6.1f} / {st.double().median():5.1f} steps'
              f'   (settled>= {SETTLE_MIN_HOLD}-step hold: {settled.double().mean() * 100:.2f} %)')
    else:
        print('  settle time         : none settled')
    print(f'  max |thdot| mean/max: {m["maxspd"].mean():5.2f} / {m["maxspd"].max():5.2f} rad/s')
    if ent.any():
        print(f'  switching failures  : {(m["demos"] > 0)[ent].double().mean() * 100:6.2f} % of '
              f'{int(ent.sum())} LQR-entering episodes '
              f'(mean demotions/episode {m["demos"].double().mean():.3f})')
    else:
        print(f'  switching failures  : no LQR entries')
    print(f'  saturated action %  : {(m["sat"].double().sum() / (n * STEPS)) * 100:5.2f} % of steps')


def suite_grid():
    """Swing-up + stabilize from the full initial angle range."""
    th0 = torch.linspace(-math.pi, math.pi, 73)
    td0 = torch.tensor([-1., -.5, 0., .5, 1.])
    TH, TD = torch.meshgrid(th0, td0, indexing='ij')
    env = _fresh_env(TH.numel(), SEED)
    ctrl = EnergyLQR(TH.numel(), env.device)
    m = rollout(env, ctrl, _inject(env, TH.reshape(-1), TD.reshape(-1)))
    _report(f'grid: th0 73 pts in [-pi,pi] x thdot0 in {{-1,-.5,0,.5,1}}', m)
    worst = m['ret'].argsort()[:5]
    th_all = TH.reshape(-1)
    td_all = TD.reshape(-1)
    for i in worst.tolist():
        print(f'    worst: th0={th_all[i]:+.3f} thdot0={td_all[i]:+.2f} ret={m["ret"][i]:.1f} '
              f'settle={int(m["settle"][i])} demos={int(m["demos"][i])}')


def suite_boundary():
    """Initial states near the switching thresholds (enter 0.35 / exit 0.55)."""
    th0 = torch.tensor([.30, .32, .34, .35, .36, .38, .40, .50, .55, .60])
    th0 = torch.cat([th0, -th0])
    td0 = torch.tensor([0., .6, 1.2, 3., -.6, -1.2, -3.])
    TH, TD = torch.meshgrid(th0, td0, indexing='ij')
    env = _fresh_env(TH.numel(), SEED)
    ctrl = EnergyLQR(TH.numel(), env.device)
    m = rollout(env, ctrl, _inject(env, TH.reshape(-1), TD.reshape(-1)))
    _report('boundary: |th0| in {.30..+.60} x |thdot0| in {0,.6,1.2,3}', m)
    fails = (m['ret'] <= PendulumEnv.success_return).nonzero().flatten()
    th_all = TH.reshape(-1)
    td_all = TD.reshape(-1)
    for i in fails.tolist():
        print(f'    fail: th0={th_all[i]:+.3f} thdot0={td_all[i]:+.2f} ret={m["ret"][i]:.1f} '
              f'settle={int(m["settle"][i])} demos={int(m["demos"][i])} entries={int(m["entries"][i])}')


def suite_saturation():
    """Hanging-region (rescue + saturated pumping) and extreme-speed states."""
    cases = [(math.pi, 0.), (-math.pi, 0.)]
    for s in (1., -1.):
        cases += [(math.pi, s), (-math.pi, s)]
    for d in (.05, .12, .3, .5, .8, 1.2):
        cases += [(math.pi - d, 0.), (-(math.pi - d), 0.)]
    cases += [(2.5, 1.), (-2.5, -1.), (2.5, -1.), (-2.5, 1.)]
    cases += [(.3, 8.), (.3, -8.), (-.3, 8.), (0., 8.), (0., -8.)]  # OOD spins
    th0 = torch.tensor([c[0] for c in cases])
    td0 = torch.tensor([c[1] for c in cases])
    env = _fresh_env(len(cases), SEED)
    ctrl = EnergyLQR(len(cases), env.device)
    m = rollout(env, ctrl, _inject(env, th0, td0))
    _report('saturation: hanging/rescue cases + |thdot0|=8 spins (latter OOD)', m)
    for i, (t0, v0) in enumerate(cases):
        tag = 'FAIL' if m['ret'][i] <= PendulumEnv.success_return else 'ok'
        print(f'    th0={t0:+.3f} thdot0={v0:+.2f} | ret={m["ret"][i]:8.1f} '
              f'settle={int(m["settle"][i]):3d} maxspd={m["maxspd"][i]:5.2f} '
              f'demos={int(m["demos"][i])} entries={int(m["entries"][i])} sat={int(m["sat"][i]):3d} {tag}')


def suite_random():
    """Benchmark reset distribution th~U[-pi,pi], thdot~U[-1,1] (auto-resets)."""
    env = _fresh_env(NUM_ENVS_RANDOM, SEED)
    ctrl = EnergyLQR(NUM_ENVS_RANDOM, env.device)
    out = rollout(env, ctrl, env.reset(), steps=STEPS * EPISODES_RANDOM)
    _report(f'random: benchmark reset, {NUM_ENVS_RANDOM} envs x {EPISODES_RANDOM} episodes', out)


if __name__ == '__main__':
    torch.manual_seed(SEED)
    print(f'controller: energy-shaping swing-up + discrete LQR '
          f'(enter {LQR_ENTER} rad / exit {LQR_EXIT} rad, Q=diag{Q_LQR}, R={R_LQR})')
    suite_grid()
    suite_boundary()
    suite_saturation()
    suite_random()
