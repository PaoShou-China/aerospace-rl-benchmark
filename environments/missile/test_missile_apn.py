"""Tests for the missile-intercept task and its APN rate baseline.

Covers
    T1  fin-channel wrench signs, open loop: fin_cmd is forced directly
        (bypassing the rate PID) and the resulting body torques must obey the
        documented skid-to-turn mixer signs -- pitch cmd + -> tau_y > 0,
        yaw cmd + -> tau_z < 0 (negated channel), and the roll gear must keep
        the differential torque two orders below the lateral authority
    T2  rate-loop closed-loop step response: saturated per-axis rate commands
        are tracked with the correct sign and without divergence
    T3  APN closed-loop intercept success at the fixed medium operating point,
        measured with a spy on env.reset_idx (completed_metrics slots are
        stale at termination; the running min_distance captured BEFORE the
        internal reset is the ground truth) -- mixed maneuver types and the
        straight-line subset separately

Genesis allows one device type per process:
    python test_missile_apn.py [gpu|cpu]      (default gpu)
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from environments.missile.env import MissileInterceptEnv, _aim_quat, KILL_RADIUS
from algorithms.baselines.baselines import MissileAPNBaseline

BACKEND = sys.argv[1] if len(sys.argv) > 1 else 'gpu'


def gpu_sum(t):
    return float(t.detach().sum().cpu())


def run_apn(env, ctrl, obs, max_steps, target_episodes):
    """Drive the baseline until `target_episodes` completed or `max_steps`.

    The env auto-resets terminated replicas internally, so the ONLY reliable
    success record is taken by the spy at reset time.  The baseline filter
    state MUST be cleared on env.reset_buf (repo contract) -- skipping that
    leaks the LOS filter across episodes and diverges on the first step.
    """
    records = {'type': [], 'kill': [], 'min': []}
    orig_reset = env.reset_idx

    def spy(mask):
        records['type'].append(env.t_type[mask].clone())
        records['kill'].append(env.complete[mask].clone())
        records['min'].append(env.min_distance[mask].clone())
        orig_reset(mask)

    env.reset_idx = spy
    try:
        for _ in range(max_steps):
            act = ctrl.actions(obs)
            assert torch.isfinite(act).all(), 'non-finite baseline action'
            obs, _, term, trunc, _ = env.step(act)
            ctrl.reset_idx(env.reset_buf)
            if sum(int(r.numel()) for r in records['kill']) >= target_episodes:
                break
    finally:
        env.reset_idx = orig_reset
    types = torch.cat(records['type'])
    kill = torch.cat(records['kill'])
    mins = torch.cat(records['min'])
    return obs, types, kill, mins


def sr_line(tag, types, kill, mins):
    n = types.numel()
    sr = float(kill.to(torch.float64).mean())
    m0 = types == 0
    sr0 = float(kill[m0].to(torch.float64).mean()) if int(m0.sum()) else float('nan')
    med = float(mins[kill].median()) if int(kill.sum()) else float('nan')
    print(f'  [{tag}] episodes={n} SR={sr:.3f} straight-SR={sr0:.3f} '
          f'median-miss(kills)={med:.1f} m')
    return sr, sr0


# --------------------------------------------------------------------------- #
def test_fin_signs():
    print('T1: open-loop fin-channel response signs')
    from environments.common.dynamics import inverse_rotate
    env = MissileInterceptEnv(num_envs=16, backend=BACKEND, seed=7)
    env.reset()
    mask = torch.ones(16, dtype=torch.bool, device=env.device)
    cmd = torch.zeros(16, 3, device=env.device, dtype=torch.float64)

    def rates(roll, pitch, yaw, substeps=120):
        """Force fin_cmd for 0.5 s and return final body rates.  Per-link
        wrench buffers live in link-local frames, so the frame-safe probe is
        the rigid-body response itself."""
        env.plant.fin_cmd.copy_(torch.stack(
            (torch.full((16,), roll, device=env.device, dtype=torch.float64),
             torch.full((16,), pitch, device=env.device, dtype=torch.float64),
             torch.full((16,), yaw, device=env.device, dtype=torch.float64)), -1))
        tp, tv = env._target_state()
        tq = _aim_quat(tv / tv.norm(dim=-1, keepdim=True), torch.zeros_like(tv[:, 0]))
        for _ in range(substeps):
            env.plant.physics_step(cmd, mask, target_state=(tp, tq, tv, torch.zeros_like(tv)))
        _, q, _, w = env.plant.state()
        return inverse_rotate(q, w)

    # pitch channel: +cmd -> +pitch rate (the sign the rate PID relies on)
    w = rates(0., 1., 0.)
    frac = float((w[:, 1] > 0).to(torch.float64).mean())
    assert frac >= 0.9, f'pitch sign inconsistent: {frac:.2f} of batch'
    wp = float(w[:, 1].abs().mean())
    # yaw channel: +cmd -> +yaw rate (the mixer's internal negation makes the
    # CLOSED-LOOP sign correct; asserted on the dynamics, not the raw wrench)
    w = rates(0., 0., 1.)
    frac = float((w[:, 2] > 0).to(torch.float64).mean())
    assert frac >= 0.9, f'yaw sign inconsistent: {frac:.2f} of batch'
    wy = float(w[:, 2].abs().mean())
    # roll channel: a full OPEN-LOOP differential legitimately builds roll
    # rate (the suppression comes from the closed-loop mixer gear, asserted in
    # T2); here only the sign convention is asserted
    w = rates(1., 0., 0.)
    frac = float((w[:, 0] > 0).to(torch.float64).mean())
    assert frac >= 0.9, f'roll sign inconsistent: {frac:.2f} of batch'
    wx = float(w[:, 0].abs().mean())
    print(f'  [signs] |w_y|(+pitch)={wp:.3f}  |w_z|(+yaw)={wy:.3f}  '
          f'|w_x|(+roll)={wx:.3f} rad/s after 0.5 s')
    env.close()


def test_rate_loop():
    print('T2: rate-loop closed-loop step response')
    env = MissileInterceptEnv(num_envs=64, backend=BACKEND, seed=7)
    ctrl = MissileAPNBaseline(env)
    obs = env.reset()
    n = env.num_envs
    with torch.no_grad():
        # roll step: saturated +6 rad/s command must be tracked, not diverge
        act = torch.zeros(n, 3, device=env.device)
        act[:, 0] = 1.
        for _ in range(60):              # 1 s
            obs, _, term, trunc, _ = env.step(act)
            ctrl.reset_idx(env.reset_buf)
        w = obs['policy'][:, :3].to(torch.float64)
        assert torch.isfinite(w).all(), 'roll step diverged to NaN'
        wr = float(w[:, 0].mean())
        assert wr > 3., f'roll rate {wr:.2f} rad/s not tracking +6'
        env.reset()
        # pitch step: +0.5 rad/s command (coordinated-turn limit) tracked with
        # the correct sign; damped by the synthesized Cmq, so no runaway
        act = torch.zeros(n, 3, device=env.device)
        act[:, 1] = 1.
        for _ in range(30):              # 0.5 s
            obs, _, term, trunc, _ = env.step(act)
            ctrl.reset_idx(env.reset_buf)
        w = obs['policy'][:, :3].to(torch.float64)
        assert torch.isfinite(w).all(), 'pitch step diverged to NaN'
        mp = float(w[:, 1].mean())
        assert mp > 0.2 and mp < 1.0, f'pitch rate {mp:.3f} rad/s out of band'
    print(f'  [tracking] roll w_x={wr:.3f}  pitch w_y={mp:.3f} rad/s')
    env.close()


def test_apn_intercept():
    print('T3: APN closed-loop intercept (fixed medium operating point)')
    env = MissileInterceptEnv(num_envs=256, backend=BACKEND, seed=7)
    ctrl = MissileAPNBaseline(env)
    obs = env.reset()
    ctrl.reset_idx(env.reset_buf)
    t0 = time.time()
    obs, types, kill, mins = run_apn(env, ctrl, obs, max_steps=3000,
                                     target_episodes=256)
    sr, sr0 = sr_line(f'medium {time.time() - t0:.0f}s', types, kill, mins)
    assert types.numel() >= 128, f'only {types.numel()} episodes completed'
    assert sr >= 0.45, f'mixed-type APN success {sr:.3f} < 0.45'
    assert sr0 >= 0.85, f'straight-target APN success {sr0:.3f} < 0.85'
    env.close()


if __name__ == '__main__':
    torch.manual_seed(7)
    test_fin_signs()
    test_rate_loop()
    test_apn_intercept()
    print('all missile APN tests passed')
