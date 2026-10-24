"""Tests for the quadx SE(3) geometric baseline (Lee, Leok & McClamroch 2010)
on the QuadX-Waypoints task.

Covers
    T0  wrench->rotor allocation exactness (no simulation)
    T1  undisturbed benchmark waypoint navigation + batch-size generality
    T2  varied initial conditions (manual spawn-noise DR knobs + injected
        positions/attitudes via plant.set_state, env code untouched)
    T3  action saturation: recoverable gust recovery, a 90 deg bank
        fall-and-catch whose transient saturates the rate-command window in
        closed loop, and a synthetic hostile observation that exercises every
        limiting path (total thrust, per-rotor pwm, rate commands)
    T4  partial batch reset: manual reset_idx on half the batch mid-episode,
        trim-integral clearing, and re-convergence of both halves

Each test reports success rate (all 4 waypoints reached), waypoints reached
and action saturation ratios; failures raise AssertionError (exit code 1).
Genesis allows one device type per process, so run the CPU pass in a separate
invocation:
    python test_quadx_se3.py [gpu|cpu]      (default gpu)
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import math

import torch

from environments.multirotor import QuadXWaypointsEnv
from algorithms.baselines.baselines import QuadXSE3Baseline

BACKEND = sys.argv[1] if len(sys.argv) > 1 else 'gpu'
EPS = 1e-6
NUM_TARGETS = 4


def quat_attitude(q_xyzw):
    """Smallest rotation angle to identity and yaw angle from [x,y,z,w] quats."""
    qw = q_xyzw[:, 3].clamp(-1., 1.)
    angle = 2. * torch.acos(qw.abs())
    yaw = torch.atan2(2. * (qw * q_xyzw[:, 2] + q_xyzw[:, 0] * q_xyzw[:, 1]),
                      1. - 2. * (q_xyzw[:, 1] ** 2 + q_xyzw[:, 2] ** 2))
    return angle, yaw


def run(env, ctrl, obs, steps, record_last=50):
    """Run the controller for `steps` agent steps; return per-phase metrics.

    pos_err is tracked against the ACTIVE waypoint each step (the target
    advances on capture, so steady-state values are diagnostics, not asserts).
    """
    pos_errs, att_errs, yaw_errs = [], [], []
    wp_count = torch.zeros(env.num_envs, device=env.device)
    prev_index = env.target_index.clone()
    with torch.no_grad():
        for k in range(steps):
            act = ctrl.actions(obs)
            assert torch.isfinite(act).all(), 'non-finite controller action'
            obs, _, term, trunc, _ = env.step(act)
            rb = term | trunc
            ctrl.reset_idx(rb)
            wp_count += (env.target_index - prev_index).clamp_min(0).to(torch.float64)
            prev_index = env.target_index.clone()
            o = obs['policy'].to(torch.float64)
            p_err = (o[:, 10:13] - env.current_target.to(torch.float64)).norm(dim=-1)
            att, yaw = quat_attitude(o[:, 3:7])
            pos_errs.append(p_err)
            att_errs.append(att)
            yaw_errs.append(yaw.abs())
    pos = torch.stack(pos_errs)
    att = torch.stack(att_errs)
    yaw = torch.stack(yaw_errs)
    n = min(record_last, steps)
    return dict(
        steady_pos=float(pos[-n:].mean()), steady_att=float(att[-n:].mean()),
        steady_yaw=float(yaw[-n:].mean()),
        max_pos=float(pos.max()), max_att=float(att.max()),
        waypoints_reached=float(wp_count.mean()),
        pos_errs=pos, att_errs=att, obs=obs)


def sat_fractions(ctrl, steps, n):
    """Cumulative saturation counters -> per-step fractions."""
    return dict(total=float(ctrl.sat_total.sum() / (steps * n)),
                rotor=float(ctrl.sat_rotor.sum() / (steps * n * 4)),
                rate=float(ctrl.sat_rate.sum() / (steps * n * 3)))


def sat_snapshot(ctrl):
    return (ctrl.sat_total.clone(), ctrl.sat_rotor.clone(), ctrl.sat_rate.clone())


def sat_delta(ctrl, snap, steps, n):
    """Per-step saturation fractions within one test phase."""
    t0, r0, w0 = snap
    return dict(total=float((ctrl.sat_total - t0).sum() / (steps * n)),
                rotor=float((ctrl.sat_rotor - r0).sum() / (steps * n * 4)),
                rate=float((ctrl.sat_rate - w0).sum() / (steps * n * 3)))


def gpu_sum(t):
    return float(t.detach().sum().cpu())


def episode_outcome(env, episodes_before, successes_before):
    """Episode deltas vs the cumulative SUMS captured earlier."""
    return (gpu_sum(env.completed_episodes) - episodes_before,
            gpu_sum(env.completed_successes) - successes_before)


def metrics_line(tag, episodes, success, run_stats, sat):
    print(f'  [{tag}] episodes={episodes} success={success / max(episodes, 1):.3f} '
          f'wp/reached={run_stats["waypoints_reached"]:.2f} '
          f'steady pos={run_stats["steady_pos"]:.4f} m att={run_stats["steady_att"]:.4f} rad | '
          f'max pos={run_stats["max_pos"]:.3f} max att={run_stats["max_att"]:.3f} | '
          f'sat total={sat["total"]:.4f} rotor={sat["rotor"]:.4f} rate={sat["rate"]:.4f}')
    return success / max(episodes, 1)


# --------------------------------------------------------------------------- #
def test_allocation_exactness():
    """Wrench -> rotor allocation must reproduce (f, M) exactly before clipping."""
    env = QuadXWaypointsEnv(num_envs=16, backend=BACKEND, seed=0)
    ctrl = QuadXSE3Baseline(env)
    g = torch.Generator().manual_seed(3)
    f = (torch.rand(16, 1, generator=g) * 1.2 + .05).double().to(env.device)
    M = ((torch.rand(16, 3, generator=g) * 2 - 1) * 5e-3).double().to(env.device)
    # allocation gains exactly as assembled in QuadXSE3Baseline.actions()
    w_alloc = torch.cat((ctrl.rotor_xy / (4. * ctrl.ARM),
                         ctrl.mixer_t[:, 2:3] / (4. * ctrl.KM_KF)), -1)   # (4, 3)
    # physical moment arms the plant realizes: arm_y = d*ay, -arm_x = -d*ax, c*spin
    w_arm = torch.cat((ctrl.rotor_xy * ctrl.ARM,
                       ctrl.mixer_t[:, 2:3] * ctrl.KM_KF), -1)           # (4, 3)
    t = f * .25 + M @ w_alloc.T                                          # (n, 4)
    f_rec = t.sum(-1, keepdim=True)
    M_rec = t @ w_arm
    assert (f_rec - f).abs().max() < 1e-12, 'thrust allocation mismatch'
    assert (M_rec - M).abs().max() < 1e-12, 'torque allocation mismatch'
    env.close()
    print('  [allocation] wrench->rotor exactness ok')


def test_hostile_observation(ctrl, env):
    """Synthetic state far outside the flight envelope: every limiter engages."""
    n = env.num_envs
    snap = sat_snapshot(ctrl)
    o = torch.zeros(n, 37, device=env.device, dtype=torch.float64)
    o[:, 0] = 400.                                      # huge body rate x
    o[:, 1:3] = torch.tensor((-15., 8.))
    o[:, 3] = 1.                                        # 180 deg roll (x,y,z,w)
    o[:, 7:10] = torch.tensor((-50., 0., -50.))         # body-frame velocity: under
                                                        # the flipped attitude this
                                                        # maps to +50 m/s climb, so
                                                        # F_des.b3 > f_max on call 1
    o[:, 10:13] = torch.tensor((9., 0., 2.))            # position
    for _ in range(8):
        act = ctrl.actions({'policy': o.to(torch.float32)})
    assert torch.isfinite(act).all() and act.abs().max() <= 1. + EPS
    sat = sat_delta(ctrl, snap, 8, n)
    assert sat['total'] > 0, 'total-thrust limiter never engaged'
    assert sat['rotor'] > 0, 'per-rotor limiter never engaged'
    assert sat['rate'] > 0, 'rate-command limiter never engaged'
    ctrl.reset_idx(torch.ones(n, dtype=torch.bool, device=env.device))
    print(f'  [hostile-obs] actions bounded, over 8 calls x {n}: '
          f'sat total={sat["total"]:.3f} rotor={sat["rotor"]:.3f} rate={sat["rate"]:.3f}')


# --------------------------------------------------------------------------- #
def test_waypoint_navigation():
    print('T1: undisturbed benchmark waypoint navigation')
    for n in (1024, 7, 1):
        env = QuadXWaypointsEnv(num_envs=n, backend=BACKEND, seed=7)
        ctrl = QuadXSE3Baseline(env)
        ep0, sc0 = gpu_sum(env.completed_episodes), gpu_sum(env.completed_successes)
        obs = env.get_observations()
        t0 = time.time()
        st = run(env, ctrl, obs, 800)
        wall = time.time() - t0
        episodes, success = episode_outcome(env, ep0, sc0)
        sr = metrics_line(f'n={n}', episodes, success, st, sat_fractions(ctrl, 800, n))
        assert torch.isfinite(st['pos_errs']).all()
        if n > 7:
            assert sr >= 0.90, f'benchmark waypoint success {sr:.3f} < 0.90'
            assert st['waypoints_reached'] >= 3.9, \
                f'mean waypoints reached {st["waypoints_reached"]:.2f} < 3.9'
        env.close()
    print(f'  [n=1024] wall={wall:.1f}s')


def test_varied_initial():
    print('T2: varied initial conditions')
    # (a) spawn noise via the env's manual domain-randomization knobs
    env = QuadXWaypointsEnv(num_envs=1024, backend=BACKEND, seed=7)
    env.spawn_xy_noise, env.spawn_z_noise = .5, .25
    env.spawn_vel_noise, env.spawn_att_noise = .5, .35
    ctrl = QuadXSE3Baseline(env)
    ep0, sc0 = gpu_sum(env.completed_episodes), gpu_sum(env.completed_successes)
    st = run(env, ctrl, env.get_observations(), 800)
    episodes, success = episode_outcome(env, ep0, sc0)
    sr = metrics_line('dr-noise', episodes, success, st, sat_fractions(ctrl, 800, 1024))
    assert sr >= 0.85, f'dr-noise waypoint success {sr:.3f} < 0.85'
    assert st['waypoints_reached'] >= 3.8, \
        f'mean waypoints reached {st["waypoints_reached"]:.2f} < 3.8'
    env.close()

    # (b) injected positions / attitudes (plant.set_state; env model untouched)
    env = QuadXWaypointsEnv(num_envs=64, backend=BACKEND, seed=7)
    ctrl = QuadXSE3Baseline(env)
    n = env.num_envs
    dev = env.device
    grid_xy = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                            [.8, .8], [-.8, -.8], [.8, -.8], [-.8, .8]], device=dev)
    p0 = torch.cat((grid_xy, torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
                                           device=dev)), -1).repeat(8, 1)
    p0[:, 2] = torch.tensor([.8, 1.3, 1.6, .7, 1.0, 1.2, .9, 1.4] * 8, device=dev)
    rpy = torch.zeros(n, 3, device=dev)
    rpy[:, 0] = torch.tensor([.5, -.5, .3, -.3, 0., 0., 0., 0.] * 8, device=dev)
    rpy[:, 1] = torch.tensor([0., 0., .4, -.4, .5, -.5, 0., 0.] * 8, device=dev)
    rpy[:, 2] = torch.tensor([0., 0., 0., 0., 0., 0., 2.5, -2.5] * 8, device=dev)
    q0 = _rpy_to_quat(rpy, dev)
    mask = torch.ones(n, dtype=torch.bool, device=dev)
    env.plant.set_state(p0, q0, torch.zeros(n, 3, device=dev), torch.zeros(n, 3, device=dev))
    env.plant.zero(mask)
    ctrl.reset_idx(mask)
    env._observe(mask)
    obs = env.get_observations()
    st = run(env, ctrl, obs, 800)
    episodes, success = episode_outcome(env, 0, 0)
    sr = metrics_line('injected IC', episodes, success, st, sat_fractions(ctrl, 800, n))
    assert sr >= 0.85, f'injected-IC success {sr:.3f} < 0.85'
    env.close()


def test_saturation():
    print('T3: action saturation')
    env = QuadXWaypointsEnv(num_envs=64, backend=BACKEND, seed=7)
    ctrl = QuadXSE3Baseline(env)
    n, dev = env.num_envs, env.device
    mask = torch.ones(n, dtype=torch.bool, device=dev)
    identity = torch.tensor([1., 0., 0., 0.], device=dev).repeat(n, 1)   # wxyz for set_state

    # (a) recoverable gusts: attitude-limited braking stays inside the 5 m
    # dome (|p| > 5 terminates), then waypoint navigation resumes
    snap = sat_snapshot(ctrl)
    v0 = torch.tensor([[-4., 0., 0.], [4., 0., -1.], [0., -3.5, -2.], [0., 3., 1.5],
                       [-3., -2.5, 0.], [2.5, 2.5, -1.5], [-3.5, 0., 0.], [0., 0., -2.5]] * 8,
                      device=dev)
    p0 = torch.tensor([0., 0., 1.3], device=dev).repeat(n, 1)
    p0[:, 0] = 0.3
    env.plant.set_state(p0, identity, v0, torch.zeros(n, 3, device=dev))
    env.plant.zero(mask)
    ctrl.reset_idx(mask)
    env._observe(mask)
    st = run(env, ctrl, env.get_observations(), 800)
    episodes, success = episode_outcome(env, 0, 0)
    sr = metrics_line('gust', episodes, success, st, sat_delta(ctrl, snap, 400, n))
    assert torch.isfinite(st['pos_errs']).all(), 'gust run produced NaN'
    assert sr >= 0.85, f'gust recovery success {sr:.3f} < 0.85'
    ep_a, sc_a = episodes, success

    # (b) 90 deg bank from 2.8 m: during the fall-and-catch the low collective
    # steepens the pwm slope so the recovery transient drives the rate command
    # past +-pi, then the controller re-catches and resumes waypoint navigation
    snap = sat_snapshot(ctrl)
    tilt = torch.tensor([.70710678, .70710678, 0., 0.], device=dev).repeat(n, 1)
    env.plant.set_state(torch.tensor([0.3, 0., 2.8], device=dev).repeat(n, 1), tilt,
                        torch.zeros(n, 3, device=dev), torch.zeros(n, 3, device=dev))
    env.plant.zero(mask)
    ctrl.reset_idx(mask)
    env._observe(mask)
    st = run(env, ctrl, env.get_observations(), 800)
    episodes, success = episode_outcome(env, ep_a, sc_a)
    sat = sat_delta(ctrl, snap, 400, n)
    sr = metrics_line('tilt-inversion', episodes, success, st, sat)
    assert sat['rate'] > 0, 'rate-command limiter never engaged in tilt-inversion test'
    assert sr >= 0.85, f'tilt-inversion success {sr:.3f} < 0.85'

    # (c) synthetic hostile state far outside the envelope: every limiter path
    test_hostile_observation(ctrl, env)
    env.close()


def test_partial_reset():
    print('T4: partial batch reset mid-episode')
    env = QuadXWaypointsEnv(num_envs=128, backend=BACKEND, seed=7)
    ctrl = QuadXSE3Baseline(env)
    n, dev = env.num_envs, env.device
    obs = env.get_observations()
    run(env, ctrl, obs, 100, record_last=1)
    trim_before = ctrl.trim.clone()
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    mask[::2] = True
    env.reset_idx(mask)
    ctrl.reset_idx(mask)
    assert bool((ctrl.trim[mask] == 0).all()), 'trim not cleared on manual reset'
    assert bool((ctrl.trim[~mask] == trim_before[~mask]).all()), 'trim disturbed on live half'
    st = run(env, ctrl, env.get_observations(), 800)
    episodes, success = episode_outcome(env, 0, 0)
    sr = success / max(episodes, 1)
    print(f'  [partial-reset] episodes={episodes} success={sr:.3f} | '
          f'wp/reached={st["waypoints_reached"]:.2f}')
    assert sr >= 0.85, f'partial-reset success {sr:.3f} < 0.85'
    env.close()


def _rpy_to_quat(rpy, dev):
    """ZYX euler [r,p,y] -> quats.  Returns [w,x,y,z] for plant.set_state."""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = torch.cos(r * .5), torch.sin(r * .5)
    cp, sp = torch.cos(p * .5), torch.sin(p * .5)
    cy, sy = torch.cos(y * .5), torch.sin(y * .5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack((qw, qx, qy, qz), -1)


if __name__ == '__main__':
    torch.manual_seed(7)
    test_allocation_exactness()
    test_waypoint_navigation()
    test_varied_initial()
    test_saturation()
    test_partial_reset()
    print('all quadx SE(3) baseline tests passed')
