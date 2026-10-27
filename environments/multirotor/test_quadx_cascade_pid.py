"""Direction, hover, waypoint and tuning tests for the QuadX cascaded PID baseline.

Covers
    T0  control-direction signs on one control call (hover, +-x, +-y, +-z, yaw)
    T1  hover station keeping at the spawn point (direct plant stepping so the
        fixed setpoint is not consumed by the env's capture logic)
    T2  single-waypoint convergence to a known +x and +y offset
    T3  four-waypoint check (512 envs, seed 7, one episode per replica; the
        official 1024-env record is produced by scripts/baseline_eval.py)
    T4  small gain sweep (256 envs, seed 7; enable with --sweep)

The controller under test is ``QuadXCascadePIDBaseline``
(algorithms/baselines/baselines.py, registry variant ``cascaded_pid``).

Usage:
    python test_quadx_cascade_pid.py [gpu|cpu] [--sweep]
"""
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import torch

from environments.multirotor import QuadXWaypointsEnv
from algorithms.baselines.baselines import QuadXCascadePIDBaseline

BACKEND = next((a for a in sys.argv[1:] if a in ('gpu', 'cpu')), 'gpu')
SWEEP = '--sweep' in sys.argv
NUM_TARGETS = 4
OUT = _ROOT / 'results' / 'quadx'


def gpu_sum(t):
    return float(t.detach().sum().cpu())


def set_target(env, target):
    """Point all four waypoint slots at ``target`` (n, 3) and rewind the index.

    Used by the direct-stepping tests, which never run the env's capture check.
    """
    env.targets.copy_(target[:, None, :].expand(-1, NUM_TARGETS, -1))
    env.target_index.zero_()


def fresh_controller(env, gains=None):
    ctrl = QuadXCascadePIDBaseline(env, gains)
    ctrl.reset_idx(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    return ctrl


def low_level_step(env, ctrl, mask):
    """One agent step through the same controller/physics sequence as env.step,
    without the waypoint capture bookkeeping (fixed-setpoint tests)."""
    act = ctrl.actions(env.get_observations())
    physical = env._scale_actions(act)
    env.action.copy_(physical)
    for _ in range(env.inner_steps):
        env.plant.clear_contacts()
        env.plant.controller(physical, mask)
        for _ in range(env.plant.physics_per_control):
            env.plant.physics_step(physical, mask)
    env._observe(mask)
    env._update_observation()
    env.count += 1
    return act


def attitude_angle(q_xyzw):
    """Smallest rotation angle to identity from [x,y,z,w] quaternions."""
    return 2. * torch.acos(q_xyzw[:, 3].clamp(-1., 1.).abs())


# --------------------------------------------------------------------------- #
def test_direction_signs():
    print('T0: control-direction signs')
    env = QuadXWaypointsEnv(num_envs=8, backend=BACKEND, seed=0)
    n, dev = env.num_envs, env.device
    mask = torch.ones(n, dtype=torch.bool, device=dev)
    p0 = env.plant.drone.get_pos().to(torch.float64).clone()
    v0 = torch.zeros((n, 3), device=dev, dtype=torch.float64)
    w0 = v0.clone()
    # start from a settled level hover (the fresh env observation carries one
    # step of fall velocity, which would bias the collective-direction checks)
    # plant.set_state takes the w-first quaternion; identity is (1, 0, 0, 0)
    q_id = torch.tensor((1., 0., 0., 0.), device=dev, dtype=torch.float64).repeat(n, 1)
    env.plant.set_state(p0, q_id, v0, w0)
    env.plant.zero(mask)
    env._observe(mask)
    env._update_observation()
    obs = env.get_observations()

    def act_for(offset):
        set_target(env, p0 + offset)
        ctrl = fresh_controller(env)
        return ctrl.actions(obs).to(torch.float64)

    hover = act_for(torch.zeros((n, 3), device=dev, dtype=torch.float64))
    checks = [
        ('hover rates', hover[:, :3].abs().max(), lambda x: x < 1e-3),
        ('+x pitch>0', act_for(torch.tensor((1., 0., 0.), device=dev))[:, 1].mean(), lambda x: x > 0.1),
        ('-x pitch<0', act_for(torch.tensor((-1., 0., 0.), device=dev))[:, 1].mean(), lambda x: x < -0.1),
        ('+y roll<0', act_for(torch.tensor((0., 1., 0.), device=dev))[:, 0].mean(), lambda x: x < -0.1),
        ('-y roll>0', act_for(torch.tensor((0., -1., 0.), device=dev))[:, 0].mean(), lambda x: x > 0.1),
        ('+z throttle up', act_for(torch.tensor((0., 0., 1.), device=dev))[:, 3].mean(),
         lambda x: x > float(hover[:, 3].mean()) + 0.02),
        ('-z throttle down', act_for(torch.tensor((0., 0., -1.), device=dev))[:, 3].mean(),
         lambda x: x < float(hover[:, 3].mean()) - 0.02),
    ]
    # positive yaw error -> negative yaw-rate command (fixed reference 0)
    half = math.sin(.25)
    q_yaw = torch.tensor((math.cos(.25), 0., 0., half), device=dev, dtype=torch.float64).repeat(n, 1)
    env.plant.set_state(p0, q_yaw, v0, w0)
    env.plant.zero(mask)
    env._observe(mask)
    env._update_observation()
    obs = env.get_observations()
    checks.append(('yaw>0 -> wz<0', act_for(torch.zeros((n, 3), device=dev, dtype=torch.float64))[:, 2].mean(),
                   lambda x: x < -0.1))

    out = {}
    out['hover_collective'] = float(hover[:, 3].mean())
    for name, value, ok in checks:
        value = float(value)
        out[name] = value
        assert ok(value), f'direction check failed: {name} = {value:+.4f}'
        print(f'  [{name}] {value:+.4f}')
    env.close()
    return out


def test_hover(num_envs=256, steps=400):
    print('T1: hover station keeping')
    env = QuadXWaypointsEnv(num_envs=num_envs, backend=BACKEND, seed=7)
    dev = env.device
    mask = torch.ones(num_envs, dtype=torch.bool, device=dev)
    p0 = env.plant.drone.get_pos().to(torch.float64).clone()
    set_target(env, p0)
    ctrl = fresh_controller(env)
    acts = []
    t0 = time.time()
    for _ in range(steps):
        acts.append(low_level_step(env, ctrl, mask))
    p = env.plant.drone.get_pos().to(torch.float64)
    q = env.plant.drone.get_quat().to(torch.float64)          # wxyz
    q_xyzw = torch.cat((q[:, 1:], q[:, :1]), -1)
    drift = (p - p0).norm(dim=-1)
    att = attitude_angle(q_xyzw)
    thr = torch.stack(acts)[:, :, 3].mean()
    out = dict(steps=steps, num_envs=num_envs,
               pos_drift_mean=float(drift.mean()), pos_drift_max=float(drift.max()),
               att_mean=float(att.mean()), att_max=float(att.max()),
               throttle_action_mean=float(thr), wall=time.time() - t0)
    print(f'  drift mean={out["pos_drift_mean"]:.4f} max={out["pos_drift_max"]:.4f} m | '
          f'att mean={out["att_mean"]:.4f} max={out["att_max"]:.4f} rad | '
          f'throttle action={out["throttle_action_mean"]:+.4f}')
    assert out['pos_drift_max'] < 0.10, f'hover drift {out["pos_drift_max"]:.4f} m >= 0.10'
    assert out['att_max'] < 0.10, f'hover attitude {out["att_max"]:.4f} rad >= 0.10'
    env.close()
    return out


def test_single_waypoint(num_envs=64, steps=400, offset=(2.5, 0., 0.6)):
    print(f'T2: single waypoint {offset}')
    env = QuadXWaypointsEnv(num_envs=num_envs, backend=BACKEND, seed=7)
    dev = env.device
    mask = torch.ones(num_envs, dtype=torch.bool, device=dev)
    p0 = env.plant.drone.get_pos().to(torch.float64).clone()
    off = torch.tensor(offset, device=dev, dtype=torch.float64)
    target = p0 + off
    set_target(env, target)
    ctrl = fresh_controller(env)
    t0 = time.time()
    for _ in range(steps):
        low_level_step(env, ctrl, mask)
    p = env.plant.drone.get_pos().to(torch.float64)
    err = (p - target).norm(dim=-1)
    delta = (p - p0).mean(dim=0)
    out = dict(steps=steps, num_envs=num_envs, offset=list(offset),
               final_err_mean=float(err.mean()), final_err_max=float(err.max()),
               dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]),
               wall=time.time() - t0)
    print(f'  final err mean={out["final_err_mean"]:.4f} max={out["final_err_max"]:.4f} m | '
          f'd=({out["dx"]:+.3f},{out["dy"]:+.3f},{out["dz"]:+.3f}) m')
    assert out['final_err_max'] < 0.20, f'single-waypoint error {out["final_err_max"]:.4f} m >= 0.20'
    for axis in range(3):
        if abs(offset[axis]) > 0.:
            assert (out[('dx', 'dy', 'dz')[axis]] * offset[axis] > 0.), \
                f'wrong travel direction on axis {axis}'
    env.close()
    return out


def run_four_waypoints(num_envs=1024, max_steps=800, seed=7, gains=None):
    env = QuadXWaypointsEnv(num_envs=num_envs, backend=BACKEND, seed=seed)
    ctrl = QuadXCascadePIDBaseline(env, gains)
    ep0 = env.completed_episodes.detach().clone()
    ok0 = env.completed_successes.detach().clone()
    ret0 = env.completed_returns.detach().clone()
    obs = env.get_observations()
    steps, t0 = 0, time.time()
    with torch.no_grad():
        while steps < max_steps and gpu_sum(env.completed_episodes - ep0) < num_envs:
            act = ctrl.actions(obs)
            obs, _, term, trunc, _ = env.step(act)
            ctrl.reset_idx(term | trunc)
            steps += 1
    wall = time.time() - t0
    episodes = gpu_sum(env.completed_episodes - ep0)
    successes = gpu_sum(env.completed_successes - ok0)
    rets = gpu_sum(env.completed_returns - ret0)
    wps = gpu_sum(env.completed_metrics['waypoints_reached']) / max(episodes, 1.)
    out = dict(num_envs=num_envs, seed=seed, env_steps=steps, episodes=episodes,
               successes=successes, success_rate=successes / max(episodes, 1.),
               mean_return=rets / max(episodes, 1.), waypoints_reached=wps,
               crashes=gpu_sum(env.completed_metrics['crashes']),
               out_of_bounds=gpu_sum(env.completed_metrics['out_of_bounds']),
               timeouts=gpu_sum(env.completed_metrics['timeouts']),
               sat_acc=float(ctrl.sat_acc.sum() / (steps * num_envs)),
               sat_rate=float(ctrl.sat_rate.sum() / (steps * num_envs * 3)),
               sat_thrust=float(ctrl.sat_thrust.sum() / (steps * num_envs)),
               wall=wall, gains=gains)
    env.close()
    return out


def test_four_waypoints(num_envs=512, max_steps=800):
    print('T3: four-waypoint benchmark')
    out = run_four_waypoints(num_envs=num_envs, max_steps=max_steps)
    print(f'  episodes={out["episodes"]:.0f} success={out["success_rate"]:.4f} '
          f'wp/reached={out["waypoints_reached"]:.3f} steps={out["env_steps"]} | '
          f'crash={out["crashes"]:.0f} oob={out["out_of_bounds"]:.0f} '
          f'timeout={out["timeouts"]:.0f} | sat acc={out["sat_acc"]:.4f} '
          f'rate={out["sat_rate"]:.4f} thrust={out["sat_thrust"]:.4f} | '
          f'wall={out["wall"]:.1f}s')
    assert out['success_rate'] >= 0.90, f'waypoint success {out["success_rate"]:.3f} < 0.90'
    assert out['waypoints_reached'] >= 3.9, f'mean waypoints {out["waypoints_reached"]:.2f} < 3.9'
    return out


# --------------------------------------------------------------------------- #
# T4: small gain sweep around the class defaults
# --------------------------------------------------------------------------- #
SWEEP_GAINS = [
    ('default', {}),
    ('kd2p5_undersaturated', dict(kd_pos=2.5)),   # original tuning: dome exits
    ('kp4_kd4', dict(kp_pos=4.0, kd_pos=4.0)),
    ('a_max5', dict(a_max=5.0)),
]


def test_tuning(num_envs=256, max_steps=800):
    print('T4: small PID gain sweep')
    rows = []
    for tag, gains in SWEEP_GAINS:
        out = run_four_waypoints(num_envs=num_envs, max_steps=max_steps, gains=gains)
        out['tag'] = tag
        rows.append(out)
        print(f'  [{tag}] success={out["success_rate"]:.4f} wp={out["waypoints_reached"]:.3f} '
              f'steps={out["env_steps"]} return={out["mean_return"]:.2f} wall={out["wall"]:.1f}s')
    return rows


if __name__ == '__main__':
    torch.manual_seed(7)
    report = {'backend': BACKEND,
              'signs': test_direction_signs(),
              'hover': test_hover(),
              'single_waypoint_x': test_single_waypoint(),
              'single_waypoint_y': test_single_waypoint(num_envs=32, steps=300,
                                                        offset=(0., 2.5, 0.6)),
              'four_waypoints': test_four_waypoints()}
    if SWEEP:
        report['tuning'] = test_tuning()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cascade_pid_tests.json').write_text(json.dumps(report, indent=2))
    print('all quadx cascaded PID tests passed')
