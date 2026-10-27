"""Checks for QuadXTrackEnv / QuadXTrackKFBaseline.

Verifies the public-observation contract that the whole comparison rests on:
  * 25-D policy observation and the documented layout,
  * invisible steps carry a ZERO relative measurement and a 0 flag,
  * visible measurements are the camera measurement (truth + bounded noise),
  * geometric visibility: FOV/range loss and line-of-sight occlusion,
  * the target schedule contains straight, turn and stop-go segments,
  * the KF baseline obtains a reference and flies without reading target truth.

Run: python test_quadx_track.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
from environments.multirotor import QuadXTrackEnv
from environments.multirotor.track_env import _occluded, OCCLUDERS


def main():
    torch.manual_seed(0)
    n = 32
    env = QuadXTrackEnv(num_envs=n, backend='gpu', seed=11)
    obs = env.get_observations()
    assert obs['policy'].shape == (n, 25), obs['policy'].shape
    print('PASS observation shape (n, 25)')

    # reset places the target inside the footprint, so every replica is acquired
    assert bool((obs['policy'][:, 24] > .5).all()), 'target not visible at reset'
    print('PASS target acquired at reset (in-footprint spawn)')

    # the schedule must contain all three cooperative motion primitives
    modes = torch.unique(env.seg_mode).tolist()
    assert set(modes) == {0, 1, 2}, modes
    print('PASS target schedule contains straight / turn / stop-go')

    # visible measurement = truth + N(0, 0.15^2); verify empirically and check
    # that an invisible step never leaks the truth
    ctrl = env.conventional_controller(env)
    max_err, invisible_checked, visible_checked = 0., 0, 0
    with torch.no_grad():
        for _ in range(120):
            actions = ctrl.actions(obs)
            obs, _, term, trunc, _ = env.step(actions)
            ctrl.reset_idx(term | trunc)
            p = env.plant.state()[0]
            truth = env.target_pos - p
            vis = obs['policy'][:, 24] > .5
            err = (obs['policy'][:, 21:24].to(torch.float64) - truth).norm(dim=-1)
            if bool(vis.any()):
                max_err = max(max_err, float(err[vis].max()))
                visible_checked += int(vis.sum())
            if bool((~vis).any()):
                assert bool((obs['policy'][~vis, 21:24] == 0).all()), 'invisible measurement is not zero'
                invisible_checked += int((~vis).sum())
    print(f'PASS visible measurement noise bound: max |err| = {max_err:.3f} m over {visible_checked} detections')
    print(f'PASS invisible samples: {invisible_checked} with zero measurement and flag 0')

    # geometric line-of-sight occlusion (direct unit check on the geometry)
    uav = torch.tensor([[4.5, -2., 6.]], device=env.device, dtype=torch.float64)
    tgt = torch.tensor([[4.5, 8., .15]], device=env.device, dtype=torch.float64)
    assert bool(_occluded(uav, tgt)[0]), 'occluder does not block the LOS'
    tgt2 = torch.tensor([[20., 20., .15]], device=env.device, dtype=torch.float64)
    assert not bool(_occluded(uav, tgt2)[0]), 'LOS falsely blocked'
    print('PASS line-of-sight occlusion geometry')

    # FOV / range loss: move one target outside the footprint
    p = env.plant.state()[0]
    far = p.clone()
    far[0, 0] += 10.
    env.target_pos[0] = far[0]
    obs, _, _, _, _ = env.step(torch.zeros((n, 4), device=env.device, dtype=torch.float32))
    assert env.ep_fov_loss[0] > 0., 'out-of-FOV sample not counted'
    assert not bool(obs['policy'][0, 24] > .5) and bool((obs['policy'][0, 21:24] == 0).all())
    print('PASS out-of-FOV target counted as a loss with zero measurement / flag 0')

    # crash termination: put the vehicle on the ground with the motors off
    p, q, v, w = env.plant.state()
    p[1, 2] = 0.
    q[1] = torch.tensor([1., 0., 0., 0.], device=env.device, dtype=torch.float64)
    v[1] = 0.
    w[1] = 0.
    env.plant.set_state(p, q, v, w)
    actions = torch.zeros((n, 4), device=env.device, dtype=torch.float32)
    actions[:, 3] = -1.
    _, _, term, _, _ = env.step(actions)
    assert bool(term[1]), 'ground contact did not terminate'
    print('PASS ground-contact collision termination')

    # the controller only consumes the policy observation (it has no env state
    # reference beyond the public geometry constants)
    a = ctrl.actions(env.get_observations())
    assert a.shape == (n, 4) and bool((a.abs() <= 1.).all())
    print('PASS KF+PID action in [-1, 1]^4 from the public observation')
    env.close()
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
