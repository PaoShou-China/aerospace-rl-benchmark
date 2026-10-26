# Multirotor

Two independent waypoint environments share the Crazyflie/Genesis physics and
PPO utilities. Select the scenario by running its training file; no sampling
mode command-line parameter is used.

```text
multirotor/
  random_waypoint/
    __init__.py
    env.py               # full original spatial waypoint environment
    train_quadx.py       # full spatial training configuration and entry
    test_quadx_se3.py    # original spatial-task SE(3) tests
  splits/
    __init__.py
    env.py               # full Split-S waypoint environment
    train.py             # full training configuration and entry
    split_s.py           # original-scale gate coordinates and sampling
```

| Scenario | Targets | Out-of-bounds radius | Training output |
| --- | --- | --- | --- |
| `random_waypoint` | Original polar sampling, radius 1..4.5 m, altitude >=0.4 m | 5 m | `results/quadx_random_waypoint/` |
| `splits` | Random 4 consecutive gates, original coordinates, Gate7 wraps to Gate1 | 15 m | `results/quadx/` |

Both use 37-D observations, the existing progress/waypoint reward, and 800
policy steps (20 s). The spatial task uses a 0.2 m capture radius. Split-S uses
QuadXSplitsEnv with directed aperture crossing and swept frame collisions;
full_track=True enables fixed Gate1..Gate7 tests, with optional laps. Its
frame/clearance geometry is an approximation. Spatial sampling is not
volume-uniform. Each training entry owns its constants, including wind settings.

From the repository root:

```powershell
python environments/multirotor/random_waypoint/train_quadx.py
python environments/multirotor/splits/train.py
python environments/multirotor/random_waypoint/test_quadx_se3.py
```

The existing `from environments.multirotor import QuadXWaypointsEnv`,
`scripts/train.py quadx`, `scripts/eval.py quadx`, and video/baseline scripts
continue to use the Split-S environment. For spatial-task evaluation, import
`environments.multirotor.random_waypoint.env.QuadXWaypointsEnv` explicitly;
the generic quadx evaluation entry does not switch scenario based on output path.

Each training file retains `CLEAN=True` by default, which clears old training
artifacts in its own output directory for a fresh run. Model weights and
existing result directories are not changed by this source reorganization.
