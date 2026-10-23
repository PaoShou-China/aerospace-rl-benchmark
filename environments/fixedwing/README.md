# Fixedwing

Fixedwing-Waypoints (`FixedwingWaypointsEnv`, task id `fixedwing`): a 2.35 kg
fixed-wing UAV capturing 4 waypoints. URDF rigid-body dynamics on the Genesis
engine (240 Hz) with blade-element aerodynamics (incl. stall) and a 30 Hz
policy; 3600 steps = 120 s per episode; success = 4/4 waypoints captured
(< 2 m). Task spec lives in the package docstring [`__init__.py`](__init__.py).

## Files

- `env.py` / `__init__.py` — environment implementation
- `train_fixedwing.py` — self-contained PPO training entry (constants
  in-file, no CLI arguments); writes `results/fixedwing/`
- `eval_fixedwing_autopilot.py` — L1 + TECS autopilot deep-dive harness
  (scenario grid, per-scenario breakdown; serialized parameters in
  `algorithms/baselines/fixedwing_autopilot.json`)

## Run

From the repository root:

```bash
python environments/fixedwing/train_fixedwing.py             # training -> results/fixedwing/
python environments/fixedwing/eval_fixedwing_autopilot.py    # autopilot deep dive
python scripts/eval.py fixedwing                # deterministic checkpoint evaluation
python scripts/baseline_eval.py fixedwing       # classical baseline, official protocol
python scripts/visualize.py fixedwing           # checkpoint replay / trajectory plot
```
