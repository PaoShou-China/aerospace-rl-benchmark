# Multirotor

QuadX-Hover (`QuadXHoverEnv`, task id `quadx`): a 27 g Crazyflie 2.x
quadrotor holding station at 1 m altitude. URDF rigid-body dynamics on the
Genesis engine (240 Hz) under an inner-loop body-rate PID with a 40 Hz
policy; 400 steps = 10 s per episode. Task spec (action/observation layout,
reward, terminations, difficulty axis) lives in the package docstring
[`__init__.py`](__init__.py).

## Files

- `env.py` / `__init__.py` — environment implementation
- `train_quadx.py` — self-contained PPO training entry (constants in-file,
  no CLI arguments); writes `results/quadx/`
- `test_quadx_se3.py` — SE(3) geometric baseline deep-dive harness

## Run

From the repository root:

```bash
python environments/multirotor/train_quadx.py        # training -> results/quadx/
python environments/multirotor/test_quadx_se3.py     # classical baseline deep dive
python scripts/eval.py quadx            # deterministic checkpoint evaluation
python scripts/baseline_eval.py quadx   # classical baseline, official protocol
python scripts/visualize.py quadx       # checkpoint replay / trajectory plot
```
