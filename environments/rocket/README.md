# Rocket

Rocket-Landing (`RocketLandingEnv`, task id `rocket`): an F9-class booster
returning to a landing pad. URDF rigid-body dynamics on the Genesis engine
(240 Hz) with grid fins and engine gimbal, 60 Hz policy; 1800 steps = 30 s
per episode; success = pad contact with |v| < 0.02 m/s, |omega| < 0.02 rad/s
and tilt < 0.1 rad. Success does not truncate: the episode continues and the
settled booster keeps earning the contact stream until the time limit. Task
spec lives in the package docstring [`__init__.py`](__init__.py).

## Files

- `env.py` / `__init__.py` — environment implementation
- `train_rocket.py` — self-contained PPO training entry (constants in-file,
  no CLI arguments); writes `results/rocket/`
- `rocket_pdg_test.py` — powered-descent-guidance (ZEM/PN braking +
  virtual-pad manifold) baseline deep-dive harness

## Baseline

Classical ZEM/ZEV powered-descent guidance (`RocketPDGBaseline`, `zem_zev_pdg`)
under the official protocol (`scripts/baseline_eval.py rocket`,
`results/rocket/baseline.json`): 1024 envs, 1800 env steps, 1024 episodes,
success rate 0.8281, mean episode return 3968.16.

## Run

From the repository root:

```bash
python environments/rocket/train_rocket.py           # training -> results/rocket/
python environments/rocket/rocket_pdg_test.py        # PDG baseline deep dive
python scripts/eval.py rocket           # deterministic checkpoint evaluation
python scripts/baseline_eval.py rocket  # classical baseline, official protocol
python scripts/visualize.py rocket      # checkpoint replay / trajectory plot
```
