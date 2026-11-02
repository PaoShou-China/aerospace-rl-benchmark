# Rocket

Rocket-Landing (`RocketLandingEnv`, task id `rocket`): an F9-class booster
returning to a landing pad. URDF rigid-body dynamics on the Genesis engine
(240 Hz) with grid fins and engine gimbal, 40 Hz policy; 1200 steps = 30 s
per episode. Success is the v2 benchmark metric `ever_stable_2s`: pad contact
with |v| < 0.02 m/s, |omega| < 0.02 rad/s and tilt < 0.1 rad held for 2 s
(81 policy samples at 40 Hz, 121 at 60 Hz); the instantaneous contact event
still drives the settle reward. Success does not truncate: the episode
continues and the settled booster keeps earning the contact stream until the
time limit. Task spec lives in the package docstring [`__init__.py`](__init__.py).

## Files

- `env.py` / `__init__.py` — environment implementation
- `train_rocket.py` — self-contained PPO training entry (constants in-file,
  no CLI arguments); writes `results/rocket/`
- `rocket_pdg_test.py` — powered-descent-guidance (ZEM/PN braking +
  virtual-pad manifold) baseline deep-dive harness

## Baseline

Classical ZEM/ZEV powered-descent guidance (`RocketPDGBaseline`, variant key
`zem_zev_pdg`) under the official protocol (`scripts/baseline_eval.py rocket`):

| Record | Replicas / seed | Steps | Metric | Success | Mean return |
|---|---|---|---|---|---|
| `results/rocket/baseline.json` | 128 / seed 13 | 1200 | `ever_stable_2s` | 100% | 903.47 |
| `results/rocket/baselines/zem_zev_pdg_seed7.json` | 256 / seed 7 | 1200 | `ever_stable_2s` | 100% | 907.31 |
| `results/rocket/baselines/pd_landing_seed7.json` | 256 / seed 7 | 1200 | `ever_stable_2s` | 100% | 22.34 |
| `results/rocket/baselines/eguidance_seed7.json` | 256 / seed 7 | 1200 | `ever_stable_2s` | 34.4% | −3332.01 |

All shipped records were produced at the current 40 Hz policy rate. The old
60 Hz numbers (success 0.8281, return 3968.16) are historical and not
comparable with the current table.

## Run

From the repository root:

```bash
python environments/rocket/train_rocket.py           # training -> results/rocket/
python environments/rocket/rocket_pdg_test.py        # PDG baseline deep dive
python scripts/eval.py rocket           # deterministic checkpoint evaluation
python scripts/baseline_eval.py rocket  # classical baseline, official protocol
python scripts/visualize.py rocket      # checkpoint replay / trajectory plot
```
