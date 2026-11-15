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

## Measured wind for the full-scale landing environment

Select a weather replay through the ordinary environment constructor. Training and
checkpoint evaluation can use the same `env_kwargs`. All measurements ship with
the repository. The default constructor continues to use calm air.

```python
from environments.rocket import RocketLandingEnv

env = RocketLandingEnv(
    num_envs=8, backend="cpu", seed=7, benchmark_version="v1",
    aero="realistic", measured_wind="kxmr-train",
)
```

| `measured_wind` | Weather applied to the plant |
|---|---|
| `kxmr-train`, `kxmr-dev`, `kxmr-test` | Cape Canaveral soundings from 2018–2021 (361), 2022–2023 (156), or 2024 (54). Each replica receives one profile per episode. |
| `cper-YYYY-MM-DD` | One archived minute of Colorado tower wind with a regional upper-air profile. All replicas start at the same UTC instant. Each restarted replica rewinds its own clock. |
| `cper-surface-YYYY-MM-DD` | The same replay with the experimental near-ground horizontal speed correction described below. |

The [weather source manifest](weather/sources.json) lists the eight CPER dates,
the UTC windows and the upper-air source for each, as well as the annual KXMR
download receipts. For example, use `cper-2025-04-21` to replay the minute
beginning at 21:11 UTC. To train using the existing PPO entry, set
`WIND["measured_wind"] = "kxmr-train"` in `train_rocket.py` and select a new
output directory for the run. Use the test split explicitly when evaluating held-out weather.

**KXMR.** The archive retains soundings with surface speed at most 6 m/s, a peak
of at least 12 m/s below or at 450 m, and a wind-vector difference of at least
9 m/s from the surface within that layer. Wind directions are converted from
meteorological bearings to east/north components in m/s. The replay interpolates
the reported levels to a 5 m grid through 1,520 m and holds the top value above
that height. Soundings supply horizontal wind. Setting `dryden=0.3` adds the
synthetic gust setting from the coastal experiment.

**CPER.** The NEON sonic anemometer at 8.32 m supplies 1,200 three-axis samples
per minute. The simulation uses the most recent 20 Hz sample at the policy
clock rate. The tower's horizontal wind blends into the first regional wind
level, then follows the upper-air grid. The vertical component tapers to zero at
the ground and at the first upper-air level. KCYS radar and NOAA HRRR analysis
supply different upper-air records across dates. The optional surface correction
scales horizontal speed by 0.625677 at and below 1.5 m and linearly returns to
1 at the mast. This factor was fitted to five-minute speeds at the nearby Nunn
USCRN station and the tower on six 2024 dates. It estimates near-ground exposure
and does not measure instantaneous wind at the landing pad.

Both modes apply wind at the simulated vehicle's height through the existing
aerodynamics. The rocket policy keeps its 30 observations. These presets describe
external weather only. For paired comparisons use the same version, weather,
seed and replica count.
