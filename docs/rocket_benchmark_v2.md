# Rocket Benchmark v2: Replacement Rationale and Protocol

**English** · [简体中文](rocket_benchmark_v2.zh-CN.md)

The old benchmark's Genesis landing-contact configuration had a structural
problem: three flat-bottomed collision bars kept chattering after touching the
platform, and even a gentle engine-off touchdown was bounced back up;
instantaneous ever-success then recorded states that briefly met the threshold
during chatter as successes. This moved scoring away from the task goal "the
rocket lands and settles". v2 fixes contact with spherical pads and scores a
continuous 2-second settle.

## Verification already completed by Claude

On the same batch of 128 random gentle-touchdown initial states, with the engine
off, control surfaces neutral, and the default solver, run for 10 seconds:

| Contact diagnostic | Original three bars | v2 six spherical pads |
|---|---:|---:|
| Lift-off above 1 cm after contact | 108/128 | 0/128 |
| Fatal ending | 88/128 | 0/128 |
| Ever met the instantaneous success condition | 85/128 | 128/128 |

Claude's full first-episode evaluation, 4×128 per column, 512 episodes total,
with the continuous 2-second settle as the primary metric:

| Condition and policy | Development set | Independent set |
|---|---:|---:|
| nominal / seed | 508/512 | 504/512 |
| nominal / PDG | 512/512 | 512/512 |
| shifted / seed | 281/512 | 293/512 |
| shifted / PDG | 290/512 | 314/512 |

Nominal conditions recover normal settling, while thrust, surface, actuator and
wind disturbances still separate control capability, so v2 provides both a
reliable contact foundation and a challenging physical generalization
evaluation. Per-environment contact records, evaluation counts, intervals and
source-file SHA-256 are in [`evidence/rocket_v2/`](evidence/rocket_v2/). This
commit directly reuses Claude's measured data.

## Corresponding locations in main

- **Contact assets**:
  `environments/common/original_pyflyt/models/vehicles/rocket/rocket_v2.urdf`.
  Each leg uses two spherical collision pads of radius 0.025 m, at local y
  coordinates ±0.225 m. Visuals, mass, inertia, lowest-point height and lateral
  support footprint are retained; the collision shape is explicitly upgraded to
  v2. The original `rocket.urdf` is retained, and both assets have SHA-256
  manifests.
- **Physics model**: `environments/common/plant.py` and `rocket_benchmark.py`.
  Per-episode physics disturbance is integrated with an independent counter
  random stream, seed offset 104729.
- **Task scoring**: `environments/rocket/env.py` uses v2, with primary metric
  `ever_stable_2s`. The existing reward, fatal criteria, initial distribution,
  30-dimensional observation, 7-dimensional action and 30-second horizon are
  retained.
- **Evaluation**: `scripts/rocket_benchmark_eval.py` counts complete first
  episodes in 4×128 batches. `eval.py` and `baseline_eval.py` also fix the
  cumulative-episode-count exit bug: each replica contributes only its first
  episode, and later episodes of early-failed replicas do not replace a
  still-running first episode.
- **Training configuration**: `algorithms/train_utils.py` saves the disturbance
  tier and policy rate before training starts. Continuation locks these two
  settings, and cross-configuration experiments use separate directories.

## Continuous-settle metric

At the end of each policy step, check platform contact, speed norm < 0.02 m/s,
angular-rate norm < 0.02 rad/s, combined roll/pitch tilt < 0.1 rad, and no fatal
or out-of-bounds termination. The span between the first and last sample that
continuously satisfies the conditions must reach 2 seconds:
**121 points at 60 Hz, 81 points at 40 Hz**. A contact break or any violated
condition resets the continuous counter. After success, simulation continues to
natural termination or the time limit.

The old instantaneous `ever_success`, terminal single-point
`terminal_single_point`, terminal continuous settle `terminal_stable_2s`,
post-success fatal/out-of-bounds `success_then_failure`, and
post-settle fatal/out-of-bounds `stable_then_failure` are recorded alongside.

## Per-episode physics disturbance

| Parameter | nominal | shifted | stress |
|---|---|---|---|
| Thrust scale | 1 | [0.92, 1.08] | [0.85, 1.15] |
| Surface force/torque scale | 1 | [0.6, 1.4] | [0.3, 1.7] |
| Engine and gimbal time constants (s, sampled independently) | 0.01 | [0.005, 0.03] | [0.005, 0.06] |
| Two-axis gimbal bias (deg, sampled independently) | 0 | [-0.3, 0.3] | [-1, 1] |
| Horizontal steady wind (m/s, uniform direction) | 0 | [0, 5] | [0, 10] |

The thrust scale represents effectiveness change while the fuel-consumption
formula is retained; the surface scale acts on both force and torque. These
parameters are simulator-internal state, and the policy still uses the original
30-dimensional observation. Main's existing `steady_wind` / `dryden` interfaces
are retained; additionally configured wind is superposed on the per-episode wind
and written into the experiment configuration. Masked resets update only the
corresponding replicas. Construction and the first reset each sample once,
matching the research version.

## Version and usage

The data above use the 40 Hz v2 snapshot of the frozen research source
`39e6bb6fd5d0f5432f7405a76946938a469492ce`. Main integration retains the current
40 Hz policy (since 2026-10-27, the same rate as the research snapshot; 50 Hz is
unavailable because it does not divide the 120 Hz control loop), 120 Hz control,
240 Hz physics, and the v4 reward (since 2026-10-28 v5: the same dense shaping
plus a feasible descent-speed corridor penalty; see
[`method/env-rocket-landing.md`](method/env-rocket-landing.md)).
Reports distinguish experiments by version, frequency, source hash, backend and
wind configuration; historical leaderboards retain their original version
labels.

```bash
python scripts/rocket_benchmark_eval.py --backend gpu --variation nominal --out results/rocket-v2-nominal
python scripts/rocket_benchmark_eval.py --backend gpu --variation shifted --out results/rocket-v2-shifted
```

Default development seeds are 1000007, 2000007, 3000007, 4000007; independent
confirmation uses a pre-fixed different `--seeds` set. Reports include
per-episode records, completeness, sample counts, Wilson 95% intervals, initial
states and source hashes. Checkpoint evaluation uses `scripts/eval.py`;
`rocket_pdg_test.py` remains the touchdown-telemetry entry point.
