# Missile

Development placeholder. No missile environment, task script or result exists
yet in this repository. Contents below are split into **Environment** and
**Algorithm**, each divided into what already exists (reusable as-is) and
what is planned.

## 1. Environment

### 1.1 Existing (shared, reusable as-is)

| Piece | Location | Role |
|---|---|---|
| Env contract | `multirotor/`, `fixedwing/`, `rocket/`, `satellite/` | batched `step`/`reset`/buffer protocol all tasks share |
| Vehicle plant | `environments/common/plant.py` | `TensorPlant` (physics, contacts, warmup), `TensorRandom` (per-replica RNG streams) |
| Dynamics & aero models | `environments/common/dynamics.py`, `environments/common/aerodynamics.py` | batched SI-unit rigid-body model, lifting-surface model (incl. stall) |
| Frozen reference assets | `environments/common/consistency.py`, `environments/common/original_pyflyt/` | URDF/YAML models, inertial parsing, SHA-256 manifests |
| Classical GNC baselines (existing tasks) | `algorithms/baselines/baselines.py` | one controller per existing task, same action convention as the learned policy |

### 1.2 Planned (missile-specific, to be added)

| Piece | Location | Role |
|---|---|---|
| Task package | `missile/__init__.py` | task docstring: action/obs/reward/termination spec |
| Environment | `missile/env.py` | tensorized missile env, same step/reset/buffer contract |
| Deep-dive harnesses | `missile/eval_missile_*.py` | scenario grids, per-scenario breakdowns |
| Generic-entry registration | `scripts/train.py`, `scripts/eval.py`, `scripts/baseline_eval.py`, `scripts/sweep_tune.py` | task id + env class |
| Results directory | `results/missile/` | `run.json`, `baseline.json`, harness reports |
| Documentation | `missile/README.md` (replace this placeholder), root `README.md` tables | task summary, run commands |

## 2. Algorithm

### 2.1 Existing (shared, reusable as-is)

| Piece | Location | Role |
|---|---|---|
| PPO (on-policy arm) | `algorithms/train_utils.py` | shared `run_training()` via rsl-rl; identical schedule across tasks |
| FlashSAC (off-policy arm) | `algorithms/flashsac/flash_sac.py` | norm-controlled SAC port on the batched protocol |
| FlashSAC trunk for PPO | `algorithms/ppo/ppo_flashnet.py` | `arch='flash_residual'` A/B arm, architecture-matched |
| BC warm start | `algorithms/warmstart/bc_warmstart.py` | behavior-cloning warm start from the classical baseline |
| Classical GNC baselines | `algorithms/baselines/baselines.py`, `fixedwing_autopilot.json` | non-learned reference controllers (PN/PDG/LQR/autopilot), official `baseline_eval` protocol |
| Sweep harness | `scripts/sweep_tune.py` | small hyperparameter sweep entry |

### 2.2 Planned (missile-specific, to be added)

| Piece | Location | Role |
|---|---|---|
| Training entry | `missile/train_missile.py` | self-contained constants (no CLI), standard 1024-env config |
| Classical baseline | register in `algorithms/baselines/baselines.py` | e.g. proportional navigation / augmented PN, same action convention |
| Baseline-driven warm start | via `bc_warmstart.py` | expert data from the PN/APN baseline controller |
| Benchmark run + report | `results/missile/`, `results/summary.md` | PPO vs FlashSAC vs classical baseline under the official eval protocol |

## Honesty note

No environment, result or claim exists yet — nothing in this file should be
read as a benchmark number. The missile Intercept task and the full
task-extension plan are tracked in the root `README.md` under
**Roadmap: planned control tasks**.
