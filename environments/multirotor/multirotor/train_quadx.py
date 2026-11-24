"""RSL-RL PPO training entry for the QuadX-Waypoints task (QuadXWaypointsEnv).

Crazyflie 2.x quadrotor waypoint navigation on the Genesis rigid-body engine:
reach a sequence of 4 waypoints sampled in a 5 m flight dome within 800 policy
steps (20 s), without ground contact or leaving the dome.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_quadx.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in algorithms/train_utils.py.
WIND configures the wind disturbance passed to the env ctor (steady_wind,
dryden, severity); None = calm air (bit-for-bit benchmark behavior).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.multirotor import QuadXWaypointsEnv

TASK = 'quadx'
NUM_ENVS = 1024          # Genesis hover tutorial uses 8192; kept at 1024 for hardware/wall time
UPDATES = 1000           # tutorial runs 301 iterations; kept for a comparable final policy
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
# Web-alignment (2026-10-21): hyperparameters follow the official Genesis hover
# tutorial (rsl-rl defaults): lr 3e-4, 5 epochs, 4 mini-batches, entropy 0.01,
# init_std 1.0, num_steps_per_env 100, gamma 0.99, lam 0.95, clip 0.2.
# Note: the earlier local 300-update sweep (results/quadx_sweep) picked fixed
# 1e-3 over 3e-4; that local result is intentionally overridden here so the
# configuration matches the published reference. Re-run the sweep if this arm
# stalls (documented failure was -260 return / 0% success at fixed 3e-4).
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
INIT_STD = 1.0
NUM_STEPS_PER_ENV = 100
NUM_LEARNING_EPOCHS = 5
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
# Wind disturbance (environments/common/disturbances.py): pure Dryden gusts
# (zero-mean, 30% of the moderate preset). None = calm air (benchmark).
WIND = dict(steady_wind=None, dryden=.3, severity='moderate')

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before
# PPO. None = off.
BC_WARMSTART = None

if __name__ == '__main__':
    run_training(TASK, QuadXWaypointsEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 entropy_coef=ENTROPY_COEF, init_std=INIT_STD,
                 num_steps_per_env=NUM_STEPS_PER_ENV,
                 num_learning_epochs=NUM_LEARNING_EPOCHS,
                 env_kwargs=WIND, bc_warmstart=BC_WARMSTART)
