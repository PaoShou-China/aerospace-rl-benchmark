"""RSL-RL training entry for QuadX-Track (QuadXTrackEnv).

Persistent observation of a cooperative ground moving target with the standard
cf2x quadrotor plant, inner rate PID, 4-D rate+collective action interface and
40 Hz policy used by the QuadX task.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments.  Set ALGORITHM = 'ppo' or 'flash_sac' and run
``python train_quadx_track.py``.

Fair-comparison protocol
    Both algorithms consume the SAME environment-interaction budget
    (TRAIN_ENV_STEPS vectorized policy steps x NUM_ENVS replicas), the same
    seed, the same reward, action interface and evaluation scenarios:
      * PPO: TRAIN_ENV_STEPS / NUM_STEPS_PER_ENV updates (100 steps each),
      * FlashSAC: TRAIN_ENV_STEPS iterations (one vectorized step each).
    The raw implementations are used unchanged: no recurrent layer, filter,
    predictor, safety layer, reward shaping or observation augmentation is
    added by this entry.  Model/optimiser hyperparameters follow the shared
    per-task defaults in algorithms/train_utils.py, identical to every other
    task entry.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))                # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.multirotor import QuadXTrackEnv

TASK = 'quadx_track'
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
NUM_ENVS = 1024
# Shared vectorized-step budget for BOTH arms (20.5M transitions at 1024 envs).
# Sized so one arm completes inside a bounded wall-clock on a shared, saturated
# GPU while still giving PPO ~195 updates and FlashSAC 20,000 gradient updates.
TRAIN_ENV_STEPS = 20_000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False
CLEAN = True             # wipe stale artifacts inside this algorithm's out_dir

# shared PPO/model hyperparameters (identical to the other task entries)
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
INIT_STD = 1.0
NUM_STEPS_PER_ENV = 100
NUM_LEARNING_EPOCHS = 5

# no wind, no BC warm start, no hand-coded controller in the loop
WIND = None
BC_WARMSTART = None

# PPO performs NUM_STEPS_PER_ENV vectorized steps per update; FlashSAC one.
UPDATES = TRAIN_ENV_STEPS // NUM_STEPS_PER_ENV if ALGORITHM == 'ppo' else TRAIN_ENV_STEPS
OUT_DIR = _ROOT / 'results' / TASK / ALGORITHM

if __name__ == '__main__':
    # PPO-specific knobs are not forwarded to FlashSAC (its original config
    # builder owns its own lr/entropy/batch hyperparameters).
    extra = dict(lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 entropy_coef=ENTROPY_COEF, init_std=INIT_STD,
                 num_steps_per_env=NUM_STEPS_PER_ENV,
                 num_learning_epochs=NUM_LEARNING_EPOCHS) if ALGORITHM == 'ppo' else {}
    run_training(TASK, QuadXTrackEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, out_dir=OUT_DIR, algo=ALGORITHM,
                 env_kwargs=WIND, bc_warmstart=BC_WARMSTART, **extra)
