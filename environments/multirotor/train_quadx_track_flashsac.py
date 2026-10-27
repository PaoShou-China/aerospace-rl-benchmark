"""FlashSAC arm for QuadX-Track (QuadXTrackEnv) -- the off-policy counterpart
of train_quadx_track.py (PPO) for the classical-baseline comparison
(cv Kalman filter + fixed-offset SE(3), algorithms/baselines/baselines.py).

Identical task, env, seed and interaction budget as the PPO entry (6.14M
transitions). FlashSAC performs one vectorized step per iteration, so its
iteration count equals the shared step budget. The raw FlashSAC
implementation is used unchanged -- no recurrent layer, filter, predictor or
safety layer is added by this entry.

FlashSAC hyperparameter tuning (deviations from paper defaults):
  g_max=20     (quadx_track returns span ~[-10, +12000]; paper default 5
                causes reward normalization to compress step rewards to ~1e-4,
                where the 101-bin critic categorical distribution loses
                resolution)
  updates_per_step=4  (4x more gradient updates than paper default; quadx_track
                       is a 4-D continuous control task with sparse crash penalty)
  n_step=3     (3-step returns for faster credit assignment on a 1200-step horizon)
  buffer_min_length=5000  (start learning earlier; 5000 iterations x 1024 envs
                           = 5.1M transitions is ample for critic bootstrapping)
  actor_update_period=1  (actor updated every gradient step instead of every 2nd;
                          paper's period-2 is for faster-warming critics on Mujoco
                          benchmarks, not needed here)

Self-contained entry: run `python train_quadx_track_flashsac.py`. Results go
to results/quadx_track/flash_sac (kept separate from the PPO run directory).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))                # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.multirotor import QuadXTrackEnv

TASK = 'quadx_track'
ALGORITHM = 'flash_sac'
NUM_ENVS = 1024
# Same shared vectorized-step budget as the PPO entry (6.1M transitions).
TRAIN_ENV_STEPS = 6_000
UPDATES = TRAIN_ENV_STEPS  # one vectorized env step per FlashSAC iteration
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False
CLEAN = True             # wipe stale artifacts inside this algorithm's out_dir
OUT_DIR = _ROOT / 'results' / TASK / ALGORITHM

# FlashSAC-specific overrides (PPO-only keys filtered by run_training):
PPO_OVERRIDES = {
    'gamma': 0.99,
    'learning_rate': 3e-4,
    'updates_per_step': 4,        # FlashSAC: more gradient updates for harder task
    'n_step': 3,                  # FlashSAC: faster credit assignment
    'g_max': 20.,                 # FlashSAC: cover quadx_track's return range
    'buffer_min_length': 5_000,   # FlashSAC: start learning earlier
    'actor_update_period': 1,     # FlashSAC: actor updated every step
}

if __name__ == '__main__':
    run_training(TASK, QuadXTrackEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, out_dir=OUT_DIR,
                 algo=ALGORITHM, env_kwargs=None, **PPO_OVERRIDES)
