"""FlashSAC arm for the CWH docking task (SpaceDockingEnv) -- the
off-policy counterpart of train_docking.py for the equal-interaction-steps
fair comparison with the PPO arm.

Identical task, env, seed and interaction budget as the PPO entry: PPO does
1000 updates x 32 steps x 1024 envs = 32,768,000 transitions. FlashSAC does
3200000 iterations x 1024 envs = 32,768,000 transitions to match exactly.

Hyperparameters:
  gamma=0.999  (docking's 500-step horizon needs high discount, same as PPO)
  learning_rate=3e-4  (same as PPO arm, fair comparison)
  updates_per_step=2  (doubled vs paper default; docking is harder than the
                       2-block FlashSAC paper benchmarks)
  n_step=3  (3-step returns for faster credit assignment on long-horizon)
  buffer_min_length=20000  (enough data before first update but not too long)
  g_max=10  (docking rewards span ~[-200, +10]; paper default 5 may clip)

Self-contained entry: run `python train_docking_flashsac.py`. Results go
to results/docking_flashsac (kept separate from the PPO run directories).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.satellite.rendezvous_docking import SpaceDockingEnv

TASK = 'docking'
NUM_ENVS = 1024
# Equal interaction budget with the PPO arm: 1000 PPO updates x 32 steps x 1024 envs
# = 32,768,000 transitions. FlashSAC does one vectorized step per iteration.
UPDATES = 32_000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale artifacts inside this algorithm's out_dir
ALGORITHM = 'flash_sac'
OUT_DIR = _ROOT / 'results' / 'docking_flashsac'

# FlashSAC-specific overrides (PPO-only keys are filtered by run_training):
# gamma=0.999 matches the PPO arm's docking discount; n_step=3 and
# updates_per_step=2 are deliberate deviations from the paper defaults to
# handle docking's 500-step horizon and sparse success signal.
PPO_OVERRIDES = dict(
    num_steps_per_env=32,           # PPO-only; filtered out for flash_sac
    num_mini_batches=1,             # PPO-only; filtered out
    lr_schedule='fixed',            # PPO-only; filtered out
    gamma=0.999,                    # PASSED to flash_sac_cfg (long-horizon task)
    learning_rate=3e-4,             # PASSED to flash_sac_cfg (fair comparison)
    entropy_coef=0.005,             # PPO-only; filtered out
    updates_per_step=2,             # PASSED to flash_sac_cfg (harder task)
    n_step=3,                       # PASSED to flash_sac_cfg (credit assignment)
    buffer_min_length=20_000,       # PASSED to flash_sac_cfg (more data before first update)
    g_max=10.,                      # PASSED to flash_sac_cfg (docking rewards ~[-200, +10])
)

if __name__ == '__main__':
    run_training(TASK, SpaceDockingEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, out_dir=OUT_DIR, **PPO_OVERRIDES)
