"""FlashSAC arm for the low-thrust perigee-raising transfer task
(LowThrustTransferEnv) -- the off-policy counterpart of train_transfer.py for
the classical-baseline comparison (Gauss greedy / Q-law) in
eval_transfer_qlaw.py.

Identical task, env, seed and PPO-invariant hyperparameters as the PPO entry
(run_training filters PPO-only keys for flash_sac, so gamma 0.9999 carries
over and the rest are dropped with a warning). The only deliberate deviation
is the update count: FlashSAC performs one vectorized env step per iteration
(1024 transitions), so 32768 iterations match the PPO arm's interaction
budget (1000 updates x 32 steps x 1024 envs = 32.8M transitions, the
project's equal-interaction-steps fair comparison).

Self-contained entry: run `python train_transfer_flashsac.py`. Results go to
results/transfer_flashsac (kept separate from the PPO run directory).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.satellite.orbit_transfer import LowThrustTransferEnv

TASK = 'transfer'
NUM_ENVS = 1024
# equal interaction steps with the PPO arm: 32768 iters x 1024 envs = 32.8M
UPDATES = 32768
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/transfer_flashsac artifacts
ALGORITHM = 'flash_sac'
OUT_DIR = _ROOT / 'results' / 'transfer_flashsac'
# same dict as train_transfer.py; run_training keeps only flash_sac-valid
# keys (gamma=0.9999) and warns about the rest
PPO_OVERRIDES = {'gamma': 0.9999, 'lr_schedule': 'fixed', 'learning_rate': 2.5e-4,
                 'lam': 0.99, 'entropy_coef': 4.75e-8, 'num_learning_epochs': 30,
                 'clip_param': 0.3, 'value_loss_coef': 0.5}

if __name__ == '__main__':
    run_training(TASK, LowThrustTransferEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, out_dir=OUT_DIR, **PPO_OVERRIDES)
