"""FlashSAC arm for the CWH rendezvous task (SpaceRendezvousEnv) -- the
off-policy counterpart of train_rendezvous.py for the PD-baseline comparison.

Identical task, env, seed and PPO-invariant hyperparameters as the PPO entry
(run_training filters PPO-only keys for flash_sac, so gamma 0.999 and lr 3e-4
carry over).  The only deliberate deviation is the update count: FlashSAC
performs one vectorized env step per iteration (1024 transitions), so 250000
iterations match the PPO arm's interaction budget (1000 updates x 250 steps x
1024 envs = 256M transitions, the project's equal-interaction-steps fair
comparison).

Self-contained entry: run `python train_rendezvous_flashsac.py`.  Results go
to results/rendezvous_flashsac (kept separate from the PPO run directory).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.satellite.rendezvous_docking import SpaceRendezvousEnv

TASK = 'rendezvous'
NUM_ENVS = 1024
# equal interaction steps with the PPO arm: 250000 iters x 1024 envs = 256M
UPDATES = 250000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'flash_sac'
OUT_DIR = _ROOT / 'results' / 'rendezvous_flashsac'
# same dict as train_rendezvous.py; run_training keeps only flash_sac-valid
# keys (gamma=.999, learning_rate=3e-4) and warns about the rest
PPO_OVERRIDES = dict(num_steps_per_env=250, gamma=.999, num_mini_batches=1,
                     lr_schedule='fixed', learning_rate=3e-4, entropy_coef=.003)

if __name__ == '__main__':
    run_training(TASK, SpaceRendezvousEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, out_dir=OUT_DIR, **PPO_OVERRIDES)
