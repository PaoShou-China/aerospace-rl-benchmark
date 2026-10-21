"""RSL-RL PPO training entry for the QuadX-Hover task (QuadXHoverEnv).

Crazyflie 2.x quadrotor station keeping on the Genesis rigid-body engine:
hover at 1 m altitude for 400 policy steps (10 s) without ground contact or
leaving the 3 m radius bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_quadx.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.quadx import QuadXHoverEnv

TASK = 'quadx'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RNN = False              # GRU actor/critic backbone via rsl-rl RNNModel
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
# Documented per-task PPO deviation (same precedent as docking): the KL-adaptive
# schedule thrashes on this task (lr slammed between its 1e-5 floor and the
# 1e-2 ceiling, then settles ~5e-5 and stalls below the fixed-lr plateau), and
# the default fixed 3e-4 gets stuck at ~-260 return with 0% success. A 300-update
# lr sweep (2026-10-20, results archived under results/quadx_sweep) picked
# fixed 1e-3: success 1.0 with mean_reward rising monotonically.
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 1e-3
# entropy_coef stays at the shared 0.001 default: a 0.005 variant was tried and
# made the late-phase behavior worse (lower peak 858 @244, earlier drift, std
# pushed back up 0.279 -> 0.311), so the docking late-collapse remedy does not
# transfer to this task.
ENTROPY_COEF = 0.001
# Third documented deviation: exponential lr half-life of 300 updates, applied
# by train_utils writing the optimizer param_groups directly (schedule='fixed'
# never reapplies alg.learning_rate). Rationale: with success saturated the
# policy mean random-walks off the hover ridge (first 1000-update attempt:
# peak 917 @417 -> ~830 @699, success still 1.0); decaying lr damps that walk.
# The half-life is deliberately long (300, not ~175) because fixed 3e-4 from
# scratch stalls at -377 with 0% success -- the climb phase (first ~400
# updates) must keep lr >= ~4e-4, and this schedule stays above 3.5e-4 through
# update 600 while reaching ~1e-4 by update 1000.
LR_DECAY_UPDATES = 300
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)

if __name__ == '__main__':
    run_training(TASK, QuadXHoverEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 entropy_coef=ENTROPY_COEF, lr_decay_updates=LR_DECAY_UPDATES)
