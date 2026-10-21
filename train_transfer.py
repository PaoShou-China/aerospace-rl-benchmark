"""RSL-RL PPO training entry for the low-thrust perigee-raising transfer task
(LowThrustTransferEnv).

Two-body point-mass dynamics with a 3-axis 10 mN electric-propulsion thruster
(pure-tensor RK4, 10 s substeps, dt = 300 s): raise perigee over 166 steps
(~5 orbits). The training reward is the exact potential-based telescoping
decomposition of the reference terminal score 1e-4 * (-|dra| + drp + 2e5 * dm),
so every step carries immediate credit and the episode return still equals the
terminal R; success = positive terminal R.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_transfer.py`.
Hyperparameters are identical across all per-task entries for fair
comparison except the deviations documented here: gamma = 1.0 (fixed-horizon
task; makes the PBRS return undiscounted and exactly telescoping over the
166-step episode) and a disabled timeout value bootstrap inside the env
(timeouts are true terminals). Shared PPO machinery lives in
src/common/train_utils.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.transfer import LowThrustTransferEnv

TASK = 'transfer'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RNN = False              # GRU actor/critic backbone via rsl-rl RNNModel
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)

# Task-specific deviations from the shared fair-comparison PPO configuration.
# lr pinned to the proven recipe (2026-10-20, 1000-update verification run:
# mean_reward 0.14->1.55, success 100% from update 31, final ckpt 100% / +20.6
# km): KL-adaptive 3e-4 is healthy on this short 166-step task (lr stayed
# 4e-4..3e-3, no floor slam, std 0.5->0.10), so neither the adaptive pathology
# nor the fixed-1e-3 default was ever beaten here; both are untested at the
# new [128,128] network scale.
PPO_OVERRIDES = {'gamma': 1.0, 'lr_schedule': 'adaptive', 'learning_rate': .0003}

if __name__ == '__main__':
    run_training(TASK, LowThrustTransferEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, **PPO_OVERRIDES)
