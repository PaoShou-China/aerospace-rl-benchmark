"""RSL-RL PPO training entry for the toy Pendulum task (PendulumEnv).

Minimal gym-classic Pendulum-v1 swing-up used as a correctness sanity check
for the training stack (rsl-rl PPO / FlashSAC on the fully tensorized
batched env protocol). Episodes truncate at 200 steps with reward
-(th^2 + 0.1 thdot^2 + 0.001 u^2); success metric = completed-episode
return > -200. A healthy implementation clears the bar within ~100-200
updates; flat or diverging curves indicate a bug in the stack, not in the
task.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_pendulum.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.pendulum import PendulumEnv

TASK = 'pendulum'
NUM_ENVS = 1024
UPDATES = 312            # PPO on the FlashSAC trunk: 312 x 32 x 1024 = 10.2M transitions,
                         # matched to the flash_sac arm (10k x 1 x 1024 = 10.24M,
                         # archived at results/pendulum_flashsac1024)
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RNN = False              # GRU actor/critic backbone via rsl-rl RNNModel
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)

if __name__ == '__main__':
    # Exploration overrides: the shared baseline (init_std=.5, entropy_coef=.001)
    # was tuned for the drone tasks and collapses to the bottom-wiggle local
    # optimum here (std -> .06, return stuck ~ -1130); rsl-rl's own defaults
    # give the swing-up behavior room to be discovered.
    # lr overrides: the previous run (KL-adaptive 3e-4, archived at
    # results/pendulum_net64) only reached return -300 with 46% of episodes
    # above the -200 success bar in 500 updates -- the same stall pattern fixed
    # 3e-4 shows on quadx (-377). Fixed 1e-3 is the shared-default A/B winner
    # on every other task, so it is pinned here explicitly.
    # PPO exploration/lr overrides (see rationale above). learning_rate=1e-3 is
    # passed for BOTH algos: build_cfg (PPO) and flash_sac_cfg both accept it,
    # and fixed 1e-3 is the shared-default A/B winner on every task -- on
    # pendulum too (flash_sac A/B, 3000 updates: 27% vs 10% success at paper
    # 3e-4; flash_sac still applies its own warmup-cosine decay on top).
    # NOTE: lr_schedule/init_std/entropy_coef below are PPO-only and filtered
    # for flash_sac (a previous leak let learning_rate through unnoticed --
    # now intentional and pinned identically for both).
    ppo_kw = {} if ALGORITHM == 'flash_sac' else dict(
        init_std=1.0, entropy_coef=.01, lr_schedule='fixed',
        arch='flash_residual')  # PPO on the FlashSAC trunk (src/common/ppo_flashnet.py)
                                # for the architecture-matched algorithm A/B
    run_training(TASK, PendulumEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 learning_rate=1e-3, **ppo_kw)
