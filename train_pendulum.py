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
PPO_UPDATES = 2500       # web-aligned rollout: 4 steps/env x 1024 envs = 4096
                         # samples/update, the same batch as RL Zoo's Pendulum-v1
                         # (4 envs x 1024 steps). 2500 updates keep the arm at
                         # 10.24M transitions, matched to flash_sac's 10k x 1 x 1024.
FLASH_SAC_UPDATES = 10000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
# Curriculum difficulty (project-wide feature, see src/common/curriculum.py):
# None = off (gym benchmark, bit-for-bit); float in [0,1] = static difficulty
# (0 easiest, 1 hardest = benchmark); True = auto-ramp difficulty 0 -> 1,
# advancing +0.1 whenever a 1024-episode window closes at >= 60% success.
CURRICULUM = None
ENV_KWARGS = ({'curriculum': True} if CURRICULUM is True
              else {} if CURRICULUM is None else {'difficulty': CURRICULUM})

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
    # Fair-config equalization (2026-10-21): BOTH algorithms now train under the
    # IDENTICAL warmup-cosine lr schedule (flash_sac's paper schedule: peak 1e-3
    # -> end 1.5e-4, lr_warmup_rate 1e-6 of total updates). For PPO this goes
    # through run_training's lr_cosine opt-in (same _warmup_cosine_decay formula,
    # param_groups written directly since schedule='fixed' never re-applies the
    # lr); for flash_sac it is its native _set_lr path (learning_rate_end
    # forwards its paper default 1.5e-4). Remaining asymmetries are
    # algorithm-native mechanics, not tunable configs: entropy/exploration (PPO
    # entropy_coef+std vs SAC auto-temperature+zeta), data re-use (PPO 10 epochs
    # vs UTD 1 replay), reward normalization (SAC's RewardNormalizer vs PPO raw),
    # grad clip (PPO 1.0 vs SAC none -- ~inert under Adam scale invariance).
    # Web-alignment (2026-10-21): gamma 0.9, entropy 0, 10 epochs, 4096-sample
    # batch follow the RL Zoo PPO config for Pendulum-v1
    # (github.com/DLR-RM/rl-baselines3-zoo, hyperparams/ppo.yml). The warmup-
    # cosine lr schedule is kept for the cross-algorithm fairness comparison.
    ppo_kw = {} if ALGORITHM == 'flash_sac' else dict(
        init_std=1.0, entropy_coef=0.0, lr_schedule='fixed',
        gamma=0.9, num_learning_epochs=10, num_steps_per_env=4)
        # arch: flash_residual trunk is the build_cfg default (project-wide since
        # 2026-10-21); pass arch='mlp' explicitly only for A/B arms
    updates = PPO_UPDATES if ALGORITHM == 'ppo' else FLASH_SAC_UPDATES
    run_training(TASK, PendulumEnv, num_envs=NUM_ENVS, updates=updates, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 learning_rate=1e-3, lr_cosine=(ALGORITHM == 'ppo'), learning_rate_end=1.5e-4,
                 env_kwargs=ENV_KWARGS,
                 **ppo_kw)
