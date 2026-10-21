"""RSL-RL PPO training entry for the CWH docking task (SpaceDockingEnv).

Clohessy-Wiltshire-Hill relative motion in the LVLH frame of a 400 km circular
orbit (pure-tensor symplectic Euler, dt = 1 s): final approach from 50-100 m
separation to range < 1 m with closing speed < 0.1 m/s within 500 steps,
without exceeding the 200 m failure bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_docking.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.docking import SpaceDockingEnv

TASK = 'docking'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
# PPO deviation under test (diagnosed docking failure mode): the KL-adaptive
# schedule multiplied lr by 1.5 whenever measured KL stayed below
# desired_kl/2, racing 3e-4 -> the hard-coded 1e-2 ceiling within 3 updates
# and holding it for 176 updates while mean_reward fell -173 -> -321. A
# fixed lr keeps every other setting at the shared fair-comparison values.
LR_SCHEDULE = 'fixed'
# PPO deviation under test #2: with the success rate saturated at ~1.0 the
# return variance (and with it the advantage signal) collapses, value loss
# spikes, and the policy mean random-walks off the precision ridge into a
# low-progress saddle it cannot escape from its own saturated data (observed
# at update ~913 of the first 1000-update run: success 1.0 -> 0 within 85
# updates while std stayed ~0.27). A slightly stronger entropy bonus keeps
# the policy distribution wide enough for the advantage signal to pull it
# back; every other setting stays at the shared fair-comparison values.
ENTROPY_COEF = 0.005
# Web-alignment (2026-10-21): lr/gamma follow "Neural-based Control for CubeSat
# Docking Maneuvers" (arXiv:2410.12703): lr 5e-5, gamma 0.999, clip 0.2, 25M
# transitions. entropy_coef stays 0.005 (the paper does not report entropy; the
# 0.005 value is this task's measured late-collapse remedy, see above).
LEARNING_RATE = 5e-5
GAMMA = 0.999
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
# Curriculum difficulty (project-wide feature, see src/common/curriculum.py):
# None = off (published benchmark, bit-for-bit); float in [0,1] = static
# difficulty (0 easiest, 1 hardest = benchmark); True = auto-ramp difficulty
# 0 -> 1, advancing +0.1 whenever a 1024-episode window closes at >= 60% success.
CURRICULUM = None
ENV_KWARGS = ({'curriculum': True} if CURRICULUM is True
              else {} if CURRICULUM is None else {'difficulty': CURRICULUM})

if __name__ == '__main__':
    run_training(TASK, SpaceDockingEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 entropy_coef=ENTROPY_COEF, gamma=GAMMA,
                 env_kwargs=ENV_KWARGS)
