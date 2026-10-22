"""RSL-RL PPO training entry for the QuadX-Hover task (QuadXHoverEnv).

Crazyflie 2.x quadrotor station keeping on the Genesis rigid-body engine:
hover at 1 m altitude for 400 policy steps (10 s) without ground contact or
leaving the 3 m radius bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_quadx.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
DIFFICULTY / ENV_KWARGS expose the static difficulty knob: None = benchmark,
float in [0, 1] = static difficulty (0 easiest, 1 hardest = benchmark).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.quadx import QuadXHoverEnv

TASK = 'quadx'
NUM_ENVS = 1024          # Genesis hover tutorial uses 8192; kept at 1024 for hardware/wall time
UPDATES = 1000           # tutorial runs 301 iterations; kept for a comparable final policy
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
# Web-alignment (2026-10-21): hyperparameters follow the official Genesis hover
# tutorial (rsl-rl defaults): lr 3e-4, 5 epochs, 4 mini-batches, entropy 0.01,
# init_std 1.0, num_steps_per_env 100, gamma 0.99, lam 0.95, clip 0.2.
# Note: the earlier local 300-update sweep (results/quadx_sweep) picked fixed
# 1e-3 over 3e-4; that local result is intentionally overridden here so the
# configuration matches the published reference. Re-run the sweep if this arm
# stalls (documented failure was -260 return / 0% success at fixed 3e-4).
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
INIT_STD = 1.0
NUM_STEPS_PER_ENV = 100
NUM_LEARNING_EPOCHS = 5
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
# Static difficulty (None = published benchmark, bit-for-bit; float in [0,1]
# = 0 easiest, 1 hardest = benchmark). Curriculum learning was removed.
DIFFICULTY = None
ENV_KWARGS = {} if DIFFICULTY is None else {'difficulty': DIFFICULTY}

# BC warm start (src/common/bc_warmstart.py): pretrain the actor/critic on the
# classical nominal controller (extras['nominal_action']) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == '__main__':
    run_training(TASK, QuadXHoverEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 entropy_coef=ENTROPY_COEF, init_std=INIT_STD,
                 num_steps_per_env=NUM_STEPS_PER_ENV,
                 num_learning_epochs=NUM_LEARNING_EPOCHS,
                 env_kwargs=ENV_KWARGS, bc_warmstart=BC_WARMSTART)
