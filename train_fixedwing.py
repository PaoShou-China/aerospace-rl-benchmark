"""RSL-RL PPO training entry for the Fixedwing-Waypoints task
(FixedwingWaypointsEnv).

2.35 kg fixed-wing UAV with blade-element aerodynamics on the Genesis
rigid-body engine: capture 4/4 waypoints (< 10 m, see src/fixedwing/env.py)
within 3600 policy steps (120 s); ground contact / out-of-bounds terminates
the episode, timeout at 3600 steps.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_fixedwing.py`.
Hyperparameters are identical across all per-task entries for fair
comparison, with two documented deviations (same precedent as train_quadx.py
and train_docking.py):

1. The KL-adaptive learning rate is replaced by a fixed 1e-3. Justification
   (2026-10-20 A/B, 150 updates x 64 envs, post physics fix): both schedules
   learn, but fixed 1e-3 reaches all-4-waypoint success (Episode/success_rate
   up to 0.33) with longer episodes by update ~137 while adaptive stalls at
   success 0 with shorter episodes; the adaptive schedule also reproduced the
   project-wide pathology of thrashing across 1e-5..1e-3.

2. num_steps_per_env 32 -> 64 (2026-10-20, user-directed follow-up, verified
   with a 1000-update rerun): at ~700+ step episodes only ~3 episodes complete
   per 32-step rollout, so per-iteration success statistics and advantages are
   noisy. Verified outcome: final-window (it 900-1000) success_rate 0.416 ->
   0.514, waypoints/episode 2.51 -> 2.78, overall run.json success 0.295 ->
   0.304 (6912 episodes, 2x the baseline count), late-training success still
   rising where the baseline plateaued; cost is +48% wall time (3803 s) and
   2x transitions. Kept. Baseline (nsp=32): results/fixedwing_baseline_nsp32.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.fixedwing import FixedwingWaypointsEnv

TASK = 'fixedwing'
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
LR_SCHEDULE = 'fixed'    # documented deviation, see module docstring
LEARNING_RATE = 1e-3
NUM_STEPS_PER_ENV = 64   # documented deviation (default 32), see module docstring

if __name__ == '__main__':
    run_training(TASK, FixedwingWaypointsEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 num_steps_per_env=NUM_STEPS_PER_ENV)
