"""RSL-RL PPO training entry for the Quadplane-VTOL Waypoints task
(QuadplaneWaypointsEnv).

2.35 kg hybrid UAV (H-layout four quad motors + fixedwing surfaces + forward
pusher) on the Genesis rigid-body engine: capture 4/4 waypoints (< 15 m)
within 1200 policy steps (30 s at 40 Hz); ground contact / out-of-bounds
(|p| > 120 m) terminates the episode. The plant has no mode switch: quad
thrust authority is independent of airspeed while lifting-surface forces
scale as v^2, so the agent must learn to hand control authority from the
rotors to the aerodynamic surfaces as it accelerates (~3-12 m/s body-x).

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run
``python environments/vtol/train_quadplane.py``.

Hyperparameters are identical to the other per-task entries for fair
comparison (fixed 3e-4 lr, 64 steps/env, 10 epochs, entropy 0, gamma 0.99,
value loss 0.5), matching train_fixedwing.py / train_quadx.py.

Known limitation (honest reporting): the shipped conventional baseline
``QuadplaneHybridBaseline`` (cascaded PID + gentle pusher) currently crashes
in 100% of episodes on the training, dev and test seed sets
(results/quadplane/cascade_report.json), so there is no validated non-learned
reference for this task yet. BC warm-starting from it stays disabled until
that controller is fixed.

WIND configures the wind disturbance passed to the env ctor (steady_wind,
dryden, severity); None = calm air (benchmark behavior).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.vtol.env import QuadplaneWaypointsEnv

TASK = 'quadplane'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 3e-4
NUM_STEPS_PER_ENV = 64
NUM_LEARNING_EPOCHS = 10
ENTROPY_COEF = 0.0
VALUE_LOSS_COEF = 0.5
# Wind disturbance (environments/common/disturbances.py): pure Dryden gusts
# (zero-mean, 30% of the moderate preset). None = calm air (benchmark).
WIND = dict(steady_wind=None, dryden=.3, severity='moderate')

# BC warm start (algorithms/warmstart/bc_warmstart.py): disabled -- the
# quadplane baseline is not yet a competent demonstrator (100% crash rate).
BC_WARMSTART = None

if __name__ == '__main__':
    run_training(TASK, QuadplaneWaypointsEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 num_steps_per_env=NUM_STEPS_PER_ENV,
                 num_learning_epochs=NUM_LEARNING_EPOCHS,
                 entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
                 env_kwargs=WIND, bc_warmstart=BC_WARMSTART)
