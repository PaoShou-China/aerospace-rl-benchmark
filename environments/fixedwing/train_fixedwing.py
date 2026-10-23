"""RSL-RL PPO training entry for the Fixedwing-Waypoints task
(FixedwingWaypointsEnv).

2.35 kg fixed-wing UAV with blade-element aerodynamics on the Genesis
rigid-body engine: capture 4/4 waypoints (< 10 m, see fixedwing/env.py)
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

DIFFICULTY / ENV_KWARGS expose the static difficulty knob: None = benchmark,
float in [0, 1] = static difficulty (0 easiest, 1 hardest = benchmark).

Geometry and diagnostics (2026-10-21): the waypoint shell is now 20-65 m
with a 20 m altitude floor (was 1-90 m and 0.5 m). The old shell sampled
targets inside one minimum turn radius of the 100 m OOB sphere and required
ground-skimming captures at z ~ 0.5 m. Termination reasons are now logged
(Episode/crash_rate, out_of_bounds_rate, timeout_rate) and aggregated in
run.json, so the remaining 4th-waypoint gap can be attributed instead of
guessed.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.fixedwing import FixedwingWaypointsEnv

TASK = 'fixedwing'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
# Web-alignment (2026-10-21): PPO hyperparameters follow "Adversarial RL for
# Robust Control of Fixed-Wing Aircraft" (arXiv:2510.16650): lr 3e-4, 10 epochs,
# entropy 0, gamma 0.99, lam 0.95, clip 0.2, vf_coef 0.5, 2048 steps/env.
# num_steps_per_env stays 64 (verified local improvement, see docstring above);
# 2048 would multiply wall time ~32x at 1024 envs. 1024 envs kept.
LR_SCHEDULE = 'fixed'
LEARNING_RATE = 3e-4
NUM_STEPS_PER_ENV = 64
NUM_LEARNING_EPOCHS = 10
ENTROPY_COEF = 0.0
VALUE_LOSS_COEF = 0.5
# Static difficulty (None = published benchmark, bit-for-bit; float in [0,1]
# = 0 easiest, 1 hardest = benchmark). Curriculum learning was removed.
DIFFICULTY = None
ENV_KWARGS = {} if DIFFICULTY is None else {'difficulty': DIFFICULTY}

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == '__main__':
    run_training(TASK, FixedwingWaypointsEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 num_steps_per_env=NUM_STEPS_PER_ENV,
                 num_learning_epochs=NUM_LEARNING_EPOCHS,
                 entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
                 env_kwargs=ENV_KWARGS, bc_warmstart=BC_WARMSTART)
