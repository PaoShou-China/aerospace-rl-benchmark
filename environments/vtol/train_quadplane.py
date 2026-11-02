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

Conventional baselines: the task default is ``QuadplaneSE3PusherBaseline``
(quadrotor SE(3) throughout + forward-pusher allocation), which reaches
87.5% on 256 replicas / seed 7 and 89.1% over the 3-seed 128-replica protocol
(``results/quadplane/baseline.json`` and
``results/quadplane/baselines/se3_pusher_seed7.json``). The former default
``QuadplaneHybridBaseline`` (mode-switching cascaded PID) scores 26.6% on the
same 256-replica set, and ``QuadplaneL1WingBaseline`` scores 49.6%; both are
retained as structural ablations. The env's ``conventional_controller``
attribute still names ``QuadplaneHybridBaseline``, so the BC collector
currently distils that weaker arm even though the default evaluation baseline
is ``se3_pusher`` -- this mismatch is a known limitation of the warm-start
recipe, not of the environment.

WIND configures the wind disturbance passed to the env ctor (steady_wind,
dryden, severity); None = calm air (benchmark behavior).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training  # noqa: E402
from environments.vtol.env import QuadplaneWaypointsEnv  # noqa: E402

TASK = "quadplane"
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = "gpu"  # 'gpu' | 'cpu'
RESUME = False  # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True  # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = "ppo"
LR_SCHEDULE = "fixed"
LEARNING_RATE = 3e-4
NUM_STEPS_PER_ENV = 64
NUM_LEARNING_EPOCHS = 10
ENTROPY_COEF = 0.0
VALUE_LOSS_COEF = 0.5
# Wind disturbance (environments/common/disturbances.py): pure Dryden gusts
# (zero-mean, 30% of the moderate preset). None = calm air (benchmark).
WIND = dict(steady_wind=None, dryden=0.3, severity="moderate")

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# env's ``conventional_controller``, which for this task is still
# QuadplaneHybridBaseline (26.6% standalone) rather than the default evaluated
# baseline se3_pusher (87.5%); driven directly by the BC collector before PPO.
# None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == "__main__":
    run_training(
        TASK,
        QuadplaneWaypointsEnv,
        num_envs=NUM_ENVS,
        updates=UPDATES,
        seed=SEED,
        backend=BACKEND,
        resume=RESUME,
        clean=CLEAN,
        algo=ALGORITHM,
        lr_schedule=LR_SCHEDULE,
        learning_rate=LEARNING_RATE,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        num_learning_epochs=NUM_LEARNING_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        value_loss_coef=VALUE_LOSS_COEF,
        env_kwargs=WIND,
        bc_warmstart=BC_WARMSTART,
    )
