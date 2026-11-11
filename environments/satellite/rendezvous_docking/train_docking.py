"""RSL-RL PPO training entry for the CWH docking task (SpaceDockingEnv).

Clohessy-Wiltshire-Hill relative motion in the LVLH frame of a 400 km circular
orbit (pure-tensor symplectic Euler, dt = 1 s): final approach from 60-120 m
separation to range < 1 m with closing speed < 0.1 m/s within 500 steps,
without exceeding the 200 m failure bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_docking.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in algorithms/train_utils.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training  # noqa: E402
from environments.satellite.rendezvous_docking import SpaceDockingEnv  # noqa: E402

TASK = "docking"
NUM_ENVS = 1024
UPDATES = 1500  # ppo_scratch barely learns (1.5% SR); more budget needed
SEED = 7
BACKEND = "gpu"  # 'gpu' | 'cpu'
RESUME = False  # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True  # wipe stale results/<task> artifacts before a fresh run
# PPO deviation under test (diagnosed docking failure mode): the KL-adaptive
# schedule multiplied lr by 1.5 whenever measured KL stayed below
# desired_kl/2, racing 3e-4 -> the hard-coded 1e-2 ceiling within 3 updates
# and holding it for 176 updates while mean_reward fell -173 -> -321. A
# fixed lr keeps every other setting at the shared fair-comparison values.
LR_SCHEDULE = "fixed"
# PPO deviation under test #2: with the success rate saturated at ~1.0 the
# return variance (and with it the advantage signal) collapses, value loss
# spikes, and the policy mean random-walks off the precision ridge into a
# low-progress saddle it cannot escape from its own saturated data (observed
# at update ~913 of the first 1000-update run: success 1.0 -> 0 within 85
# updates while std stayed ~0.27). A slightly stronger entropy bonus keeps
# the policy distribution wide enough for the advantage signal to pull it
# back; every other setting stays at the shared fair-comparison values.
ENTROPY_COEF = 0.005
# gamma 0.999 follows "Neural-based Control for CubeSat Docking Maneuvers"
# (arXiv:2410.12703); its lr 5e-5 does NOT transfer -- see the measured note
# below. entropy_coef stays 0.005 (the paper does not report entropy; the
# 0.005 value is this task's measured late-collapse remedy, see above).
# Web-alignment NOTE (measured 2026-10-22): the 10-21 attempt to align lr to
# "Neural-based Control for CubeSat Docking Maneuvers" (arXiv:2410.12703,
# lr 5e-5) BROKE the task -- 0-2.9% success for BOTH architectures at 64 and
# 1024 envs (results/diag_docking_*). The paper's lr does not transfer to this
# task's batch scale. 3e-4 (the pre-alignment value, and the build_cfg shared
# default) recovers: mlp 79.4% @64 / 77.9% @1024 envs. gamma 0.999 stays.
LEARNING_RATE = 3e-4
GAMMA = 0.999
ALGORITHM = "ppo"

OUT_DIR = str(Path(__file__).resolve().parents[3] / "results" / "docking_mlp")

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)


def main(**overrides):
    """Run one training. Keyword overrides (seed / out_dir / bc_warmstart /
    env_wrapper / ...) let the 5-seed orchestrator drive the same constants."""
    kwargs = dict(
        num_envs=NUM_ENVS,
        updates=UPDATES,
        seed=SEED,
        backend=BACKEND,
        resume=RESUME,
        clean=CLEAN,
        algo=ALGORITHM,
        out_dir=OUT_DIR,
        lr_schedule=LR_SCHEDULE,
        learning_rate=LEARNING_RATE,
        entropy_coef=ENTROPY_COEF,
        gamma=GAMMA,
        bc_warmstart=BC_WARMSTART,
    )
    kwargs.update(overrides)
    run_training(TASK, SpaceDockingEnv, **kwargs)


if __name__ == "__main__":
    main()
