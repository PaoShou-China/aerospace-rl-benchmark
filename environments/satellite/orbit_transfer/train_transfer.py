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
algorithms/train_utils.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training  # noqa: E402
from environments.satellite.orbit_transfer import LowThrustTransferEnv  # noqa: E402

TASK = "transfer"
NUM_ENVS = 1024
UPDATES = 500  # converges immediately (potential-based reward); 500 updates enough
SEED = 7
BACKEND = "gpu"  # 'gpu' | 'cpu'
RESUME = False  # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True  # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = "ppo"

# Task-specific deviations from the shared fair-comparison PPO configuration.
# Web-alignment (2026-10-21): values follow "RL for Low-Thrust Trajectory Design
# of Interplanetary Missions" (arXiv:2008.08501): gamma 0.9999, lam 0.99,
# lr 2.5e-4, entropy 4.75e-8, clip 0.2, vf_coef 0.5. 2026-11-03 5-seed
# protocol: 10 learning epochs (shared convention) and the paper's lr decay
# implemented as the warmup-cosine schedule 2.5e-4 -> 2.5e-5 (lr_cosine).
PPO_OVERRIDES = {
    "gamma": 0.9999,
    "lr_schedule": "fixed",
    "learning_rate": 2.5e-4,
    "lr_cosine": True,
    "learning_rate_end": 2.5e-5,
    "lam": 0.99,
    "entropy_coef": 4.75e-8,
    "num_learning_epochs": 10,
    "clip_param": 0.2,
    "value_loss_coef": 0.5,
}

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
        bc_warmstart=BC_WARMSTART,
        **PPO_OVERRIDES,
    )
    kwargs.update(overrides)
    run_training(TASK, LowThrustTransferEnv, **kwargs)


if __name__ == "__main__":
    main()
