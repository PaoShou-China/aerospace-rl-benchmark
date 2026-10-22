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
src/common/train_utils.py.
ALGORITHM / DIFFICULTY / ENV_KWARGS expose the shared switches: the
algorithm arm ('ppo' | 'flash_sac') and the static difficulty knob:
None = benchmark, float in [0, 1] = static difficulty (0 easiest,
1 hardest = benchmark).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.transfer import LowThrustTransferEnv

TASK = 'transfer'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)

# Task-specific deviations from the shared fair-comparison PPO configuration.
# lr pinned to the proven recipe (2026-10-20, 1000-update verification run:
# mean_reward 0.14->1.55, success 100% from update 31, final ckpt 100% / +20.6
# km): KL-adaptive 3e-4 is healthy on this short 166-step task (lr stayed
# 4e-4..3e-3, no floor slam, std 0.5->0.10), so neither the adaptive pathology
# nor the fixed-1e-3 default was ever beaten here; both are untested at the
# new [128,128] network scale.
# Web-alignment (2026-10-21): values follow "RL for Low-Thrust Trajectory Design
# of Interplanetary Missions" (arXiv:2008.08501): gamma 0.9999, lam 0.99,
# lr 2.5e-4, entropy 4.75e-8, 30 epochs, clip 0.3, vf_coef 0.5. The paper
# linearly decays lr (2.5e-4 -> 0) and clip (0.3 -> 0); rsl-rl only supports
# fixed/adaptive schedules, so the constant start values are used here.
PPO_OVERRIDES = {'gamma': 0.9999, 'lr_schedule': 'fixed', 'learning_rate': 2.5e-4,
                 'lam': 0.99, 'entropy_coef': 4.75e-8, 'num_learning_epochs': 30,
                 'clip_param': 0.3, 'value_loss_coef': 0.5}

# Static difficulty (None = published benchmark, bit-for-bit; float in [0,1]
# = 0 easiest, 1 hardest = benchmark). Curriculum learning was removed.
DIFFICULTY = None
ENV_KWARGS = {} if DIFFICULTY is None else {'difficulty': DIFFICULTY}

# BC warm start (src/common/bc_warmstart.py): pretrain the actor/critic on the
# classical nominal controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == '__main__':
    run_training(TASK, LowThrustTransferEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, env_kwargs=ENV_KWARGS, bc_warmstart=BC_WARMSTART,
                 **PPO_OVERRIDES)
