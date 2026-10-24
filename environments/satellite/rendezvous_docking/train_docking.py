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

import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.satellite.rendezvous_docking import SpaceDockingEnv

# Zero-init the final linear head of FlashResMLPModel (actor mean head and
# critic head) so the initial trunk policy is exactly do-nothing (zero thrust).
# Rationale (measured 2026-10-22): with the default init the trunk's RMSNorm /
# weight-renormalization stack starts from a badly non-neutral behavior
# (iter-0 mean reward -343 vs the MLP arm's +23), so the first ~80 updates are
# spent just climbing back to neutral approach; afterwards the run plateaus at
# the hover reward (~58-60) for 900+ updates with ZERO successes in 65937
# episodes -- the critic (value loss flat at ~0.9, never spiking like the
# MLP's 127 at discovery) smooths away the 1 m capture cliff and the policy
# gradient toward capture stays under the noise floor. A neutral start removes
# the wasted recovery phase and gives the advantage signal an unbiased
# do-nothing baseline, mirroring the zero-init precedent in train_rocket.py.
from algorithms.ppo.ppo_flashnet import FlashResMLPModel as _FlashResMLPModel

_flash_res_init_orig = _FlashResMLPModel.__init__

def _flash_res_init_zero_head(self, *args, **kwargs):
    _flash_res_init_orig(self, *args, **kwargs)
    nn.init.zeros_(self.mlp.weight)
    nn.init.zeros_(self.mlp.bias)

_FlashResMLPModel.__init__ = _flash_res_init_zero_head

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
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)

# Architecture A/B (user decision 2026-10-22: docking keeps BOTH arms). CAUTION
# (measured 2026-10-22): the archived 94% MLP reference predates the 1024-env
# fair-comparison change -- it ran at num_envs=64 (possibly lr 3e-4); every
# 1024-env arm so far (trunk AND mlp) plateaus at ~2-3%, so the dominant
# variable is the config vintage (batch size / lr), NOT the architecture.
ARMS = (('flash_residual', None), ('mlp', str(Path(__file__).resolve().parents[2] / 'results' / 'docking_mlp')))

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = None

if __name__ == '__main__':
    for arch, out_dir in ARMS:
        run_training(TASK, SpaceDockingEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                     backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                     out_dir=out_dir, arch=arch,
                     lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                     entropy_coef=ENTROPY_COEF, gamma=GAMMA,
                     bc_warmstart=BC_WARMSTART)
