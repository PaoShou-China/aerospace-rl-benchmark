"""RSL-RL PPO training entry for the Rocket-Landing task (RocketLandingEnv).

F9-class booster with gimballed main engine and grid fins on the Genesis
rigid-body engine: soft landing on the pad -- contact with |v| < 0.02 m/s,
|omega| < 0.02 rad/s and tilt < 0.1 rad -- within 1200 policy steps (30 s).

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rocket.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
CURRICULUM / ENV_KWARGS expose the project-wide difficulty contract
(src/common/curriculum.py): None = benchmark, float = static difficulty
in [0, 1], True = success-rate-gated auto-ramp.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.rocket import RocketLandingEnv

# Zero-init the last linear layer of every MLPModel (actor mean head and critic
# head) so the initial policy is exactly do-nothing: ignition cmd 0.5 (raw 0 ->
# [0,1] box midpoint, at the 0.5 relight threshold so the engine stays off on
# the mean action), throttle 0.5, zero fins and gimbal. Measured 2026-10-20 on
# the previous staged task: the default kaiming-style init gives initial mean
# actions ~N(0, 0.6) per dim, which ignited immediately and tumbled the
# booster; zero-init is the standard do-nothing policy start. With the
# upstream [0,1] ignition box, exploration still fires on ~50% of sampled
# steps (raw > 0), exactly like upstream's own exploration regime.
import torch.nn as nn
from rsl_rl.models import MLPModel as _MLPModel

_mlp_init_orig = _MLPModel.__init__

def _mlp_init_zero_last(self, *args, **kwargs):
    _mlp_init_orig(self, *args, **kwargs)
    last = None
    for m in self.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

_MLPModel.__init__ = _mlp_init_zero_last

# The trunk default (arch='flash_residual', FlashResMLPModel) REPLACES self.mlp
# with a fresh nn.Linear AFTER super().__init__ returns, so the MLPModel patch
# above is a no-op for trunk models (verified 2026-10-22: the Sequential head
# it zeroes is discarded). Patch FlashResMLPModel too: zero its final linear
# head, which is the actual actor-mean/critic output layer on the trunk latent.
from common.ppo_flashnet import FlashResMLPModel as _FlashResMLPModel

_flash_res_init_orig = _FlashResMLPModel.__init__

def _flash_res_init_zero_head(self, *args, **kwargs):
    _flash_res_init_orig(self, *args, **kwargs)
    nn.init.zeros_(self.mlp.weight)
    nn.init.zeros_(self.mlp.bias)

_FlashResMLPModel.__init__ = _flash_res_init_zero_head

TASK = 'rocket'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # upstream-aligned task rewrite (2026-10-20): old
                         # checkpoints belong to the dropped staged task, and
                         # the action box changed (ignition/throttle in [0,1]),
                         # so training restarts from scratch
CLEAN = True             # wipe results/rocket before this fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
# Curriculum difficulty (project-wide feature, see src/common/curriculum.py):
# None = off (published benchmark, bit-for-bit); float in [0,1] = static
# difficulty (0 easiest, 1 hardest = benchmark); True = auto-ramp difficulty
# 0 -> 1, advancing +0.1 whenever a 1024-episode window closes at >= 60% success.
CURRICULUM = None
ENV_KWARGS = ({'curriculum': True} if CURRICULUM is True
              else {} if CURRICULUM is None else {'difficulty': CURRICULUM})


class _JsonStd(list):
    """Per-dim init std as a JSON-serializable list that rsl_rl still accepts.

    rsl_rl GaussianDistribution computes `init_std * torch.ones(output_dim)`,
    which requires a tensor (a plain list crashes there), but run_training
    dumps its config to config.json with json.dumps, which crashes on a
    tensor ("Object of type Tensor is not JSON serializable", measured at the
    END of a full 1000-iter run 2026-10-20 -- the training itself completed).
    This list subclass converts on multiply, satisfying both.
    """
    def __mul__(self, other):
        return torch.as_tensor(self, dtype=other.dtype, device=other.device) * other
    __rmul__ = __mul__

INIT_STD = _JsonStd([0.05, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05])
# Task-specific per-dim exploration std (documented deviation from the shared
# scalar 0.5 default). Rationale (measured 2026-10-20):
#   * a scalar 0.2 kept the ignition/depth exploration alive but the fin/gimbal
#     noise alone tumbling the booster in free-fall: PD-feasibility sweep showed
#     zero-mean sigma=0.2 fin noise drives omega^2 to 1.65 rad^2/s^2 (a trivial
#     PD reflex on the same noise: 0.0016 -- the fins have ample authority), and
#     training time agreed (attitude cost ~-14/episode, reward sinking to -33);
#   * during powered flight the same noise made 94% of pad touches tumbling
#     (fatal). Noise on fins (0-2) and gimbal (5-6) is pure disturbance -- the
#     attitude reflex is dense-reward and needs no exploration -- while the
#     ignition (3, hysteretic +0.5/-0.5 switch) and throttle depth (4) dims DO
#     need sigma~0.2 to discover the suicide-burn timing. 0.05 on the
#     disturbance dims cuts the noise-induced tumble ~16x (scales ~sigma^2).

if __name__ == '__main__':
    # num_steps_per_env=64 (documented deviation from the shared 32 default,
    # user-sanctioned fallback knob, 2026-10-20): at 32 the batch holds only
    # ~8 episodes, and the advantage signal that separates a 4 m/s touch from
    # a 12 m/s touch (terminal delta ~+3) is estimated from so few episodes
    # that touch-quality clusters keep regressing (measured: touch_speed
    # bounces 4.2 <-> 14 across iterations while its mean grinds down).
    # 64 doubles the batch (16 episodes/update), halving gradient noise.
    # Web-alignment (2026-10-21): PPO values follow "Rocket Landing Control with
    # Random Annealing Jump Start RL" (arXiv:2407.15083): lr 3e-4, gamma 0.995,
    # lam 0.97, entropy 0.007, clip 0.2, target KL 0.01. The paper uses batch
    # 20000 x 30 gradient steps; 64 steps/env x 1024 envs x 10 epochs gives
    # ~655k gradient samples/update, close to the paper's ~600k (10 epochs is
    # the equivalent of its 30 gradient steps at this batch size).
    run_training(TASK, RocketLandingEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, init_std=INIT_STD, num_steps_per_env=64,
                 lr_schedule='fixed', learning_rate=3e-4,
                 gamma=0.995, lam=0.97, entropy_coef=0.007,
                 num_learning_epochs=10,
                 clip_param=0.2, value_loss_coef=1.0, desired_kl=0.01,
                 env_kwargs=ENV_KWARGS)
