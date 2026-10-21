"""RSL-RL PPO training entry for the Rocket-Landing task (RocketLandingEnv).

F9-class booster with gimballed main engine and grid fins on the Genesis
rigid-body engine: soft landing on the pad -- contact with |v| < 0.02 m/s,
|omega| < 0.02 rad/s and tilt < 0.1 rad -- within 1200 policy steps (30 s).

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rocket.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.rocket import RocketLandingEnv

# Zero-init the last linear layer of every MLPModel (actor mean head and critic
# head) so the initial policy is exactly do-nothing: engine off (dim 3 identity
# mapping, fire threshold 0.5), min throttle depth (dim 4 dead zone), zero fins
# and gimbal. Measured 2026-10-20: the default kaiming-style init gives initial
# mean actions ~N(0, 0.6) per dim, and THIS task's action structure turns a
# positive dim-3/4 draw into an immediate un-commanded 3-TWR burn (deterministic
# model_0 rollout: 100% ignite at step 1, depth 0.56 -> climb/tumble, episodes
# 375 steps at reward -216). Zero-init is the standard do-nothing policy start.
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

TASK = 'rocket'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RNN = False              # GRU actor/critic backbone via rsl-rl RNNModel
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1
RESUME = False           # upstream-aligned task rewrite (2026-10-20): old
                         # checkpoints belong to the dropped staged task, and
                         # the action box changed (ignition/throttle in [0,1]),
                         # so training restarts from scratch
CLEAN = True             # wipe results/rocket before this fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
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
    run_training(TASK, RocketLandingEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, init_std=INIT_STD, num_steps_per_env=64)
