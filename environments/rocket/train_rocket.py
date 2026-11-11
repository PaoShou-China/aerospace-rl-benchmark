"""RSL-RL PPO training entry for the Rocket-Landing task (RocketLandingEnv).

F9-class booster with gimballed main engine and grid fins on the Genesis
rigid-body engine: soft landing on the pad -- contact with |v| < 0.02 m/s,
|omega| < 0.02 rad/s and tilt < 0.1 rad -- within 1200 policy steps (30 s).

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rocket.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in algorithms/train_utils.py.
WIND configures the wind disturbance passed to the env ctor (steady_wind,
dryden, severity); None = calm air (bit-for-bit benchmark behavior).
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training  # noqa: E402
from environments.rocket import RocketLandingEnv  # noqa: E402

# Zero-init the last linear layer of every MLPModel (actor mean head and critic
# head) so the initial policy is exactly do-nothing: ignition cmd 0.5 (raw 0 ->
# [0,1] box midpoint, at the 0.5 relight threshold so the engine stays off on
# the mean action), throttle 0.5, zero fins and gimbal. Measured 2026-10-20 on
# the previous staged task: the default kaiming-style init gives initial mean
# actions ~N(0, 0.6) per dim, which ignited immediately and tumbled the
# booster; zero-init is the standard do-nothing policy start. With the
# upstream [0,1] ignition box, exploration still fires on ~50% of sampled
# steps (raw > 0), exactly like upstream's own exploration regime.
import torch.nn as nn  # noqa: E402
from rsl_rl.models import MLPModel as _MLPModel  # noqa: E402

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

TASK = "rocket"
NUM_ENVS = 1024
UPDATES = 2000
SEED = 7
BACKEND = "gpu"  # 'gpu' | 'cpu'
RESUME = False  # upstream-aligned task rewrite (2026-10-20): old
# checkpoints belong to the dropped staged task, and
# the action box changed (ignition/throttle in [0,1]),
# so training restarts from scratch
CLEAN = True  # wipe OUT_DIR before this fresh run
ALGORITHM = "ppo"
# Wind disturbance (environments/common/disturbances.py): calm air for the
# four-arm comparison (the classical-baseline records in results/rocket were
# evaluated with env_kwargs={} calm air -- same caliber required; the dryden
# robustness arm is a separate study, not this benchmark).
WIND = dict(steady_wind=None, dryden=0.0, severity="moderate", aero="realistic")
# Rocket benchmark version:
#   'v2' (default, current Falcon9): 7-dim finlet mixer (finlet x/y/yaw, ignition, throttle, 2 gimbals),  # noqa: E501
#     40 Hz, rocket_v2.urdf with 6 spherical footpads, ever_stable_2s primary metric
#   'v1' (real-scale): single-Merlin terminal landing, dual-tank variable mass, 60 s horizon
BENCHMARK_VERSION = "v1"  # 'v1' (real-scale F9, lenient success) | 'v2' (scaled legacy, strict)
# separate from results/rocket, which holds the classical-baseline eval
# records (baseline.json / baseline_pd_landing.json)
OUT_DIR = (
    "results/rocket_v1_ppo" if BENCHMARK_VERSION == "v1" else "results/rocket_ppo"
)
# reward (2026-10-22): the redesigned shaping (impact-speed potential, off-axis
# cost, first-touch bonus, success +100) is now the env's ONLY implementation --
# the verbatim-upstream arm was deleted from rocket/env.py after the
# 1000-update run that finished today with success 0.000 (vs classical baseline
# 0.668) converged to a reward-farming local optimum on it: hover on the pad
# axis farming the absolute holding term (+0.3/(||p_xy||+0.1) -> +1810 of the
# +1471 mean episode return), burn the 5% fuel in ~305 steps, then hit the pad
# at ~40 m/s (fatal, touch_speed 41.7, success_rate 0 across all 1000 updates).


class _JsonStd(list):
    """Per-dim init std as a JSON-serializable list that rsl_rl still accepts.

    Only relevant to the optional Gaussian policy: rsl_rl
    GaussianDistribution computes `init_std * torch.ones(output_dim)`, which
    requires a tensor (a plain list crashes there), but run_training dumps its
    config to config.json with json.dumps, which crashes on a tensor
    ("Object of type Tensor is not JSON serializable", measured at the END of
    a full 1000-iter run 2026-10-20 -- the training itself completed).
    This list subclass converts on multiply, satisfying both.

    The default policy is BetaDistribution (bounded, rescaled to `[-1, 1]`),
    which has no `init_std` knob at all: INIT_STD below is inert unless a
    caller switches `distribution="gaussian"`.
    """

    def __mul__(self, other):
        return torch.as_tensor(self, dtype=other.dtype, device=other.device) * other

    __rmul__ = __mul__


# Per-dim init_std for the unified 7-dim finlet convention:
#   [finlet_x, finlet_y, finlet_yaw, ignition, throttle, gimbal_x, gimbal_y]
# Historical (Gaussian-policy era): the default BetaDistribution ignores
# init_std, so this vector is inert in shipped runs; it only takes effect if
# a caller passes distribution="gaussian" to build_cfg.
# Same rationale as before: 0.2 on ignition/throttle (exploration-critical),
# 0.05 on finlets/gimbals (attitude reflex is dense-reward, no exploration).
INIT_STD = _JsonStd([0.05, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05])
# Task-specific per-dim exploration std (documented deviation from the shared
# scalar 0.5 default; Gaussian-policy-only, measured 2026-10-20):
#   * a scalar 0.2 kept the ignition/depth exploration alive but the fin/gimbal
#     noise alone tumbling the booster in free-fall: PD-feasibility sweep showed
#     zero-mean sigma=0.2 fin noise drives omega^2 to 1.65 rad^2/s^2 (a trivial
#     PD reflex on the same noise: 0.0016 -- the fins have ample authority), and
#     training time agreed (attitude cost ~-14/episode, reward sinking to -33);
#   * during powered flight the same noise made 94% of pad touches tumbling
#     (fatal). Noise on fins (0-3) and gimbal (6-7) is pure disturbance -- the
#     attitude reflex is dense-reward and needs no exploration -- while the
#     ignition (4, hysteretic +0.5/-0.5 switch) and throttle depth (5) dims DO
#     need sigma~0.2 to discover the suicide-burn timing. 0.05 on the
#     disturbance dims cuts the noise-induced tumble ~16x (scales ~sigma^2).

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)


def main(**overrides):
    """Run one training. Keyword overrides (seed / out_dir / bc_warmstart /
    env_wrapper / ...) let the 5-seed orchestrator drive the same constants.

    Hyperparameters (2026-11-03 5-seed protocol, arXiv:2407.15083 alignment):
    num_steps_per_env 100 (matches other tasks; at 64 the batch holds only ~8
    episodes and touch-quality clusters keep regressing);
    lr 3e-4, gamma 0.995, lam 0.97, entropy 0.007, clip 0.2, 5 epochs --
    100 steps/env x 1024 envs x 5 epochs gives 512k gradient samples/update."""
    env_kwargs = dict(WIND, benchmark_version=BENCHMARK_VERSION)
    kwargs = dict(
        num_envs=NUM_ENVS,
        updates=UPDATES,
        seed=SEED,
        backend=BACKEND,
        resume=RESUME,
        clean=CLEAN,
        out_dir=OUT_DIR,
        algo=ALGORITHM,
        init_std=INIT_STD,
        num_steps_per_env=100,
        lr_schedule="fixed",
        learning_rate=3e-4,
        gamma=0.995,
        lam=0.97,
        entropy_coef=0.007,
        num_learning_epochs=5,
        clip_param=0.2,
        value_loss_coef=1.0,
        desired_kl=0.01,
        env_kwargs=env_kwargs,
        bc_warmstart=BC_WARMSTART,
    )
    kwargs.update(overrides)
    run_training(TASK, RocketLandingEnv, **kwargs)


if __name__ == "__main__":
    main()
