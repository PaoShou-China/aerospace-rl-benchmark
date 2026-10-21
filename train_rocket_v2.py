"""RSL-RL PPO training entry for the Rocket-Landing task -- v2 reward arm.

Identical to train_rocket.py (the upstream-faithful reward arm): the PPO
configuration, per-dim exploration stds and the do-nothing zero-init patch
are imported from there, so the A/B can never drift apart. The ONLY
difference is the reward, via reward_version='v2' (2026-10-21 redesign,
rationale documented in src/rocket/env.py):
  * impact-speed potential  brake = 10*(Phi_prev - Phi),
    Phi = sqrt(vz^2 + 2*9.81*z)  replaces the exp(z) deceleration term
    (numerically dead above ~10 m while the suicide burn must start at
    z ~150-300 m; paid free fall through the +1 offset; its sign(vz) branch
    slammed any upward motion, settle bounces included, with -hundreds/step);
  * off-axis cost -0.1*||p_xy|| replaces the farmable holding term
    +0.3/(||p_xy||+0.1) (hover-on-the-pad-axis exploit; hovering 30 s at
    r=0.5 m now nets ~-1300 instead of ~+9720);
  * first-touch quality bonus 50*max(0, 1 - v_first) replaces the
    per-contact-step +5-0.3|vz|; success +3 -> +100.
Episode semantics (touch/settle/fatal) are unchanged. Per-component reward
accounting lands in Episode/rew_* tensorboard scalars and run.json
completed_metrics keys for both arms. Results go to results/rocket_v2 so
the upstream arm in results/rocket stays intact.

Run: python train_rocket_v2.py
"""
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

# importing train_rocket applies its zero-init patch and pins the exact
# upstream-arm hyperparameters (fair A/B by construction)
import train_rocket as upstream

from common.train_utils import run_training
from pyflyt_genesis.rocket import RocketLandingEnv

if __name__ == '__main__':
    run_training(upstream.TASK, partial(RocketLandingEnv, reward_version='v2'),
                 num_envs=upstream.NUM_ENVS, updates=upstream.UPDATES, seed=upstream.SEED,
                 backend=upstream.BACKEND, resume=upstream.RESUME, clean=upstream.CLEAN,
                 out_dir=str(Path(__file__).resolve().parent / 'results' / 'rocket_v2'),
                 algo=upstream.ALGORITHM, init_std=upstream.INIT_STD, num_steps_per_env=64,
                 lr_schedule='fixed', learning_rate=3e-4,
                 gamma=0.995, lam=0.97, entropy_coef=0.007,
                 num_learning_epochs=10,
                 clip_param=0.2, value_loss_coef=1.0, desired_kl=0.01,
                 env_kwargs=upstream.ENV_KWARGS)
