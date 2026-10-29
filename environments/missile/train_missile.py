"""RSL-RL PPO training entry for the Missile-Intercept task
(MissileInterceptEnv).

6-DOF tail-controlled missile vs a kinematic 3-DOF target (straight / weave /
barrel at 250 m/s, 4 km): proximity-fuse kill (miss < 15 m) at any physics
substep. The policy commands body rates (roll +-6 rad/s, pitch/yaw +-1.0
rad/s); the boost-coast engine is autonomous, so there is no throttle dim.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_missile.py`.
Shared PPO machinery lives in algorithms/train_utils.py.

VANILLA PPO, from scratch (2026-10-24 decision): no BC warm start, no
sigma pinning, no obs-normalizer freeze -- the shared shared-defaults PPO
exactly as the other task entries run it. The two learnability levers live in
the ENV instead (measured history in .trae/documents/missile-6dof-env.md):

  1. Reward redesigned for from-scratch guidance learning: the old
     -0.5*|LOS rate| penalty accumulated to -92/episode (vs the +50 kill
     bonus) and dominated the return; it is replaced by a dense
     velocity/LOS-alignment potential (+0.02 * cos(v_hat, r_hat)) that rewards
     flying the collision course and cannot be gamed by flying away.
  2. Maneuver authority raised: OMEGA_LAT 0.5 -> 1.0 rad/s (transient
     terminal-correction authority above the steady a/V turn rate) and
     KILL_RADIUS 8 -> 15 m. With these, from-scratch PPO has a smooth
     progress+alignment gradient from the ballistic zero policy all the way
     to the fuse.

Zero-init output head (rocket-style patch below): the initial policy is
exactly do-nothing -- zero body-rate commands -- which here is NOT a degenerate
start: the fire-control lead aim puts the unguided missile on a collision
course, so the zero policy already scores progress + alignment reward on
straight-line targets and PPO bootstraps from there.

Direct benchmark training (2026-10-24, user decision): one run at the
fixed medium engagement geometry, from scratch, vanilla PPO. The earlier
attempt failed with the OLD reward (the -0.5*|LOS rate|
penalty dominated the return at -92/episode); the dense alignment potential
below now gives a gradient even when rollouts contain no kills, so the
benchmark run is viable.

PPO deviations from the shared fair-comparison baseline (starting point; to be
sweep-tuned like every other task entry -- see .trae/documents/
missile-6dof-env.md):
  num_steps_per_env 64 : 900-step episodes; the shared 32-step rollout would
      hold ~37 episodes/batch, 64 doubles it and halves advantage noise.
  gamma 0.995          : the kill lands ~250-400 steps into a 900-step
      episode; 0.99 discounts a 300-step-horizon bonus by 95%, 0.995 by 78%.
  lam 0.97, 10 learning epochs : rocket-entry values under the same
      long-horizon sparse-bonus structure (arXiv:2407.15083 alignment).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

# Zero-init the final linear head of MLPModel so the initial policy is exactly
# do-nothing (zero body-rate commands).  See train_rocket.py for the full
# rationale; here do-nothing additionally rides the fire-control lead aim
# (straight-line collision course).
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

from algorithms.train_utils import run_training
from environments.missile import MissileInterceptEnv

TASK = 'missile'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # fresh from-scratch run; True resumes results/missile checkpoint
CLEAN = True             # wipe results/missile before this fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
# Dense guidance shaping variant (see env.py reward block / _zem docstring):
#   'graded_terminal' velocity/LOS alignment over the whole flight (default)
#   'align_gated'     same signal, only inside ZEM_GATE (800 m). Measured on
#                     APN vs zero-policy: discriminative gap unchanged
#                     (0.205 vs 0.184/ep) while the policy-insensitive
#                     plateau drops 4.36 -> 1.18/ep and its std falls 33%.
#   'zem'             zero-effort-miss telescoping. REJECTED by measurement
#                     (gap/sd 0.17, magnitude 1/10 of align): the instantaneous
#                     ZEM is dominated by velocity-direction noise and ignores
#                     the target's future maneuver.
#   'terminal_gauss'  hard +100 fuse replaced by a soft KILL_BONUS*exp(-(d/r)^2)
#                     payoff. Screening on the APN endgame-ablation family
#                     (88% vs 33% kill rate): span/sd 1.40 vs 1.19 for the hard
#                     bonus and per-episode std 51.6 -> 25.9.
# Everything else (PPO hyperparameters, seed, reward terminal grading) is held
# fixed so the arms differ in exactly one shaping term.
REWARD_VARIANT = 'terminal_gauss'
ENV_KWARGS = {'reward_variant': REWARD_VARIANT}
# BC warm start: OFF for vanilla PPO (enable by setting a dict here and passing
# bc_warmstart=BC_WARMSTART below; 300 epochs at seed 7 previously reached
# 0.62 greedy SR at d=0 -- but the fit is brittle under PPO updates).
BC_WARMSTART = None

if __name__ == '__main__':
    run_training(TASK, MissileInterceptEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, bc_warmstart=None,
                 out_dir=_ROOT/'results'/f'missile_{REWARD_VARIANT}',
                 env_kwargs=ENV_KWARGS,
                 num_steps_per_env=64,
                 lr_schedule='fixed', learning_rate=3e-4,
                 # 2026-10-28, adjustment 2 of 2 (root-cause reward rebalance in
                 # env.py: KILL_BONUS 50->100, K_ALIGN 0.02->0.01, K_CTRL
                 # 0.01->0.003; see constants there). Run 1 (gamma .995, ent
                 # .007): kills decayed 11%->1% as ctrl collapsed -26->-1.7 and
                 # std 0.46->0.21 -- the discounted kill bonus was too weak vs
                 # the constant ~24/ep shaping plateau. Run 2 (gamma .999, ent
                 # .015): std exploded 0.58->5.2, saturated noise. So: gamma back
                 # to 0.995 (stable advantages), entropy to the 0.010 midpoint,
                 # and the kill differential carried by the env weights instead.
                 # 2026-10-28 run 4: entropy 0.010->0.007 -- run 3 showed the std
                 # ratchet (0.57->160) under entropy 0.010 (clipped samples give
                 # zero std-gradient); 0.007 kept std bounded in run 1. Env
                 # reward rebalance kept as-is.
                 # 2026-10-29 comparison arm: identical PPO hyperparameters and
                 # seed; only the env terminal-miss grading and ctrl-penalty
                 # removal change. Baseline remains in results/missile.
                 gamma=0.995, lam=0.97, entropy_coef=0.007,
                 init_std=0.5,
                 num_learning_epochs=10,
                 clip_param=0.2, value_loss_coef=1.0, desired_kl=0.01)
