"""RSL-RL PPO training entry for the CWH rendezvous task
(SpaceRendezvousEnv).

Clohessy-Wiltshire-Hill relative motion in the LVLH frame of a 400 km circular
orbit (pure-tensor symplectic Euler, dt = 1 s): medium-range approach from
800-1200 m to range < 10 m with speed < 0.5 m/s within 1000 steps, while
avoiding the 5 m keep-out sphere around the target and the 2000 m failure
bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rendezvous.py`.
Shared PPO machinery lives in algorithms/train_utils.py.
CURRICULUM / ENV_KWARGS expose the project-wide difficulty contract
(no separate module; DIFFICULTY constant in each train entry): None = benchmark, float = static difficulty
in [0, 1], True = success-rate-gated auto-ramp (this task's proven
default, shipped enabled).

Task-specific deviations from the shared fair-comparison baseline (all
driven by the 1000-step horizon and verified in short sweep runs; everything
not listed here is identical to the other task entries):
  num_steps_per_env 250 : rollout covers 1/4 episode; 32-step slices force
      ~31 bootstraps per episode and cannot credit a ~400-step maneuver.
  gamma 0.999           : with gamma 0.99 the discounted value of the
      accelerate-then-coast approach equals slow creeping (~5 return either
      way); 0.999 restores the long-horizon incentive.
  num_mini_batches 1    : full-batch gradient steps; minibatch noise plus a
      critic lagging the success-phase state shift caused boom-bust policy
      collapse (success ~0.9 then bust) in every other variant tested.
  lr fixed 1e-3         : KL-adaptive rides its 1e-2 ceiling on this task and
      amplifies the boom-bust loop; fixed lr keeps updates even.
  entropy_coef 0.003    : keeps exploration alive without the std runaway
      seen at 0.01 (and without the premature std collapse at 0.001).
  lr_decay_updates 400  : exponential lr half-life (1e-3 at start, ~2.5e-4 by
      update 800). With a constant lr the policy locks onto the maneuver by
      ~update 400, then late-phase advantage noise slowly erodes it (success
      oscillates); the decay keeps the discovery phase near full speed and
      makes the second half converge instead of drift.
The env's progress-potential brake_ref (0.1 m/s^2, soft dead-zone) and the
20-point terminal bonus/violation spread are set on SpaceRendezvousEnv
itself; see satellite/rendezvous_docking/env.py.

Difficulty (2026-10-21): the 800-1200 m spawn makes the terminal "d<10 m and
v<0.5 m/s" event too rare to bootstrap (0.8% success with the shared wide
net) while the same CWH machinery reaches ~94% on docking at 50-100 m. A
success-rate-gated curriculum (150-300 m start, auto-widening to benchmark)
was tried and later removed; training now runs directly at the benchmark
spawn, with the static `difficulty` kwarg kept for controlled experiments.
The task table (goal, keep-out, 1000-step horizon, reward) is unchanged, and
eval.py (no env kwargs) always evaluates the full 800-1200 m benchmark.

Narrow net: 64x64 was the only arm that learned this task (22% success vs
0.8% for 512x256x128), so the shared residual trunk is pinned to 64.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.common.difficulty import resolve_env_kwargs
from environments.satellite.rendezvous_docking import SpaceRendezvousEnv

TASK = 'rendezvous'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
# Web-alignment (2026-10-21): lr 3e-4 follows the hierarchical multiphase
# rendezvous paper (arXiv:2610.17468), which also uses gamma 0.99, lam 0.95,
# entropy 0.01, target KL 0.03. gamma/entropy keep the locally verified 0.999/
# 0.003 (see docstring: gamma 0.99 makes the maneuver look like slow creeping;
# entropy 0.01 caused std runaway). The paper's constant lr replaces the old
# 1e-3 + 400-update decay, so the decay knob is dropped.
PPO_OVERRIDES = dict(num_steps_per_env=250, gamma=.999, num_mini_batches=1,
                     lr_schedule='fixed', learning_rate=3e-4, entropy_coef=.003,
                     trunk_hidden=64, critic_trunk_hidden=64)
# Static difficulty (None = published benchmark, bit-for-bit; float in [0,1]
# = 0 easiest, 1 hardest = benchmark). Curriculum learning was removed.
DIFFICULTY = None
ENV_KWARGS = resolve_env_kwargs('rendezvous', DIFFICULTY)   # validated against environments/common/difficulty.py

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == '__main__':
    run_training(TASK, SpaceRendezvousEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, env_kwargs=ENV_KWARGS, bc_warmstart=BC_WARMSTART,
                 **PPO_OVERRIDES)
