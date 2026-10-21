"""RSL-RL PPO training entry for the CWH rendezvous task
(SpaceRendezvousEnv).

Clohessy-Wiltshire-Hill relative motion in the LVLH frame of a 400 km circular
orbit (pure-tensor symplectic Euler, dt = 1 s): medium-range approach from
800-1200 m to range < 10 m with speed < 0.5 m/s within 1000 steps, while
avoiding the 5 m keep-out sphere around the target and the 2000 m failure
bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rendezvous.py`.
Shared PPO machinery lives in src/common/train_utils.py.

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
itself; see src/docking/env.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.docking import SpaceRendezvousEnv

TASK = 'rendezvous'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RNN = False              # GRU actor/critic backbone via rsl-rl RNNModel
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
PPO_OVERRIDES = dict(num_steps_per_env=250, gamma=.999, num_mini_batches=1,
                     lr_schedule='fixed', learning_rate=.001, entropy_coef=.003)

if __name__ == '__main__':
    run_training(TASK, SpaceRendezvousEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, rnn=RNN, rnn_hidden_dim=RNN_HIDDEN_DIM,
                 rnn_num_layers=RNN_NUM_LAYERS, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, lr_decay_updates=400, **PPO_OVERRIDES)
