"""RSL-RL PPO training entry for the CWH rendezvous task
(SpaceRendezvousEnv).

Clohessy-Wiltshire-Hill relative motion in the LVLH frame of a 400 km circular
orbit (pure-tensor symplectic Euler, dt = 1 s): medium-range approach from
1000-1500 m to range < 10 m with speed < 0.5 m/s within 1000 steps, while
avoiding the 5 m keep-out sphere around the target and the 2000 m failure
bound.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_rendezvous.py`.
Shared PPO machinery lives in algorithms/train_utils.py.

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

Spawn distribution (2026-10-21): the 800-1200 m spawn makes the terminal
"d<10 m and v<0.5 m/s" event too rare to bootstrap (0.8% success with the
shared wide net) while the same CWH machinery reaches ~94% on docking at
50-100 m; training runs directly at the benchmark spawn. The
task table (goal, keep-out, 1000-step horizon, reward) is unchanged, and
eval.py (no env kwargs) always evaluates the full benchmark spawn
(1000-1500 m since 2026-10-24, widened from 800-1200 m after the baseline
sat at 100% success).

Narrow net: 64x64 was the only arm that learned this task (22% success vs
0.8% for 512x256x128) under the retired residual trunk; with the trunk removed
(2026-10-28) the shared 2x128 tanh MLP is used, i.e. the archived 64x64 recipe
is no longer reproducible.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

from algorithms.train_utils import run_training
from environments.satellite.rendezvous_docking import SpaceRendezvousEnv

TASK = 'rendezvous'
NUM_ENVS = 1024
UPDATES = 1000
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # resume from the newest model_*.pt / checkpoint.pt
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (algorithms/flashsac/flash_sac.py)
# 2026-10-28: BC warm start proved the init/reward fine (93-98% rolling success,
# iterations 1-5) and lr 1e-3 (+400-update decay) destroyed it within ~5 updates;
# with the warm start on, the lr drops to the gentler fixed 3e-4 (decay dropped).
# 2026-10-28 final arm: sigma_pin=0.2 + lr_sigma_coupling=1.5e-3 (lr=3e-4 at the
# pinned sigma 0.2, drift rate lr/sigma held constant) -- the BC actor is near-
# deterministic (sigma->0), the mean's policy gradient scales ~1/sigma^2 and the
# bust is lr- (1e-4..1e-3) and entropy- (0/.003) independent; pinning sigma
# keeps the noisy-rollout objective aligned with greedy behavior (measured:
# sigma 0.2 holds ~60-64%) and entropy 0.0 is moot since log_std is overwritten.
# 2026-10-28 variant: sigma_pin 0.2 -> 0.05 (0.2 only slowed the bust: ~10%
# rolling plateau then 0 by end, cumulative 0.98%, greedy 0.0%); tighter pin
# sits closer to the greedy maneuver. lr = 1.5e-3 * 0.05 = 7.5e-5.
# 2026-10-29 ARM A: PPO knobs unchanged (this is the best prior arm); the lever
# moves to the env -- time_penalty=.01/step in env.py makes the ~190 m hover
# bust target unprofitable (SpaceGym CVPRW 2026; AAS 23-026).
LR_SIGMA_COUPLING = 1.5e-3
SIGMA_PIN = 0.2
# 2026-10-29 no-BC from-scratch sweep (all with time_penalty=.01 + spawn easing
# + 100-pt spread, seed 7): fixed lr 3e-4/ent .003 -> greedy 11% (boom 99%,
# tail 50%); coupling 6e-4/ent .003 (sigma stuck 0.552) -> 21%; coupling 6e-4/
# ent .0005 (sigma anneals to 0.23-0.28, lr follows) -> 24% @1500 upd, 10% @3000;
# pin 0.05 -> 0% flat (no discovery); BEST: pin 0.2 + coupling 1.5e-3 (lr 3e-4)
# @1500 upd -> rolling 40%, greedy 26%. Conclusion: without the BC warm start
# the sampled-rollout noise keeps re-eroding the discovered maneuver -- no PPO
# schedule among those tested holds it; the hot start (sigma pin 0.05 + BC) is
# what held 99.5% greedy. BC_WARMSTART stays None per user decision.
# 2026-10-29 PPO-knob + env-param sweep (5 arms, seed 7, 1024 envs, 1500
# updates each, PPO core = the pin-0.2/coupling-1.5e-3 arm above; greedy =
# deterministic mean action at the BENCHMARK spawn 1000-1500 m, 1024 envs):
#   a1 num_steps_per_env 250->500, env unchanged        : greedy 30.3%
#   a2 spawn (300,600)->(200,400) + time_penalty .01->.02: greedy 40.6%  <- WINNER
#   a3 a1+a2 combined                                    : greedy  3.1%  (overfits
#       the eased spawn; 2x data at 500-step rollouts on the short-spawn env)
#   a4 terminal_bonus 100->200 / viol_penalty 100->50    : greedy 23.7%
#   a5 a2 env + num_learning_epochs 4->10 + clip 0.2->0.1: greedy  0.2%  (boom
#       to 100% rolling then full collapse)
# Every arm boomed to ~99-100% rolling at its eased training spawn, then
# eroded (tails 43-51%); the levers that moved benchmark greedy are env-side
# (closer spawn + harder hover floor). New default = a2: the PPO constants
# below are UNCHANGED, the env moves to (200,400) + time_penalty .02 (see
# satellite/rendezvous_docking/env.py).
PPO_OVERRIDES = dict(num_steps_per_env=250, gamma=.999, num_mini_batches=1,
                     entropy_coef=0.0)
UPDATES = 1500

# BC warm start (algorithms/warmstart/bc_warmstart.py): pretrain the actor/critic on the
# conventional controller (driven directly by the BC collector) before PPO. None = off.
# 2026-10-28: enabled (shared defaults) -- pure PPO from scratch busts back to ~0 after
# booming to 78% success (100-pt spread); BC inherits the 100%-success expert's ceiling.
# 2026-10-29: OFF by user decision ("默认不用热启动") -- time_penalty + spawn easing
# (env.py) must bootstrap from scratch on their own.
BC_WARMSTART = None

# 2026-10-29: training-only spawn easing as a local subclass. The env default
# stays at the benchmark (1000,1500) so scripts/eval.py (which rebuilds the env
# from config.json env_kwargs, empty here) always evaluates benchmark spawn.
class _EasedSpawnEnv(SpaceRendezvousEnv):
    init_range=(200.,400.)   # sweep winner a2 (see the sweep table above)

if __name__ == '__main__':
    run_training(TASK, _EasedSpawnEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN,
                 algo=ALGORITHM, bc_warmstart=BC_WARMSTART,
                 lr_sigma_coupling=LR_SIGMA_COUPLING, sigma_pin=SIGMA_PIN,
                 **PPO_OVERRIDES)
