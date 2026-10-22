"""RSL-RL PPO training entry for the Satellite CMG attitude-control task
(SatelliteAttitudeEnv).

Rigid-body satellite slewing under a 4- or 6-CMG control-moment-gyro
cluster (pure-tensor RK4 at 2.5 ms): drive the attitude error below 0.05 rad
with |omega| < 0.02 rad/s by the 1200-step (30 s) episode limit; success is
evaluated only on the final step.

Self-contained entry: every run parameter is hard-coded below, no CLI
arguments. Edit the constants and run `python train_satellite.py`.
Hyperparameters are identical across all per-task entries for fair
comparison; shared PPO machinery lives in src/common/train_utils.py.
DIFFICULTY / ENV_KWARGS expose the static difficulty knob: None = benchmark,
float in [0, 1] = static difficulty (0 easiest, 1 hardest = benchmark).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))  # exposes `common`

from common.train_utils import run_training
from pyflyt_genesis.satellite import SatelliteAttitudeEnv

TASK = 'satellite'
NUM_ENVS = 1024
UPDATES = 1000           # fair-comparison budget (matches the other 6 task entries)
SEED = 7
BACKEND = 'gpu'          # 'gpu' | 'cpu'
RESUME = False           # pure-RL from scratch (warm-start protocol superseded)
CLEAN = True             # wipe stale results/<task> artifacts before a fresh run
ALGORITHM = 'ppo'        # 'ppo' (rsl-rl) | 'flash_sac' (src/common/flash_sac.py)
# 2026-10-22 warm-start refinement (run-9): the archived BC+DAgger checkpoint
# (results/satellite, seed-7 1024-env run) re-evaluates at 94.8% deterministic
# on the benchmark (results/satellite/eval.json) -- far above every pure-RL
# from-scratch attempt (17.4% benchmark; curriculum run 5.5%, stalled at d=0.5,
# archived as results/satellite_rl_benchmark_20261022). Refine it with the
# gate-proven run-8 recipe: sigma pinned 0.1, lr = 1.5e-4 * sigma (constant
# mean-update drift K), obs normalizers frozen (the warm start fitted them),
# 150 updates at benchmark difficulty.
LEARNING_RATE = 3e-4
INIT_STD = 1.0
# Pure-RL entry (arXiv:2505.00165 alignment): gamma 0.999 because 1200-step
# episodes make 0.99 blind to the settle bonus; nsp 150 matches the paper's
# 150-step rollout; 5 mini-batches mirror its ~32-sample minibatches.
# The fixed schedule replaces SB3's KL-adaptive schedule (adaptive is proven
# to clamp to 1e-5 on this task). lr/sigma coupling and obs-norm freezing are
# warm-start-specific and removed.
PPO_OVERRIDES = dict(gamma=0.999, num_steps_per_env=150, num_mini_batches=5,
                     num_learning_epochs=40, entropy_coef=0.0, init_std=INIT_STD,
                     clip_param=0.2)
# 2026-10-20 ROOT-CAUSE FIX: all 7 prior runs failed because the telescoped
# shaping sign was INVERTED (negative potential with Phi_prev - Phi subtraction):
# approaching the target and braking were penalized, so PPO plateaued at
# att_err~0.3-0.6 with <=5% success. env.py now uses the positive
# distance-to-go potential (docking 'progress' convention) with the absolute
# per-step state costs folded into the potential.
LR_SCHEDULE = 'fixed'
# Run 7 (2026-10-20): 100-update dose-response gates at 256 envs isolated the
#   refinement-destroyer: the EXPLORATION SIGMA. At sigma=0.5 the in-rollout
#   success collapses to 2-25% vs 50%+ greedy, so PPO's gradient chases
#   noise-injected return differences instead of policy quality -- greedy slid
#   monotonically 50 -> 59(upd10) -> 12%(upd59) even with frozen obs normalizers
#   and lr/sigma held constant at 1.5e-4 (normalizer drift and sigma-anneal
#   step growth are NOT the mechanism). sigma=0.35 slides slower (66.8 upd10 ->
#   34.8% upd99); sigma=0.2 HOLDS and improves: 50 -> 62-65% band across all
#   100 updates (post99 62.5%). So run 7 pins sigma=0.2 (re-pinned every
#   25-update chunk), couples lr = 1.5e-4*sigma (the empirically stable
#   mean-update-step band, applied before the first chunk too), and freezes
#   the obs normalizers (eval mode skips stat updates, keeping the
#   DAgger-fitted input mapping). Documented deviations: pinned sigma 0.2
#   replaces the shared init_std 0.5 (it IS the variable under test,
#   gate-verified); effective lr 3e-5 replaces 1e-3 (warm-start precision
#   regime, gate-verified); BC+DAgger warm start replaces random init.
#   Every other PPO hyperparameter unchanged.
# Run 7 (2026-10-20, 300 upd, 1024 envs): the full pipeline HELD and refined the
#   warm start through 200 updates (greedy 50 -> 58% @100 -> 57% @200, in-rollout
#   success 25-31% vs run-6's 2-25% collapse, return 364 -> 372, mean att_err
#   0.176 -> 0.147) but then slid late: 54% @240 -> 46% @300. The failure mode
#   is PRECISION SATURATION, not instability: failing envs park at att_err
#   0.10-0.15, inside the 0.2 gate but outside the 0.05 success window, with
#   omega fully damped. At sigma=0.2 the noise-robust optimal mean sits at that
#   offset -- the noisy-rollout objective cannot see the 0.05 window, so late
#   gradient pressure drifts the park point outward. Best checkpoint model_200.
# Run 8 (2026-10-20): sigma_pin 0.2 -> 0.1, resume from model_200. Halving the
#   exploration std halves the noise-robust offset so the noisy objective
#   aligns with greedy precision; the lr/sigma coupling keeps the mean-update
#   drift rate at the gate-stable K=1.5e-4 (lr follows 1.5e-5). Earlier
#   small-sigma collapses (run-5 gate: sigma 0.05 destroyed the policy in one
#   update) were UNCOUPLED -- lr 1e-3 / sigma 0.05 puts the drift at 2e-2,
#   133x the stable band; coupled at K the drift rate is identical to the
#   sigma=0.2 gates that held 62-65%. 150 updates with per-chunk greedy evals.
# Run 8 result: FLAT 55-58% -- stability achieved, ceiling not broken.
# 2026-10-20 PHYSICS CEILING (probe-verified, /tmp/sat_probes/): R(q)(I*w +
#   h_dyn*Sum r_i(theta)) is conserved EXACTLY (residual 4e-10). The reset
#   gimbals theta0~0 give |H0| = 46.2 (4-CMG pyramid net rotor momentum
#   [0,0,4cos54.74deg]), so the bus can rest at the target attitude ONLY with
#   Sum r_i(theta_f) = R(target)^T Sum r_i(0) within |I*w| < 1.2. Optimizer-
#   exact per-env solvability on the seed-99 eval: 206/256 strictly solvable
#   (80.5% CEILING), 231/256 with 0.05-rad attitude slack. All six expert
#   architectures (PD, anti-windup, momentum dump x2, theta*-servo, manifold
#   tracking) converge to 70-71.5% and succeed ONLY on feasible inits (88.3%
#   of strict-solvable, 0% of infeasible). The >80% policy target therefore
#   requires solving 99.5% of feasible inits -- beyond DAgger distillation
#   (student <= expert). The blocking env-level defect: the reset leaves 46
#   N*m*s stored momentum (unphysical for a quiescent spacecraft); a
#   momentum-neutral home theta0 = (pi,0,pi,0) (Sum r_i = 0 exactly) makes
#   every target solvable. User decision pending.
# HISTORICAL: tuning log for the BC+DAgger warm-start runs 1-8 (superseded by
# the pure-RL block above; kept for the reward/physics findings below).
# - num_steps_per_env 250 / gamma 0.999 / num_mini_batches 1 / entropy 0.003 /
#   lr half-life 400: the EXACT rendezvous recipe (structurally identical task:
#   approach a target, brake, hold precision over a 1000-step episode) that
#   converged to 75-90% success. Run 1 under the sign-corrected reward
#   (nsp=64/gamma=0.999/mb=1/entropy=0.005) peaked at reward 76 / 1.9% success
#   and parked the greedy mean dead just outside the brake gate (att_err~0.36):
#   rare-capture credit too sparse at 4096 samples/update and entropy 0.005
#   kept std too high for the precision hold.
# Run 3 (2026-10-20): added near-field potential boost (slope 30 -> 100 below
#   0.5 rad; the assist diagnostic proved capture is worth +560 return but the
#   maneuver earned only +0.2/step shaping, invisible under advantage noise)
#   and entropy 0.001 so std reaches the precision regime before the lr
#   half-life freezes learning. Broke the 2% ceiling (peak 8.3% success,
#   reward 124.6) but the hard settle cliff made near-target values bimodal
#   and the policy retreated to the park equilibrium late.
# Run 4 (2026-10-20): settle bonus replaced by a smooth Gaussian surrogate of
#   the success indicator (same +1 peak at the region center): kills the
#   critic's boundary variance bomb and adds a direct damping gradient.
#   STILL COLLAPSED (peak 10.6% -> 0.78% at update 594): critic_sanity.py proved
#   the decisive blocker is critic MIXTURE-BLINDNESS -- V converged to the
#   outcome mixture (V(e<0.06)=+48 vs +315 empirical on the capture path, 6.6x
#   under-priced) so the per-step capture advantage (+0.17) drowns in advantage
#   noise (+/-3-5) and the mean policy never learns capture.
# Run 5 (2026-10-20): BC + DAgger warm start from the scripted PD + damped-pinv
#   steering controller (/tmp/sat_probes/bc_warmstart.py; probe2: 70.7% success,
#   +596 return). Actor regressed (MSE) onto 1.2M expert (obs, action) pairs,
#   then 9 DAgger rounds (learner-visited states labeled with expert actions,
#   pooled retraining) -> 61.3% greedy success, +431 return, tail att_err med
#   0.032 INSIDE the 0.05 window. Critic regressed onto the empirical
#   gamma=0.999 return-to-go of the same rollouts (capture-path-accurate from
#   step 0). Gate-tested PPO stability before launch (dbg_update.py):
#   exploration std is DECISIVE because the policy-gradient step scales as
#   1/sigma^2 -- sigma=0.05 (BC-precision) amplified the effective lr ~400x and
#   one update destroyed the 59% policy; sigma=0.2 amplified ~25x and degraded
#   to 21%; sigma=0.5 (= the shared init_std default, no deviation) held
#   59 -> 54 -> 51 -> 56% over 3 updates. RESUME=True starts refinement from
#   results/satellite/model_0.pt. Documented deviation: the warm start replaces
#   the random network initialization only; every PPO hyperparameter is
#   unchanged.
# Run 5b (2026-10-20, lr 1e-3, 400 upd): the 3-update gate was too short --
#   greedy success slid 58% -> 29% (upd 99) -> 32% (198) -> 8% (396) while the
#   NOISY-rollout metrics improved (reward 101 -> 293, success 2 -> 25%): at
#   lr 1e-3 the per-update mean drift (step ~ lr/sigma with sigma~0.5)
#   accumulates into degradation faster than the advantage signal re-sharpens
#   the policy; the late sigma anneal to 0.24 amplified the drift further.
# Run 6 (2026-10-20): lr 1e-4 -- the extended 6-update gate at lr 1e-4 HELD the
#   warm start and improved it (55 -> 53 -> 53 -> 62 -> 67 -> 61% greedy, net
#   +6%): the update step drops 10x below the drift threshold. Documented
#   deviation: docking precedent (per-task proven optimum, 3e-4) generalized
#   one notch for the warm-started precision policy; every other PPO
#   hyperparameter unchanged.
# - gamma 0.999 (default 0.99): 1200-step episodes; at gamma=0.99 the settle-
#   bonus value of capturing is discounted to ~5% by the slew phase
#   (gamma^300), leaving no gradient toward capture.
# Static difficulty (None = published benchmark, bit-for-bit; float in [0,1]
# = 0 easiest, 1 hardest = benchmark). Curriculum learning was removed.
DIFFICULTY = None
ENV_KWARGS = {} if DIFFICULTY is None else {'difficulty': DIFFICULTY}

# BC warm start (src/common/bc_warmstart.py): pretrain the actor/critic on the
# classical nominal controller (driven directly by the BC collector) before PPO. None = off.
BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)

if __name__ == '__main__':
    run_training(TASK, SatelliteAttitudeEnv, num_envs=NUM_ENVS, updates=UPDATES, seed=SEED,
                 backend=BACKEND, resume=RESUME, clean=CLEAN, algo=ALGORITHM,
                 lr_schedule=LR_SCHEDULE, learning_rate=LEARNING_RATE,
                 env_kwargs=ENV_KWARGS, bc_warmstart=BC_WARMSTART,
                 **PPO_OVERRIDES)
