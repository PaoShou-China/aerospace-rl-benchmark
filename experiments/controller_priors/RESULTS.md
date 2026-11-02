# RESULTS — first-round pilot (controller priors)

**Status: PILOT-SCALE, not the final JGCD budget.** Every number below was read
from the run outputs under `results/controller_priors/` and
`results/controller_priors_scans/` when the study was executed; those
directories are **not shipped in this checkout**, so the recorded tables are
the aggregate evidence (regenerate with the commands in `PROGRESS.md` §10).
The generating tables/figures are `_analysis/summary.csv`,
`_analysis/stage_summary.csv`, `_analysis/figures/`.

Protocol: all arms in a task share the same PPO recipe, the same evaluation
seed (`eval_seed=7`), the same env construction and the same first-episode
protocol. `paired_condition_check.json` is empty (no init-digest mismatch)
and `protocol_consistency.json` has one protocol digest per task/condition.
Baselines re-verified at 100% for docking/satellite/rendezvous.

Training seeds {7,13,21}; budgets: docking 1024 envs / 250 updates, satellite
512 / 100, rendezvous 512 / 200. BC: `fit_critic=False`,
`normalizer_mode='fit_once'`, 70/15/15 episode-disjoint split.

## 1. Final deterministic first-episode results (mean ± sd over 3 seeds)

| Task | Arm | Success | Return | Physical cost (all completed first episodes) |
|---|---|---|---|---|
| docking | traditional | 1.000 ± 0.000 | 189.3 | 0.713 kg, 189.6 s, sat 0.758, terminal 0.970 m / 0.057 m/s |
| docking | bc_only | 1.000 ± 0.000 | 189.3 | 0.710 kg, 188.4 s, sat 0.406, terminal 0.965 m |
| docking | ppo_scratch | 0.071 ± 0.122 | 83.0 | failures: 495 s, 0.129 kg, sat 0.025 |
| docking | bc_ppo | 0.000 ± 0.000 | 68.6 | failures: 500 s, 0.079 kg, sat 0.014 |
| docking | residual_ppo | 1.000 ± 0.000 | 189.3 | **0.696 kg, 188.6 s, sat 0.436**, terminal 0.970 m |
| satellite | traditional | 1.000 ± 0.000 | 64.7 | 6.11 s, gimbal ∫rate² 63.67, sat 0.348, att 0.0196 rad |
| satellite | bc_only | 0.108 ± 0.087 | 30.7 | failures: 28.5 s, ∫ 529.6, sat 0.680, att 0.445 rad |
| satellite | ppo_scratch | 0.993 ± 0.007 | 65.7 | 6.69 s, **∫ 24.57**, sat 0.128, att 0.0192 rad |
| satellite | bc_ppo | 0.992 ± 0.003 | 65.9 | **5.90 s**, ∫ 54.05, sat 0.329, att 0.0130 rad |
| satellite | residual_ppo | 1.000 ± 0.000 | 65.1 | 6.27 s, ∫ 59.61, sat 0.251, att 0.0167 rad |
| rendezvous | traditional | 1.000 ± 0.000 | 184.9 | 1.434 kg, 510.7 s, sat 0.173, terminal 9.93 m / 0.142 m/s |
| rendezvous | bc_only | 0.976 ± 0.025 | 180.2 | 1.430 kg, 613.2 s, sat 0.078, terminal 9.88 m |
| rendezvous | ppo_scratch | 0.000 ± 0.000 | 17.7 | failures: 991.9 s, 1.086 kg, terminal 662 m |
| rendezvous | bc_ppo | 0.195 ± 0.075 | 95.1 | failures dominate: 941.7 s, 1.098 kg, terminal 23.8 m |
| rendezvous | residual_ppo | 1.000 ± 0.000 | 184.9 | 1.436 kg, 511.1 s, sat **0.092**, terminal 9.91 m |

`sat` = fraction of first-episode env steps with any channel at full authority.
Docking/rendezvous fuel uses the env's `Isp=300 s`, 10 N/axis, 500 kg model.
Satellite `gimbal ∫rate²` is an **actuator-usage proxy**, not electrical energy.
Costs of arms with collapsed success are shown only to expose the failure mode;
they must not be used as efficiency evidence.

Stochastic evaluation (same protocol seed, sampled policy):

| Task | Arm | Success | env-action RMS perturbation |
|---|---|---|---|
| docking | bc_only / bc_ppo / scratch / residual | 1.000 / 0.000 / 0.071 / **1.000** | 0.500 / 0.382 / 0.359 / **0.042** |
| satellite | bc_only / bc_ppo / scratch / residual | 0.001 / **1.000** / 0.999 / **1.000** | 1.000 / 0.321 / 0.372 / **0.034** |
| rendezvous | bc_only / bc_ppo / scratch / residual | 0.365 / 0.212 / 0.001 / **1.000** | 0.500 / 0.497 / 0.514 / **0.042** |

Residual scaling makes stochastic exploration far smaller than direct PPO;
the measured environment perturbation (`exploration_env_action_rms`) is
reported rather than assumed comparable.

The BC fit (shared recipe, `fit_critic=False`, `fit_once`):

| task | train pairs | train actor MSE | val actor MSE | episodes train/val/test | BC-only closed-loop | initial std |
|---|---|---|---|---|---|---|
| docking | 463,349 | 0.00333 | 0.00335 | 2460 / 527 / 527 | 1.000 | 0.5 |
| satellite | 272,341 | 0.0189 | 0.0210 | 1122 / 240 / 240 | 0.108 | 1.0 |
| rendezvous | 366,494 | 0.00175 | 0.00203 | 716 / 154 / 154 | 0.976 | 0.5 |

The `initial std` column is the configured `init_std` value recorded for
provenance. Under the shipped default `BetaDistribution` that field is inert
(the policy has α/β, not a free std), so the actual stochastic exploration is
the measured `exploration_env_action_rms` above, not the column value.

Satellite has the largest normalized action regression error among the three:
its expert CMG commands are near bang-bang and the learned mean is smooth, so
a modest MSE still leaves the 0.05 rad / 0.02 rad/s terminal window hard to
reach (BC terminal attitude error 0.445 rad). This is a precision/covariate-
shift failure of plain BC, not a controller failure.

## 2. Learning curves (deterministic first-episode success, 3-seed mean)

* docking: BC→PPO **1.000** at stage 0/25 → 0.840 (50) → 0.081 (100) → 0.000
  (150/200/final). Scratch 0 until 1.5% at 200. Residual 1.000 at every stage.
* satellite: BC-only 0.108 at stage 0; BC→PPO 1.000 at 25, 0.988 at 50,
  0.992 final; scratch 0 → 0.870 (25) → 0.987 (50) → 0.993 final; residual
  1.000 throughout.
* rendezvous: BC-only 0.976 at stage 0; BC→PPO 0.003 (25) → 0.001 (50) →
  0.025 (100) → 0.111 (150) → 0.195 final; scratch 0 throughout; residual
  1.000 throughout.

The BC→PPO collapse is therefore **not universal**: on satellite the same
mechanism recovers within 25 updates; on docking it collapses under the shared
recipe but a pinned-sigma/lr-coupling ablation keeps 100% (below).

## 3. Single-factor scans (docking, seed 7, 512 envs, 100 updates)

This scan budget is separate from the pilot; compare only inside the scan.
Controller quality keeps the TRUE plant fixed; mismatch conditions pin the
controller's design model explicitly.

| Condition | Arm | Success | Fuel kg | Time s | sat |
|---|---|---|---|---|---|
| quality_sluggish | traditional / bc_only / bc_ppo / residual | 1.0 / 1.0 / **0.504** / 1.0 | 0.566 / 0.560 / 0.220* / 0.560 | 177.7 / 177.9 / 461.5* / 176.9 | 0.545 / 0.353 / — / 0.285 |
| quality_aggressive | traditional / bc_only / bc_ppo / residual | 1.0 / 1.0 / **0.172** / 1.0 | 0.665 / 0.661 / 0.179* / 0.661 | 183.2 / 185.5 / 494.3* / 183.0 | 0.758 / 0.360 / — / 0.544 |
| quality_model_bias | traditional / bc_only / bc_ppo / residual | 1.0 / 1.0 / **0.217** / 1.0 | 0.677 / 0.674 / 0.191* / 0.673 | 183.5 / 183.5 / 489.3* / 183.7 | 0.749 / 0.423 / — / 0.527 |
| coverage_narrow | bc_only / bc_ppo | 1.0 / **0.330** | 0.710 / 0.208* | 189.1 / 481.9* | 0.433 / — |
| coverage_wide | bc_only / bc_ppo | 1.0 / **0.113** | 0.709 / 0.164* | 188.7 / 494.6* | 0.446 / — |
| samples_1x | bc_only / bc_ppo | 1.0 / **0.129** | 0.706 / 0.166* | 190.4 / 495.0* | 0.474 / — |
| samples_6x | bc_only / bc_ppo | 1.0 / **0.275** | 0.708 / 0.195* | 187.9 / 486.2* | 0.433 / — |
| mismatch_plant | traditional / scratch / bc_only / bc_ppo / residual | 1.0 / **0.000** / 1.0 / **0.018** / 1.0 | 0.929 / 0.101* / 0.926 / 0.171* / 0.929 | 215.5 / 500* / 215.3 / 499.5* / 216.9 | 0.802 / 0.019 / 0.395 / — / 0.560 |
| mismatch_design | traditional / scratch / bc_only / bc_ppo / residual | 1.0 / **0.000** / 1.0 / **0.217** / 1.0 | 0.677 / 0.094* / 0.674 / 0.191* / 0.673 | 183.5 / 500* / 183.5 / 489.3* / 183.7 | 0.749 / 0.016 / 0.423 / — / 0.527 |
| mismatch_both | traditional / scratch / bc_only / bc_ppo / residual | 1.0 / **0.000** / 1.0 / **0.051** / 1.0 | 0.889 / 0.101* / 0.887 / 0.194* / 0.881 | 212.5 / 500* / 212.1 / 497.9* / 212.7 | 0.777 / 0.019 / 0.431 / — / 0.522 |

`*` marks costs of runs whose success collapsed (not efficiency evidence).
Mismatch plant = true mass 550 kg, thrust 9 N/axis, Isp 250 s; controller design
model fixed to the nominal 500 kg / 10 N / 300 s values. `mismatch_design` =
nominal plant, controller thinks 550/9/250. `mismatch_both` = shifted plant +
shifted design model.

Residual authority scan (docking, ResidPPO, rho = limit / full action range):

| rho | success | fuel kg | time s | sat | rms residual | max residual | cancellation |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.0 | 0.710 | 188.9 | 0.546 | 0.008 | 0.013 | 0.432 |
| 0.025 | 1.0 | 0.710 | 189.4 | 0.522 | 0.021 | 0.039 | 0.448 |
| 0.05 | 1.0 | 0.704 | 189.1 | 0.488 | 0.045 | 0.063 | 0.510 |
| 0.10 | 1.0 | 0.681 | 186.6 | 0.425 | 0.106 | 0.154 | 0.535 |
| 0.20 | 1.0 | 0.630 | 181.6 | 0.397 | 0.225 | 0.294 | 0.639 |
| 0.40 | 1.0 | **0.451** | **182.8** | 0.220 | 0.737 | 0.737 | 0.734 |

At rho 0.4 the residual is no longer a small correction (80% of the channel
range) and the benefit is near a direct-policy regime; success was still 100%
in this one seed. **No failure boundary was found on docking up to rho 0.4.**

Satellite residual authority (same scan protocol, seed 7):
rho 0.05: 1.000 success, 6.29 s, ∫ 59.3, sat 0.247;
rho 0.20: **0.939 success**, 8.40 s, ∫ 55.8, sat 0.219, att error 0.036 rad;
rho 0.40: **0.912 success**, 10.08 s, ∫ 57.2, sat 0.196, att error 0.036 rad.
On this precision-sensitive task a large residual **degrades** success and
settling time — a failure boundary absent on docking. Even at rho 0.40 the
residual's actuator integral (57.2) stays far above the from-scratch policy
(24.6), so authority alone does not let the residual recover scratch-level
actuator efficiency.

## 4. Optimizer ablation: the BC→PPO collapse is protocol-dependent

Docking, same BC checkpoint/data, 250 updates:
* shared protocol: success 0.000;
* pinned sigma 0.05 + lr-sigma coupling: **success 1.000, fuel 0.654 kg,
  time 181.1 s** (better than the controller 0.713 kg / 189.6 s).
Rendezvous, same BC checkpoint/data, 200 updates:
* shared protocol: success 0.195;
* pinned sigma 0.05: 0.000;  pinned sigma 0.2: **0.020**.
So the docking collapse is optimizer-induced and the mechanism is viable;
the rendezvous collapse is robust to these two pins at this spawn/budget and
must be reported as a genuine task-level failure condition.

## 5. Independent validation (held-out training seed 101, docking)

Held-out seed was not used to formulate any rule:
traditional 1.000 (0.713 kg, 189.6 s, sat 0.758); bc_only 1.000 (0.712 kg,
188.7 s, sat 0.477); residual 1.000 (**0.692 kg, 187.9 s, sat 0.434**);
ppo_scratch 0.000; bc_ppo 0.000. The pilot ranking transfers exactly.

## 6. Decision rules supported by this pilot (hypotheses, not laws)

1. **Success-saturated + actuator-saturated controller (docking, rendezvous):**
   residual PPO is the safest knowledge mechanism: it keeps 100% success and
   cuts actuator chattering, with 2–4% fuel benefit at rho 0.05 and much larger
   fuel benefit only at high authority. BC-only also keeps success, but BC→PPO
   needs a stable optimizer.
2. **Task where from-scratch PPO is stable and efficient (satellite):** direct
   PPO finds a far more actuator-efficient policy (∫ 24.6 vs controller 63.7)
   with ~99.3% success. Small-authority residual inherits the controller's
   actuator usage (∫ 59.6); BC→PPO reaches 99.2% and ∫ 54.1. Here the
   controller prior is not automatically better than scratch.
3. **Model mismatch:** controller-anchored priors preserve success (BC-only and
   residual 100% under the shifted plant); scratch collapses to 0%; BC→PPO
   remains optimizer-fragile. This is the strongest case for keeping the
   conventional controller in the loop.
4. **Controller quality:** a slower/lower-fuel controller produces a
   lower-cost BC student and a less severe BC→PPO collapse (0.50 vs 0.17 at
   aggressive gains), but residual success stays 1.0 in all quality arms.
5. **Demonstration coverage/sample count:** BC-only is insensitive in this
   range; BC→PPO improves with more samples (0.13 → 0.28 from 1x to 6x) but
   degrades with wider coverage (0.33 → 0.11 narrow → wide). These are
   optimizer-tolerance effects, not BC capacity effects.

## 7. Failure conditions found

* BC→PPO under the shared PPO recipe: docking (all 3 seeds, after ~50 updates),
  rendezvous (all seeds, within 25 updates), but not satellite.
* Residual authority too large on a precision task: satellite rho 0.2 → 93.9%
  success, longer settling, worse terminal attitude error.
* PPO from scratch: docking 7.1% (250 updates), rendezvous 0% (200), satellite
  fine (99.3%).
* Plain MSE BC alone on satellite: 10.8% (precision/covariate shift); not a
  controller failure.
* No failure of residual on docking/rendezvous at rho ≤0.4 in this pilot.

## 8. What is NOT verified

* Formal (1000–1500-update, 1024-env) budget, more seeds, satellite/rendezvous
  full factor scans, rocket/fixedwing/quadplane, critic pretraining,
  frozen-BC residual (arm F), DAgger, and the global
  optimality gap (no trusted reference solution).
* Satellite authority scan uses one seed and two authority points; docking
  scans use one seed and a 100-update budget.
* Runs were executed concurrently on one RTX 3090; wall/throughput numbers
  include contention, though transitions and evaluation are unaffected.
