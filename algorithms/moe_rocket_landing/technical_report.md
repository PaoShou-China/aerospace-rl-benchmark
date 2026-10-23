# MoE-Based Rocket Landing: Technical Report

Technical report for the experiment package. Evidence snapshot: 2026-10-23.

MoE denotes the phase-routed mixture of action experts developed for this task. This package ships the measured policy and repair-stage training, while documenting the broader study that produced its frozen experts.

## 1. Result and scope

The highest recorded point estimate among the inspected, completed CPU float32 evaluations under the fixed 4×128 development protocol is **444/512 = 86.71875%**, obtained by the first counterfactual feedback repair policy. Its frozen parent scored **438/512 = 85.546875%** in a fresh reference run. The observed difference is six episodes, or **1.171875 percentage points**.

This is the candidate included in this experiment package. It is not an established improvement in generalization, a 95%/99% result, or a stable-landing success rate. Final strict settling was **0/512** for both policies. The repair candidate reached the recorded success condition and subsequently failed in **271/512** episodes; the corresponding parent count was **253/512**.

The earlier 79.10% reference was itself produced during this work by the new phase-aware policy and DAgger pipeline; it was not supplied by the original repository. The subsequent method inherits those newly trained experts and a learned router, then adds simulator search and a learned intervention gate. Its score must not be attributed entirely to PPO hyperparameter or reward tuning.

Primary evidence: [candidate evaluation](evidence/candidate.json), [fresh parent evaluation](evidence/parent.json), and [paired comparison](evidence/paired_comparison.json).

### 1.1 What existed before this work

The comparison starts at upstream commit `39e6bb6fd5d0f5432f7405a76946938a469492ce`. Its rocket task already supplied the Genesis plant, actuator/contact model, 30-dimensional observation, seven action channels, natural initialization, and success/termination rules. It also supplied the classical PDG teacher, BC warm start, RSL-RL PPO, an alternative FlashSAC implementation, batched GPU simulation, checkpoints, and generic evaluation. These are inherited capabilities, not contributions introduced here.

The anchored rocket entry point selected a single FlashResMLP actor (width 128) and critic (width 256). Its default run used 1024 environments, 1000 PPO updates, and 64 steps per environment per update: 65,536,000 online transitions. BC collected 512 steps per environment and ran 100 fitting epochs. PPO defaults included learning rate 0.0003, gamma 0.995, GAE lambda 0.97, entropy coefficient 0.007, ten learning epochs, clipping 0.2, and desired KL 0.01. The anchor already contained redesigned reward terms and per-action exploration scales; neither reward shaping nor task-specific exploration originated with this extension.

Sources inspected directly from Git: [anchored rocket entry](https://github.com/PaoShou-China/aerospace-rl-benchmark/blob/39e6bb6fd5d0f5432f7405a76946938a469492ce/scripts/train_rocket.py) and [anchored training utilities](https://github.com/PaoShou-China/aerospace-rl-benchmark/blob/39e6bb6fd5d0f5432f7405a76946938a469492ce/src/common/train_utils.py).

### 1.2 Contributions relative to that starting point

**Training-data coverage and policy-state coverage.** The small local reproduction's 512-step BC collection produced 65,536 transitions but zero observed pad-contact samples. Extending collection to 1200 steps produced 153,600 transitions, including 21,590 pad-contact observations. Complete-horizon BC alone still scored 0/512. We therefore added DAgger collection: run the student, query the existing PDG teacher at the visited states, aggregate labels, and refit. Some collection rounds mixed teacher and student execution; queries and executed teacher actions were accounted for separately. This changes how the existing teacher is used, without claiming a new teacher controller.

**Policy representation, architecture, and supervised objective.** We implemented `RocketPhaseModel`: 33 scaled state-derived features, a width-256 shared trunk, and three heads routed by height. The feature map excludes previous action-command echoes to reduce action-copy shortcuts and adds world velocity and near-ground scales. Fitting combines weighted continuous-action regression and an ignition classification loss. The phase architecture and DAgger were developed together, so their benefits have not been isolated by an equal-budget ablation.

The r3 policy used 614,400 labeled transitions: 153,600 inherited full-horizon examples plus three rounds of 153,600 additional visited-state labels. Of the cumulative physical execution, 269,191 steps used teacher actions and 345,209 used student actions. The r3 model had **zero PPO updates** and reached **405/512 (79.10%)**. Later expert training extended this lineage; one continued DAgger expert scored **432/512 (84.38%)** on CPU. The 79.10% and 84.38% policies are products of this work, not pre-existing author checkpoints.

**PPO implementation and reward experiments.** We found that the inherited Flash trunk projected parameters inside a gradient-enabled forward pass but not during the usual no-gradient rollout. This could change an action's likelihood before any optimizer update. `StableFlashResMLPModel` and optimizer hooks moved projection to explicit initialization and post-update boundaries while retaining the original network shape and direct-action PPO. In equal local budgets, the repaired implementation with the anchor reward, reward scaled by 0.01, and bounded potential/event reward each scored 0/512. These establish implementation work and negative experiments, not a success-rate gain.

Further PPO continuation introduced reference-policy KL constraints, critic warmup, smaller learning rates, longer credit assignment, phase/channel-specific updates, mixed ignition distributions, and retention replay. The local 512-episode terminal-BC PPO candidate tied r3 at 405/512; the later long-credit/ignition candidate scored 424/512 against its 432/512 start. These experiments remain separate from the DAgger gains. The PPO expert used by the final combination has its own training provenance and is not evidence that all PPO variants improved.

**Learned expert selection.** We added a ridge success-difference gate over two frozen experts trained during this work. The gate selects one policy from the initial state and holds it for the episode. Its recorded combination result is 438/512 (85.55%). This includes both inherited expert-training gains and any routing effect; the routing contribution alone is not established by comparing it with the older 405/512 model.

**State-dependent feedback repair.** On top of that combination, we added actual simulator branch search, a finite library of twelve-parameter feedback programs, and a learned intervention gate. This is the final addition behind the highest recorded point estimate, 444/512 (86.72%). Sections 3–5 describe the implementation, cost, and limited evidence for its incremental gain over 438/512.

**Experiment execution and measurement.** We added configuration-driven runs, isolated source snapshots, transition/provenance ledgers, CPU process-based sampling, and GPU inference/training chunk adjustment to the existing batched simulator. We also added fixed first-episode evaluation, matched initial-state comparisons, terminal-state capture, and separate reporting of ever-success, final settling, and later failure. Expanded development/stress suites are supporting infrastructure; they do not replace the original-distribution score reported here. GPU batching itself was already upstream, and no throughput or accuracy gain is claimed merely from these engineering additions.

Primary evidence and implementation entry points: [upstream local reproduction](evidence/history/validation__upstream-ppo-local.json), [equal-budget PPO/reward experiments](evidence/history/validation__direct-ppo-local-v1.json), [phase/DAgger milestone ledger](evidence/history/validation__aggressive-local-v1.json), phase training and collection (historical research source; outside this package), PPO implementation repair (historical research source; outside this package), and reward candidates (historical research source; outside this package).

### 1.3 How to interpret the progression

```text
Inherited environment + PDG teacher + BC/PPO training foundation
    -> full-horizon demonstrations and DAgger
    -> phase-aware learned actor (r3: 405/512)
    -> further expert training (continued DAgger on CPU: 432/512)
    -> initial-state selection between DAgger/PPO experts (438/512)
    -> gated local feedback repair (444/512)
```

The arrow chain describes implementation and checkpoint ancestry. It is not a sequence of single-variable, equal-cost causal comparisons. In particular, **79.10% is already our intermediate result**. The clean small-budget reproduction of upstream PPO was 0/512 using 128 environments and 128 updates (1,048,576 online transitions, about 1.6% of the default budget); it cannot establish the original full-budget PPO performance or a like-for-like 86.72-point gain. The upstream README's 66.8% refers to its classical PDG evaluation with a different window, so it is not a comparable PPO denominator either.

The minimum submission should therefore describe the complete contribution chain—data/architecture changes, expert training and combination, and local repair—while preserving the distinction between measured gains and attempted modifications that did not improve the chosen metric.

## 2. Evaluation contract

| Item | Recorded setting |
| --- | --- |
| Environment anchor | `39e6bb6fd5d0f5432f7405a76946938a469492ce` |
| Protocol identifier | `upstream_source_first_episode_full_horizon_v1` |
| Backend / physics precision | Genesis CPU / float32 |
| Cohorts | Four batches of 128 environments |
| Starting seeds | `1000007`, `2000007`, `3000007`, `4000007` |
| Horizon | At most 1200 control steps per scored first episode, 40 Hz |
| Initial conditions | Natural initialization, `difficulty=None` |
| Counting | One first episode per environment; subsequent automatic-reset episodes do not enter the denominator |
| Early endings | Natural fatal collision and out-of-bounds termination remain active |
| Model selection boundary | Candidate checkpoint frozen before its development evaluation |
| Sealed test | Not used |

The primary metric is the environment's latched `complete` flag: at some point during the episode, pad contact satisfies the specified speed (<0.02 m/s), angular speed (<0.02 rad/s), tilt (<0.1 rad), and nonfatal-contact conditions. The environment uses its original inner-step timing for those measurements. Achieving success does not end the rollout; a later failure does not erase this flag.

`final_settled_successes` separately checks strict contact and motion conditions at the end of the first episode, before reset. Consequently, the two success metrics answer different questions and must be reported together.

The candidate's four cohort counts were **107, 107, 116, 114**. All 512 first episodes finished, and the parent/candidate initial-state hashes matched. Evaluation did not replace natural initial conditions with constructed stress cases. Different backends, altered precision, and 128-episode screens are not pooled into this ranking.

## 3. Policy architecture

### 3.1 Frozen parent: initial-state routing between two experts

The parent contains two previously trained direct-action networks: a DAgger expert and a PPO expert. Both use the same phase-aware actor architecture: 33 engineered observation features, a shared width-256 representation, and three action heads selected by height above the landing surface (>10 m, 2–10 m, ≤2 m). Each head produces the seven original action channels.

At the first observation of an episode, the parent router constructs six normalized position/orientation coordinates and their 21 quadratic products. A ridge model fits the paired training target

`success(DAgger) - success(PPO)`.

The router standardizes those 27 features and uses 28 coefficients including the intercept, with ridge coefficient 64 and an unpenalized intercept. A positive prediction chooses DAgger; a nonpositive prediction chooses PPO. This choice is held until the environment naturally resets.

The router's fitting arithmetic uses float64; this does not change Genesis physics from float32. The current parent implementation evaluates both experts and selects their output with a mask, so this implementation does not establish single-expert inference cost.

The parent gate was fitted using 512 new initial conditions with both experts executed, costing 1,228,800 control transitions. Its earlier independent confirmation used counterfactual selection from executed expert outcomes; its reported 438/512 development score came from an actual routed rollout. The experts' earlier training costs are inherited, not zero.

Sources: [phase policy](phase_policy.py), [expert selector](expert_router.py), and [parent report](evidence/history/local-group-selector-v1__report.json).

### 3.2 Finite feedback programs

The repair layer modifies the selected parent's action for a finite interval. The action order is:

```text
[fin_x, fin_y, fin_yaw, ignition, throttle, gimbal_x, gimbal_y]
```

Each program contains 12 continuous parameters: throttle bias, descent feedback gain, near-ground target speed, position gain, lateral-velocity gain, tilt gain, angular-rate gain, fin gain, yaw damping, two gimbal biases, and throttle rate limit. Search also selects intervention onset, duration, and one of five ignition modes: inherit, off, on, low-altitude hysteresis pulse, or cut on currently observed pad contact.

The program reads the current state every step. It normalizes orientation, rotates body velocity into world coordinates, computes descent and lateral corrections, and applies bounded fin/TVC/throttle commands. Parent actions are clipped before correction so a small correction is not lost behind an already saturated network output. The environment still executes fuel consumption, actuator response, thrust noise, collision, and contact dynamics.

The final library contains ten programs. These are short feedback controllers, not stored action sequences and not a network capable of generating arbitrary new programs at inference time. Search uses existing elite-distribution optimization ideas; no algorithmic priority claim is made.

Source: [feedback program implementation](feedback_program.py).

### 3.3 Learned intervention gate

The gate receives 30 features derived only from the current observation and parent action. These include height, world velocity, body angular velocity, orientation, lateral position, actuator/fuel/contact channels, parent commands, and heuristic braking/landing features. Seed identifiers and future outcomes are not inputs.

| Component | Frozen v1 setting |
| --- | --- |
| Network | 30 → 96 → 96 → 11, SiLU hidden activations |
| Outputs | Retain parent, or invoke one of ten programs |
| Input normalization | Training mean/std, std floor 0.1, normalized input clipped to [-8, 8] |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.01 |
| Fitting | 800 updates per fit; 64 positive + 192 retention rows per minibatch |
| Gradient clipping | Norm 1.0 |
| Final training rows | 36 positive and 23,746 retention rows, including repeated emphasis rows; not independent episode counts |
| Activation probability | 0.98 for the recorded checkpoint |
| Local support | RMS distance ≤1.5 in the fixed physically scaled feature space |
| Height eligibility | ≤150 m above the landing surface |
| Intervention count | At most one per episode |

The gate is first fitted from successful search branches and retention observations on successful parent trajectories. It is then refitted using additional paired parent/student training episodes. Label 0 on a retained trajectory expresses the objective to preserve the parent, not proof that every alternative action fails.

Let `p` be the gate's softmax probabilities, `j` the highest-scoring nonzero class, and `d_j` its distance to the corresponding program's center. An unused episode can trigger only if

```text
height <= 150 m
p[j] >= 0.98
p[j] > p[0]
d_j <= 1.5
```

After triggering, the program remains a state-feedback controller for its saved duration. The policy then returns to the parent, without permitting another intervention in that episode. Natural environment resets also reset the parent's route and the intervention memory.

The probability threshold was derived only from training retention scores: their 99.9th percentile, clipped to [0.65, 0.98]. The probability and distance rules are empirical restrictions, not a guarantee of no degradation.

Evidence: [frozen gate implementation](repair_policy.py) and [final fitting record](evidence/gate_training.json).

## 4. Training and interaction budget

Every physical branch starts from natural initialization and replays its prefix. No incomplete simulator-state restore or free rewind is used. Search candidates share matched prefixes and paired future noise streams; evaluation restores the unchanged natural random streams.

| Stage | Actual new control transitions |
| --- | ---: |
| Screen 256 training initial conditions | 307,200 |
| Two rounds of candidate search, including reference branches and prefixes | 1,228,800 |
| 32 new paired parent/student training conditions | 76,800 |
| 128 additional paired parent/student training conditions | 307,200 |
| **New repair training total** | **1,920,000** |
| Frozen candidate development evaluation | 614,400 |
| Fresh parent reference evaluation | 614,400 |
| **Repair experiment plus fresh reference** | **3,148,800** |

Screening found 45 failures among 256 episodes and selected 16 cases for search. The first round used 16 candidates per case and two future-noise replicas. The second round redistributed candidate counts with a minimum of eight per case, keeping the total search budget fixed. Candidate onset choices included 16, 64, 256, and 512 control steps before the observed failure.

Candidates were ranked using actual success, net gain over their executed reference, terminal motion error, and intervention duration. Ten programs were retained with success under both training-noise replicas and positive paired search gain. Those twenty successes are repeated training branches of ten cases, not twenty independently sampled generalization successes.

Before the final refit, paired student checks scored 25→24 on 32 conditions and 101→100 on 128 conditions. These are training data subsequently used for fitting. They are not independent confirmation of the frozen final model.

Batch size began at 128 and could adapt within a 256-environment cap using measured rollout throughput. Physics used four threads. This describes the CPU experiment, not a GPU scaling or multi-host scheduler result.

The recorded candidate run, including development evaluation, took 902.34 seconds. Its development stage overlapped with another evaluation process, and competing runs used different thread allocations. These wall-clock observations do not establish an algorithmic speedup.

All numbers count `env.step` control transitions, including full prefixes and later automatic-reset advances. Internal reset warmup is not a separate control transition. The ledger excludes earlier expert and parent-router training; it must not be described as training from scratch in 1.92 million transitions.

Evidence: [manifest](evidence/training_manifest.json) and [run report](evidence/history/local-counterfactual-repair-v1-run2__report.json).

## 5. Effect size and attribution

| Comparison item | Recorded value |
| --- | ---: |
| Parent successes | 438/512 |
| Candidate successes | 444/512 |
| Failed parent episodes that became successes | 24 |
| Successful parent episodes that became failures | 18 |
| Net gain | 6 episodes |
| Candidate nominal 95% Wilson interval | 83.50%–89.39% |
| Nominal paired McNemar exact two-sided p | 0.4408 |
| Episodes with an intervention | 38/512 |
| Gains / losses among triggered episodes | 2 / 1 |
| Gains / losses among untriggered episodes | 22 / 17 |

The parent rerun reproduced all historical success labels. However, most parent/candidate disagreements occurred in episodes where the repair gate never triggered. The total six-episode difference therefore cannot be explained as six directly repaired failures.

One concrete mechanism under suspicion is environment reset warmup: resetting a subset advances the shared scene, then restores selected visible states of other environments. Preservation of all internal contact-solver state has not been established. The current evidence does not identify a unique cause, but it does show that cross-environment effects cannot be ignored.

The nominal interval and p-value assume statistical properties that these batch effects may violate. Even under their nominal assumptions the observed improvement is not significant at 0.05. The evidence supports “highest recorded development point estimate,” not a reliable promotion over the parent.

Only one training run is represented, and this development suite has been used across many experiments. Freezing this candidate before its evaluation does not make the repeatedly consulted development suite a sealed test.

Evidence: [trigger attribution](evidence/interventions.json) and [fresh comparison](evidence/paired_comparison.json).

## 6. PPO comparison and other routes

The most direct PPO control starts from the same 438/512 parent and spends the same **1,920,000 additional training transitions**. It retains the initial-state route and trains expert mean networks with a reference-KL penalty, a new conditional critic, and the anchored environment reward. Its final result is **406/512 = 79.296875%**, with 51 gains and 83 losses relative to the parent. Final strict settling is 3/512.

This comparison establishes the outcome of one concrete PPO continuation configuration. It does not establish superiority over all PPO settings, equal total historical training cost across all experiments, or faster attainment of a target success rate. The comparison is complete in the later PPO record; the repair run's earlier `matched_budget_ppo_control_completed: false` field predates that completion.

Other routes are summarized once in the [README](README.md#other-explored-routes). Their evidence sources are:

- PPO continuation: [matched-budget control](evidence/history/local-selector-ppo-control-v1__development512.json), [single-expert starting policy](evidence/history/parent_reference__development.json), and [long-credit/ignition candidate](evidence/history/long_credit_ignition__development512.json).
- Alternative policy updates: [RLOO](evidence/history/local-group-return-v1__development512.json) and [restricted replay comparison](evidence/history/local-replay-refine-v1__comparison.json).
- Alternative routing: [touchdown selector](evidence/history/local-touchdown-selector-v1__development512.json) and [event repair](evidence/history/local-repair-event-v1__development512.json).
- Other completed repair comparisons: [aggregate record](evidence/history/local-repair-comparison-v1__report.json).
- Independent GPU route: [unified physics policy evaluation](evidence/history/gpu-unified-physics-v6-final__development512.json).

## 7. Exact artifact and source version

The measured policy is [student-final.pt](assets/phase_aware_feedback_repair.pt), **2,294,170 bytes**, containing the parent expert configurations and weights, parent router, repair gate, program parameters, ignition modes, durations, centers, and activation thresholds.

```text
Candidate checkpoint SHA-256
1d5fcc65a01271325673e764d80771e01db2ba6af61abadaadb70d9e102a38b5

Parent selector checkpoint SHA-256
68397471e5093b630c321e2617d2a6d75316ec8348986ba20952be07120abf78

Anchored environment archive SHA-256
9043ffc4d10ce94ae9416b9f385a960928bf4c44fcee3b32d70b2e79a64c8ace
```

The measured implementation was preserved with its original run. The included [source provenance](evidence/source_provenance.json) records its core source hashes; [train.py](train.py) packages that run's repair procedure with portable artifact lookup. A later research-workspace implementation removed the hard-coded 150 m activation ceiling and expanded retention sampling beyond that height. Retraining the current entry point is consequently not an exact reproduction of the measured v1 training recipe.

The current `load_repair_policy` loader restores a missing legacy `activation_height` as 150 m. A saved [compatibility check](evidence/loader_compatibility.json) compared 15,360 action rows against the old snapshot with maximum absolute difference 0. This was an offline action comparison, not a new simulation evaluation. Three other core modules—phase policy, expert selector, and feedback program—matched their saved source hashes when this report was prepared.

The experiment requires Python 3.11. Its [requirements](requirements.txt) pin PyTorch 2.14.1, Genesis 1.4.4, rsl-rl-lib 5.5.1, tensordict 0.15.2, and the supporting libraries. These pins describe the packaged runtime; a fresh dependency installation was not performed for this submission.

## 8. Package scope and verification

The experiment lives under `algorithms/moe_rocket_landing/`. It adds no replacement environment to the benchmark. `runtime.py` extracts the historical environment at the recorded commit into a new output directory before simulation; the later `environments/` reorganization on main therefore does not silently replace the measured dynamics.

The four policy modules implement phase routing, episode-level expert selection, bounded feedback programs, and the learned repair gate. `train.py` contains the measured failure-screening, paired branch-search, retention fitting, and final development protocol. It extracts the frozen parent from the supplied self-contained checkpoint; it does not retrain the earlier DAgger/PPO experts. The historical expert-training runs and alternative research routes are documented as provenance, not shipped as a complete sweep framework.

`evaluate.py` reconstructs the supplied model and records first-episode outcomes in four batches of 128. The optional `--cohorts 1` is a diagnostic subset and is marked `full_development_protocol=false`; it must not be compared as a full 512-episode score. Both entry points require a new output directory and retain the original 150 m repair eligibility and retention-sampling limit. Commands are in the [README](README.md#run).

The original checkpoint is included unchanged. Packaging moves module import paths and artifact lookup into this directory, adds a legacy-compatible loader, and replaces dependencies on private working directories with the checkpoint and included reference outcomes. This is a source/layout adaptation, not a new training result. [Source provenance](evidence/source_provenance.json) identifies the measured source hashes and adaptation boundaries.

[Package validation](evidence/package_validation.json) records deterministic action and intervention-memory equivalence against the saved measured modules on 15,360 stored training-observation rows, including explicit episode-memory resets. This check creates no new environment transitions. Entry-point help, Python syntax, repository-root lookup, evidence links, and the unchanged checkpoint digest were also checked. Full physical 512-episode reevaluation and retraining of the packaged entry points were not performed during submission preparation; the reported score is the preserved completed experiment result.

The entire submission is an experiment, not a new environment, a replacement for the upstream PPO package, or a claim of a new general RL algorithm. The research contributions and unsuccessful paths remain distinguishable from the compact executable release.
