# Observable-belief, survival-aware rocket landing

Research prototype based on `Yxuan-Gong` at `91a17f96e2ee289720f6bee521aab35f0ceb7647`.
See `results/validation_status.json` for the exact tested revision and CI evidence.

## What has and has not been executed

Actual source integration tests, a small synthetic identification check, and a
bounded **CPU Genesis training smoke** have run. The smoke uses four environments,
four policy steps and one training update: **16 new transitions and zero completed
landing episodes**. It verifies collection, initialization from existing weights,
backpropagation, the PPO/auxiliary update and checkpoint serialization. It is not
an evaluation, a useful trained landing policy, or evidence of improved performance.

At tested code revision `ac77c1a0331d1d9475706cadf7dc770aef08aa5d`, all **21 tests
passed, with zero skips**, in CI run `40057451572`. The Genesis CPU smoke also
passed. Its artifact SHA256 is recorded in `results/validation_status.json`.

The supplied remote GPU was not contacted: the authoring runtime could not resolve
the SSH host and had no SSH client. No credentials are in source, logs or CI.
There is no new landing success rate, multi-seed ablation or OOD performance result.
The original branch, simulator archive and historical checkpoints are unchanged.

This model has **state observations, not visual/language inputs**. It ports the
physical-belief/control kernel of the VLA-Physical idea, not a full VLA backbone.
No disruptive novelty, speedup, calibrated safety probability or flight readiness
is claimed. Use this research prototype in simulation only.

## Finding that motivates this experiment

The existing comparison file reports DAgger 419/512 official successes and the
published physical policy 428/512. For the latter, only 3/512 are settled at the
final state, and 214 episodes succeed then fail. Paired outcomes contain 79 gains
and 70 losses: exact two-sided McNemar p=0.5123564769. These are old repository
records, not new experiments or a multi-training-seed architecture comparison.
A final settled state does not establish a two-second dwell.

The research question is whether factual online parameter inference and prediction
beyond first contact can improve **remaining landed**, not merely touching the
success condition once. Original success is always retained as a separate metric.

## Candidate architecture

`ObservableSurvivalActor` extends the existing `ConfidenceSequenceActor` and reuses
its neural seed policy, rigid-body/contact operators and PPO training loop.

1. **Evidence-grounded physical memory.** Twelve effective coefficients and a
   full 12x12 covariance are updated with a Joseph-form filter using the local
   parameter Jacobian of the physical predictor. Evidence consists of the actual
   preceding state, executed action and subsequent observation. Task reward cannot
   directly change factual parameter estimates. These coefficients are effective
   corrections, not separately identifiable mass/thrust measurements.
2. **Observable directions only.** Missing/reset transitions leave the belief
   unchanged. Zero sensitivity supplies no information. Contact transitions are
   excluded from flight identification; the contact parameters retain prior
   uncertainty and are NOT identified in this version. Observation noise is
   predicted from past context before the outcome is seen.
3. **Physical action inference.** Twenty-five positive-weight cubature points
   represent all twelve covariance directions. Candidate continuous controls and
   the two ignition branches are evaluated through the shared physical model.
4. **Counterfactual settling continuation.** Near the pad, a common engine-off
   continuation adds predicted bounce, tipping and residual-motion cost. Research
   configuration uses forty steps (one second); smoke tests use two. This fixed
   continuation is a heuristic, not an optimal terminal value or a safety proof.

Covariance is approximate. Clipping, state-observation error and structural model
bias prevent unconditional Gaussian coverage claims. Cubature points are not
confidence bounds. A wrong contact model can make the continuation harmful. Its
no-tail ablation is mandatory. Twenty-five hypotheses and a longer rollout may
cost substantially more than the existing policy; runtime must be measured.

The default forty-step tail has regression tests for higher-order gradients near
contact. A norm-at-zero second-derivative failure found during testing was fixed
by using squared radial excess in the new tail cost. No simulator physics was
altered to make this test pass. Finite gradients at test points are not global
numerical-stability guarantees.

## Environment and action interface

For comparability, this experiment uses the existing package's archived environment
at `39e6bb6fd5d0f5432f7405a76946938a469492ce`, with its original complete episode
horizon and success/termination checks. It does not silently rewrite the live
`environments/rocket` package or substitute a different landing task.

Both inspected environment implementations accept normalized policy actions in
[-1,1] and internally map ignition/throttle to [0,1]. An earlier documentation
draft overstated an interface difference; comparing `_scale_actions` corrected
that statement. Physical actuator ranges must not be confused with policy input
ranges. Execution through the current live package's different import layout has
not been separately validated here.

## Metrics

`StableLandingMonitor` records the first episode only. Supplemental durable
success requires the FINAL eighty consecutive 40-Hz observations to satisfy the
settled condition, with no fatal/out-of-bounds failure. A two-second temporary
dwell followed by a crash does not pass. This is a sampled condition, not a
continuous-time certificate. Terminal samples are recorded before auto-reset,
and nonterminal samples afterward; episodes are not ended on first success.

## Reproduce

Use a clean checkout of `experiment/observable-survival-rocket-v1` and the existing
experiment's dependencies. The CI workflow records exact installed versions.
A CPU Genesis run also needs a functioning headless EGL implementation; CI uses
software Mesa. No changes to simulation physics are required for that renderer.

```bash
python -m pytest algorithms/observable_survival_rocket/tests -q -ra
python algorithms/observable_survival_rocket/synthetic_probe.py \
  --out /tmp/rocket_synthetic_probe.json
```

Actual-source tests explicitly skip if their real dependencies are missing. A
skip is not a pass. The synthetic probe is just a noisy linear acceleration
model with known inputs and the correct observation equation. It demonstrates
filter arithmetic, not rocket control or superiority over adaptive MPC.

On the GPU machine, with its environment activated:

```bash
bash algorithms/observable_survival_rocket/run_gpu_smoke.sh \
  /root/autodl-tmp/observable-survival-smoke-01
```

The output directory must be NEW. The launcher checks dependencies, runs tests,
creates an isolated experiment bundle and runs a foreground smoke with a
45-minute timeout. It does not install packages, use SSH credentials, overwrite
old experiments or start a background job. GPU smoke: 16 environments x 8 steps
x 2 updates = 256 NEW transitions. It is still not a performance experiment.

The preparer generates full/no-tail/fixed-belief/no-solver/confidence-control
configs, with the same initial checkpoint, twenty-five hypotheses, persistent
training reward and **131,072 additional transitions**. Existing history is
accounted separately: the shipped initialization already has 6,225,920 recorded
transitions. None of these runs is learning from scratch or a small-total-data claim.
An explicit additional budget prevents short continuation runs from accidentally
being interpreted as a total budget smaller than their inherited training history.

```bash
python /root/autodl-tmp/observable-survival-smoke-01/bundle/scripts/run.py train \
  --config /root/autodl-tmp/observable-survival-smoke-01/full.json \
  --out /root/autodl-tmp/observable-survival-full-seed730101 \
  --skip-evaluation
python /root/autodl-tmp/observable-survival-smoke-01/bundle/scripts/evaluate.py \
  --run-dir /root/autodl-tmp/observable-survival-full-seed730101 \
  --out /root/autodl-tmp/observable-survival-eval-seed730101 \
  --backend gpu --workers 1
```

Evaluate the newly trained run directly. `prepare-eval` targets the historical
packaged checkpoint, not this new architecture. The no-solver setting removes
solver execution, not all physical auxiliary losses. Preparation verifies source
hashes and stops on source drift instead of applying an unreviewed patch.

## Required research validation, not yet completed

Use at least three training seeds, equal additional interactions and identical
initialization, sensor access, actuator limits and full episode lengths. Compare
full versus no-tail and fixed-belief, as well as the existing confidence policy
and a conventional/adaptive controller with the same information budget.

Report official success, final settle, terminal dwell, success-then-failure,
terminal velocities/tilt, predictive errors, empirical interval coverage AND
width, end-to-end latency and training time. Pair initial states by hashes;
report training-seed variability separately. Time steps are not independent
landing trials. The historical four 128-environment cohorts are development
sets, not a fresh sealed test set. A new sealed-test runner is not implemented.

Thrust, wind/aerodynamic and actuator-delay distribution shifts require a separately
labeled evaluation extension with feasible, pre-specified ranges. They are NOT
silently added to the unchanged reference benchmark and are not implemented here.

The central hypothesis remains unproven: better factual uncertainty and a
post-contact continuation might improve durable landing, but the contact surrogate,
compute cost, partial observability and model mismatch can also make it worse.
