# Observable-belief, survival-aware rocket landing (research prototype)

## Actual execution status

Based on `Yxuan-Gong` commit `91a17f96e2ee289720f6bee521aab35f0ceb7647`.
**The authoring session has not connected to the supplied GPU and has not run
Genesis training or evaluated a new landing policy.** The local execution
sandbox cannot resolve the supplied SSH hostname and has no SSH client.
No credentials are stored in the repository or passed to CI. The original
`Yxuan-Gong` branch, remote machine and historical checkpoints are unchanged.
CPU tests and a synthetic identification probe are provided; execution evidence
must come from actual logs, not this README. See `results/validation_status.json`
and the bounded CPU validation workflow. CPU tests do not certify a landing policy.

This is a **state-based physical-intelligence component**, not a vision-language
action model: the repository actor consumes 30 state channels. No image/language
VLA training is claimed. Novelty, speed and better landing performance remain
unproven. Kalman inference, cubature and terminal MPC costs are established tools;
combining them is not evidence of a disruptive invention.

## What the existing results actually say

The stored comparison reports 419/512 official event successes for DAgger and
428/512 for the published physical actor. The latter has only 3/512 terminal
settled episodes and 214 episodes that succeeded and later failed. Discordant
pairs are 79 gains and 70 losses; the paired exact test is recomputed in the
research audit. These are EXISTING results, not a fresh run, and they do not
isolate architecture effects or variation across training seeds.

This prototype asks whether (1) evidence-grounded uncertainty helps and (2)
reasoning beyond first contact helps the rocket remain landed. It preserves
original success and adds a separate metric: final 80 consecutive 40-Hz samples
settled, without fatal/out-of-bounds failure in the first episode. This is sampled,
not continuous-time stability. A temporary two-second dwell followed by a crash
is not durable success. A historical final-settled count does not establish a
two-second dwell; that metric needs fresh full-episode traces.

## Candidate architecture

`ObservableSurvivalActor` reuses `ConfidenceSequenceActor`, its rigid-body and
contact operators, mixed discrete/continuous action distribution and PPO loop.

* **Observable physical belief:** 12 effective coefficients with a full 12x12
  Joseph-form covariance. Local parameter sensitivities are computed through
  the physical predictor from the actual previous state/action and measured
  successor. No task-loss gradient changes factual parameter estimates. Invalid
  reset transitions cannot shrink covariance. Zero sensitivity supplies no
  information. Contact transitions are excluded from flight identification;
  contact coefficients are not identified in this version. Noise is predicted
  from past context, before seeing the new outcome. These are effective gains,
  not uniquely recovered physical mass and thrust.
* **Settling continuation:** near the pad, candidate actions are additionally
  evaluated with a shared engine-off continuation through the existing contact
  surrogate. This is a fixed-policy counterfactual heuristic, not an optimal
  value function, invariant set or proof of safety. Wrong contact physics can
  make it harmful; the no-tail ablation is mandatory.

A 25-point positive-weight cubature rule represents all 12 covariance directions.
The points are not calibrated posterior samples or confidence bounds. Model bias,
parameter clipping and robust innovations mean nominal Gaussian intervals require
empirical coverage checks. State-observation noise remains an important limitation.
The method may be substantially slower than the seven-hypothesis baseline; no
speedup is claimed. It does not infer new contact mechanisms.

## Relationship to the current repository environment

The existing packaged actor uses the preserved upstream snapshot from commit
`39e6bb6fd5d0f5432f7405a76946938a469492ce`. This extension deliberately evaluates
on that SAME snapshot for comparability. The current `environments/rocket/env.py`
has different action-interface/reward revisions, so the old normalized ignition
and throttle channels must NOT be passed to it without a separately tested
adapter. No claim of evaluation on the current live environment is made here.

## Code

- `observable_belief.py`: full-covariance update and covariance-matching cubature.
- `rocket_observable_survival.py`: actor extension with factual inference and tail cost.
- `survival_metrics.py`: separate first-episode terminal-dwell metrics.
- `prepare_experiment.py`: hash-checks reviewed sources, creates a private experiment
  copy, patches the architecture registry and evaluator, records hashes.
- `preflight.py`: read-only CUDA/dependency check; no installs or workload launch.
- `run_gpu_smoke.sh`: explicit bounded foreground smoke training.
- `synthetic_probe.py`: linear acceleration identification; NOT Genesis or a VLA.
- `tests/`: unit tests and actual-checkout forward/backward integration tests.

## Local tests

```bash
python -m pytest algorithms/observable_survival_rocket/tests -q -ra
python algorithms/observable_survival_rocket/synthetic_probe.py \
  --out /tmp/rocket_synthetic_probe.json
```

The actual-source integration test explicitly skips when the checkout or `rsl_rl`
is unavailable. A skip is not a pass. The CPU CI workflow installs the published
experiment's versions; dependency-install failure must be resolved, not hidden.

## GPU smoke test — not already run

Use a clean checkout of this feature branch and the repository's existing
experiment dependencies. No SSH credentials belong in these commands or files.

```bash
bash algorithms/observable_survival_rocket/run_gpu_smoke.sh \
  /root/autodl-tmp/observable-survival-smoke-01
```

The output directory must be new. This neither overwrites previous experiments
nor changes the reference environment. The wall-time limit is 45 minutes. The
smoke run uses 16 environments x 8 steps x 2 updates = 256 new transitions and a
2-step tail. It is only a software test. Research configs use a 40-step tail.

Preparation generates `full.json`, `no_tail.json`, `fixed_belief.json`,
`no_solver.json`, and `confidence_control.json`, all with the same initialization,
25 hypotheses, persistent training reward and 131,072 new-transition budget.
The no-solver setting removes solver execution, not every physical auxiliary loss.

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

Evaluate the trained run directly. Do not use `prepare-eval` for a new architecture;
that entrypoint is for the historical packaged checkpoint. Terminal samples are
recorded before auto-reset; nonterminal samples afterward. Episodes do not end
early merely because success occurred once.

## Pre-registered research comparison

Use at least three training seeds, equal new transitions, identical initial
checkpoints, sensor information, action limits and complete original horizons.
Keep the four historical 128-environment cohorts as development sets. Freeze
hyperparameters before using new pre-specified held-out evaluation seeds; a new
sealed-test runner is not implemented here. Include conventional/adaptive control
with the same information budget. Primary comparisons are full vs no-tail and
fixed-belief on durable success without material loss of official success.

Report original success, terminal settle, terminal two-second dwell,
success-then-failure, speed/tilt, predictive errors, covariance coverage and interval
width, action latency including optimization, and total training time. Pair
initial states and report training-seed variability separately. Correlated
transitions are not independent landing trials.

Thrust effectiveness, aerodynamic disturbance and actuator-lag shifts need a
separately labeled evaluation extension, not hidden edits to the frozen benchmark.
They are NOT implemented here. Specify feasible ranges before testing.

## Limits

The contact surrogate is not the Genesis contact solver. A short engine-off tail
cannot guarantee sustained stability. Full covariance is still approximate and
does not solve structural model mismatch or unobservable parameter separation.
The actor requires actual-source integration and GPU smoke tests before research
training. The synthetic probe only tests elementary filter behavior under a
correct linear measurement equation; it says nothing about rocket success or
superiority over adaptive MPC. Use this prototype in simulation only.
