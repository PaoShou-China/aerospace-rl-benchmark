# pyflyt-genesis Technical Highlights and Reproducibility Notes

**English** · [简体中文](technical-highlights.zh-CN.md)

Supplementary material for reviewers. Initial-version anchor: commit `6b02383`
(2026-10-21); this document was originally added on top of that commit. Paths are
relative to the repository root; `#L...` line anchors come from that commit and
drift as the worktree evolves. On 2026-11-02 the task count (10), wind
configuration, evaluation conventions and baseline records were updated to the
current worktree; when citing a specific location, use the file and
function/symbol in the code index at the end, not the line number.

This document states only facts that **land on a code line and carry a
re-verification condition**. Each highlight follows "claim → mechanism →
evidence → significance for reviewers → boundary", followed by separate
"Non-claims" and "Claim status" sections, so design intent is not misread as a
verified conclusion.

---

## Abstract

Compared with general RL environment libraries (Gymnasium / Brax / Isaac Lab,
etc.), this suite's increment is not the number of environments but six kinds of
**infrastructure for controlled experiments**:

1. **Hash-manifested reference baseline**: upstream PyFlyt 0.29.0 task code is
   shipped as a frozen copy; source/asset SHA-256 verification primitives are
   also provided. Tensor-plant inertias are parsed directly from the reference
   URDF, not hand-copied.
2. **Multiple execution paths**: 4 tasks are pure tensor implementations
   (satellite / transfer / docking / rendezvous), 6 are Genesis rigid-body
   implementations (quadx / quadx_track / fixedwing / quadplane / rocket /
   missile); the latter also has a shared-scene batch bridge that runs the
   **unmodified** original NumPy task code, enabling implementation-level
   cross-validation.
3. **Training infrastructure**: warmup-cosine LR scheduling, deterministic
   evaluation, and controlled hyperparameter knobs shared across tasks
   (`lr_sigma_coupling`, `sigma_pin`, `freeze_obs_norm`).
4. **Result provenance**: `run.json` records the task configuration, real
   transitions, physics-step call counts, device-side completion metrics, and an
   explicit numerical non-equivalence boundary.
5. **Classical (non-learned) baseline library**: each of the 10 tasks has an
   analytic GNC baseline (geometric control, L1+TECS, PDG, CWH-LQR, two-impulse
   guidance, etc.), giving a non-learned reference under the same environment,
   reward and success criterion, while retaining the baselines' honest failure
   modes as surfaces for RL to attack (all three quadplane controller arms are
   evaluated under the official convention; see Highlight 6).

---

## Highlight 1: An auditable reference baseline, not "a rewrite"

**Claim**: the reference implementation exists as "frozen copy + hash manifest +
inertia parsing"; the audit primitives can be run manually but are not currently
runtime gates.

**Mechanism and evidence**

- The original code path does not depend on the PyFlyt installed in the
  environment: the bridge imports the repository's frozen copy
  `environments/common/original_pyflyt`
  ([original_backend.py:455-468](../environments/common/original_backend.py#L455-L468)).
- `verify_installed_reference()` verifies the **installed** `PyFlyt==0.29.0`:
  the version, plus the SHA-256 of the manifest's source files and URDF/YAML
  assets; any mismatch raises immediately
  ([consistency.py:26-44](../environments/common/consistency.py#L26-L44)).
- `contract()` hashes the repository's
  `environments/common/assets/reference_*` and serializes vehicle, `dt`, control
  extraction ratio, coordinate conventions and noise/wind settings into a
  `plant_id` ([consistency.py:97-127](../environments/common/consistency.py#L97-L127)).
- Inertias are not hand-written constants: `asset_inertials()` parses the
  reference URDF for tensor dynamics
  ([consistency.py:63-95](../environments/common/consistency.py#L63-L95),
  [dynamics.py:25-27](../environments/common/dynamics.py#L25-L27)); the same
  inertias pass through `inertial_basis()` to rotate "principal axes → author
  axes" before being fed to the Genesis rigid body
  ([plant.py:441-462](../environments/common/plant.py#L441-L462)).

**Significance**: reviewers can distinguish "the code reproduces the task
definition" from "the code happens to run"; the reference version, asset hashes
and inertia provenance are all checkable.

**Boundary (must read together)**

- The two verification functions are **not called by the training/evaluation
  entry points**; this development environment also does not have PyFlyt
  installed, so the hash re-verification command requires installing 0.29.0
  first.
- Coverage differs: `verify_installed_reference()` verifies the **installed
  package**, not the repository's frozen `environments/common/original_pyflyt`
  copy; `contract()` verifies the repository's reference assets. They are not
  substitutes for each other.
- Hash agreement **does not equal** physical agreement. The Genesis compatibility
  backend declares on its first line: "API compatibility backend, not a claim of
  bitwise physics equivalence"
  ([original_backend.py:1-5](../environments/common/original_backend.py#L1-L5)).

---

## Highlight 2: Cross-validatable multiple execution paths sharing one protocol

**Claim**: the environment layer provides not just "a physics-engine wrapper" but
three mutually verifiable execution modes — pure tensor (no engine), tensorized
plant (Genesis rigid body), and the bridged original NumPy task code; upper-level
training/evaluation code does not need to perceive the difference.

**Mechanism and evidence**

1. **Pure tensor path** (satellite / transfer / docking / rendezvous)
   - No engine: RK4 / symplectic Euler and other integrators live directly in
     each env; state, random numbers and statistics all stay on device
     ([README.md §Package layout](../README.md#package-layout)).
   - The suite-level convention is that every step is a masked tensor kernel:
     inactive replicas freeze and **do not consume random numbers**; the
     training loop has no gym and no CPU round-trips ([README.md](../README.md)).

2. **Tensor plant path** (quadx / quadx_track / fixedwing / quadplane / rocket /
   missile)
   - `TensorPlant` uniformly carries geometry, actuation (rate-PID mixer, lifting
     surfaces, booster/fuel), physics stepping, contact checks and reset/warmup;
     reward/termination/observation remain in each env
     ([plant.py:1-9](../environments/common/plant.py#L1-L9)).
   - `TensorRandom` maintains an independent counter stream per replica; the
     documentation explicitly states it is **distribution-compatible but not
     NumPy PCG64 bitstream-compatible** with upstream
     ([plant.py:60-72](../environments/common/plant.py#L60-L72)).
   - Actuators, contacts and aerodynamics are tensorized layers; observations,
     rewards and resets follow the same protocol.
   - The integrator choice is explicitly recorded with a measured justification:
     explicit Euler makes landing contact chatter to ~0.1 m/s, crossing the
     0.02 m/s criterion, so Genesis's default `approximate_implicitfast` is
     retained ([original_backend.py:95-100](../environments/common/original_backend.py#L95-L100)).
     Such "measurements behind a parameter" records are more valuable to
     reviewers than the parameter itself.

3. **Original-code bridge** (quadx / fixedwing / rocket only):
   `OriginalTaskBatch` hosts N replicas in one Genesis scene, while the Python
   side remains **unmodified** PyFlyt task/controller/aero objects
   ([original_batch.py:1-7](../environments/common/original_batch.py#L1-L7),
   [original_batch.py:95-113](../environments/common/original_batch.py#L95-L113)).
   - On selective reset, non-participating replicas' physics state is rolled back
     exactly via `scene.get_state()` / `scene.reset(state=..., envs_idx=...)`,
     and their Aviary-observed physics time is unchanged
     ([original_batch.py:142-185](../environments/common/original_batch.py#L142-L185)).
   - Contact information is read **before** state rollback to avoid losing it
     after the cache is cleared
     ([original_batch.py:165-181](../environments/common/original_batch.py#L165-L181)).
   - `loadURDF` verifies that all replicas load the same-named URDF, preventing
     mixed batches ([original_batch.py:62-73](../environments/common/original_batch.py#L62-L73)).

**Significance**: the 4 engine-free tasks and 6 engine tasks share one protocol
and can be verified separately; 3 atmospheric tasks (quadx / fixedwing / rocket)
can also be cross-validated between "original implementation / tensorized
implementation" rather than trusting a single implementation.

**Boundary**

- Correctness comes first for this bridge: the original NumPy code still runs on
  host and is **not the throughput-optimal path**; its value is a correctness
  comparison, not performance numbers.
- "Pausing non-participating replicas" is **post-step state-rollback** semantics,
  not letting the engine skip them; NumPy/controller/RNG state is untouched, but
  the physics solver did advance the whole batch.

---

## Highlight 3: Wind disturbance is a unified plant hook, not a per-env force patch

**Claim**: the 5 atmospheric tasks (QuadX-Waypoints, Fixedwing-Waypoints,
Quadplane-VTOL, Rocket-Landing, Missile-Intercept) share one wind-disturbance
path — aerodynamic forces are uniformly evaluated with **airspeed**
`v_air = v - wind`; quadx / fixedwing / quadplane training entries hard-code a
30% Dryden gust `WIND` constant as the default wind field, rocket fixes
`dryden=0.0`, and missile stays in calm air. Calm air (default) is
**bit-identical** to the published baseline.

**Mechanism and evidence**

- `TensorPlant` holds `wind` (world-frame (n,3) float64, `None` = calm air);
  `_air_velocity` subtracts wind in world frame then rotates to body axes
  ([plant.py](../environments/common/plant.py)), so fixedwing lifting surfaces
  and rocket fin/body-drag aerodynamic terms automatically sense wind;
- QuadX has no lifting surfaces; its existing body-drag term's variable is
  already called `vb_air` — before integration it equaled body-axis velocity,
  after integration it becomes `vb - R^T·wind`, and in calm air it returns
  bit-for-bit to the old implementation;
- Wind generation uniformly goes through
  `environments/common/disturbances.py`'s `WindDisturbance` (steady wind + Dryden
  gusts, MIL-F-8785C low-altitude shaping filters, seeded RNG streams); each env
  updates `plant.wind` once per control step (gusts freeze between physics
  substeps), and `reset_idx` mirrors the filter state reset;
- Training entries hard-code `WIND`: quadx / fixedwing / quadplane use pure
  Dryden gusts (zero mean, 30% moderate); rocket uses `dryden=0.0` (same
  convention as the calm-air baseline); missile passes no wind parameters;
  quadx_track's constructor does not accept wind parameters. Satellite tasks are
  in vacuum with no wind.

**Invariant (the most critical point)**

- The calm-air path never enters the wind branch: with `wind=None`,
  `_air_velocity` and QuadX's drag term follow the same bytecode path as the old
  implementation; the first reset and subsequent steps with the same seed are
  **bit-identical**;
- `eval.py` restores the training `env_kwargs` from
  `results/<task>/config.json` when it exists, so a wind-trained policy is
  evaluated under the same wind configuration; only when the config file is
  missing does it use the environment's default calm air. Cross-method
  comparisons must explicitly fix the same wind configuration instead of
  assuming evaluation is always calm.

**Significance**: wind disturbance is not "a hand-written patch to the
dynamics", but the physically correct formulation of evaluating all aerodynamic
forces against airspeed; the effect of "disturbance robustness" on a policy can
be tested separately without suspecting the evaluation standard was quietly
relaxed.

**Boundary**

- Gust filters advance once per control step and freeze between physics
  substeps (Dryden bandwidth is far below the control frequency, so the error is
  negligible);
- `WindDisturbance`'s RNG is a per-replica counter stream (seed
  `mix32(seed ^ i*GOLDEN)` + per-replica counter, all on device, no CPU→GPU
  transfer); a `step` with `env_ids` advances only the selected replicas'
  counters and does not change other replicas' noise sequences; the calm-air
  path has zero overhead.

---

## Highlight 4: Training infrastructure — reusable LR scheduling and controlled knobs

**Claim**: the repository provides a reusable training schedule and controlled
hyperparameter knobs that guarantee fair cross-task comparisons.

**Mechanism and evidence**

- **warmup-cosine LR scheduling**: the PPO side has a built-in
  `_warmup_cosine_decay` formula (warmup fraction 1e-6 × total updates,
  peak→end span); because rsl-rl never resets lr under `schedule='fixed'`, the
  code writes `optimizer.param_groups` directly
  ([train_utils.py:27-36](../algorithms/train_utils.py#L27-L36),
  [train_utils.py:681-697](../algorithms/train_utils.py#L681-L697)).
- **Other controlled knobs**: exponential decay, `lr_sigma_coupling` (makes lr ∝
  σ, keeping per-step policy drift rate constant), `sigma_pin` (re-pins
  exploration variance), `freeze_obs_norm` (freezes warm-start observation
  normalization); all are opt-in with explicit mutual-exclusion checks
  ([train_utils.py:551-566](../algorithms/train_utils.py#L551-L566),
  [train_utils.py:713-730](../algorithms/train_utils.py#L713-L730)).
  `lr_sigma_coupling` / `sigma_pin` read `log_std_param` and therefore apply only
  to `distribution='gaussian'`; the default `BetaDistribution` has no such
  parameter, so those knobs are inactive with it.

**Boundary**

- Wall-clock values are recorded on the same machine and are affected by
  concurrent work on that machine; they are order-of-magnitude references only.
- Deterministic evaluation is run uniformly by `eval.py` and written to each
  arm's `eval.json`; however, existing records do not all use the same evaluation
  protocol (different env counts/step counts), so **success rates from different
  `eval.json` files must not be compared directly**.

---

## Highlight 5: Result provenance and reproducibility

**Claim**: the record produced by one training run is sufficient to answer
"what ran, how much ran, and whether it was computed correctly".

**Mechanism and evidence**

- `run.json` records: task configuration (env kwargs written as-is), algorithm,
  iteration count, total transitions, wall time, per-replica
  `completed_episodes` / `completed_successes`, real `physics_step_calls`,
  device-side metrics for completed episodes, and a numerical non-equivalence
  statement ([train_utils.py:750-801](../algorithms/train_utils.py#L750-L801)).
- `config.json` is deep-copied before being handed to the runner, preventing
  `OnPolicyRunner` from mutating the config in place so that what is written is
  not the real training configuration
  ([train_utils.py:318-352](../algorithms/train_utils.py#L318-L352)).
- `eval.py` uses a frozen deterministic policy (mean actions, no exploration
  noise); success/return are accumulated on device by the environment and written
  to `eval.json` ([eval.py:1-7](../scripts/eval.py#L1-L7),
  [eval.py:123-139](../scripts/eval.py#L123-L139)). When constructing the
  evaluation env, it restores the training `env_kwargs` (wind, benchmark version,
  etc.) from `results/<task>/config.json`, so evaluation matches the training
  distribution; if that file is missing, it uses the environment defaults.
- Each task entry hard-codes hyperparameters and exposes no CLI; flexibility is
  traded for reproducibility ([README.md §Usage](../README.md#usage)).

**Significance**: the training/evaluation "input — configuration — accounting
convention" is all reviewable.

**Boundary**

- This is not a full experiment-management platform: there is no multi-seed
  orchestration and no automatic significance testing.
- **Recording gap**: training-entry-level switches (e.g. `lr_cosine`,
  `freeze_obs_norm`) are not written to `config.json`/`run.json`; reproduction
  must use the corresponding `train_<task>.py`'s hard-coded constants.
- `update()` performs one small scalar synchronization at episode boundaries (at
  most once per step and only when an episode ends); the step-by-step training
  path itself has no host synchronization.

---

## Highlight 6: A non-learned baseline per task, and the baselines are not strawmen

**Claim**: each of the 10 tasks has an analytic classical GNC default baseline
(plus 20 structural variants, 30 controller arms total), sharing the same
environment, reward and success criterion as learning algorithms; baselines
reach 100% on some tasks and retain **documented failure modes** on others.

**Mechanism and evidence**

- Unified contract: all baselines are implemented in
  [baselines.py](../algorithms/baselines/baselines.py) (`QuadXSE3Baseline` …
  `RendezvousTwoImpulseBaseline`,
  [baselines.py:527-3253](../algorithms/baselines/baselines.py#L527-L3253)), with
  factory entry `build_baseline(vehicle, env)`
  ([baselines.py:3939](../algorithms/baselines/baselines.py#L3939)). Each
  controller **reads only the `'policy'` observation group**, reads environment
  constants at construction (mass, thrust authority, CMG geometry, nominal orbit
  — the model knowledge a classical GNC designer would have), **reads no
  internal state buffer**, and does not change environment semantics
  ([baselines.py:1-26](../algorithms/baselines/baselines.py#L1-L26)).
- Unified evaluation protocol: [baseline_eval.py](../scripts/baseline_eval.py)
  defaults to evaluating any task's baseline at 1024 envs / 20000-step safety
  cap / seed 7, writing `results/<task>/baseline.json` (or
  `baseline_<controller>.json` with `--baseline`), with controller metadata
  marked `classical, policy-observation-only`. Shipped artifacts use the smaller
  windows recorded in their own files (not a uniform 1024 replicas); citations
  must state `num_envs`, `seed` and `env_steps`
  ([baseline_eval.py](../scripts/baseline_eval.py),
  [baseline_test_report.md](baseline_test_report.md)).
- Methodological coverage (default arms): SE(3) geometric control (Lee et al.
  2010, quadx), CV Kalman filter + SE(3) (quadx_track), L1 lateral + TECS
  longitudinal (fixedwing), quadrotor SE(3) + pusher (quadplane), ZEM/PN
  powered-descent guidance (rocket), augmented proportional navigation APN
  (missile), quaternion PD + singularity-robust CMG damped pseudo-inverse
  (satellite), Gauss-equation tangential guidance (transfer), exactly
  symplectic-discretized CWH-LQR (docking), recursively re-solved two-impulse
  CWH guidance (rendezvous). Structural variants cover cascaded PID,
  sliding-mode outer loop, pure pursuit, EMA filtering, NLGL, E-guidance,
  PPN/TPN, L1 wing mode, hybrid-mode autopilot, ordinary/weighted pseudo-inverse
  CMG, Edelbaum, Q-law, pole placement, CWH ZEM/ZEV, glideslope and CWH
  compensated PD ([baselines.py](../algorithms/baselines/baselines.py),
  [baseline_test_report.md](baseline_test_report.md) §5).
- **Baseline reference records** (multi-seed protocol: 128 replicas, first
  episode per replica, seeds 7/13/21 + held-out 101; see
  [baseline_test_report.md](baseline_test_report.md) §2): satellite / docking /
  rendezvous / rocket 100%; quadx 99.7% (3-seed mean); transfer 99.2%; quadplane
  89.1%; fixedwing 87.0%; quadx_track 85.9%; missile 77.3%. Held-out seed 101
  shows no systematic difference from the protocol seeds. Single shipped
  `results/<task>/baseline.json` windows differ: quadx is 32 replicas / seed 7 /
  800 steps (90.6%), most tasks are 128 replicas / seed 13, quadx_track is seed
  21; these numbers are not directly comparable with the protocol means above.
- **Failure modes are recorded item by item**: fixedwing residuals are the three
  variants' OOB (`l1_tecs` 2.5%, `nlgl_tecs` 6.8%, `pid_cascade` 10.1%);
  quadplane's weak arms are mostly timeouts (`hybrid_cascade` 68.5%, `l1_wing`
  50.4%, while default `se3_pusher` is only 12.0%); missile timeouts are 0–3.1%.
  Specialized harnesses
  ([eval_fixedwing_autopilot.py](../environments/fixedwing/eval_fixedwing_autopilot.py),
  [rocket_pdg_test.py](../environments/rocket/rocket_pdg_test.py)) provide
  scenario-level decomposition, but their result JSONs are not shipped.
- **Two-way evidence (only citing delivered results)**: the baselines are not
  strawmen — in `benchmark_report.md` §6.3–6.5's short 200/150-update budget,
  pure PPO's cumulative success on satellite is 0.43–0.69 and on docking 0.11,
  far below the baseline's 100%; BC warm-start raises satellite to 0.709 and
  docking to 1.000, and residual RL further reaches satellite 0.985 and docking
  1.000. The baselines are also not a ceiling — missile 77.3%, quadx_track 85.9%
  and fixedwing 87.0% all leave attackable failure space.

**Significance**: learning algorithms now have a non-learned reference frame;
the reference frame's failure modes are attributed (authority saturation /
geometric infeasibility / numerical pathology) rather than glossed over, so
"RL surpasses classical methods" can be grounded in specific scenarios.

**Boundary**

- Baseline formulas are analytic, with no optimizer and no tuning loop; their
  hyperparameters (gains, bandwidths, servo bands) are **task-level manual
  designs**, not claimed globally optimal.
- Success numbers depend on the evaluation window (steps/env count); different
  harnesses use different windows, so direct cross-harness success comparisons
  are invalid (consistent with Highlight 4's evaluation convention boundary).
- Rocket baseline success depends strongly on the evaluation window and version:
  v1's instantaneous touch-pad criterion is not comparable to v2's "continuous
  settle for 2 s" criterion. The current v2 artifact
  `results/rocket/baseline.json` (128 replicas / seed 13 / 1200 steps,
  `ever_stable_2s=1.0`) is 100%; citing historical v1 numbers requires stating
  the version and window.

---

## Non-claims

Reviewers are most easily misled by the word "reproduction"; the following
boundaries are explicit design statements:

1. **We do not claim bitwise physical equivalence with PyFlyt/Bullet**. The
   Genesis backend is an API compatibility layer with different engine numerics
   ([original_backend.py:1-5](../environments/common/original_backend.py#L1-L5)).
2. **We do not claim NumPy random bitstream identity**. `TensorRandom` is
   distribution-compatible with upstream but has a different bitstream
   ([plant.py:60-72](../environments/common/plant.py#L60-L72)).
3. **We do not claim optimality beyond baseline-vs-RL comparability**. Baselines
   are manually designed and do not represent a performance upper bound (see §6).
4. **We do not claim the original-code bridge is a high-performance path**. It is
   a correctness-first host-side bridge.
5. **Example numbers are not statistical conclusions**. Single-seed results are
   records and order-of-magnitude references only.
6. **We do not claim classical baselines are optimal**. They are analytic manual
   designs (no optimizer, no tuning loop); their value is a non-learned reference
   under the same criterion and attributed failure modes, not a performance
   ceiling.

---

## Claim status table

| Claim | Status | Evidence/re-verification |
|---|---|---|
| Upstream PyFlyt 0.29.0 source and assets are hash-verifiable | Implemented; **not auto-wired**; requires installing PyFlyt 0.29.0 | `python -c "from environments.common.consistency import verify_installed_reference as v; print(v())"` |
| `plant_id` plant certificate | Implemented; **not called by entry points** | `contract()`, [consistency.py:97-127](../environments/common/consistency.py#L97-L127) |
| Tensor-plant inertias come from the reference URDF | Wired | [plant.py:441-462](../environments/common/plant.py#L441-L462) |
| Inactive replicas consume no random numbers / do not advance state | Design guarantee + code path | [plant.py:1-9](../environments/common/plant.py#L1-L9) |
| Baseline difficulty is bit-reproducible | Verified (10 tasks, same-seed first reset and state after 3 steps) | Ad-hoc re-check script not retained |
| Evaluation uses the training-recorded configuration | Guaranteed (eval restores `env_kwargs` from `config.json`) | [eval.py](../scripts/eval.py) |
| PPO warmup-cosine LR schedule | Wired (opt-in `lr_cosine`) | [train_utils.py:27-36](../algorithms/train_utils.py#L27-L36), [train_utils.py:681-697](../algorithms/train_utils.py#L681-L697) |
| One classical baseline per task, policy observations only | Implemented + measured under the official convention | `python scripts/baseline_eval.py <task>`, [baselines.py:3939](../algorithms/baselines/baselines.py#L3939) |
| Bitwise physical equivalence with upstream | **Not claimed** | See Non-claims #1 |

> Note: the two "implemented but not wired" audit primitives (hash verification,
> plant_id) remain optional tools rather than mandatory training-time gates, so
> environments without the reference package installed are not blocked from
> normal training. If a review process requires mandatory verification, calling
> them once at the entry point is enough.

---

## Re-verification checklist

```bash
# 1) Reference hash verification (requires installing PyFlyt 0.29.0; not installed in this repo's dev env)
python -c \
  "from environments.common.consistency import verify_installed_reference as v; print(v())"

# 2) Restore the training-recorded env_kwargs for deterministic evaluation (quadx example)
python scripts/eval.py quadx --num-envs 16 --max-steps 400

# 3) Inspect one training run's record (the archived one in this repo is a smoke run)
python -c "import json;r=json.load(open('results/smoke/quadx_ppo/run.json'));print(r['task']);print(r['transitions'], r['physics_step_calls'])"

# 4) Locate the wind-disturbance hook implementation
grep -n "wind" environments/common/plant.py environments/common/disturbances.py | head -20

# 5) A classical baseline for any task (defaults: 1024 envs / 20000-step safety cap / seed 7,
#    writes results/<task>/baseline.json; shipped artifacts use the smaller window in their own JSON)
python scripts/baseline_eval.py quadx
```

---

## Code index

| Capability | File |
|---|---|
| Reference hash verification, URDF inertia parsing, `plant_id` | [environments/common/consistency.py](../environments/common/consistency.py) |
| Pure tensor plant, per-replica random streams | [environments/common/plant.py](../environments/common/plant.py) |
| Genesis compatibility backend (BulletClient API) | [environments/common/original_backend.py](../environments/common/original_backend.py) |
| Original-task shared-scene batching | [environments/common/original_batch.py](../environments/common/original_batch.py) |
| Training/record pipeline, LR schedule alignment | [algorithms/train_utils.py](../algorithms/train_utils.py) |
| 10-task classical baseline library (30 controller arms) | [algorithms/baselines/baselines.py](../algorithms/baselines/baselines.py) |
| Unified baseline evaluation entry | [baseline_eval.py](../scripts/baseline_eval.py) |
| Baseline-specific harnesses (scenario grids/ablations) | [eval_docking_lqr.py](../environments/satellite/rendezvous_docking/eval_docking_lqr.py), [eval_fixedwing_autopilot.py](../environments/fixedwing/eval_fixedwing_autopilot.py), [eval_rendezvous_baseline.py](../environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py), [rocket_pdg_test.py](../environments/rocket/rocket_pdg_test.py), [test_quadx_se3.py](../environments/multirotor/test_quadx_se3.py) |
| Deterministic evaluation | [eval.py](../scripts/eval.py) |
