# MoE-Based Rocket Landing: Policy Learning and Feedback Repair

[中文说明](README.zh-CN.md) · [Technical report](technical_report.md)

A focused rocket-landing experiment built on AerospaceRL's existing PDG teacher, BC warm start, PPO training, and simulator. We developed a **phase-routed mixture-of-experts (MoE) policy**, improved its training-state coverage with DAgger, and added expert selection and state-triggered feedback repair. Here, MoE means three action heads selected by flight phase; it does not denote a large pretrained foundation model.

The highest recorded CPU development result is **444/512 (86.72%)** under the historical benchmark protocol. This is the environment's **ever-success** metric; final strict settling was **0/512**. The six-episode gain over the 438/512 parent remains exploratory.

## What we added to the original project

| Area | Original starting point | Work completed in this study |
| --- | --- | --- |
| Training data | 512-step PDG demonstrations for BC | Complete 1200-step demonstrations and DAgger labels on student-visited states |
| Policy | One Flash residual actor | 33 state-derived features, a shared trunk, and high-altitude / approach / near-ground action heads; weighted action regression plus ignition classification |
| PPO and rewards | Existing RSL-RL PPO and task reward | Forward/projection consistency repair; reward scaling and bounded event/potential rewards; learning-rate, GAE, exploration, reference-KL and selective-update experiments |
| Policy combination | One trained actor per run | Initial-state success-difference routing between frozen DAgger and PPO experts |
| Failure repair | No learned finite-intervention layer | Search ten feedback programs and learn when to invoke one, with at most one intervention per episode |
| Experiment tooling | Batched simulation and generic evaluation already existed | Source snapshots, fixed first-episode cohorts, paired outcomes, transition budgets, and separate final-settling diagnostics |

The **79.10% MoE/DAgger policy was already a result of our work**, not an author-provided baseline. The later CPU records are:

```text
Phase MoE + DAgger r3       405/512  (79.10%)
Continued DAgger expert     432/512  (84.38%)
Frozen expert combination   438/512  (85.55%)
Gated feedback repair      444/512  (86.72%)
```

These stages used different training budgets and are not isolated component ablations. A small-budget reproduction of upstream PPO scored 0/512 using about 1.6% of its default online budget; that does not establish the original full-budget performance. The upstream 66.8% reference belongs to the classical PDG controller under a different evaluation window.

## What this package contains

This submission contains the best-recorded policy's inference code, its frozen weights, the **feedback-repair training stage from the supplied frozen parent**, and evaluation evidence. The historical DAgger/PPO sweeps are summarized in the report; this is not a complete retraining package for every earlier expert.

| File | Purpose |
| --- | --- |
| [phase_policy.py](phase_policy.py) | Shared representation and three phase-specific action heads |
| [expert_router.py](expert_router.py) | Initial-state router over the two frozen experts |
| [feedback_program.py](feedback_program.py) | Twelve-parameter bounded feedback programs and observed-state features |
| [repair_policy.py](repair_policy.py) | Learned intervention gate, episode memory, and checkpoint loader |
| [train.py](train.py) | Failure screening, paired branch search, gate fitting, and full development evaluation |
| [evaluate.py](evaluate.py) | Standalone evaluation of the supplied checkpoint |
| [runtime.py](runtime.py), [metrics.py](metrics.py) | Historical environment isolation and first-episode accounting |
| [assets/phase_aware_feedback_repair.pt](assets/phase_aware_feedback_repair.pt) | Self-contained policy, including both experts and gates |
| [evidence/](evidence/) | Recorded outcomes, comparison, provenance, and training metadata |

## Run

Use Python **3.11** and a checkout containing commit `39e6bb6fd5d0f5432f7405a76946938a469492ce`. The entry points restore that recorded environment into a new output directory, so the repository's later environment reorganization does not silently change the benchmark.

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r algorithms/moe_rocket_landing/requirements.txt

# Full development evaluation: four fixed cohorts of 128 first episodes.
.venv/bin/python algorithms/moe_rocket_landing/evaluate.py \
  --out results/moe_landing_evaluation

# Retrain the repair stage from the frozen parent embedded in the checkpoint.
.venv/bin/python algorithms/moe_rocket_landing/train.py \
  --out results/moe_landing_repair_training \
  --max-envs 256 --physics-threads 4
```

Output directories must be new. Evaluation defaults to CPU, float32 physics, seeds `1000007 / 2000007 / 3000007 / 4000007`, and a 1200-step control horizon. `--cohorts 1` runs a diagnostic subset and is explicitly marked as an incomplete development protocol. A shallow clone may first need `git fetch origin 39e6bb6fd5d0f5432f7405a76946938a469492ce`.

The measured repair policy preserves its **150 m activation ceiling**, probability threshold 0.98, local support radius 1.5, and one-intervention-per-episode rule. Retraining consumes up to **1,920,000 new control transitions**, followed by 614,400 evaluation transitions; historical expert/router training is additional. The adaptive search batch layout and simulator batch effects mean retraining is not promised to reproduce identical weights or outcomes.

## Other explored routes

| Route | Recorded outcome | Interpretation |
| --- | --- | --- |
| PPO projection repair and two reward alternatives | Three small-budget CPU candidates: 0/512 | No success-rate gain at that budget |
| Long-credit / ignition PPO | CPU 424/512 (82.81%) | Below its 432/512 starting policy |
| Reference-KL PPO continuation | CPU 406/512 (79.30%) | Same parent and 1.92M new transitions as repair; this configuration degraded |
| Episode-return RLOO | CPU 397/512 (77.54%) | Below its 405/512 starting policy |
| Touchdown expert selection | CPU 423/512 (82.62%) | Below initial-state routing |
| Restricted replay refinement | CPU 97/128 vs. 100/128 | Screening only; no full 512-episode result |
| Program-gain selector v2 | Fresh 96-pair confirmation: 83→81 | Did not proceed to full development evaluation |
| Event repair selector v3 | CPU 408/512 (79.69%) | Below its 438/512 parent |
| Unified physics policy + PPO | GPU 428/512 vs. GPU start 419/512 | Separate backend; excluded from CPU ranking |

## Result interpretation

The repair candidate improved 24 parent failures and lost 18 parent successes, a net **+1.17 percentage points**. Only 38/512 episodes triggered a repair, and many outcome changes occurred in untriggered episodes. These observations indicate batch-coupling effects and prevent attributing the full gain to direct repairs. The nominal paired p-value is 0.4408; even its usual independence assumptions require caution here.

The development suite has been reused across experiments; the sealed test was not used. No 95%/99% success, reliable improvement over the parent, or final stable-landing result is claimed. See the [technical report](technical_report.md) for the exact architecture, training budget, per-cohort results, and limitations.
