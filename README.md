<p align="center">
<a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><img src="docs/assets/aerospacerl-cover.webp" alt="AerospaceRL Benchmark" width="100%"></a>
</p>

<p align="center"><strong>Aerospace control · GPU-parallel simulation · Reinforcement learning</strong></p>

<p align="center"><a href="README.zh-CN.md">简体中文</a> · <a href="#quick-start">Quick start</a> · <a href="#task-overview">Tasks</a> · <a href="#documentation">Documentation</a> · <a href="#acknowledgments">Acknowledgments</a></p>

<p align="center"><a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><strong>Explore the project and task replays ↗</strong></a><br><sub>Cover: brand illustration. Task demonstrations: simulation replays.</sub></p>

---

**GPU-parallel reinforcement learning environments for aerospace control.**
Ten single-agent tasks — six atmospheric vehicles on a URDF rigid-body backend,
four spacecraft missions on explicit tensor dynamics — behind one batched
device-tensor interface. No Gym / Gymnasium wrappers.

Each task shares the same `step` / `reset` / buffer contract, ships a classical
GNC baseline on the same reward and success bar, and runs physics, contacts,
reward, reset and RNG as masked tensor kernels. Host decisions such as
episode-completion checks can still trigger CPU/GPU synchronization; the accelerated path reduces repeated dispatch through caching
and segmented CUDA graphs.

## Quick start

```bash
git clone https://github.com/PaoShou-China/aerospace-rl-benchmark.git
cd aerospace-rl-benchmark
python -m pip install -r requirements.txt

python environments/multirotor/train_quadx.py   # train (parameters live in the file)
python scripts/eval.py quadx                     # evaluate a checkpoint
python scripts/baseline_eval.py quadx            # evaluate the classical baseline
```

A CUDA build of PyTorch is required for the six Genesis tasks on the GPU
backend; the four tensor-dynamics tasks and every classical baseline also run
on CPU-only PyTorch. Exact versions, the environment contract and the optional
Dryden wind interface are in
[environment-contract.md](docs/environment-contract.md); all training
entries, run artifacts and the repository tree are in
[Repository layout](#repository-layout) below.

## Benchmark experiment

One command runs the full protocol — train, classical baselines, and
evaluation — across all 10 tasks. Each stage is resumable: runs with an
existing `run.json` / `eval_seed_*.json` / `baseline.json` are skipped.

```bash
# all stages: train -> baselines -> evaluation (~2-3 days, one GPU)
python scripts/run_experiment.py

# single stage
python scripts/run_experiment.py --phase train        # 150 training runs
python scripts/run_experiment.py --phase baselines    # classical controllers
python scripts/run_experiment.py --phase eval         # evaluate every checkpoint

# subset of tasks / seeds
python scripts/run_experiment.py --tasks quadx missile --seeds 11 22
```

**Experiment matrix** (fair comparison: every other hyperparameter is shared)

| Axis | Value |
|---|---|
| Method | `ppo_scratch` · `bc_ppo` (BC warm start) · `residual_ppo` (ρ=0.05) |
| Training seeds | 11, 22, 33, 44, 55 |
| Evaluation seeds | 1011, 1022, 1033, 1044, 1055 |
| Parallelism | 3 workers (best on one GPU; more only adds CPU overhead) |
| Output | `results/5seed/<task>/<method>/seed_<s>/` |

**Run artifacts** (per run directory)

| File | Contents |
|---|---|
| `run.json` | training cost (transitions, wall clock, success rate) |
| `eval_seed_<n>.json` | success rate, return and related metrics for one eval seed |
| `checkpoint.pt` | final policy weights |
| `config.json` | environment and training configuration (for reproduction) |

Aggregate the runs into a report:

```bash
python scripts/aggregate_5seed.py
# -> results/5seed/summary.json  +  results/5seed/report.md
```

> **Long runs:** detach from the terminal with `setsid` so a dropped session
> does not kill the job:
> ```bash
> setsid python scripts/run_experiment.py > results/5seed/run.log 2>&1 &
> ```
> The log keeps key events only (start / finish / fail / progress) and does not
> capture the child training output.

## CUDA acceleration

All ten tasks support opt-in execution adapters for expert data collection,
behavior cloning (BC) and PPO training. Task equations, rewards, model
precision and training budgets keep their existing configuration. There are
no GPU-model or VRAM-capacity branches.

From the repository root, enable training acceleration through the task's
existing `main` function (Rocket-Landing shown here):

```python
from environments.rocket.train_rocket import main
main(performance=True, out_dir="results/cuda_rocket", clean=False)
```

Evaluate the resulting checkpoint:

```bash
python scripts/eval.py rocket --performance --out results/cuda_rocket
```

These acceleration options default to off. Unsupported PPO configurations
retain the upstream update implementation.

### Execution layers

With `performance=True` the environment step runs as a short sequence of CUDA
graph replays (`algorithms/performance.py`):

1. **Genesis performance mode.** Genesis compiles static-array kernels
   specialised to the scene. For these single-body (6-DOF) vehicles it then
   solves contacts with its one-kernel *monolith* Newton arm. The generic
   kernels instead pin the *decomposed* arm, whose iteration loop runs on the
   host with a synchronization per iteration on GPUs below SM 9.0 (RTX 30/40
   series). Both arms use the same physical problem and solver tolerance, but their
   execution paths differ and may produce floating-point differences. Equal
   tolerances alone do not establish equivalence of complete trajectories or
   training behavior.
2. **Whole-substep graphs with Genesis kernels inside.** Quadrants kernels
   are recorded into the same CUDA graph as the Torch math: Quadrants' own
   graph launches are switched to plain launches while capturing, the launch
   contexts whose host argument buffers back the recorded copies are kept
   alive, and Genesis' periodic error check and host step counter are replayed
   at Genesis' own cadence.
3. **Graphed step segments.** Each `env.step` is split at its host decisions
   (pause-restore, reset, episode logging) into segments that update
   persistent buffers in place (`_step_begin`, `_control_step`, `_step_end`,
   `_reset_replicas`; `_step_core` for the spacecraft tasks). Each segment
   replays as one graph. Before a segment is captured, its second warmup and
   the capture are audited: integer counters are replayed, and any other host
   state change (a tensor attribute replaced instead of updated in place)
   keeps that segment eager with a message instead of replaying stale state.
4. **Optional fused task math** (`AEROSPACE_FUSE=1`). Inductor fuses the
   elementwise plant, controller, observation and integrator math into Triton
   kernels that the segment graphs then record. Fusion contracts
   multiply-adds and may reorder reductions (float32-rounding differences),
   and adds a one-time compilation at startup.

| Variable | Default | Effect |
| --- | --- | --- |
| `AEROSPACE_GENESIS_PERFORMANCE` | `1` | `0` keeps Genesis' generic kernels (bitwise as before; layers 2-3 then fall back to Torch-only segments) |
| `AEROSPACE_NATIVE_CAPTURE` | `1` | `0` keeps Genesis calls outside CUDA graphs |
| `AEROSPACE_SEGMENT_GRAPHS` | `1` | `0` disables graphed step segments |
| `AEROSPACE_FUSE` | `0` | `1` enables fused task math (layer 4) |
| `AEROSPACE_SEGMENT_DEBUG` | `0` | `1` reports why a segment re-warms |

## Task overview

### Atmospheric flight · 6 tasks

| Task | Obs. dim. | Act. dim. | Policy rate / step | Success summary |
|---|---:|---:|---|---|
| QuadX-Waypoints | 37 | 4 | 40 Hz | 4/4 waypoints < 0.2 m |
| QuadX-Track | 25 | 4 | 40 Hz | visible fraction ≥ 0.85, no crash |
| Fixedwing-Waypoints | 39 | 4 | 30 Hz | 4/4 waypoints < 10 m |
| Quadplane-VTOL | 47 | 8 | 40 Hz | 4/4 waypoints < 15 m |
| Rocket-Landing | 30 | 7 | 40 Hz | settled for 2 s; full v2 criterion in task specification |
| Missile-Intercept | 34 | 3 | 60 Hz | miss distance < 15 m |

### Spacecraft · 4 tasks

| Task | Obs. dim. | Act. dim. | Policy rate / step | Success summary |
|---|---:|---:|---|---|
| Satellite attitude (CMG) | 26 | 4 | 40 Hz | attitude < 0.05 rad, \|ω\| < 0.02 rad/s |
| Low-thrust transfer | 8 | 3 | 300 s/step | terminal orbital-element return |
| Docking | 6 | 3 | 1 s/step | range < 1 m, speed < 0.1 m/s |
| Rendezvous | 6 | 3 | 1 s/step | range < 10 m, speed < 0.5 m/s, keep-out |

Full dynamics, episode lengths, observation layouts and reward decompositions:
[tasks.md](docs/tasks.md) and
[method/README.md](docs/method/README.md).

Baseline success rates and their evaluation conditions are in
[baselines.md](docs/baselines.md). Numbers are per task and per
controller; they are not a cross-task ranking. The rocket v1 figures are
instantaneous rates and are not comparable to the v2 settled criterion.

## Repository layout

```text
environments/               # everything that defines a task
├── multirotor/             # QuadX-Waypoints + QuadX-Track
├── fixedwing/              # Fixedwing-Waypoints
├── vtol/                   # Quadplane-VTOL
├── rocket/                 # Rocket-Landing
├── missile/                # Missile-Intercept
├── satellite/              # attitude_control, orbit_transfer, rendezvous_docking
└── common/                 # plant, dynamics, aero, contacts, disturbances
algorithms/                 # everything that produces actions
├── train_utils.py          # shared run_training() used by every entry
├── ppo/                    # rsl-rl PPO config builder
├── warmstart/              # behavior-cloning warm start from baselines
├── residual/               # residual-RL action wrapper
└── baselines/              # classical GNC controllers, one per task
experiments/                # controller-prior studies (BC, residual, gain scheduling)
scripts/                    # generic train / eval / baseline / sweep / summarize tools
docs/                       # method-level write-ups, protocol
showcase/                   # replay clips used by the GitHub Pages landing page
tests/                      # unit, env-contract and benchmark tests
```

Per-task deep-dive harnesses live next to their environments (e.g.
`environments/missile/eval_missile_scenarios.py`,
`environments/rocket/rocket_pdg_test.py`,
`environments/satellite/rendezvous_docking/eval_docking_lqr.py`).

### Training entries

Training entries are self-contained: every parameter is hard-coded in the file,
no CLI arguments. Edit the constants and run:

```bash
python environments/multirotor/train_quadx.py
python environments/multirotor/train_quadx_track.py
python environments/fixedwing/train_fixedwing.py
python environments/vtol/train_quadplane.py
python environments/rocket/train_rocket.py
python environments/missile/train_missile.py
python environments/satellite/attitude_control/train_satellite.py
python environments/satellite/orbit_transfer/train_transfer.py
python environments/satellite/rendezvous_docking/train_docking.py
python environments/satellite/rendezvous_docking/train_rendezvous.py
```

Evaluate a checkpoint, or the classical baseline:

```bash
python scripts/eval.py quadx                # restores results/<task>/config.json
python scripts/baseline_eval.py quadx       # any registered task
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/aggregate_5seed.py           # multi-seed protocol summary
```

The legacy CLI aggregator still works for quick sweeps
(`python scripts/train.py quadx`; registers quadx, fixedwing, rocket, missile,
satellite, transfer, docking, rendezvous — `quadx_track` and `quadplane` use
their per-task entries above).

### Run artifacts

Each run writes `results/<task>/`: `run.json` (config, real transition counts,
physical-step counts, device-side completion metrics), `config.json`,
`checkpoint.pt`, plus `eval.json` when evaluated. Baseline artifacts are
`baseline.json` (default controller) and `baselines/<controller>_seed7.json`
(within-task ablations).

## Documentation

| Document | Contents |
|---|---|
| [tasks.md](docs/tasks.md) | task table, dynamics, observation and action layouts |
| [baselines.md](docs/baselines.md) | classical controller results (incl. multi-seed data and reproduction commands) |
| [method/README.md](docs/method/README.md) | per-task and per-algorithm method docs (notation, equations, constants) |
| [environment-contract.md](docs/environment-contract.md) | tensor contract, installation, wind disturbance |
| [evaluation_protocol.md](docs/evaluation_protocol.md) | evaluation protocol and reporting rules |
| [benchmark_report.md](benchmark_report.md) | benchmark technical report + infrastructure/reproducibility appendix + references |

## Acknowledgments

> ### Special thanks · [Baidu Famou](https://cloud.baidu.com/product/famou.html)
> With special thanks to **Baidu Famou**. [Visit the official website ↗](https://cloud.baidu.com/product/famou.html)

We thank the following teachers and fellow students for their support and help.

**Teachers:** Lin Cheng · Shengping Gong · Peng Shi

**Fellow students:** Wang Han · Shi Yunhan · Gong Yuxuan · Dai Yuhang · Qi Keyu

**Related projects:** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ·
[PyTorch](https://pytorch.org) ·
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) ·
[PyFlyt](https://github.com/jbpalmer/PyFlyt) ·
[TorchRL](https://github.com/pytorch/rl) ·
[gym-satellite-trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) ·
SpaceGym (Supreeth & Mehta, CVPR 2026 Workshop on AI4Space)

Task definitions and defaults follow the published work listed in the
[References](benchmark_report.md#8-references) section of the benchmark
report. Upstream licenses and citation requirements apply to the
corresponding models and benchmarks.
