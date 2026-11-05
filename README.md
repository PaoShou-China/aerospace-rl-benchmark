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
reward, reset and RNG as masked tensor kernels. The only per-step host sync is
`bool(done.any())`.

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
[repository-layout.md](docs/repository-layout.md).

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
retain the upstream update implementation. Standalone inference uses
`algorithms.inference.InferencePolicy(actor.eval())`: recreate the snapshot
after changing source weights, call each instance serially, and match the
input device and dtype to the snapshot.

### Execution layers

With `performance=True` the environment step runs as a short sequence of CUDA
graph replays instead of thousands of individually launched kernels
(`algorithms/performance.py`):

1. **Genesis performance mode.** Genesis compiles static-array kernels
   specialised to the scene. For these single-body (6-DOF) vehicles it then
   solves contacts with its one-kernel *monolith* Newton arm. The generic
   kernels instead pin the *decomposed* arm, whose iteration loop runs on the
   host with a synchronization per iteration on GPUs below SM 9.0 (RTX 30/40
   series). Both arms solve the same problem to the same tolerance, so
   trajectories agree to float32 rounding, not bitwise.
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

Layers 2 and 3 replay exactly the kernels eager execution launches: rollouts
with and without them are bitwise identical on all ten tasks, and the
refactored eager step is bitwise identical to the previous one. For the four
spacecraft tasks PPO losses are identical to the previous release update for
update.

| Variable | Default | Effect |
| --- | --- | --- |
| `AEROSPACE_GENESIS_PERFORMANCE` | `1` | `0` keeps Genesis' generic kernels (bitwise as before; layers 2-3 then fall back to Torch-only segments) |
| `AEROSPACE_NATIVE_CAPTURE` | `1` | `0` keeps Genesis calls outside CUDA graphs |
| `AEROSPACE_SEGMENT_GRAPHS` | `1` | `0` disables graphed step segments |
| `AEROSPACE_FUSE` | `0` | `1` enables fused task math (layer 4) |
| `AEROSPACE_SEGMENT_DEBUG` | `0` | `1` reports why a segment re-warms |

### Measured speed-up

PPO update throughput relative to the previous acceleration release
([`d7ead04`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/d7ead044bdc22ddb11b1adefa4569bdd3b7c8b5a)),
measured with its benchmark protocol: 1024 parallel environments, seed 7,
each task's shipped PPO configuration, BC excluded, median of five updates
after two warmup updates (rollout, logging and PPO update included).

| Task | `d7ead04` s/update | This branch s/update | Speed-up | `AEROSPACE_FUSE=1` s/update | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuadX-Waypoints | 2.415 | 0.579 | **4.17x** | 0.598 | 4.04x |
| QuadX-Track | 2.781 | 0.724 | **3.84x** | 0.786 | 3.54x |
| Fixedwing-Waypoints | 1.977 | 0.545 | **3.63x** | 0.632 | 3.13x |
| Quadplane-VTOL | 1.461 | 0.520 | **2.81x** | 0.443 | 3.30x |
| Rocket-Landing | 2.533 | 1.029 | **2.46x** | 0.862 | 2.94x |
| Missile-Intercept | 1.350 | 0.345 | **3.91x** | 0.328 | 4.12x |
| Satellite attitude (CMG) | 0.763 | 0.599 | **1.27x** | 0.337 | 2.26x |
| Low-thrust transfer | 0.150 | 0.132 | **1.14x** | 0.0810 | 1.85x |
| Docking | 0.0808 | 0.0365 | **2.21x** | 0.0492 | 1.64x |
| Rendezvous | 0.818 | 0.276 | **2.96x** | 0.299 | 2.74x |

Both arms ran interleaved on the same machine (two rounds each, best round
reported, which damps contention from other tenants of the cloud host). The
fused column is a single run of the same protocol and carries about +/-15%
run-to-run noise; its satellite and orbit-transfer gains are well outside that
noise, the others are not. Fusion also adds a one-time compilation of one to
three minutes.

The previous release measured 1.88-15.29x over
[`4439708`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/4439708d3bfaf97c585be84727fd7df48eedefaa)
with the same protocol; the factors above compound with it.

Quadx from-scratch PPO (no BC warm start) can drive the Genesis contact
solver to non-finite constraint forces, which Genesis reports as
`GenesisException: Invalid constraint forces causing 'nan'`. This occurs with
both solver arms (2 of 3 seeds in performance mode and 1 of 3 with the generic
kernels within 30 updates) and predates these changes. Time to convergence
and success-rate equivalence over complete training runs remain unvalidated.

Measurements used an NVIDIA GeForce RTX 4080 SUPER (driver 595.71), Python
3.12.3, PyTorch 2.8.0+cu128, Genesis 1.4.4, Quadrants 1.3.3 and rsl-rl 5.3.0.
The PyTorch 2.11 pin in `requirements.txt` has not been validated for these
acceleration paths.

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
| Satellite attitude (CMG) | 26 or 32 | 4 or 6 | 40 Hz | attitude < 0.05 rad, \|ω\| < 0.02 rad/s |
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

## Documentation

| Document | Contents |
|---|---|
| [tasks.md](docs/tasks.md) | task table, dynamics, observation and action layouts |
| [baselines.md](docs/baselines.md) | classical controller results and comparability notes |
| [method/README.md](docs/method/README.md) | per-task and per-algorithm method docs (notation, equations, constants) |
| [environment-contract.md](docs/environment-contract.md) | tensor contract, installation, wind disturbance |
| [evaluation_protocol.md](docs/evaluation_protocol.md) | evaluation protocol and reporting rules |
| [baseline_test_report.md](docs/baseline_test_report.md) | multi-seed baseline reanalysis |
| [technical-highlights.md](docs/technical-highlights.md) | reviewer-oriented infrastructure claims |
| [repository-layout.md](docs/repository-layout.md) | repository tree, training entries, run artifacts |
| [roadmap.md](docs/roadmap.md) | planned tasks and landed infrastructure |
| [references.md](docs/references.md) | publications and upstream projects |
| [benchmark_report.md](benchmark_report.md) | Benchmark technical report |
| [rocket_benchmark_v2.md](docs/rocket_benchmark_v2.md) | Rocket v2 criterion and reproduction notes |

## Acknowledgments

> ### Special thanks · [Baidu Famou](https://cloud.baidu.com/product/famou.html)
> With special thanks to **Baidu Famou**. [Visit the official website ↗](https://cloud.baidu.com/product/famou.html)

We thank the following teachers and fellow students for their support and help.

**Teachers:** Lin Cheng · Shengping Gong · Peng Shi

**Fellow students:** Wang Han · Shi Yunhan · Gong Yuxuan · Dai Yuhang

**Related projects:** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ·
[PyTorch](https://pytorch.org) ·
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) ·
[PyFlyt](https://github.com/jbpalmer/PyFlyt) ·
[TorchRL](https://github.com/pytorch/rl) ·
[gym-satellite-trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) ·
SpaceGym (Supreeth & Mehta, CVPR 2026 Workshop on AI4Space)

Task definitions and defaults follow the published work listed in
[references.md](docs/references.md). Upstream licenses and citation
requirements apply to the corresponding models and benchmarks.
