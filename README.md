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
