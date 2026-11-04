<p align="center">
  <a href="https://paoshou-china.github.io/aerospace-rl-benchmark/">
    <img src="docs/assets/aerospacerl-cover.webp" alt="AerospaceRL Benchmark — aerospace control project cover" width="100%">
  </a>
  <br>
  <sub>Brand illustration, not a simulation capture.</sub>
</p>

<p align="center">
  <a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><b>paoshou-china.github.io/aerospace-rl-benchmark</b></a>
  <br>
  <a href="README.zh-CN.md">Chinese</a> ·
  <a href="docs/technical-highlights.md">Technical highlights</a> ·
  <a href="docs/evaluation_protocol.md">Evaluation protocol</a> ·
  <a href="benchmark_report.md">Benchmark report (Chinese)</a>
</p>

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
python -m pip install -r requirements.txt

python environments/multirotor/train_quadx.py   # train (parameters live in the file)
python scripts/eval.py quadx                     # evaluate a checkpoint
python scripts/baseline_eval.py quadx            # evaluate the classical baseline
```

A CUDA build of PyTorch is required for the six Genesis tasks on the GPU
backend; the four tensor-dynamics tasks and every classical baseline also run
on CPU-only PyTorch. Exact versions, the environment contract and the optional
Dryden wind interface are in
[docs/environment-contract.md](docs/environment-contract.md); all training
entries, run artifacts and the repository tree are in
[docs/repository-layout.md](docs/repository-layout.md).

## Task overview

| Task | Group | Obs / Act | Policy rate | Success |
|---|---|---|---|---|
| QuadX-Waypoints | Atmospheric | 37 / 4 | 40 Hz | 4/4 waypoints < 0.2 m |
| QuadX-Track | Atmospheric | 25 / 4 | 40 Hz | visible fraction ≥ 0.85, no crash |
| Fixedwing-Waypoints | Atmospheric | 39 / 4 | 30 Hz | 4/4 waypoints < 10 m |
| Quadplane-VTOL | Atmospheric | 47 / 8 | 40 Hz | 4/4 waypoints < 15 m |
| Rocket-Landing | Atmospheric | 30 / 7 | 40 Hz | settled landing, \|v\| < 0.02 m/s, tilt < 0.1 rad |
| Missile-Intercept | Atmospheric | 34 / 3 | 60 Hz | miss distance < 15 m |
| Satellite attitude (CMG) | Spacecraft | 26 / 32 · 4 / 6 | 40 Hz | attitude < 0.05 rad, \|ω\| < 0.02 rad/s |
| Low-thrust transfer | Spacecraft | 8 / 3 | 300 s/step | terminal orbital-element return |
| Docking | Spacecraft | 6 / 3 | 1 s/step | range < 1 m, speed < 0.1 m/s |
| Rendezvous | Spacecraft | 6 / 3 | 1 s/step | range < 10 m, speed < 0.5 m/s, keep-out |

Full dynamics, episode lengths, observation layouts and reward decompositions:
[docs/tasks.md](docs/tasks.md) and
[docs/method/README.md](docs/method/README.md).

Baseline success rates and their evaluation conditions are in
[docs/baselines.md](docs/baselines.md). Numbers are per task and per
controller; they are not a cross-task ranking. The rocket v1 figures are
instantaneous rates and are not comparable to the v2 settled criterion.

## Documentation

| Document | Contents |
|---|---|
| [docs/tasks.md](docs/tasks.md) | task table, dynamics, observation and action layouts |
| [docs/baselines.md](docs/baselines.md) | classical controller results and comparability notes |
| [docs/method/README.md](docs/method/README.md) | per-task and per-algorithm method docs (notation, equations, constants) |
| [docs/environment-contract.md](docs/environment-contract.md) | tensor contract, installation, wind disturbance |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | evaluation protocol and reporting rules |
| [docs/baseline_test_report.md](docs/baseline_test_report.md) | multi-seed baseline reanalysis |
| [docs/technical-highlights.md](docs/technical-highlights.md) | reviewer-oriented infrastructure claims |
| [docs/repository-layout.md](docs/repository-layout.md) | repository tree, training entries, run artifacts |
| [docs/roadmap.md](docs/roadmap.md) | planned tasks and landed infrastructure |
| [docs/references.md](docs/references.md) | publications and upstream projects |
| [benchmark_report.md](benchmark_report.md) | Benchmark technical report |
| [docs/rocket_benchmark_v2.md](docs/rocket_benchmark_v2.md) | Rocket v2 criterion and reproduction notes |

## Acknowledgments

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
[docs/references.md](docs/references.md). Upstream licenses and citation
requirements apply to the corresponding models and benchmarks.
