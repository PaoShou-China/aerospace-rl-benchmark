# Environment Contract and Wind Disturbance

**English** · [简体中文](environment-contract.zh-CN.md)

## Contract

- `obs_buf` / `rew_buf` are `float32`, internal physics state is `float64`.
- `extras['time_outs']` marks pure time-limit truncations (for rsl-rl
  time-limit bootstrapping); `extras['terminal_observations']` carries the
  pre-reset observation.
- `completed_episodes` / `completed_successes` / `completed_returns`
  accumulate on device; `complete` is the per-replica success flag.
- Actions arrive on `env.device`; implicit host/device copies raise.
- Physics, contacts, reward, reset and RNG are masked tensor kernels, so
  inactive replicas are frozen and draw no random numbers. The only per-step
  host sync is `bool(done.any())`, which gates episode-boundary bookkeeping.

Design claims and their verification conditions are written up in
[benchmark_report.md §Appendix A](../benchmark_report.md#appendix-a-infrastructure-and-reproducibility).

## Wind disturbance

Five atmospheric tasks (QuadX-Waypoints, Fixedwing-Waypoints, Quadplane-VTOL,
Rocket-Landing, Missile-Intercept) accept optional wind:

- `steady_wind` (tuple, m/s) — constant world-frame wind velocity;
- `dryden` (0–1 scale of the severity presets) + `severity`
  (`light | moderate | severe`) — Dryden gusts from
  [environments/common/disturbances.py](../environments/common/disturbances.py),
  advanced once per control step.

Aerodynamic terms always evaluate air-relative velocity `v_air = v - wind`,
so calm air (`steady_wind=None, dryden=0`) reproduces the benchmark
bit-for-bit. `scripts/eval.py` restores each run's recorded `env_kwargs`, so a
checkpoint trained in wind is evaluated in the same wind.

| Task | Training-entry wind |
|---|---|
| quadx, fixedwing, quadplane | pure Dryden gusts, `dryden=0.3, severity="moderate"` |
| rocket | `dryden=0.0` (same caliber as the calm-air baseline) |
| missile, quadx_track | calm air |
| satellite, transfer, docking, rendezvous | vacuum |

## Installation

```bash
python -m pip install -r requirements.txt
```

Exact versions are pinned in [requirements.txt](../requirements.txt); the
reference environment is Python 3.10.20 with PyTorch 2.11.0 (CUDA 12.8),
`genesis-world` 1.4.4, `rsl-rl-lib` 5.3.0, NumPy 2.2.6 and SciPy 1.15.3 on
Linux (RTX 3090, driver 580.173.02).

A CUDA build of PyTorch is required for the six Genesis tasks on the GPU
backend; the four tensor-dynamics tasks and every classical baseline also run
on CPU-only PyTorch. `gymnasium` / `pettingzoo` are not dependencies of this
suite.
