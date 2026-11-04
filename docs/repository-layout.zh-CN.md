# 仓库结构

**简体中文** · [English](repository-layout.md)

```text
environments/               # 定义任务所需的一切
├── multirotor/             # QuadX-Waypoints + QuadX-Track
├── fixedwing/              # Fixedwing-Waypoints
├── vtol/                   # Quadplane-VTOL
├── rocket/                 # Rocket-Landing
├── missile/                # Missile-Intercept
├── satellite/              # attitude_control、orbit_transfer、rendezvous_docking
└── common/                 # plant、动力学、气动、接触、扰动
algorithms/                 # 产生动作的一切
├── train_utils.py          # 所有入口共享的 run_training()
├── ppo/                    # rsl-rl PPO 配置构建器
├── warmstart/              # 从基线做行为克隆热启动
├── residual/               # 残差 RL 动作包装器
└── baselines/              # 经典 GNC 控制器，每任务一个
experiments/                # 控制器先验研究（BC、残差、增益调度）
scripts/                    # 通用 train / eval / baseline / sweep / summarize 工具
docs/                       # 方法文档、评测协议、技术亮点
showcase/                   # GitHub Pages 首页使用的回放片段
tests/                      # 单元、环境契约与基准测试
```

各任务的深入调试 harness 与环境放在一起（例如
`environments/missile/eval_missile_scenarios.py`、
`environments/rocket/rocket_pdg_test.py`、
`environments/satellite/rendezvous_docking/eval_docking_lqr.py`）。

## 训练入口

训练入口是自包含的：所有参数都硬编码在文件中，没有 CLI 参数。修改常量后直接运行：

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

评测一个 checkpoint，或经典基线：

```bash
python scripts/eval.py quadx                # 恢复 results/<task>/config.json
python scripts/eval.py quadplane
python scripts/baseline_eval.py quadx       # 任意已注册任务
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/aggregate_5seed.py           # 多种子协议汇总
```

旧的 CLI 聚合器仍可用于快速扫描
（`python scripts/train.py quadx`；它注册 quadx、fixedwing、rocket、missile、
satellite、transfer、docking、rendezvous —— `quadx_track` 与 `quadplane`
使用上面的各自入口）。

## 运行产物

每次运行写入 `results/<task>/`：`run.json`（配置、真实转场数、物理步调用数、
设备端完成指标）、`config.json`、`checkpoint.pt`，评测后还有 `eval.json`。
基线产物为 `baseline.json`（默认控制器）与
`baselines/<controller>_seed7.json`（任务内消融）。
