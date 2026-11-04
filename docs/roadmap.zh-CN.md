# 路线图

**简体中文** · [English](roadmap.md)

每个未来任务都遵循同一套配方：张量化环境 + 规格 docstring →
在 `algorithms/baselines/` 注册经典基线 → 自包含的 `train_<task>.py` →
`results/<task>/` → 更新 README。所有任务均为单智能体。

## 优先级 1

| 任务 | 环境 | 基线 | 关注点 |
|---|---|---|---|
| Droneship Landing | Rocket-Landing + 6-DOF 移动平台（海浪谱） | PDG + 滞后补偿重定向 | 移动触地目标 |
| Docking-Fault | Docking + 卡死/失效推力器 + 测量噪声 | LQR + 控制重分配 | 故障注入、安全 RL |
| Autoland | 下滑道 + 侧风 + 拉平，跑道接触判据 | 下滑道制导 + TECS | 自驾仪已存在 |
| Slung Payload | QuadX + 缆绳悬挂摆负载 | SE(3) + 摆动阻尼 | 欠驱动耦合 |

## 优先级 2

| 任务 | 环境 | 基线 | 关注点 |
|---|---|---|---|
| Terrain Following | DEM 采样、雷达高度表观测 | TF 横向制导 | 局部观测 |
| Inspection | CWH + 视场约束、覆盖率奖励 | ZEM/PN 绕飞 | 长时域 |
| RTB Boost-back | 上升 + 180° 翻转 + 着陆，约 90 s | 重力转弯 + 两段 PDG | 长时域 shaping |
| HGV Glide | 高超声速滑翔，热流/载荷因子走廊 | BTT 走廊制导 | 约束满足 |
| Racing Gate | 门序列、时间惩罚 + 速度上限 | 最小时间跟踪 | 激进飞行 |

## 优先级 3

| 任务 | 环境 | 基线 | 关注点 |
|---|---|---|---|
| GEO Station-keeping | 受扰二体（J2 + 日月），±0.05° 死区 | 脉冲 Euler–Hill 修正 | 极低控制预算 |
| Ascent to Orbit | 重力转弯，动压/热流限制 | 线性切线制导 | 完成发射 → 入轨 → 交会 |
| Detumble & Capture | 翻滚非合作目标，接触动力学 | 角动量匹配 | 接触丰富的操作 |
| Precision Landing | 移动小车平台 + 模拟视觉延迟 | MPC 式跟踪 | 延迟鲁棒观测 |

已落地基础设施：`environments/common/disturbances.py` 中统一的 `WindDisturbance`，
以及基线注册表（`@register_baseline('<task>')` —— 一行即可把新控制器接入
`scripts/baseline_eval.py`）。
