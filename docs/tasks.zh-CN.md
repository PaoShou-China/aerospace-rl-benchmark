# 任务总览

**简体中文** · [English](tasks.md)

本基准包含十个单智能体航空航天控制任务：六个大气层飞行器运行在 Genesis 刚体后端，
四个航天器任务使用显式张量动力学。所有任务都暴露同一套 `step` / `reset` / 缓冲区契约。

| 任务 | 类 | 动力学 | 观测 | 动作 | 策略频率 | 回合 | 成功判据 |
|---|---|---|---|---|---|---|---|
| QuadX-Waypoints | `QuadXWaypointsEnv` | URDF Crazyflie 2.x（240 Hz）+ 角速率 PID | 37 | 4 | 40 Hz | 800 步 · 20 s | 4/4 航点 < 0.2 m |
| QuadX-Track | `QuadXTrackEnv` | URDF Crazyflie 2.x（240 Hz）+ 角速率 PID，运动学地面目标，几何相机 | 25 | 4 | 40 Hz | 1200 步 · 30 s | 可见时间占比 ≥ 0.85，无碰撞 |
| Fixedwing-Waypoints | `FixedwingWaypointsEnv` | URDF 2.35 kg 固定翼（240 Hz）+ 叶素气动 | 39 | 4 | 30 Hz | 3600 步 · 120 s | 4/4 航点 < 10 m |
| Quadplane-VTOL | `QuadplaneWaypointsEnv` | URDF 混合构型（240 Hz）：4 个旋翼电机、固定翼舵面、前向推进器 | 47 | 8 | 40 Hz | 1200 步 · 30 s | 4/4 航点 < 15 m |
| Rocket-Landing | `RocketLandingEnv` | URDF F9 级助推器（240 Hz）+ 栅格舵、万向节 | 30 | 7 | 40 Hz | 1200 步 · 30 s | 平台接触，\|v\| < 0.02 m/s，\|ω\| < 0.02 rad/s，倾角 < 0.1 rad 并持续 2 s（81 个采样） |
| Missile-Intercept | `MissileInterceptEnv` | URDF 150 kg 导弹（240 Hz）+ 尾舵气动、助推-滑翔发动机、运动学 3-DOF 目标 | 34 | 3 | 60 Hz | 900 步 · 15 s | 脱靶距离 < 15 m（近炸引信，任意物理子步） |
| Satellite attitude (CMG) | `SatelliteAttitudeEnv` | 刚体 + 4-CMG 群，RK4 2.5 ms | 26 | 4 | 40 Hz | 1200 步 · 30 s | 姿态误差 < 0.05 rad 且 \|ω\| < 0.02 rad/s；\|ω\| > 10 rad/s 判失败 |
| Low-thrust transfer | `LowThrustTransferEnv` | 二体引力 + 三轴 10 mN 电推进，RK4 10 s | 8 | 3 | 300 s/步 | 166 步 ≈ 5 圈 | 极限处终端轨道根数返回 R > 0 |
| Docking | `SpaceDockingEnv` | Clohessy–Wiltshire–Hill，辛 Euler 1 s | 6 | 3 | 1 s/步 | 500 步 ≈ 8.3 min | 距离 < 1 m，速度 < 0.1 m/s |
| Rendezvous | `SpaceRendezvousEnv` | Clohessy–Wiltshire–Hill，辛 Euler 1 s | 6 | 3 | 1 s/步 | 1000 步 ≈ 16.7 min | 距离 < 10 m，速度 < 0.5 m/s，遵守 keep-out |

六个 Genesis 刚体任务（`quadx`、`quadx_track`、`fixedwing`、`quadplane`、`rocket`、`missile`）
使用完全张量化的作动与接触层；导弹的机动目标、QuadX-Track 的地面目标与相机是解析几何，
每个物理子步推进一次。四个太空任务（`satellite`、`transfer`、`docking`、`rendezvous`）
是纯张量积分器，不依赖任何引擎。

完整的观测/动作布局、奖励分解、终止条件与初始分布位于各包的 docstring：

[multirotor](../environments/multirotor/__init__.py) ·
[fixedwing](../environments/fixedwing/__init__.py) ·
[vtol](../environments/vtol/__init__.py) ·
[rocket](../environments/rocket/__init__.py) ·
[missile](../environments/missile/__init__.py) ·
[attitude_control](../environments/satellite/attitude_control/__init__.py) ·
[orbit_transfer](../environments/satellite/orbit_transfer/__init__.py) ·
[rendezvous_docking](../environments/satellite/rendezvous_docking/__init__.py)。

论文式方法文档索引见 [method/README.md](method/README.md)
（中文索引：[method/README.zh-CN.md](method/README.zh-CN.md)）。
