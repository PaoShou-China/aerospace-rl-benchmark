# 环境契约与风扰

**简体中文** · [English](environment-contract.md)

## 契约

- `obs_buf` / `rew_buf` 为 `float32`，内部物理状态为 `float64`。
- `extras['time_outs']` 标记纯时间上限截断（供 rsl-rl 做 time-limit bootstrap）；
  `extras['terminal_observations']` 携带重置前的观测。
- `completed_episodes` / `completed_successes` / `completed_returns` 在设备上累计；
  `complete` 是逐副本的成功标志。
- 动作到达 `env.device`；隐式 host/device 拷贝会直接报错。
- 物理、接触、奖励、重置与随机数都是带掩码的张量核，未激活副本被冻结且不消耗随机数。
  每步唯一的 host 同步是 `bool(done.any())`，仅用于回合边界记账。

设计主张及其验证条件见 [technical-highlights.zh-CN.md](technical-highlights.zh-CN.md)。

## 风扰

五个大气任务（QuadX-Waypoints、Fixedwing-Waypoints、Quadplane-VTOL、
Rocket-Landing、Missile-Intercept）接受可选风场：

- `steady_wind`（tuple，m/s）—— 世界系定常风速；
- `dryden`（0–1 的强度预设比例）+ `severity`
  （`light | moderate | severe`）—— 来自
  [environments/common/disturbances.py](../environments/common/disturbances.py)
  的 Dryden 阵风，每个控制步推进一次。

气动项始终使用空速 `v_air = v - wind`，因此静风
（`steady_wind=None, dryden=0`）可逐位复现基准。`scripts/eval.py` 会恢复每次运行
记录的 `env_kwargs`，因此带风训练的 checkpoint 会在同一风场中评测。

| 任务 | 训练入口风配置 |
|---|---|
| quadx、fixedwing、quadplane | 纯 Dryden 阵风，`dryden=0.3, severity="moderate"` |
| rocket | `dryden=0.0`（与静风基线同口径） |
| missile、quadx_track | 静风 |
| satellite、transfer、docking、rendezvous | 真空 |

## 安装

```bash
python -m pip install -r requirements.txt
```

精确版本锁定在 [requirements.txt](../requirements.txt)；参考环境为 Linux 上的
Python 3.10.20、PyTorch 2.11.0（CUDA 12.8）、`genesis-world` 1.4.4、
`rsl-rl-lib` 5.3.0、NumPy 2.2.6 与 SciPy 1.15.3（RTX 3090，驱动 580.173.02）。

六个 Genesis 任务在 GPU 后端需要 CUDA 版 PyTorch；四个张量动力学任务以及所有
经典基线也可以在纯 CPU 的 PyTorch 上运行。本套件不依赖 `gymnasium` / `pettingzoo`。
