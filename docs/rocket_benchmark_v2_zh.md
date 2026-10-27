# Rocket benchmark v2：替换原因与协议

旧 benchmark 的 Genesis 着陆接触配置存在结构性问题：三根平底碰撞条与平台接触后持续抖动，
关机轻放也会被弹起；瞬时 ever-success 又会把抖动过程中短暂满足阈值的状态记为成功。
这让评分偏离了“火箭着陆并停稳”的任务目标。v2 用球形脚垫修复接触，并以连续停稳 2 秒计分。

## Claude 已完成的验证

同一批 128 个随机轻放初态，发动机关机、舵面中立、默认求解器，运行 10 秒：

| 接触诊断 | 原三根碰撞条 | v2 六球形脚垫 |
|---|---:|---:|
| 接触后抬离超过 1 cm | 108/128 | 0/128 |
| 致命结束 | 88/128 | 0/128 |
| 曾满足瞬时成功条件 | 85/128 | 128/128 |

Claude 的完整首回合评估，每列 4×128，共 512 回合，主指标为连续停稳 2 秒：

| 条件与策略 | 开发集 | 独立集 |
|---|---:|---:|
| nominal / seed | 508/512 | 504/512 |
| nominal / PDG | 512/512 | 512/512 |
| shifted / seed | 281/512 | 293/512 |
| shifted / PDG | 290/512 | 314/512 |

名义条件下恢复正常停稳，加入推力、翼面、执行器和风扰动后仍能区分控制能力，
因此 v2 同时提供了可靠的接触基础和有挑战性的物理泛化评估。
逐环境接触记录、评估计数、区间及源文件 SHA-256 见
[`evidence/rocket_v2/`](evidence/rocket_v2/)。本次提交直接复用 Claude 的这些实测数据。

## 接入 main 的对应位置

- **接触资产**：`environments/common/original_pyflyt/models/vehicles/rocket/rocket_v2.urdf`。
  每根腿使用两个半径 0.025 m 的球形碰撞脚垫，局部 y 坐标为 ±0.225 m。
  视觉、质量、惯量、最低点高度和横向支撑范围保留；碰撞形状明确升级为 v2。
  原 `rocket.urdf` 保留，两个资产均有 SHA-256 清单。
- **物理模型**：`environments/common/plant.py` 与 `rocket_benchmark.py`。
  接入每回合物理扰动，采用独立计数器随机流，种子偏移 104729。
- **任务计分**：`environments/rocket/env.py` 使用 v2，主指标为 `ever_stable_2s`。
  保留现有奖励、致命判据、初始分布、30 维观测、7 维动作和 30 秒时域。
- **评估**：`scripts/rocket_benchmark_eval.py` 按 4×128 完整首回合统计。
  `eval.py` 与 `baseline_eval.py` 同步修正累计回合总数退出的问题：每个副本只采首回合，
  提前失败副本的后续回合不会替代尚未结束的首回合。
- **训练配置**：`algorithms/train_utils.py` 在开始训练前保存扰动档位和策略频率。
  续训锁定这两项配置，跨配置实验使用独立目录。

## 连续停稳指标

每个策略步末检查平台接触、速度范数 <0.02 m/s、角速度范数 <0.02 rad/s、
roll/pitch 合成倾角 <0.1 rad，且没有致命或越界结束。连续满足条件的样本首末跨度达到 2 秒：
**60 Hz 为 121 点，40 Hz 为 81 点**。接触间断或任一条件超限会重置连续计数。
成功后继续仿真至自然结束或时限。

同步记录旧瞬时 `ever_success`、末端单点 `terminal_single_point`、
末端连续停稳 `terminal_stable_2s`、成功后致命/越界 `success_then_failure`，
以及连续停稳后致命/越界 `stable_then_failure`。

## 每回合物理扰动

| 参数 | nominal | shifted | stress |
|---|---|---|---|
| 推力比例 | 1 | [0.92, 1.08] | [0.85, 1.15] |
| 翼面力/矩比例 | 1 | [0.6, 1.4] | [0.3, 1.7] |
| 发动机、万向节时间常数（秒，各自采样） | 0.01 | [0.005, 0.03] | [0.005, 0.06] |
| 两轴万向节偏置（度，各自采样） | 0 | [-0.3, 0.3] | [-1, 1] |
| 水平定常风速（m/s，方向均匀） | 0 | [0, 5] | [0, 10] |

推力比例表示效能变化，燃油消耗公式保留；翼面比例同时作用于力和矩。
这些参数属于模拟器内部状态，策略仍使用原 30 维观测。
main 已有 `steady_wind`、`dryden` 接口保留，额外配置的风与回合风叠加并写入实验配置。
掩码重置只更新对应副本。构造和首次 reset 各采样一次，与研究版本一致。

## 版本与使用

Claude 上述数据采用冻结研究源 `39e6bb6fd5d0f5432f7405a76946938a469492ce` 的
40 Hz v2 快照。main 接入保留当前 40 Hz 策略（2026-10-27 起，与研究快照同款频率；
50 Hz 因不能整除 120 Hz 控制环不可用）、120 Hz 控制、240 Hz 物理频率和 v4 奖励。
报告通过版本、频率、源代码哈希、后端和风配置区分各实验；历史榜单保留原版本标签。

```bash
python scripts/rocket_benchmark_eval.py --backend gpu --variation nominal --out results/rocket-v2-nominal
python scripts/rocket_benchmark_eval.py --backend gpu --variation shifted --out results/rocket-v2-shifted
```

默认开发种子为 1000007、2000007、3000007、4000007，独立确认使用预先固定的另一组
`--seeds`。报告包含逐回合记录、完整性、样本数、Wilson 95% 区间、初态和源代码哈希。
训练检查点评估使用 `scripts/eval.py`；`rocket_pdg_test.py` 保留为触地遥测入口。
