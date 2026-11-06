# 冻结评测协议（2026-10-22）

**简体中文** · [English](evaluation_protocol.md)

正式实验前冻结的判定、预算与报告口径。冻结后只允许以追加条目的方式修订，且必须在正文注明修订日期与理由；不允许回溯修改已发布结果对应的条目。

## 1. 成功定义与终止优先级（已冻结进代码）

- 终止优先级：**违约 > 成功**。回合终步若同时满足目标状态与任一失败条件，记为失败（terminated），不计入 `completed_successes`，不发放终端奖励。
- 该优先级在 `satellite/rendezvous_docking/env.py` 的 `_CWHBase.step` 实现（`success = goal & ~violation`）。修复动机：rendezvous 的 keep-out 球（5 m）位于目标球（10 m）内部，旧行为允许"低速冲进 keep-out"被同时记为 success + terminated，成功率不可解释。
- docking 无 keep-out，语义变化仅可能出现在"燃料恰好在捕获步耗尽"的边角（基线最大燃料 0.912 kg < 2 kg 预算，不受影响）。
- 语义修正后复测：rendezvous 官方基线 1024/1024 = 100% 不变（hold 点 9 m 在 keep-out 外）。**语义修正前的 RL 训练成功率（rendezvous 1.5% 等）与新口径不可直接比较，正式实验一律用新口径重跑。**

## 2. 传统控制器（BC 数据源）审计

BC warm-start 数据源 = 各任务 `conventional_controller`（BC 采集器自建控制器实例直接驱动，env 的 info 不含控制器动作）。各控制器均在训练/验证初始条件上调参并冻结后才开始采集 BC 数据，冻结状态如下：

| 任务 | 控制器 | train 场景成功率 | dev/test | 状态 | BC 决策 |
|---|---|---|---|---|---|
| quadx | SE(3) 几何（Lee et al. 2010） | 100% | 全场景 100% | 冻结 | 启用 |
| docking | CWH-LQR + shield | 100%（d=0/.5/1 × 3 seeds） | 64 场景网格 63/64 | 冻结 | 启用 |
| rendezvous | 两脉冲 CWH 制导 | 100%（4096 eps） | 150–1900 m 网格全 100% | 冻结 | 启用 |
| transfer | 单步贪心 Gauss 门控 | 100%（d=0/.5/1） | 最优性探针已做 | 冻结 | 启用 |
| satellite | 四元数 PD + CMG 阻尼伪逆 | 100%（1024 eps） | 随机复位 100%；奇异/180° 专项记录未随仓库发布 | 冻结 | 启用 |
| fixedwing | L1 + TECS | 74.6% | dev 74.4% / test 73.1% | 冻结（21.3%→74.9% 迭代后定稿；残余 = OOB 极限环的结构性局限） | 启用，但报告须注明专家天花板 ~75%，BC 继承该上限 |
| rocket | ZEM/PN + 脉冲伺服 PDG | 84.1%（v1 瞬时 ever-success，2000 步口径） | 700 步套件 A 69%/B 34% | 冻结（v3 定稿；B_corners 为饱和权限的结构性局限） | 启用，但报告须注明专家天花板（v1 84.1%，v2 `ever_stable_2s` 100%） |

结论：无"表现很差"的失控控制器；fixedwing / rocket 的天花板为控制器结构性局限（已记录成因），BC 只能继承、不会放大这些上限，报告时须与专家成功率并列呈现。

> 注：上表是 2026-10-22 冻结时点的审计快照，行内数字不是当前发布产物。当前可核验值见 [baselines.zh-CN.md](baselines.zh-CN.md)：`results/fixedwing/baseline.json` 为 97.0%（128 副本 / seed 13 / 1500 步，`complete=false`），`results/rocket/baseline.json` 为 100%（v2 `ever_stable_2s`）。表中已删除不存在的 `pendulum` 任务——本仓库只注册 10 个任务（见 `scripts/baseline_eval.py` 的 `ENVS`）。

satellite 转向律消融（不改变冻结基线）：同一四元数 PD 与增益下，把阻尼伪逆换成普通 Moore–Penrose 伪逆，注册名 `quat_pd_mp_pinv_cmg`，评测入口 `scripts/satellite_steering_compare.py`；配置与近奇异/精确奇异场景见 [baseline-satellite-mp-pinv.zh-CN.md](method/baseline-satellite-mp-pinv.zh-CN.md)。

## 3. 冻结评测协议

对每个任务（10 任务统一）：

- **训练预算**：冻结口径为 1024 并行 envs × 1000 PPO updates；达到预算即停，不做 early stopping，不依据训练中指标挑选 checkpoint（只评最终 checkpoint）。当前各 `train_<task>.py` 实际硬编码的预算如下（与冻结口径的偏差已登记在 §5 的 2026-11-02 条目）：

  | 任务 | updates | steps/env | transitions |
  |---|---:|---:|---:|
  | quadx | 1000 | 100 | 102.4 M |
  | quadx_track | 200 | 100 | 20.5 M |
  | fixedwing | 1000 | 64 | 65.5 M |
  | quadplane | 1000 | 64 | 65.5 M |
  | rocket | 2000 | 64 | 131.1 M |
  | missile | 1000 | 64 | 65.5 M |
  | satellite | 1000 | 150 | 153.6 M |
  | transfer | 1000 | 32 | 32.8 M |
  | docking | 1000 | 32 | 32.8 M |
  | rendezvous | 1500 | 250 | 384.0 M |

  所有入口均为 1024 并行副本、seed 7。
- **收敛阈值（绝对成功率，训练分布、训练口径）**：

  | 任务 | 收敛阈值 | 依据 |
  |---|---|---|
  | quadx | ≥ 60% | 当前最佳 84.9% |
  | docking | ≥ 60% | MLP 臂 77.9% |
  | transfer | ≥ 90% | 当前 98.9% |
  | rendezvous | ≥ 60% | 当前 82.2%（新成功口径下重测） |
  | fixedwing | ≥ 60% | 当前 76.2% |
  | satellite | ≥ 10% | 当前 17.4%（基线 100%，RL 明显未解，阈值仅作地板） |
  | rocket | ≥ 30% | 2026-10-22 快照：v1 基线 84.1%（瞬时 ever-success；当前 v2 `ever_stable_2s` 基线为 100%） |

  达到阈值记为**已收敛**；未达到记为**未收敛（not converged）**并原样报告，不得以"仍在上升"等叙事替代。阈值修订需在第 5 节登记。表中"当前"数字是 2026-10-22 冻结时点值，不是当前发布产物；另有不存在的 `pendulum` 行已删除（本仓库没有该任务）。
- **评测间隔**：训练过程每 200 updates 保存一次 `model_*.pt`（rsl-rl save_interval），仅在训练结束后对最终 checkpoint 做一次确定性评测（eval.py，冻结 config.json 里的 env_kwargs）；训练中不做验证集模型选择。
- **种子与区间**：正式实验每任务 **3 seeds {7, 13, 21}** 训练；报告 mean 与 [min, max] 区间，不报单 seed 结果。
- **Held-out 测试集**：训练/验证只用 seed 7/13/21 的 IC 流；最终泛化评测用 eval.py `--seed 101`（未在训练中出现的 RNG 流），每任务只跑一次，只报不改。基线（baseline_eval.py）同样补测 seed 101，保证同分布可比。
- **报告口径**：成功率必须注明①训练口径还是确定性评测口径②步数窗口（rocket 的 700 步与 2000 步口径差异巨大，报告默认 2000 步/官方 reset 分布）③是否带风扰。训练可以带风；`eval.py` 在 `results/<task>/config.json` 存在时会恢复该次训练的 `env_kwargs`，因此评测使用训练时的风配置，而不是恒为静风。跨方法比较必须显式固定同一风配置。

## 4. 成本与并行性能的报告口径（学习成本 ≠ 并行性能）

RL 主结果同时报告两类比较，不得混用：

- **等交互步数**：固定 transitions（`1024 envs × updates × steps/env`；冻结口径为 1024 × 1000 × 32 = 32.8 M，当前各入口的实际值见 §3 预算表），比最终成功率/return —— 衡量样本效率。
- **等墙钟时间**：报告 `train_wall_seconds` 与 `train_throughput_steps_per_s`（run.json 现已记录），在相同墙钟预算下比较可达成功率 —— 衡量工程吞吐。吞吐量、完整训练墙钟、显存峰值（`peak_vram_gib`）三项分别报告（summarize.py 已加 k steps/s 与 vram GiB 列）。

BC 另行报告独立成本，不摊入 RL 训练预算：示范采集墙钟（`bc_warmstart.collect_wall_seconds`）、预训练墙钟（`bc_warmstart.wall_seconds`）、最终 actor/critic MSE、采集 transitions 数（config.json `bc_warmstart` 节）。使用 BC 的臂在等交互步数比较中须注明 PPO 交互步不含 BC 采集步。

## 5. 协议修订记录

精简 changelog。旧条目细节见 git 历史。

- **2026-10-22 — 初始冻结。** 成功定义（违规 > 成功）、控制器审计、各任务训练预算/收敛阈值、种子 {7,13,21} + 留出 {101}、成本记账规则（§3–§4）冻结。
- **2026-10-24 — env RNG 流修复。** `TensorRandom` 种子分配由 `arange(n)+seed` 改为 `mix32(seed ^ (i·0x9e3779b1))`，使 {7,13,21,101} 的随机流两两不相交。2026-10-22/23 的所有 env-RNG 相关数字不再可逐位复现，基线已重跑。
- **2026-10-24 — 任务难度修订。** 对 5 个基线 100% 的任务（quadx、satellite、transfer、docking、rendezvous）仅放宽初始/任务参数分布；成功判据、奖励、时域与失败边界未动。基线已重跑。
- **2026-10-26 — Rocket 基准 v2。** 六球形垫接触模型 + 连续 2 秒停稳主指标（`ever_stable_2s`）。历史 v1 瞬时分数不可比。见 [env-rocket-landing.zh-CN.md](method/env-rocket-landing.zh-CN.md) §14。
- **2026-10-27 — Rocket 策略频率 60→40 Hz。** Rocket 策略/奖励统一为 40 Hz（30 s 时域不变，1800→1200 步）。物理步频率改由 `PHYSICS_HZ` 读取，不再硬编码。40 Hz 之前的 rocket 数字仅作历史参考。
- **2026-11-02 — 文档与代码一致性（不改变结果）。** 任务数确认为 10；§3 预算表列出各 `train_<task>.py` 的实际硬编码值；评测风协议更正（`eval.py` 从 `config.json` 恢复训练 `env_kwargs`）；rocket 奖励为 v5；rocket 84.1% 为 v1 瞬时而非 v2；基线样本量为 128（协议）/256（变体），非 1024；sigma-pin 机制退役（默认 Beta 策略）。
