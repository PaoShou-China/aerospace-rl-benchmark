# pyflyt-genesis 技术亮点与可复验性说明

面向审稿人的补充材料。初始版本锚点：commit `6b02383`（2026-10-21）；
本文档最初为该提交之上新增的文件。文中路径相对仓库根目录，带 `#L...`
的行号锚点来自该提交，会随工作区演进漂移。2026-11-02 已按当前工作区
更新任务计数（10 个）、风配置、评测口径与基线战绩；引用具体位置时请以
文末代码索引中的文件与函数/符号名为准，不要依赖行号。

本文只写**能落到代码行、并注明复验条件**的东西。每条亮点按
“断言 → 机制 → 证据 → 对审稿人的意义 → 边界”组织，最后单列
“不声称什么”和“断言状态表”，以免把设计意图误读为已验证结论。

---

## 摘要

与一般 RL 环境库（Gymnasium / Brax / Isaac Lab 等）相比，本套件的增量不在环境数量，
而在六类**面向对照实验的基础设施**：

1. **哈希清单化的参考基准**：上游 PyFlyt 0.29.0 的任务代码以冻结副本形式随仓库提供；
   另有源码/资产 SHA-256 校验原语。张量 plant 的惯量直接解析自参考 URDF，而非手工抄数。
2. **多执行路径**：4 个任务是纯张量实现（satellite / transfer / docking /
   rendezvous），6 个是 Genesis 刚体实现（quadx / quadx_track / fixedwing /
   quadplane / rocket / missile）；后者另有一条“原版 NumPy 任务代码不改一行”
   的共享场景批处理桥，可做实现级交叉校验。
3. **训练基础设施**：warmup-cosine LR 调度、
   确定性评测，以及跨任务共享的受控超参旋钮（lr_sigma_coupling、sigma_pin、freeze_obs_norm）。
4. **结果溯源**：`run.json` 记录任务配置、真实转场数、物理步调用数、设备端完成指标，
   并显式声明数值不等价边界。
5. **经典（非学习）基线库**：10 个任务各配一个解析 GNC 基线（几何控制、L1+TECS、
   PDG、CWH-LQR、两脉冲制导等），同一环境、同一奖励、同一成功判据下给出
   非学习参照，并保留基线的诚实失败模式作为 RL 的攻击面
   （quadplane 的三个控制器臂均通过官方口径评测，见亮点六）。

---

## 亮点一：可审计的参考基准，而不是“重写了一遍”

**断言**：参考实现以“冻结副本 + 哈希清单 + 惯量解析”的形式存在；
审计原语可手动执行，但目前不是运行时强制门禁。

**机制与证据**

- 原版代码路径不依赖环境里安装的 PyFlyt：桥接直接导入仓库内的冻结副本
  `environments/common/original_pyflyt`
  （[original_backend.py:455-468](../environments/common/original_backend.py#L455-L468)）。
- `verify_installed_reference()` 校验**已安装**的 `PyFlyt==0.29.0`：
  版本号，以及清单中源码文件与 URDF/YAML 资产的 SHA-256；任一不符立即抛错
  （[consistency.py:26-44](../environments/common/consistency.py#L26-L44)）。
- `contract()` 对仓库内 `environments/common/assets/reference_*` 做哈希校验，并把车辆、
  `dt`、控制抽取比、坐标系约定、噪声/风设置序列化成一个 `plant_id`
  （[consistency.py:97-127](../environments/common/consistency.py#L97-L127)）。
- 惯量不是手写常数：`asset_inertials()` 解析参考 URDF 供张量动力学使用
  （[consistency.py:63-95](../environments/common/consistency.py#L63-L95)、
  [dynamics.py:25-27](../environments/common/dynamics.py#L25-L27)）；
  同一套惯量经 `inertial_basis()` 做“主轴 → 作者轴”旋转后喂给 Genesis 刚体
  （[plant.py:441-462](../environments/common/plant.py#L441-L462)）。

**对审稿人的意义**：可以区分“代码复现了任务定义”与“代码碰巧能跑”；
参考版本、资产哈希与惯量来源都可核查。

**边界（必须同时读）**

- 两个校验函数目前**没有被训练/评测入口调用**；本开发环境也未安装 PyFlyt，
  因此哈希复验命令需先安装 0.29.0 才能执行。
- 覆盖范围不同：`verify_installed_reference()` 校验**已安装包**，
  不校验仓库内的 `environments/common/original_pyflyt` 冻结副本；`contract()` 校验仓库内参考资产。
  两者不能互相替代。
- 哈希一致 **不等于** 物理一致。Genesis 兼容后端在模块首行即声明
  “API compatibility backend, not a claim of bitwise physics equivalence”
  （[original_backend.py:1-5](../environments/common/original_backend.py#L1-L5)）。

---

## 亮点二：可交叉校验的多执行路径，共享一套协议

**断言**：环境层提供的不只是“一个物理引擎包装”，而是三种可互相校验的执行方式——
纯张量（无引擎）、张量化 plant（Genesis 刚体）、原版 NumPy 任务代码桥接；
上层训练/评测代码不需要感知差异。

**机制与证据**

1. **纯张量路径**（satellite / transfer / docking / rendezvous）
   - 无引擎：RK4 / 辛 Euler 等积分器直接写在各自 env 内，
     状态、随机数与统计全在设备上（[README.md §Package layout](../README.md#package-layout)）。
   - 套件级约定是每一步都是带掩码的张量核，未激活副本冻结、
     且**不消耗随机数**；训练循环内无 gym、无 CPU 往返
     （[README.md](../README.md)）。

2. **张量 plant 路径**（quadx / quadx_track / fixedwing / quadplane / rocket / missile）
   - `TensorPlant` 统一承载几何、作动（rate-PID mixer、升力面、助推器/燃料）、
     物理步进、接触检查与 reset/warmup；reward/终止/观测仍留在各 env
     （[plant.py:1-9](../environments/common/plant.py#L1-L9)）。
   - `TensorRandom` 为每个副本维护独立计数器随机流；文档明确其与上游
     **分布兼容、但非 NumPy PCG64 位流兼容**
     （[plant.py:60-72](../environments/common/plant.py#L60-L72)）。
   - 作动器、接触、气动均为张量化层，观测/奖励/重置沿用同一协议。
   - 积分器选择被显式记录并给出实测理由：显式 Euler 会让着陆接触抖动到
     ~0.1 m/s，越过 0.02 m/s 的判定阈值，因此保留 Genesis 默认的
     `approximate_implicitfast`
     （[original_backend.py:95-100](../environments/common/original_backend.py#L95-L100)）。
     这类“参数背后有测量”的记录，比参数本身更有审稿价值。

3. **原版代码桥接**（仅 quadx / fixedwing / rocket）：`OriginalTaskBatch` 用一个
   Genesis scene 托管 N 个副本，Python 侧仍是**未修改的** PyFlyt
   task/controller/aero 对象
   （[original_batch.py:1-7](../environments/common/original_batch.py#L1-L7)、
   [original_batch.py:95-113](../environments/common/original_batch.py#L95-L113)）。
   - 选择性重置时，非参与副本的物理状态通过
     `scene.get_state()` / `scene.reset(state=..., envs_idx=...)` 精确回滚，
     其 Aviary 观测到的物理时间不变
     （[original_batch.py:142-185](../environments/common/original_batch.py#L142-L185)）。
   - 接触信息在状态回滚**之前**读取，避免缓存被清除后丢失
     （[original_batch.py:165-181](../environments/common/original_batch.py#L165-L181)）。
   - `loadURDF` 校验所有副本加载同名 URDF，避免混批
     （[original_batch.py:62-73](../environments/common/original_batch.py#L62-L73)）。

**对审稿人的意义**：4 个无引擎任务与 6 个引擎任务共享同一套协议，可分别验证；
其中 3 个大气任务（quadx / fixedwing / rocket）还能做“原版实现/张量化实现”
的交叉校验，而不是只能相信单一实现。

**边界**

- 该桥接的正确性优先：原版 NumPy 代码仍跑在 host，**不是吞吐最优路径**；
  其价值是正确性对照，不是性能数字。
- “暂停非参与副本”是**步进后状态回滚**语义，不是让引擎跳过它们；
  NumPy/控制器/RNG 状态不被触碰，但物理求解器确实推进过整批。

---

## 亮点三：风扰是统一 Plant 钩子，不是 per-env 力补丁

**断言**：5 个大气任务（QuadX-Waypoints、Fixedwing-Waypoints、Quadplane-VTOL、
Rocket-Landing、Missile-Intercept）共享同一套风扰接入路径——气动力统一用
**空速** `v_air = v - wind` 评估；其中 quadx / fixedwing / quadplane 的训练入口
硬编码 30% Dryden 阵风的 `WIND` 常量为默认风场，rocket 固定 `dryden=0.0`，
missile 保持静风。静风（默认）与发表基准**逐位一致**。

**机制与证据**

- `TensorPlant` 持有 `wind`（world 系 (n,3) float64，`None` = 静风）；
  `_air_velocity` 在世界系减去风后再旋到体轴
  （[plant.py](../environments/common/plant.py)），因此
  fixedwing 升力面与 rocket 鳍/体阻力的气动项自动感知风；
- QuadX 无升力面，其既有体阻力项的变量就叫 `vb_air`——接入前恒等于体轴
  速度，接入后变为 `vb - R^T·wind`，静风时逐位回到原实现；
- 风的生成统一走 `environments/common/disturbances.py` 的
  `WindDisturbance`（定常风 + Dryden 阵风，MIL-F-8785C 低空成形滤波、
  种子化 RNG 流），env 每个控制步更新一次 `plant.wind`（阵风在物理子步间
  冻结），`reset_idx` 镜像重置滤波器状态；
- 训练入口硬编码 `WIND` 常量：quadx / fixedwing / quadplane 为纯 Dryden 阵风
  （零均值，30% moderate 档）；rocket 为 `dryden=0.0`（与静风基线同口径）；
  missile 不传风参数；quadx_track 的构造器不接受风参数。卫星任务在真空中无风。

**不变式（这条最关键）**

- 静风路径完全不进入风分支：`wind=None` 时 `_air_velocity` 与 QuadX 阻力
  项的字节码路径与旧实现一致，同 seed 的首个 reset 及后续 step **逐位一致**；
- `eval.py` 在 `results/<task>/config.json` 存在时恢复该次训练记录的
  `env_kwargs`，因此带风训练的策略会在同一风配置下评测；不存在配置文件时
  才使用环境默认的静风。跨方法比较必须显式固定同一风配置，而不是假定评测
  恒为静风。

**对审稿人的意义**：风扰不是"改 dynamics 的手写补丁"，而是把所有气动力
改为对空速评估这一物理上正确的表述；可以单独检验"扰动鲁棒性"对策略的
影响，而不用怀疑评测标准被偷偷放宽。

**边界**

- 阵风滤波每控制步推进一次，物理子步间冻结（Dryden 带宽远低于控制频率，
  误差可忽略）；
- `WindDisturbance` 的 RNG 是逐副本计数器流（种子 `mix32(seed ^ i*GOLDEN)` +
  每副本计数器，全部留在 device 上，无 CPU→GPU 传输）；带 `env_ids` 的
  `step` 只推进被选副本的计数器，不会改变其它副本的噪声序列；
  静风路径零开销。

---

## 亮点四：训练基础设施——可复用的 LR 调度与受控旋钮

**断言**：仓库提供可复用的训练调度与受控超参旋钮，保证跨任务的公平对照。

**机制与证据**

- **warmup-cosine LR 调度**：PPO 侧内置 `_warmup_cosine_decay` 公式（
  warmup 比例 1e-6 × total updates，peak→end 跨度）；由于 rsl-rl 在
  `schedule='fixed'` 下永不重设 lr，代码直接写 `optimizer.param_groups`
  （[train_utils.py:27-36](../algorithms/train_utils.py#L27-L36)、
  [train_utils.py:681-697](../algorithms/train_utils.py#L681-L697)）。
- **其他受控旋钮**：指数衰减、`lr_sigma_coupling`（令 lr ∝ σ，保持每步策略漂移率恒定）、
  `sigma_pin`（重钉探索方差）、`freeze_obs_norm`（冻结 warm start 的观测归一化），
  均为 opt-in 且互斥关系显式检查
  （[train_utils.py:551-566](../algorithms/train_utils.py#L551-L566)、
  [train_utils.py:713-730](../algorithms/train_utils.py#L713-L730)）。
  其中 `lr_sigma_coupling` / `sigma_pin` 读取 `log_std_param`，只适用于
  `distribution='gaussian'`；默认的 `BetaDistribution` 无此参数，相关旋钮随之失效。

**边界**

- wall-clock 为同机记录值，受同批并发影响，只作量级参考。
- 确定性评测由 `eval.py` 统一执行并写入各臂 `eval.json`；
  但现存记录的评测协议不完全一致（env 数/步数不同），
  **不应把不同 `eval.json` 的成功率直接横比**。

---

## 亮点五：结果溯源与可复现性

**断言**：一次训练产出的记录，足以回答“跑了什么、跑了多少、算得对不对”。

**机制与证据**

- `run.json` 记录：任务配置（env kwargs 原样落盘）、算法、迭代数、
  总 transitions、wall time、逐副本 `completed_episodes` / `completed_successes`、
  真实 `physics_step_calls`、完成 episode 的设备端指标，以及一条数值不等价声明
  （[train_utils.py:750-801](../algorithms/train_utils.py#L750-L801)）。
- `config.json` 在交给 runner 前做深拷贝，避免 `OnPolicyRunner` 原地修改配置
  导致落盘的不是真实训练配置
  （[train_utils.py:318-352](../algorithms/train_utils.py#L318-L352)）。
- `eval.py` 使用冻结的确定性策略（均值动作、无探索噪声），
  成功/回报由环境自身在设备端累计，结果写入 `eval.json`
  （[eval.py:1-7](../scripts/eval.py#L1-L7)、[eval.py:123-139](../scripts/eval.py#L123-L139)）。
  评测构造 env 时从 `results/<task>/config.json` 恢复训练记录的 `env_kwargs`
  （风、benchmark 版本等），因此评测与训练分布一致；若该文件不存在则使用
  环境默认配置。
- 每个任务入口的超参硬编码在文件中、不暴露 CLI；牺牲灵活性换取可复现性
  （[README.md §Usage](../README.md#usage)）。

**对审稿人的意义**：训练与评测的“输入—配置—统计口径”都可复核。

**边界**

- 这不是完整的实验管理平台：没有多 seed 编排、没有自动显著性检验。
- **记录缺口**：训练入口级开关（如 `lr_cosine`、`freeze_obs_norm`）不写入
  `config.json`/`run.json`，复现时须以对应 `train_<task>.py` 的硬编码常量为准。
- `update()` 在 episode 边界会做一次小规模标量同步
  （每步至多一次，且仅当有 episode 结束）；逐步训练路径本身无 host 同步。

---

## 亮点六：每个任务配一个非学习基线，且基线不是稻草人

**断言**：10 个任务各有一个解析式经典 GNC 默认基线（另有 20 个结构变体，
合计 30 个控制器臂），与学习算法共享同一环境、同一奖励、同一成功判据；
基线在部分任务上达到 100%，在另一些任务上保留**有取证记录的失败模式**。

**机制与证据**

- 统一契约：所有基线实现于 [baselines.py](../algorithms/baselines/baselines.py)
  （`QuadXSE3Baseline` … `RendezvousTwoImpulseBaseline`，
  [baselines.py:527-3253](../algorithms/baselines/baselines.py#L527-L3253)），工厂入口
  `build_baseline(vehicle, env)`
  （[baselines.py:3939](../algorithms/baselines/baselines.py#L3939)）。每个控制器
  **只读 `'policy'` 观测组**、构造时读取环境常数（质量、推力权限、CMG 几何、
  名义轨道——即经典 GNC 设计者本会掌握的模型知识），**不读任何内部状态缓冲**，
  不改环境语义（[baselines.py:1-26](../algorithms/baselines/baselines.py#L1-L26)）。
- 统一评测协议：[baseline_eval.py](../scripts/baseline_eval.py) 默认以
  1024 envs / 20000 步安全上限 / seed 7 评测任意任务的基线，结果写
  `results/<task>/baseline.json`（指定 `--baseline` 时写
  `baseline_<controller>.json`），控制器信息标记为
  `classical, policy-observation-only`。仓库内已发布的产物使用各自文件里
  记录的较小窗口（并非统一 1024 副本），引用时必须同时写明 `num_envs`、
  `seed` 与 `env_steps`
  （[baseline_eval.py](../scripts/baseline_eval.py)、
  [docs/baseline_test_report.md](baseline_test_report.md)）。
- 覆盖的方法学谱系（默认臂）：SE(3) 几何控制（Lee et al. 2010, quadx）、
  CV 卡尔曼滤波 + SE(3)（quadx_track）、L1 横向 + TECS 纵向（fixedwing）、
  四旋翼 SE(3) + 前推（quadplane）、ZEM/PN 动力下降制导（rocket）、
  增广比例导引 APN（missile）、四元数 PD + 奇异鲁棒 CMG 阻尼伪逆
  （satellite）、Gauss 方程切向制导（transfer）、精确辛离散化 CWH-LQR
  （docking）、递归重解两脉冲 CWH 制导（rendezvous）。结构变体覆盖
  级联 PID、滑模外环、纯追踪、EMA 滤波、NLGL、E-guidance、PPN/TPN、
  L1 机翼模式、混合模式自驾、普通/加权伪逆 CMG、Edelbaum、Q-law、
  极点配置、CWH ZEM/ZEV、glideslope 与 CWH 补偿 PD
  （[baselines.py](../algorithms/baselines/baselines.py)、
  [docs/baseline_test_report.md](baseline_test_report.md) 第 5 节）。
- **基线的参考战绩**（多 seed 协议：128 副本、每副本首回合、种子
  7/13/21 + 留出 101；详见
  [docs/baseline_test_report.md](baseline_test_report.md) 第 2 节）：
  satellite / docking / rendezvous / rocket 100%；quadx 99.7%（3-seed 均值）；
  transfer 99.2%；quadplane 89.1%；fixedwing 87.0%；quadx_track 85.9%；
  missile 77.3%。留出种子 101 与协议种子无系统性差异。
  单次发布的 `results/<task>/baseline.json` 窗口不同：quadx 为 32 副本 /
  seed 7 / 800 步（90.6%），多数任务为 128 副本 / seed 13，
  quadx_track 为 seed 21；这些数字与上表的 protocol 均值不可直接横比。
- **失败模式已被分项记录**：fixedwing 的残余是三变体的 OOB
  （`l1_tecs` 2.5%、`nlgl_tecs` 6.8%、`pid_cascade` 10.1%）；quadplane 的
  弱臂以超时为主（`hybrid_cascade` 68.5%、`l1_wing` 50.4%，而默认
  `se3_pusher` 仅 12.0%）；missile 的超时率为 0–3.1%。
  专项 harness（[eval_fixedwing_autopilot.py](../environments/fixedwing/eval_fixedwing_autopilot.py)、
  [rocket_pdg_test.py](../environments/rocket/rocket_pdg_test.py)）给出
  场景级分解，但这些 harness 的结果 JSON 未随仓库发布。
- **双向证据（只引用已落盘结果）**：基线并非稻草人——`benchmark_report.md`
  §6.3–6.5 的 200/150 updates 短预算中，纯 PPO 在 satellite 上累计成功率
  0.43–0.69、docking 上 0.11，明显低于基线的 100%；BC warm-start 将
  satellite 提到 0.709、docking 提到 1.000，residual RL 进一步达到
  satellite 0.985、docking 1.000。基线也并非天花板——missile 77.3%、
  quadx_track 85.9%、fixedwing 87.0% 都留出了可攻击的失败空间。

**对审稿人的意义**：学习算法的成绩有了非学习参照系；且参照系的失败
模式被归因（权限饱和 / 几何不可达 / 数值病态）而非含糊带过，
"RL 超越经典方法"的主张可以落到具体场景。

**边界**

- 基线公式是解析式，无优化器、无调参循环；它们的超参（增益、带宽、
  伺服带）是**任务级手工设计**，不声称全局最优。
- 成功率数字依赖评测窗口（步数/env 数），不同 harness 的窗口不同；
  跨 harness 直接横比成功率无效（与亮点四的评测口径边界一致）。
- rocket 基线的成功率强依赖评测窗口与版本：v1 的瞬时触垫判据与 v2 的
  “连续停稳 2 s”判据不可比。当前 v2 产物
  `results/rocket/baseline.json`（128 副本 / seed 13 / 1200 步，
  `ever_stable_2s=1.0`）为 100%；引用历史 v1 数字时必须声明版本与窗口。

---

## 不声称什么（Non-claims）

审稿人最容易被“复现”二字误导，以下边界均为显式设计声明：

1. **不声称与 PyFlyt/Bullet 逐位物理等价**。Genesis 后端是 API 兼容层，
   引擎数值不同（[original_backend.py:1-5](../environments/common/original_backend.py#L1-L5)）。
2. **不声称 NumPy 随机位流一致**。`TensorRandom` 与上游分布兼容，
   但位流不同（[plant.py:60-72](../environments/common/plant.py#L60-L72)）。
3. **不声称基线与 RL 可比性之外的最优性**。基线是手工设计，不代表性能上界（见第 6 条）。
4. **不声称原版代码桥接是高性能路径**。它是正确性优先的 host 端桥接。
5. **示例数字不构成统计结论**。单 seed 结果只作记录与量级参考。
6. **不声称经典基线最优**。基线是解析式手工设计（无优化器、无调参循环）；
   它们的价值是同一判据下的非学习参照与已归因的失败模式，不是性能上界。

---

## 断言状态表

| 主张 | 状态 | 证据/复验方式 |
|---|---|---|
| 上游 PyFlyt 0.29.0 源码与资产可哈希校验 | 已实现；**未自动接线**；需安装 PyFlyt 0.29.0 | `python -c "from environments.common.consistency import verify_installed_reference as v; print(v())"` |
| `plant_id` 受控对象证书 | 已实现；**未被入口调用** | `contract()`，[consistency.py:97-127](../environments/common/consistency.py#L97-L127) |
| 张量 plant 惯量取自参考 URDF | 已接线 | [plant.py:441-462](../environments/common/plant.py#L441-L462) |
| 未激活副本不消耗随机数/不推进状态 | 设计保证 + 代码路径 | [plant.py:1-9](../environments/common/plant.py#L1-L9) |
| 基准难度逐位复现 | 已验证（10 任务，同 seed 首个 reset 及 3 步后状态） | 临时复核脚本未随仓库保留 |
| 评测使用训练记录的配置 | 已保证（eval 恢复 `config.json` 的 `env_kwargs`） | [eval.py](../scripts/eval.py) |
| PPO warmup-cosine LR 调度 | 已接线（opt-in `lr_cosine`） | [train_utils.py:27-36](../algorithms/train_utils.py#L27-L36)、[train_utils.py:681-697](../algorithms/train_utils.py#L681-L697) |
| 每任务一个经典基线、只读 policy 观测 | 已实现 + 官方口径实测 | `python scripts/baseline_eval.py <task>`，[baselines.py:3939](../algorithms/baselines/baselines.py#L3939) |
| 与上游物理逐位等价 | **不声称** | 见 Non-claims 第 1 条 |

> 说明：两张“已实现但未接线”的审计原语（哈希校验、plant_id）保留为可选工具，
> 而不是训练时的强制门禁，以免在没有安装参考包的环境里阻塞正常训练。
> 若审稿流程要求强制校验，只需在入口处调用一次即可。

---

## 复验清单

```bash
# 1) 基准哈希校验（需先安装 PyFlyt 0.29.0；本仓库开发环境未安装）
python -c \
  "from environments.common.consistency import verify_installed_reference as v; print(v())"

# 2) 恢复训练记录的 env_kwargs 做确定性评测（以 quadx 为例）
python scripts/eval.py quadx --num-envs 16 --max-steps 400

# 3) 查看一次训练的实验记录（仓库内已归档的是 smoke 运行）
python -c "import json;r=json.load(open('results/smoke/quadx_ppo/run.json'));print(r['task']);print(r['transitions'], r['physics_step_calls'])"

# 4) 定位风扰钩子的实现位置
grep -n "wind" environments/common/plant.py environments/common/disturbances.py | head -20

# 5) 任一任务的经典基线（默认 1024 envs / 20000 步安全上限 / seed 7，
#    写 results/<task>/baseline.json；已发布产物用的是各自 JSON 内的较小窗口）
python scripts/baseline_eval.py quadx
```

---

## 代码索引

| 能力 | 文件 |
|---|---|
| 基准哈希校验、URDF 惯量解析、`plant_id` | [environments/common/consistency.py](../environments/common/consistency.py) |
| 纯张量 plant、逐副本随机流 | [environments/common/plant.py](../environments/common/plant.py) |
| Genesis 兼容后端（BulletClient API） | [environments/common/original_backend.py](../environments/common/original_backend.py) |
| 原版任务共享场景批处理 | [environments/common/original_batch.py](../environments/common/original_batch.py) |
| 训练/记录管线、LR 调度对齐 | [algorithms/train_utils.py](../algorithms/train_utils.py) |
| 10 任务经典基线库（30 个控制器臂） | [algorithms/baselines/baselines.py](../algorithms/baselines/baselines.py) |
| 基线统一评测入口 | [baseline_eval.py](../scripts/baseline_eval.py) |
| 基线专项 harness（场景网格/消融） | [eval_docking_lqr.py](../environments/satellite/rendezvous_docking/eval_docking_lqr.py)、[eval_fixedwing_autopilot.py](../environments/fixedwing/eval_fixedwing_autopilot.py)、[eval_rendezvous_baseline.py](../environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py)、[rocket_pdg_test.py](../environments/rocket/rocket_pdg_test.py)、[test_quadx_se3.py](../environments/multirotor/test_quadx_se3.py) |
| 确定性评测 | [eval.py](../scripts/eval.py) |
