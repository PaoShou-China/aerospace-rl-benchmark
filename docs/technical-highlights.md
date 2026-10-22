# pyflyt-genesis 技术亮点与可复验性说明

面向审稿人的补充材料。版本锚点：commit `6b02383`（2026-10-21）；
本文档为该提交之上新增的文件。文中路径相对仓库根目录，行号指向该提交。

本文只写**能落到代码行、并注明复验条件**的东西。每条亮点按
“断言 → 机制 → 证据 → 对审稿人的意义 → 边界”组织，最后单列
“不声称什么”和“断言状态表”，以免把设计意图误读为已验证结论。

---

## 摘要

与一般 RL 环境库（Gymnasium / Brax / Isaac Lab 等）相比，本套件的增量不在环境数量，
而在五类**面向对照实验的基础设施**：

1. **哈希清单化的参考基准**：上游 PyFlyt 0.29.0 的任务代码以冻结副本形式随仓库提供；
   另有源码/资产 SHA-256 校验原语。张量 plant 的惯量直接解析自参考 URDF，而非手工抄数。
2. **多执行路径**：5 个任务是无引擎的纯张量实现，3 个大气任务是 Genesis 刚体实现；
   后者另有一条“原版 NumPy 任务代码不改一行”的共享场景批处理桥，可做实现级交叉校验。
3. **全库统一的 0–1 课程契约**：只改初始分布，不动成功判据；基准难度处逐位复现；
   成功率门控自动爬升，进度写入 `run.json`。
4. **算法对照基础设施**：FlashSAC 完整移植、PPO 架构匹配臂、跨算法同一 LR 调度、
   确定性评测，以及把“失败配置为什么失败”写进代码的负结果记录。
5. **结果溯源**：`run.json` 记录任务配置、真实转场数、物理步调用数、设备端完成指标，
   并显式声明数值不等价边界。

---

## 亮点一：可审计的参考基准，而不是“重写了一遍”

**断言**：参考实现以“冻结副本 + 哈希清单 + 惯量解析”的形式存在；
审计原语可手动执行，但目前不是运行时强制门禁。

**机制与证据**

- 原版代码路径不依赖环境里安装的 PyFlyt：桥接直接导入仓库内的冻结副本
  `src/original_pyflyt`
  （[original_backend.py:198-209](../src/common/original_backend.py#L198-L209)）。
- `verify_installed_reference()` 校验**已安装**的 `PyFlyt==0.29.0`：
  版本号，以及清单中源码文件与 URDF/YAML 资产的 SHA-256；任一不符立即抛错
  （[consistency.py:19-32](../src/common/consistency.py#L19-L32)）。
- `contract()` 对仓库内 `src/common/assets/reference_*` 做哈希校验，并把车辆、
  `dt`、控制抽取比、坐标系约定、噪声/风设置序列化成一个 `plant_id`
  （[consistency.py:56-71](../src/common/consistency.py#L56-L71)）。
- 惯量不是手写常数：`asset_inertials()` 解析参考 URDF 供张量动力学使用
  （[consistency.py:35-53](../src/common/consistency.py#L35-L53)、
  [dynamics.py:25-27](../src/common/dynamics.py#L25-L27)）；
  同一套惯量经 `inertial_basis()` 做“主轴 → 作者轴”旋转后喂给 Genesis 刚体
  （[plant.py:136](../src/common/plant.py#L136)）。

**对审稿人的意义**：可以区分“代码复现了任务定义”与“代码碰巧能跑”；
参考版本、资产哈希与惯量来源都可核查。

**边界（必须同时读）**

- 两个校验函数目前**没有被训练/评测入口调用**；本开发环境也未安装 PyFlyt，
  因此哈希复验命令需先安装 0.29.0 才能执行。
- 覆盖范围不同：`verify_installed_reference()` 校验**已安装包**，
  不校验仓库内的 `src/original_pyflyt` 冻结副本；`contract()` 校验仓库内参考资产。
  两者不能互相替代。
- 哈希一致 **不等于** 物理一致。Genesis 兼容后端在模块首行即声明
  “API compatibility backend, not a claim of bitwise physics equivalence”
  （[original_backend.py:1-5](../src/common/original_backend.py#L1-L5)）。

---

## 亮点二：可交叉校验的多执行路径，共享一套协议

**断言**：环境层提供的不只是“一个物理引擎包装”，而是三种可互相校验的执行方式——
纯张量（无引擎）、张量化 plant（Genesis 刚体）、原版 NumPy 任务代码桥接；
上层训练/评测代码不需要感知差异。

**机制与证据**

1. **纯张量路径**（satellite / transfer / docking / rendezvous / pendulum）
   - 无引擎：RK4 / 辛 Euler 等积分器直接写在各自 env 内，
     状态、随机数与统计全在设备上（[README.md:63-68](../README.md#L63-L68)）。
   - 套件级约定是每一步都是带掩码的张量核，未激活副本冻结、
     且**不消耗随机数**；训练循环内无 gym、无 CPU 往返
     （[README.md:10-14](../README.md#L10-L14)）。

2. **张量 plant 路径**（quadx / fixedwing / rocket）
   - `TensorPlant` 统一承载几何、作动（rate-PID mixer、升力面、助推器/燃料）、
     物理步进、接触检查与 reset/warmup；reward/终止/观测仍留在各 env
     （[plant.py:1-9](../src/common/plant.py#L1-L9)）。
   - `TensorRandom` 为每个副本维护独立计数器随机流；文档明确其与上游
     **分布兼容、但非 NumPy PCG64 位流兼容**
     （[plant.py:23-30](../src/common/plant.py#L23-L30)）。
   - 作动器、接触、气动均为张量化层，观测/奖励/重置沿用同一协议。
   - 积分器选择被显式记录并给出实测理由：显式 Euler 会让着陆接触抖动到
     ~0.1 m/s，越过 0.02 m/s 的判定阈值，因此保留 Genesis 默认的
     `approximate_implicitfast`
     （[original_backend.py:50-58](../src/common/original_backend.py#L50-L58)）。
     这类“参数背后有测量”的记录，比参数本身更有审稿价值。

3. **原版代码桥接**（仅 quadx / fixedwing / rocket）：`OriginalTaskBatch` 用一个
   Genesis scene 托管 N 个副本，Python 侧仍是**未修改的** PyFlyt
   task/controller/aero 对象
   （[original_batch.py:1-7](../src/common/original_batch.py#L1-L7)、
   [original_batch.py:71-78](../src/common/original_batch.py#L71-L78)）。
   - 选择性重置时，非参与副本的物理状态通过
     `scene.get_state()` / `scene.reset(state=..., envs_idx=...)` 精确回滚，
     其 Aviary 观测到的物理时间不变
     （[original_batch.py:113-143](../src/common/original_batch.py#L113-L143)）。
   - 接触信息在状态回滚**之前**读取，避免缓存被清除后丢失
     （[original_batch.py:130-141](../src/common/original_batch.py#L130-L141)）。
   - `loadURDF` 校验所有副本加载同名 URDF，避免混批
     （[original_batch.py:50-59](../src/common/original_batch.py#L50-L59)）。

**对审稿人的意义**：5 个无引擎任务与 3 个引擎任务共享同一套协议，可分别验证；
3 个大气任务还能做“原版实现/张量化实现”的交叉校验，而不是只能相信单一实现。

**边界**

- 该桥接的正确性优先：原版 NumPy 代码仍跑在 host，**不是吞吐最优路径**；
  其价值是正确性对照，不是性能数字。
- “暂停非参与副本”是**步进后状态回滚**语义，不是让引擎跳过它们；
  NumPy/控制器/RNG 状态不被触碰，但物理求解器确实推进过整批。

---

## 亮点三：课程学习是全库契约，不是 per-env 随机化

**断言**：8 个任务共享同一难度语义，且难度是可对照的实验变量。

**机制与证据**

- 统一构造参数：`difficulty=None`（基准）/ `difficulty=d∈[0,1]`（静态）/
  `curriculum=True`（自动爬升）；每个 env 自带 `benchmark_difficulty`
  与自己的难度映射（`_apply_difficulty`，或等价的构造期解析）
  （[curriculum.py:1-17](../src/common/curriculum.py#L1-L17)；
  各任务实现见
  [quadx/env.py:37](../src/quadx/env.py#L37)、
  [fixedwing/env.py:51](../src/fixedwing/env.py#L51)、
  [rocket/env.py:145](../src/rocket/env.py#L145)、
  [satellite/env.py:115](../src/satellite/env.py#L115)、
  [transfer/env.py:64](../src/transfer/env.py#L64)、
  [docking/env.py:77](../src/docking/env.py#L77)、
  [pendulum/env.py:47](../src/pendulum/env.py#L47)）。
- 共享 tracker 只负责统计与门控，难度轴映射归各 env 所有，
  因此**所有任务的爬升语义完全一致**
  （[curriculum.py:20-52](../src/common/curriculum.py#L20-L52)）。
- 爬升规则：每 1024 个完成 episode 结算一次，滚动成功率 ≥ 0.6 则难度 +0.1，
  **只进不退**，并在累加时做 `round(...,10)` 以避免
  `0.30000000000000004` 这类浮点残差写进实验记录
  （[curriculum.py:43-52](../src/common/curriculum.py#L43-L52)）。
- 难度轴（d=0 → d=1，均为“易 → 难”；d=1 即发表基准，quadx 例外——
  其确定性 spawn 基准在 d=0，轴向右延伸为更难）：

| 任务 | 旋钮 | d=0（易） → d=1（难） |
|---|---|---|
| Pendulum | reset 分布 | th0 ±30° → ±180°，thdot0 ±0.25 → ±1 rad/s |
| QuadX-Hover | spawn 扰动 | 确定性 spawn（基准） → ±0.5 m xy / ±0.25 m z / ±0.5 m/s / ±0.35 rad |
| Fixedwing | 航点壳层 | 最小半径 50 m → 20 m（最大 65 m 固定） |
| Rocket-Landing | 初始条件 | xy ±2 m，z 440–450 m，euler ±0.05 → xy ±20 m，z 400–450 m，euler ±0.3 |
| Satellite | 目标姿态 | 距 home ≤0.3 rad、无噪声 → 均匀 SO(3)、±0.001 噪声 |
| Transfer | 初始偏心率 | 0.02 → 0.1 |
| Docking | 初始距离/速度 | 15–30 m，0.1 m/s → 50–100 m，0.5 m/s |
| Rendezvous | 初始距离 | 150–300 m → 800–1200 m（vmax 不变） |

（与 [README.md:99-110](../README.md#L99-L110) 一致。）

**不变式（这条最关键）**

- 只调节**初始状态分布 / 任务参数分布**；成功判据、终止条件、reward 结构永不动，
  因此任意训练难度下的评测仍与发表基准可比。
- `difficulty=benchmark_difficulty` 时，同 seed 的首个 reset 及后续 step
  与旧实现**逐位一致**（`None` 路径完全不进入难度分支）。

**对审稿人的意义**：课程不是“改题”，而是“改题面难度”；可以单独检验
“训练分布变化”对最终策略的影响，而不用怀疑评测标准被偷偷放宽。

**边界**

- 课程是训练期分布调度器，**不声称**提升最终性能；它只是把难度变成可控变量。
- 自动爬升的统计窗（1024 episodes）与阈值（0.6）是固定超参，跨任务相同，
  没有按任务调优；这是为可比性做的取舍。

---

## 亮点四：算法对照基础设施——把“为什么失败”也写进代码

**断言**：仓库不只提供环境，还提供可复用的算法对照装置与负结果记录。

**机制与证据**

- **FlashSAC 完整移植**（Kim et al., 2026）：UnitLinear/BN/RMSNorm 的权重、
  特征、梯度范数控制；101-bin 分布式 ensemble critic；tanh-Gaussian actor；
  温度自调；zeta 噪声重复；return-RMS 奖励归一化；低 UTD 快训；EMA target。
  与官方实现的偏差被逐条列出，且声明“none algorithmic”
  （[flash_sac.py:1-38](../src/common/flash_sac.py#L1-L38)）。
  回放缓冲是设备常驻环形缓冲，循环内无 gym、无 CPU 往返
  （[flash_sac.py:348](../src/common/flash_sac.py#L348)）。
- **PPO + FlashSAC trunk 架构匹配臂**：让 PPO 使用与 FlashSAC 相同的残差 trunk，
  并把三处**算法性**偏差写在模块文档里（状态无关高斯、标量 critic、
  每个优化步后首前向才重归一化）
  （[ppo_flashnet.py:1-28](../src/common/ppo_flashnet.py#L1-L28)）。
- **负结果记录**：该文件明确记录了“BatchNorm + GAE 自举”的失败机制——
  批统计让 critic 输入归一化非平稳，而 GAE 目标又要通过 critic 自举，
  形成反馈回路压平价值面；同一 trunk 在**离线平稳数据**上反而比 MLP 好 3 倍，
  因此问题在耦合而非架构，故 critic 在 on-policy 任务上关闭 BN
  （[ppo_flashnet.py:80-101](../src/common/ppo_flashnet.py#L80-L101)）。
  这类“已排除的解释”对审稿人比调参日志有用。
- **跨算法同一 LR 调度**：PPO 侧复用 FlashSAC 的 `_warmup_cosine_decay` 公式、
  同一 warmup 比例与 peak→end 跨度；由于 rsl-rl 在 `schedule='fixed'` 下
  永不重设 lr，代码直接写 `optimizer.param_groups`
  （[train_utils.py:216-230](../src/common/train_utils.py#L216-L230)）。
- **其他受控旋钮**：指数衰减、`lr_sigma_coupling`（令 lr ∝ σ，保持每步策略漂移率恒定）、
  `sigma_pin`（重钉探索方差）、`freeze_obs_norm`（冻结 warm start 的观测归一化），
  均为 opt-in 且互斥关系显式检查
  （[train_utils.py:96-150](../src/common/train_utils.py#L96-L150)、
  [train_utils.py:246-270](../src/common/train_utils.py#L246-L270)）。

**示例：同一协议下的三方对照（Pendulum，训练栈自检，非论文结论）**

三臂均为 1024 envs、seed 7、≈10.2M transitions；
数字直接由各 `results/*/run.json` 重算（成功率 = Σcompleted_successes / Σcompleted_episodes）：

| 臂 | 算法 | 迭代 | transitions | 训练期成功率 | 训练期 mean return | wall-clock |
|---|---|---|---|---|---|---|
| `pendulum_flashsac1024` | FlashSAC | 10,000 steps | 10,240,000 | 59.7% | −275.3 | 251.7 s |
| `pendulum_ppo_flashres` | PPO + FlashRes trunk | 312 updates | 10,223,616 | 52.9% | −320.8 | 77.9 s |
| `pendulum_ppo_mlp1024` | PPO + MLP | 312 updates | 10,223,616 | 52.4% | −293.6 | 54.1 s |

**边界**

- 单 seed；Pendulum 是训练栈自检，不是方法学基准结论。
- wall-clock 为同机记录值，受同批并发影响，只作量级参考。
- 确定性评测由 `eval.py` 统一执行并写入各臂 `eval.json`；
  但现存记录的评测协议不完全一致（env 数/步数不同），
  **不应把不同 `eval.json` 的成功率直接横比**。

---

## 亮点五：结果溯源与可复现性

**断言**：一次训练产出的记录，足以回答“跑了什么、跑了多少、算得对不对”。

**机制与证据**

- `run.json` 记录：任务配置（显式设置难度/课程时含对应字段）、算法、迭代数、
  总 transitions、wall time、逐副本 `completed_episodes` / `completed_successes`、
  真实 `physics_step_calls`、完成 episode 的设备端指标，以及一条数值不等价声明
  （[train_utils.py:276-286](../src/common/train_utils.py#L276-L286)）。
- `config.json` 在交给 runner 前做深拷贝，避免 `OnPolicyRunner` 原地修改配置
  导致落盘的不是真实训练配置
  （[train_utils.py:178-180](../src/common/train_utils.py#L178-L180)）。
- `eval.py` 使用冻结的确定性策略（均值动作、无探索噪声），
  成功/回报由环境自身在设备端累计，结果写入 `eval.json`
  （[eval.py:1-7](../eval.py#L1-L7)、[eval.py:66-90](../eval.py#L66-L90)）。
  评测构造 env 时不传任何 kwargs，因此**始终落在基准难度**，不受训练课程影响。
- 每个任务入口的超参硬编码在文件中、不暴露 CLI；牺牲灵活性换取可复现性
  （[README.md:125-126](../README.md#L125-L126)）。

**对审稿人的意义**：训练与评测的“输入—配置—统计口径”都可复核；
课程难度不是隐藏状态，而是实验记录的一部分。

**边界**

- 这不是完整的实验管理平台：没有多 seed 编排、没有自动显著性检验。
- **记录缺口**：训练入口级开关（如 `lr_cosine`、`freeze_obs_norm`）不写入
  `config.json`/`run.json`，复现时须以对应 `train_<task>.py` 的硬编码常量为准。
- 为兼容课程统计，`update()` 在 episode 边界会做一次小规模标量同步
  （每步至多一次，且仅当有 episode 结束）；逐步训练路径本身无 host 同步。

---

## 不声称什么（Non-claims）

审稿人最容易被“复现”二字误导，以下边界均为显式设计声明：

1. **不声称与 PyFlyt/Bullet 逐位物理等价**。Genesis 后端是 API 兼容层，
   引擎数值不同（[original_backend.py:1-5](../src/common/original_backend.py#L1-L5)）。
2. **不声称 NumPy 随机位流一致**。`TensorRandom` 与上游分布兼容，
   但位流不同（[plant.py:23-30](../src/common/plant.py#L23-L30)）。
3. **不声称课程提升性能**。课程只改初始/任务参数分布，成功判据不变。
4. **不声称原版代码桥接是高性能路径**。它是正确性优先的 host 端桥接。
5. **不声称算法等价**。PPO+FlashNet 与 FlashSAC 的偏差已在模块文档中列明；
   FlashSAC 移植与官方仓库的偏差也已列明。
6. **示例数字不构成统计结论**。Pendulum 三方对照为单 seed 训练栈自检。

---

## 断言状态表

| 主张 | 状态 | 证据/复验方式 |
|---|---|---|
| 上游 PyFlyt 0.29.0 源码与资产可哈希校验 | 已实现；**未自动接线**；需安装 PyFlyt 0.29.0 | `PYTHONPATH=src python -c "from common.consistency import verify_installed_reference as v; print(v())"` |
| `plant_id` 受控对象证书 | 已实现；**未被入口调用** | `contract()`，[consistency.py:56-71](../src/common/consistency.py#L56-L71) |
| 张量 plant 惯量取自参考 URDF | 已接线 | [dynamics.py:25-27](../src/common/dynamics.py#L25-L27) |
| 未激活副本不消耗随机数/不推进状态 | 设计保证 + 代码路径 | [plant.py:1-9](../src/common/plant.py#L1-L9) |
| 基准难度逐位复现 | 已验证（8 任务，同 seed 首个 reset 及 3 步后状态） | 临时复核脚本未随仓库保留 |
| 课程自动爬升（0.1 步长、只进不退） | 已验证（8 任务 0.1→0.5，cfg 同步） | [curriculum.py:43-52](../src/common/curriculum.py#L43-L52) |
| 课程难度写入 `run.json` | 已接线 | [train_utils.py:276-286](../src/common/train_utils.py#L276-L286) |
| 评测始终在基准难度 | 已保证（eval 不传 kwargs） | [eval.py:54](../eval.py#L54) |
| 双算法共用同一 LR 调度 | 已接线（opt-in `lr_cosine`） | [train_utils.py:216-230](../src/common/train_utils.py#L216-L230) |
| 与上游物理逐位等价 | **不声称** | 见 Non-claims 第 1 条 |

> 说明：两张“已实现但未接线”的审计原语（哈希校验、plant_id）保留为可选工具，
> 而不是训练时的强制门禁，以免在没有安装参考包的环境里阻塞正常训练。
> 若审稿流程要求强制校验，只需在入口处调用一次即可。

---

## 复验清单

```bash
# 1) 基准哈希校验（需先安装 PyFlyt 0.29.0；本仓库开发环境未安装）
PYTHONPATH=src python -c \
  "from common.consistency import verify_installed_reference as v; print(v())"

# 2) 默认路径 = 基准难度；确定性评测（以 pendulum 为例）
python eval.py pendulum --num-envs 16 --steps 200

# 3) 查看一次训练的实验记录
python -c "import json;r=json.load(open('results/pendulum_flashsac1024/run.json'));print(r['task']);print(r['transitions'], r['physics_step_calls'])"

# 4) 定位课程契约的实现位置
grep -n "benchmark_difficulty\|_apply_difficulty" src/*/env.py
```

---

## 代码索引

| 能力 | 文件 |
|---|---|
| 基准哈希校验、URDF 惯量解析、`plant_id` | [src/common/consistency.py](../src/common/consistency.py) |
| 纯张量 plant、逐副本随机流 | [src/common/plant.py](../src/common/plant.py) |
| Genesis 兼容后端（BulletClient API） | [src/common/original_backend.py](../src/common/original_backend.py) |
| 原版任务共享场景批处理 | [src/common/original_batch.py](../src/common/original_batch.py) |
| 统一课程 tracker | [src/common/curriculum.py](../src/common/curriculum.py) |
| FlashSAC 移植 | [src/common/flash_sac.py](../src/common/flash_sac.py) |
| PPO 架构匹配臂与负结果记录 | [src/common/ppo_flashnet.py](../src/common/ppo_flashnet.py) |
| 训练/记录管线、LR 调度对齐 | [src/common/train_utils.py](../src/common/train_utils.py) |
| 确定性评测 | [eval.py](../eval.py) |
| 任务难度轴与使用约定 | [README.md](../README.md) |
