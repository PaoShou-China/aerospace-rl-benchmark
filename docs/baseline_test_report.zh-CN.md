# 传统（非学习）基线完整测试报告

**简体中文** · [English](baseline_test_report.md)

## 1. 数据来源与口径

本报告只使用仓库内**实际存在**的 JSON 产物，不沿用任何未落盘的旧数字。两类产物对应两种口径：

| 数据集 | 路径 | 并行副本 | 种子 | 记账 |
|---|---|---|---|---|
| A：协议多 seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101（held-out） | 每个副本只采信首回合（`FirstEpisodes`），每 seed 128 个首回合 |
| B：控制器变体 | `results/<task>/baselines/<controller>_seed7.json` | 256 | 7 | 每副本首回合，单 seed |

统一口径：

- 成功率 = `successes / episodes_finished`（即每个副本首回合的 `completed_successes / completed_episodes` 记账）。
- 数据集 A、B 的 JSON 中 `env_kwargs` 均为 `{}`，即静风、任务默认难度。生成脚本的口径不同：`scripts/baseline_suite.py`（数据集 B）的当前版本还会显式写入 `domain_randomize=False`，因此重新生成的 B 类产物会带该键；已随仓库发布的旧产物仍是 `{}`。引用任何产物时必须同时引用其 `env_kwargs`。
- 步数窗口取各 JSON 的 `env_steps`（不是统一硬编码值）。
- 成功率分母为 128（A）或 256（B）个首回合；`episodes_completed_total` 可能更大，因为副本在首回合结束后仍会继续推进直到安全上限。

> **与旧版报告的差异**：旧版声称 “1024 并行环境、3 seeds、3072 回合”。当前仓库中不存在 1024 副本的基线产物：数据集 A 全部为 128 副本，数据集 B 为 256 副本（单 seed）。因此 1024/3072 的样本量声明已删除，改为按实际 JSON 元数据报告。

## 2. 默认控制器：协议种子 {7, 13, 21} + 留出种子 101

每个任务使用 `BUILDERS` 中注册的默认基线。均值与标准差只对协议种子 7/13/21 计算；Wilson 95% 区间对三个种子的回合合并（384 回合）计算。

| 任务 | 默认控制器 | seed 7 | seed 13 | seed 21 | seed 101 | mean±sd (7/13/21) | [min,max] | Wilson 95% CI (pooled 384) | 平均 return | 回合 | 步数窗口 |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| quadx | `se3_geometric` | 100.0 | 100.0 | 99.2 | 100.0 | 99.7±0.5 | [99.2, 100.0] | [98.5, 100.0] | 72.5 | 384 | 648 |
| quadx_track | `cv_kf_pid` | 89.1 | 85.9 | 82.8 | 82.0 | 85.9±3.1 | [82.8, 89.1] | [82.1, 89.1] | 86.5 | 384 | 1200 |
| fixedwing | `l1_tecs` | 89.8 | 87.5 | 83.6 | 90.6 | 87.0±3.2 | [83.6, 89.8] | [83.2, 90.0] | 208.8 | 384 | 3600 |
| rocket | `zem_zev_pdg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 899.5 | 384 | 1200 |
| missile | `apn_rate` | 75.0 | 71.1 | 85.9 | 70.3 | 77.3±7.7 | [71.1, 85.9] | [72.9, 81.2] | 103.3 | 384 | 607 |
| quadplane | `se3_pusher` | 85.9 | 89.1 | 92.2 | 85.2 | 89.1±3.1 | [85.9, 92.2] | [85.5, 91.8] | 268.6 | 384 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 65.2 | 384 | 320 |
| transfer | `gauss_tangential` | 100.0 | 100.0 | 97.7 | 100.0 | 99.2±1.4 | [97.7, 100.0] | [97.7, 99.7] | 0.0 | 384 | 166 |
| docking | `cwh_lqr` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 189.7 | 384 | 267 |
| rendezvous | `cwh_two_impulse` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 184.8 | 384 | 601 |

说明：

- `rocket` 的成功率按 v2 主指标 `ever_stable_2s` 记账（连续停稳 2 s），不是旧版瞬时 `ever_success`。
- `quadx_track` 的成功判据为可见时间比例 ≥ 0.85 且无碰撞；留出种子 101 的 82.0% 是三个协议种子之外的独立确认。
- 所有行均满足 `complete = true`（每个副本的首回合在窗口内结束）。

## 3. 全部 30 个控制器变体（seed 7，256 副本）

数据集 B 为单 seed 单次评测，**不提供跨种子区间**；它用于同任务内的控制器排序，不应与第 2 节的 3-seed 均值混用。成功率的分母固定为 256（每副本首回合），表末的“总回合”是 `episodes_completed_total`，仅用于说明窗口内额外推进了多少回合。

| 任务 | 控制器 | 成功率 | 平均 return | 总回合 (`episodes_completed_total`) | 步数窗口 |
|---|---|---|---:|---:|---:|---:|
| quadx | `cascaded_pid` | 99.2% | 72.74 | 257 | 800 |
| quadx | `se3_geometric` | 100.0% | 72.86 | 278 | 648 |
| quadx | `smc` | 100.0% | 72.86 | 267 | 557 |
| quadx_track | `cv_kf_pid` | 85.2% | 85.68 | 256 | 1200 |
| quadx_track | `pid_ema` | 75.8% | 75.98 | 256 | 1200 |
| quadx_track | `pursuit` | 75.8% | 71.85 | 256 | 1200 |
| fixedwing | `l1_tecs` | 98.6% | 230.09 | 280 | 1500 |
| fixedwing | `nlgl_tecs` | 94.5% | 222.94 | 292 | 1500 |
| fixedwing | `pid_cascade` | 91.6% | 219.07 | 188 | 1500 |
| rocket | `eguidance` | 34.4% | −3332.01 | 267 | 1200 |
| rocket | `pd_landing` | 100.0% | 22.34 | 256 | 1200 |
| rocket | `zem_zev_pdg` | 100.0% | 907.31 | 256 | 1200 |
| missile | `apn_rate` | 76.2% | 101.90 | 480 | 637 |
| missile | `ppn` | 2.0% | 27.21 | 704 | 900 |
| missile | `tpn` | 76.2% | 102.13 | 488 | 580 |
| quadplane | `hybrid_cascade` | 26.6% | 173.94 | 257 | 1200 |
| quadplane | `l1_wing` | 49.6% | 226.20 | 256 | 1200 |
| quadplane | `se3_pusher` | 87.5% | 271.97 | 258 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0% | 64.85 | 258 | 330 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0% | 64.54 | 260 | 366 |
| satellite | `quat_pd_wpinv_cmg` | 100.0% | 65.03 | 259 | 337 |
| transfer | `edelbaum` | 100.0% | 0.22 | 256 | 166 |
| transfer | `gauss_tangential` | 100.0% | 0.01 | 256 | 166 |
| transfer | `q_law_petropoulos` | 100.0% | 1.15 | 256 | 166 |
| docking | `cwh_lqr` | 100.0% | 188.62 | 261 | 274 |
| docking | `cwh_pole_placement` | 100.0% | 189.30 | 257 | 236 |
| docking | `zem_zev` | 100.0% | 190.06 | 256 | 313 |
| rendezvous | `cwh_compensated_pd` | 100.0% | 179.44 | 256 | 875 |
| rendezvous | `cwh_two_impulse` | 100.0% | 184.91 | 256 | 594 |
| rendezvous | `glideslope` | 100.0% | 180.96 | 256 | 805 |

注：`fixedwing` 的 `env_steps = 1500` 小于 3600 步任务时域，因此 256 个副本中只有 216 / 237 / 179 个在窗口内完成首回合（`episodes_finished`），成功率分母就是这些已完成首回合；表中“总回合”更大是因为已完成的副本继续推进。`missile` 与 `rocket` 的总回合数由任务自身终止条件决定（如 `ppn` 大量超时导致更多回合在窗口内滚动）。

## 4. 失败模式（数据集 B 的 `task_metrics`，seed 7）

| 任务 | 控制器 | crash | out-of-bounds | timeout |
|---|---|---:|---:|---:|
| quadx | `cascaded_pid` | 0.0% | 0.0% | 0.8% |
| quadx | `se3_geometric` | 0.0% | 0.0% | 0.0% |
| quadx | `smc` | 0.0% | 0.0% | 0.0% |
| quadx_track | `cv_kf_pid` | — | 0.0% | 14.8% |
| quadx_track | `pid_ema` | — | 0.0% | 24.2% |
| quadx_track | `pursuit` | — | 0.0% | 24.2% |
| fixedwing | `l1_tecs` | 0.0% | 2.5% | 0.0% |
| fixedwing | `nlgl_tecs` | 0.0% | 6.8% | 0.0% |
| fixedwing | `pid_cascade` | 0.0% | 10.1% | 0.0% |
| quadplane | `hybrid_cascade` | 0.0% | 4.7% | 68.5% |
| quadplane | `l1_wing` | 0.0% | 0.0% | 50.4% |
| quadplane | `se3_pusher` | 0.4% | 0.4% | 12.0% |
| missile | `apn_rate` | — | — | 0.0% |
| missile | `ppn` | — | — | 3.1% |
| missile | `tpn` | — | — | 0.0% |

- `quadx_track` 的失败全部是超时（未在 30 s 内达到可见时间比例），不是碰撞或出界。
- `quadplane` 的 `hybrid_cascade` / `l1_wing` 失败以超时为主（68.5% / 50.4%），说明模式切换自驾在 30 s 内无法稳定完成 4 个航点；`se3_pusher` 则主要在接近航点时超时。
- `rocket` / `satellite` / `transfer` / `docking` / `rendezvous` 的 JSON 未提供分项失败计数，故不列表。

## 5. 方法族谱（数据集 B 标签；括号内为 `--baseline` CLI 键，若存在）

| 任务 | 控制器（标签 / CLI 变体键） | 方法 |
|---|---|---|
| quadx | `se3_geometric` | Lee SE(3) 几何控制（默认，`BUILDERS`） |
| quadx | `cascaded_pid` | 位置→姿态级联 PID（Euler） |
| quadx | `smc` | 滑模位置外环（指数趋近律 + 边界层）+ Euler 姿态级联 |
| quadx_track | `cv_kf_pid` | CV 卡尔曼滤波估计目标 + SE(3) 跟踪（默认） |
| quadx_track | `pid_ema` | 一阶 EMA 目标滤波 + SE(3) 跟踪 |
| quadx_track | `pursuit` | 无滤波纯追踪（原始含噪相对测量） |
| fixedwing | `l1_tecs` | L1 横向制导 + TECS 纵向（默认） |
| fixedwing | `nlgl_tecs` | Park 非线性制导律（NLGL）+ TECS |
| fixedwing | `pid_cascade` | 常规级联 PID 姿态/高度/速度 |
| rocket | `zem_zev_pdg` | ZEM/ZEV 中段 + 虚拟平台 PDG + 速度带脉冲伺服（默认） |
| rocket | `pd_landing` | 纯 PD 着陆（无制导层） |
| rocket | `eguidance` | Apollo E-guidance 二次多项式加速制导 + 终端脉冲伺服 |
| missile | `apn_rate` | 增广比例导引 APN（N=15）+ half-N 重力补偿（默认） |
| missile | `tpn` | 真比例导引 TPN（无 (N/2)g 增广） |
| missile | `ppn` | 纯追踪导引（无前置量） |
| quadplane | `se3_pusher` | 四旋翼 SE(3) 全程 + 前向推进器分配（默认） |
| quadplane | `l1_wing` | 机翼模式 L1 横向 + 四旋翼低速助推 |
| quadplane | `hybrid_cascade` | 模式切换混合自驾（低速四旋翼/高速机翼），含机翼升力卸荷项 |
| satellite | `quat_pd_damped_cmg` | 四元数 PD + Schaub-Junkins 阻尼伪逆 CMG（默认） |
| satellite | `quat_pd_mp_pinv_cmg` | 四元数 PD + 普通 Moore–Penrose 伪逆 CMG |
| satellite | `quat_pd_wpinv_cmg` | 四元数 PD + Oh–Fleming 加权阻尼最小范数 + Vadali 零空间奇异规避 |
| transfer | `gauss_tangential` | Gauss 行星方程切向推力单步贪心（默认） |
| transfer | `edelbaum` | Edelbaum 解析 Δv 预算开环切向推力 |
| transfer | `q_law_petropoulos`（CLI `q_law`） | Petropoulos Q-law Lyapunov 制导 |
| docking | `cwh_lqr` | CWH 全状态 LQR + 走廊/捕获网/燃料护栏（默认） |
| docking | `cwh_pole_placement`（CLI `pole_placement`） | CWH 极点配置（无附加约束层） |
| docking | `zem_zev` | CWH ZEM/ZEV 单脉冲滚动时域制导 + 燃料/距离护栏 |
| rendezvous | `cwh_two_impulse` | CWH 滚动时域两脉冲终端制导（默认） |
| rendezvous | `cwh_compensated_pd`（CLI `cwh_pd`） | CWH 补偿 PD 直趋原点 |
| rendezvous | `glideslope` | 距离—速率刹车走廊（glideslope）V-bar 逼近 |

> 数据集 B 的文件名与 `controller` 字段来自控制器类属性 `name`，而 `baseline_eval.py:82-85` 的 `--baseline` 只接受 `BASELINE_VARIANTS` 的字典键。两者并不总是同名：`q_law_petropoulos` ↔ `q_law`、`cwh_compensated_pd` ↔ `cwh_pd`、`cwh_pole_placement` ↔ `pole_placement`。卫星三个臂的标签与变体键同名；其余任务的默认控制器（`se3_geometric`、`cv_kf_pid`、`l1_tecs`、`zem_zev_pdg`、`apn_rate`、`se3_pusher`、`gauss_tangential`、`cwh_lqr`、`cwh_two_impulse`）来自 `BUILDERS`，没有 `--baseline` 键，复现时应省略该参数。

## 6. 每任务最优经典方法（数据集 B，seed 7）

| 任务 | 最优 | 次优 | 最弱 | 极差 |
|---|---|---|---|---|
| quadx | `se3_geometric` / `smc` (100.0%) | `cascaded_pid` (99.2%) | `cascaded_pid` | 0.8 pp |
| quadx_track | `cv_kf_pid` (85.2%) | `pid_ema` / `pursuit` (75.8%) | `pid_ema` / `pursuit` | 9.4 pp |
| fixedwing | `l1_tecs` (98.6%) | `nlgl_tecs` (94.5%) | `pid_cascade` (91.6%) | 7.0 pp |
| rocket | `zem_zev_pdg` / `pd_landing` (100.0%) | `eguidance` (34.4%) | `eguidance` | 65.6 pp |
| missile | `apn_rate` / `tpn` (76.2%) | `ppn` (2.0%) | `ppn` | 74.2 pp |
| quadplane | `se3_pusher` (87.5%) | `l1_wing` (49.6%) | `hybrid_cascade` (26.6%) | 60.9 pp |
| satellite | 三个 CMG 臂均 100.0% | — | — | 0.0 pp |
| transfer | 三个臂均 100.0% | — | — | 0.0 pp |
| docking | 三个臂均 100.0% | — | — | 0.0 pp |
| rendezvous | 三个臂均 100.0% | — | — | 0.0 pp |

## 7. 口径边界与已知缺口

1. **没有 1024 副本的基线产物**。`results/<task>/baseline.json` 的元数据各自不同：`quadx` 为 32 副本 / seed 7，其余任务多为 128 副本 / seed 13，`quadx_track` 为 seed 21。引用这些文件时必须同时引用 `num_envs`、`seed`、`env_steps`，不能统一写成 “1024 envs, seed 7”。
2. **数据集 B 只有 seed 7**。它不能提供跨种子方差；第 2 节的 3-seed 表才是默认控制器的方差来源。
3. **`results/quadx/baseline_cascaded_pid.json` 是旧口径遗留**：512 副本、seed 7、成功率 0.119，与同控制器的 `results/quadx/baselines/cascaded_pid_seed7.json`（256 副本、99.2%）相差极大。前者早于级联 PID 的修复，不应再被引用。
4. **变体 JSON 的窗口不统一**：`fixedwing` 为 1500 步（任务时域 3600 步），`missile` 为 580–900 步，`rocket` 为 1200 步。跨变体比较成功率应注明窗口。
5. **未落盘的专项 harness 报告**。以下路径在当前仓库中不存在，方法文档引用它们时必须标注“未随仓库发布”，不能当作可复验结果：
   - `results/rocket/pdg_test_report.json`、`results/rocket_pd_tune/tune_report.json`
   - `results/fixedwing/autopilot_report.json`、`results/fixedwing_pid/comparison.json`、`results/fixedwing_pid/scenarios_pid.json`、`results/fixedwing_ppo/checkpoint.pt`
   - `results/quadplane/cascade_report.json`
   - `results/quadx/cascade_pid_tests.json`、`results/quadx/config.json`、`results/quadx/run.json`
   - `results/quadx_track/kf_tuning.json`、`results/quadx_track/ppo/config.json`、`results/quadx_track/ppo/eval.json`
   - `results/satellite/steering_compare.json`、`results/satellite/steering_compare.md`、`results/satellite_ppo/checkpoint.pt`
   - `results/transfer/comparison.json`、`results/transfer/baseline_q_law_petropoulos.json`
   - `results/missile/scenarios.json`
   与这些路径对应的可复验替代品是数据集 A/B 的 JSON（第 2、3 节）以及 `docs/method/` 中标注为 shipped 的产物。
6. **旧版 3072 回合 / 3×1024 的声明已废弃**。若要恢复该样本量，需要重新运行 `scripts/baseline_eval.py --num-envs 1024` 并为每个 seed 单独落盘。

## 8. 复验命令

```bash
# 数据集 A：默认控制器，每个 seed 单独目录；baseline_eval 在 --out 下写 baseline.json
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --out results/baseline_protocol/seed101/<task>

# 数据集 B：默认控制器 + 全部变体，写入 results/<task>/baselines/<controller>_seed7.json
python scripts/baseline_suite.py run --tasks <task> --num-envs 256 --seeds 7
```

注意 `baseline_eval.py --baseline <variant>` 会把报告写成 `results/<task>/baseline_<name>.json`（`<name>` 是控制器类属性），而不是数据集 B 的 `baselines/<name>_seed7.json`；要复现数据集 B 的目录结构应使用 `baseline_suite.py`。可用的 `<variant>` CLI 键见第 5 节，或运行：

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```
