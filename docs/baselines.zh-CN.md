# 经典基线结果

**简体中文** · [English](baselines.md)

每个任务都附带一个非学习的 GNC 基线
（[algorithms/baselines/](../algorithms/baselines/baselines.py)），与策略共享同一环境、
同一奖励与同一成功门槛。控制器只读取 `'policy'` 观测组，构造时读取环境常数，
从不访问内部状态缓冲区。

| 任务 | 基线 | 参考成功率 |
|---|---|---|
| QuadX-Waypoints | SE(3) 几何控制 + 积分配平 | 90.6%（32 副本，seed 7）；多种子协议均值 99.7%（128 副本 × seed 7/13/21；留出 seed 101 = 100%） |
| QuadX-Waypoints | 位置 PID → 姿态级联 | 99.2%（256 副本，seed 7，0 碰撞 / 0 出界） |
| QuadX-Track | CV 卡尔曼滤波 + 固定偏移 SE(3) | 82.8%（128 副本，seed 21）；可见时间 90.2%，间距 0.258 m |
| Fixedwing-Waypoints | L1 横向 + TECS 纵向自驾 | 97.0%（128 副本，seed 13；窗口在 120 s 时限前结束） |
| Quadplane-VTOL | SE(3) + 前向推进器 | 89.1%（128 副本，seed 13） |
| Rocket-Landing | PDG：ZEM/PN 制动 + 虚拟平台流形 + 脉冲伺服 | v2 判据下 100%（40 Hz，连续停稳 2 s） |
| Satellite (CMG) | 四元数 PD + Schaub–Junkins 阻尼伪逆 | 100%（128 副本，seed 13） |
| Transfer | Gauss 单步贪心切向制导 | 100%（128 副本，seed 13） |
| Docking | CWH 离散 LQR + 护栏/燃料降额 | 100%（128 副本，seed 13） |
| Rendezvous | 两脉冲 CWH ZEM/PN，递归重解 | 100%（128 副本，seed 13） |
| Missile-Intercept | 增广比例导引（N = 15）→ 体轴角速率指令 | 71.1%（128 副本，seed 13） |

表中数字来自随仓库发布的 `results/<task>/baseline.json`；每个文件记录自己的副本数、
种子与步数窗口。任务内消融（256 副本，seed 7）位于 `results/<task>/baselines/`。

两点解读值得保留：

- 这些是参照系，不是稻草人。固定翼与火箭基线保留了有记录的残余失败
  （末端极限环、权限饱和制动），学习策略正是被期望去攻击这些失败。
- 旧笔记中的火箭 v1 数字（82.8% / 84.1%）是**瞬时**成功率，与 v2 的稳定停稳判据
  不可比。背景与复现命令见
  [env-rocket-landing.md](method/env-rocket-landing.md)。

成功率按任务、按控制器分别报告。它们不构成跨任务排名：任务难度、观测预算与成功判据
各不相同，数字也来自不同的副本数与种子。

---

## 附录：多种子协议与完整变体数据

### A.1 数据来源与口径

两类产物对应两种口径：

| 数据集 | 路径 | 并行副本 | 种子 | 记账 |
|---|---|---|---|---|
| A：协议多 seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101（held-out） | 每个副本只采信首回合，每 seed 128 个首回合 |
| B：控制器变体 | `results/<task>/baselines/<controller>_seed7.json` | 256 | 7 | 每副本首回合，单 seed |

- 成功率 = `successes / episodes_finished`（每副本首回合记账）。
- 数据集 A、B 的 JSON 中 `env_kwargs` 均为 `{}`（静风、默认难度）。
- 步数窗口取各 JSON 的 `env_steps`。
- 成功率分母为 128（A）或 256（B）个首回合。

### A.2 默认控制器：协议种子 {7, 13, 21} + 留出种子 101

均值与标准差只对协议种子 7/13/21 计算；Wilson 95% 区间对三个种子的回合合并（384 回合）计算。

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
- `rocket` 按 v2 主指标 `ever_stable_2s`（连续停稳 2 s）记账。
- `quadx_track` 成功判据为可见时间比例 ≥ 0.85 且无碰撞。
- 所有行均满足 `complete = true`。

### A.3 全部 30 个控制器变体（seed 7，256 副本）

数据集 B 为单 seed，不提供跨种子区间；仅用于同任务内控制器排序。

| 任务 | 控制器 | 成功率 | 平均 return | 步数窗口 |
|---|---|---|---:|---:|
| quadx | `cascaded_pid` | 99.2% | 72.74 | 800 |
| quadx | `se3_geometric` | 100.0% | 72.86 | 648 |
| quadx | `smc` | 100.0% | 72.86 | 557 |
| quadx_track | `cv_kf_pid` | 85.2% | 85.68 | 1200 |
| quadx_track | `pid_ema` | 75.8% | 75.98 | 1200 |
| quadx_track | `pursuit` | 75.8% | 71.85 | 1200 |
| fixedwing | `l1_tecs` | 98.6% | 230.09 | 1500 |
| fixedwing | `nlgl_tecs` | 94.5% | 222.94 | 1500 |
| fixedwing | `pid_cascade` | 91.6% | 219.07 | 1500 |
| rocket | `eguidance` | 34.4% | −3332.01 | 1200 |
| rocket | `pd_landing` | 100.0% | 22.34 | 1200 |
| rocket | `zem_zev_pdg` | 100.0% | 907.31 | 1200 |
| missile | `apn_rate` | 76.2% | 101.90 | 637 |
| missile | `ppn` | 2.0% | 27.21 | 900 |
| missile | `tpn` | 76.2% | 102.13 | 580 |
| quadplane | `hybrid_cascade` | 26.6% | 173.94 | 1200 |
| quadplane | `l1_wing` | 49.6% | 226.20 | 1200 |
| quadplane | `se3_pusher` | 87.5% | 271.97 | 1200 |
| satellite | `quat_pd_damped_cmg` | 100.0% | 64.85 | 330 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0% | 64.54 | 366 |
| satellite | `quat_pd_wpinv_cmg` | 100.0% | 65.03 | 337 |
| transfer | `edelbaum` | 100.0% | 0.22 | 166 |
| transfer | `gauss_tangential` | 100.0% | 0.01 | 166 |
| transfer | `q_law_petropoulos` | 100.0% | 1.15 | 166 |
| docking | `cwh_lqr` | 100.0% | 188.62 | 274 |
| docking | `cwh_pole_placement` | 100.0% | 189.30 | 236 |
| docking | `zem_zev` | 100.0% | 190.06 | 313 |
| rendezvous | `cwh_compensated_pd` | 100.0% | 179.44 | 875 |
| rendezvous | `cwh_two_impulse` | 100.0% | 184.91 | 594 |
| rendezvous | `glideslope` | 100.0% | 180.96 | 805 |

### A.4 失败模式（数据集 B，seed 7）

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

- `quadx_track` 失败全部是超时。
- `quadplane` 弱臂以超时为主（模式切换自驾 30 s 内无法完成 4 航点）。
- `rocket` / `satellite` / `transfer` / `docking` / `rendezvous` 未提供分项失败计数。

### A.5 口径边界

1. 没有 1024 副本的基线产物。`results/<task>/baseline.json` 元数据各异
   （`quadx` 32 副本/seed 7，其余多为 128/seed 13，`quadx_track` seed 21）。
   引用时必须同时注明 `num_envs`、`seed`、`env_steps`。
2. 数据集 B 只有 seed 7；§A.2 才是跨种子方差来源。
3. 变体 JSON 窗口不统一（`fixedwing` 1500 步，`missile` 580–900，`rocket` 1200）。
   跨变体比较必须注明窗口。

### A.6 复验命令

```bash
# 数据集 A：默认控制器，每个 seed 单独目录
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --out results/baseline_protocol/seed101/<task>

# 数据集 B：默认 + 全部变体，写入 results/<task>/baselines/<controller>_seed7.json
python scripts/baseline_suite.py run --tasks <task> --num-envs 256 --seeds 7
```

可用的 `<variant>` CLI 键：

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```
