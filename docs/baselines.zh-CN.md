# 经典基线结果

**简体中文** · [English](baselines.md)

每个任务都附带一个非学习的 GNC 基线
（[algorithms/baselines/](../algorithms/baselines/baselines.py)），与策略共享同一环境、
同一奖励与同一成功门槛。控制器只读取 `'policy'` 观测组，构造时读取环境常数，
从不访问内部状态缓冲区。

下列数字全部来自 2026-11-12 的重新实测（每格 128 副本，协议种子 7/13/21 +
留出种子 101，静风、标称 plant）。含 Wilson 区间与失败模式的全量控制器表见
[baseline_test_report.md](baseline_test_report.md)；本轮应用的增益调整与回归修复
见 §A.7。

| 任务 | 基线 | 参考成功率 |
|---|---|---|
| QuadX-Waypoints | SE(3) 几何控制 + 积分配平 | 100%（128 副本，seed 13，shipped `baseline.json`）；协议种子 7/13/21 均值 99.7±0.5%，留出 101 = 100% |
| QuadX-Waypoints | 位置 PID → 姿态级联 | 100%（128 副本，seed 7/13/21/101；阻尼重调后 0 碰撞 / 0 出界） |
| QuadX-Track | CV 卡尔曼滤波 + 固定偏移 SE(3) | 75.8%（128 副本，seed 13）；协议种子均值 79.2±3.5%。任务现在包含**实体**圆柱遮挡物，与改动前 85.9% 的差距来自新增的碰撞/可见性约束 |
| Fixedwing-Waypoints | L1 横向 + TECS 纵向自驾 | 87.5%（128 副本，seed 13，3616 步窗口）；协议种子均值 87.0±3.2% |
| Quadplane-VTOL | SE(3) + 前向推进器 | 98.4%（128 副本，seed 13）；协议种子均值 98.2±0.5% |
| Rocket-Landing | PDG：ZEM/ZEV 制动 + 虚拟平台流形 + 脉冲伺服 | v2 判据下 100%（40 Hz，稳定停稳接触） |
| Satellite (CMG) | 四元数 PD + Schaub–Junkins 阻尼伪逆 | 100%（128 副本，四个种子） |
| Transfer | Gauss 单步贪心切向制导 | 100%（128 副本，四个种子） |
| Docking | CWH 离散 LQR + 护栏/燃料降额 | 100%（128 副本，四个种子） |
| Rendezvous | 两脉冲 CWH ZEM/PN，递归重解 | 100%（128 副本，四个种子） |
| Missile-Intercept | 增广比例导引（N = 15）→ 体轴角速率指令 | 71.1%（128 副本，seed 13）；协议种子均值 77.6±8.1% |

表中数字来自随仓库发布的 `results/<task>/baseline.json`（seed 13、128 副本、标称
plant、静风）；任务内变体位于 `results/<task>/baselines/`。

两点解读值得保留：

- 这些是参照系，不是稻草人。固定翼 `pid_cascade` 保留了有记录的出界/超时残余，
  导弹制导律在大交叉角几何下仍受权限饱和限制，`eguidance` / `ppn` / 四旋翼机模式
  切换臂是各自方法族的刻意下界消融。
- 旧笔记中的火箭 v1 数字（82.8% / 84.1%）是**瞬时**成功率，与 v2 的稳定停稳判据
  不可比。背景与复现命令见
  [env-rocket-landing.md](method/env-rocket-landing.md)。

成功率按任务、按控制器分别报告。它们不构成跨任务排名：任务难度、观测预算与成功判据
各不相同。

---

## 附录：多种子协议与完整变体数据

### A.1 数据来源与口径

两类产物，同一口径：

| 数据集 | 路径 | 并行副本 | 种子 | 记账 |
|---|---|---|---|---|
| A：协议多 seed | `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 128 | 7 / 13 / 21 / 101（held-out） | 每个副本只采信首回合，每 seed 128 个首回合 |
| B：控制器变体 | `results/<task>/baselines/<controller>_seed<seed>.json` | 128 | 7 / 13 / 21 / 101 | 每副本首回合 |

- 成功率 = `successes / episodes_finished`（每副本首回合记账）。
- 两个数据集都评测**标称 plant**：静风、默认难度、`domain_randomize = False`
  （冻结的 benchmark 契约）。需要与策略同条件的对照（域随机化开启）时，用
  `scripts/baseline_eval.py --domain-randomize on`。
- 步数窗口取各 JSON 的 `env_steps`。
- 每个种子 128 个首回合。

### A.2 默认控制器：协议种子 {7, 13, 21} + 留出种子 101

均值与标准差只对协议种子 7/13/21 计算；Wilson 95% 区间对三个种子的回合合并（384 回合）计算。

| 任务 | 默认控制器 | seed 7 | seed 13 | seed 21 | seed 101 | mean±sd (7/13/21) | [min,max] | Wilson 95% CI (pooled 384) | 平均 return | 回合 | 步数窗口 |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| quadx | `se3_geometric` | 100.0 | 100.0 | 99.2 | 100.0 | 99.7±0.5 | [99.2, 100.0] | [98.5, 100.0] | 72.5 | 384 | 672 |
| quadx_track | `cv_kf_pid` | 82.8 | 75.8 | 78.9 | 75.0 | 79.2±3.5 | [75.8, 82.8] | [74.8, 82.9] | 84.7 | 384 | 1216 |
| fixedwing | `l1_tecs` | 89.8 | 87.5 | 83.6 | 90.6 | 87.0±3.2 | [83.6, 89.8] | [83.2, 90.0] | 208.8 | 384 | 3616 |
| rocket | `zem_zev_pdg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 900.2 | 384 | 1216 |
| missile | `apn_rate` | 75.0 | 71.1 | 86.7 | 71.9 | 77.6±8.1 | [71.1, 86.7] | [73.2, 81.5] | 103.5 | 384 | 672 |
| quadplane | `se3_pusher` | 97.7 | 98.4 | 98.4 | 96.1 | 98.2±0.5 | [97.7, 98.4] | [96.3, 99.1] | 274.4 | 384 | 1216 |
| satellite | `quat_pd_damped_cmg` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 65.2 | 384 | 352 |
| transfer | `gauss_tangential` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 0.8 | 384 | 192 |
| docking | `cwh_lqr` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 190.6 | 384 | 288 |
| rendezvous | `cwh_two_impulse` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0±0.0 | [100.0, 100.0] | [99.0, 100.0] | 184.8 | 384 | 640 |

说明：
- `rocket` 按 v2 主指标 `ever_stable_2s`（稳定停稳接触）记账。
- `quadx_track` 成功判据为可见时间比例 ≥ 0.85 且无碰撞。
- 所有行均满足 `complete = true`。

### A.3 全部 30 个控制器变体（128 副本，seeds 7/13/21 + 101）

每行为协议种子 7/13/21 的均值；留出列为 seed 101。该表仅用于同任务内控制器排序。

| 任务 | 控制器 | 成功率 mean±sd | [min, max] | 留出 101 | 平均 return | 步数窗口 |
|---|---|---|---:|---:|---:|---:|
| quadx | `cascaded_pid` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 72.59 | 544 |
| quadx | `se3_geometric` | 99.7 ± 0.5 | [99.2, 100.0] | 100.0 | 72.49 | 672 |
| quadx | `smc` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 72.59 | 576 |
| quadx_track | `cv_kf_pid` | 79.2 ± 3.5 | [75.8, 82.8] | 75.0 | 84.72 | 1216 |
| quadx_track | `pid_ema` | 74.7 ± 4.0 | [70.3, 78.1] | 78.1 | 78.06 | 1216 |
| quadx_track | `pursuit` | 74.2 ± 4.3 | [69.5, 78.1] | 78.1 | 72.80 | 1216 |
| fixedwing | `l1_tecs` | 87.0 ± 3.2 | [83.6, 89.8] | 90.6 | 208.84 | 3616 |
| fixedwing | `nlgl_tecs` | 94.3 ± 1.2 | [93.0, 95.3] | 90.6 | 217.62 | 3616 |
| fixedwing | `pid_cascade` | 76.8 ± 4.6 | [73.4, 82.0] | 76.6 | 193.25 | 3616 |
| rocket | `eguidance` | 34.4 ± 5.1 | [28.9, 39.1] | 33.6 | −3304.80 | 1216 |
| rocket | `pd_landing` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 7.27 | 1216 |
| rocket | `zem_zev_pdg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 900.15 | 1216 |
| missile | `apn_rate` | 77.6 ± 8.1 | [71.1, 86.7] | 71.9 | 103.52 | 672 |
| missile | `ppn` | 2.3 ± 0.8 | [1.6, 3.1] | 2.3 | 27.58 | 928 |
| missile | `tpn` | 77.6 ± 7.4 | [71.9, 85.9] | 71.9 | 103.70 | 576 |
| quadplane | `hybrid_cascade` | 26.8 ± 2.4 | [24.2, 28.9] | 22.7 | 167.48 | 1216 |
| quadplane | `l1_wing` | 49.2 ± 5.6 | [43.0, 53.9] | 39.1 | 217.27 | 1216 |
| quadplane | `se3_pusher` | 98.2 ± 0.5 | [97.7, 98.4] | 96.1 | 274.35 | 1216 |
| satellite | `quat_pd_damped_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 65.22 | 352 |
| satellite | `quat_pd_mp_pinv_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 64.92 | 384 |
| satellite | `quat_pd_wpinv_cmg` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 65.41 | 352 |
| transfer | `edelbaum` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 0.20 | 192 |
| transfer | `gauss_tangential` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 0.77 | 192 |
| transfer | `q_law_petropoulos` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 1.01 | 192 |
| docking | `cwh_lqr` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 190.59 | 288 |
| docking | `cwh_pole_placement` | 89.6 ± 3.0 | [87.5, 93.0] | 86.7 | 169.30 | 224 |
| docking | `zem_zev` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 191.21 | 320 |
| rendezvous | `cwh_compensated_pd` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 179.29 | 928 |
| rendezvous | `cwh_two_impulse` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 184.79 | 640 |
| rendezvous | `glideslope` | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 | 180.82 | 864 |

### A.4 失败模式（每个控制器取最差协议种子）

| 任务 | 控制器 | crash | out-of-bounds | timeout |
|---|---|---:|---:|---:|
| quadx | `cascaded_pid` | 0.0% | 0.0% | 0.0% |
| quadx | `se3_geometric` | 0.0% | 0.7% | 0.0% |
| quadx | `smc` | 0.0% | 0.0% | 0.0% |
| quadx_track | `cv_kf_pid` | — | 0.0% | 8.5% |
| quadx_track | `pid_ema` | — | 0.0% | 23.4% |
| quadx_track | `pursuit` | — | 0.0% | 24.2% |
| fixedwing | `l1_tecs` | 0.0% | 2.2% | 5.2% |
| fixedwing | `nlgl_tecs` | 0.0% | 7.0% | 1.0% |
| fixedwing | `pid_cascade` | 0.0% | 12.9% | 6.8% |
| quadplane | `hybrid_cascade` | 0.0% | 4.7% | 71.1% |
| quadplane | `l1_wing` | 0.0% | 0.0% | 57.0% |
| quadplane | `se3_pusher` | 2.0% | 0.0% | 0.7% |
| missile | `apn_rate` | — | — | 0.0% |
| missile | `ppn` | — | — | 3.9% |
| missile | `tpn` | — | — | 0.0% |

- `quadx_track` 失败全部是超时（在实体圆柱之间保持可见时间比例 ≥ 0.85 不达标）。
- `quadplane` 弱臂以超时为主（模式切换自驾 30 s 内无法完成 4 航点）；
  `se3_pusher` 在最高速进近中仍保留少量碰撞残余。
- `rocket` / `satellite` / `transfer` / `docking` / `rendezvous` 未提供分项失败计数。

### A.5 口径边界

1. 没有 1024 副本的基线产物。`results/<task>/baseline.json` 统一为 128 副本 /
   seed 13；协议目录为 128 副本 × seeds 7/13/21/101。引用时必须同时注明
   `num_envs`、`seed`、`env_steps`。
2. 数据集 B 现在覆盖全部四个种子，因此每个控制器都有跨种子方差，不再只有默认臂。
3. 各任务窗口不同（`fixedwing` 3616，`missile` 576–928，`rocket` 1216，
   空间任务 192–928）。跨控制器比较必须注明窗口。

### A.6 复验命令

```bash
# 数据集 A：默认控制器，每个 seed 单独目录（标称 plant）
python scripts/baseline_eval.py <task> --num-envs 128 --seed 7   --max-steps 5000 --out results/baseline_protocol/seed7/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 13  --max-steps 5000 --out results/baseline_protocol/seed13/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 21  --max-steps 5000 --out results/baseline_protocol/seed21/<task>
python scripts/baseline_eval.py <task> --num-envs 128 --seed 101 --max-steps 5000 --out results/baseline_protocol/seed101/<task>

# 数据集 B：默认 + 全部变体，四个种子
python scripts/baseline_suite.py run --tasks <task> --num-envs 128 --seeds 7,13,21,101 --max-steps 5000

# 汇总报告（即上方各表）
python scripts/baseline_suite.py report --out docs/baseline_test_report.md

# 增益扫描（仅调参用；写入 results/tuning/<task>/<controller>/）
python scripts/baseline_tune.py <task> --grid '{"KX": [3, 4, 5], "KV": [2, 3]}' --seeds 7
python scripts/baseline_tune.py docking --grid '{"w_vel": [3, 8, 15]}' --seeds 7,13
python scripts/baseline_tune.py fixedwing --config-json --grid '{"lateral.l1_distance": [40, 50]}'
python scripts/baseline_tune.py transfer --design --grid '{"rp0": [8500000, 8800000]}'
```

可用的 `<variant>` CLI 键：

```bash
python -c "from algorithms.baselines.baselines import BASELINE_VARIANTS; print({k: sorted(v) for k, v in BASELINE_VARIANTS.items()})"
```

与策略同条件（域随机化开启）的评测在 `baseline_eval.py` 追加
`--domain-randomize on`。

### A.7 2026-11-12 调参与回归修复

| 项目 | 现象 | 根因 | 修改 |
|---|---|---|---|
| rocket 三个臂 | 0%（原 100%） | 2026-11-05 的“禁用接触约束求解”对所有机型生效，助推器穿过着陆台而无法停稳 | legacy 模式下火箭保持刚体接触求解器开启（`plant.py`） |
| docking `cwh_lqr` | 0.4% | 新增的空间站实体外壳使“质心对准原点”的进近把对接鼻锥推入舱体；`w_vel = 3` 在 0.02 m/s² 权限下制动过晚 | 终端目标改为让鼻锥停在端口平面（`docking_standoff`），`w_vel` 3 → 15 → 四个种子均 100% |
| quadx `cascaded_pid` | 64.1% | 2026-11-02 仅基于 CPU 的扫描把 `KD_POS` 降到 2.5，在 GPU plant 上欠阻尼（42.6% 出界） | 恢复 `KD_POS = 4.0` → 四个种子均 100% |
| quadplane `se3_pusher` | 88.1% | `A_MAX = 6.5` 无法在 30 s 内完成高/远航点 | `A_MAX` 6.5 → 12、`V_MAX` 17 → 20 → 98.2% |
| transfer `gauss_tangential` | 74.0% | 单步贪心欠交付近地点抬升，终端目标裕量为负 | 内部目标近地点抬高 300 km（`rp0` 8.5e6 → 8.8e6）→ 100% |
| `baseline_eval.py` | 口径歧义 | CLI 继承 `domain_randomize = None`（开启），而文档的参考协议是标称 plant | 新增 `--domain-randomize {off,on,default}`，默认 `off`；`run_5seed.py` 显式用 `on` 做同条件对照 |

`quadx_track`（静态偏移扫描均低于出厂的 1.5 m 偏移）、`fixedwing pid_cascade`、
`missile` APN/TPN（N ≥ 10 时指令均已饱和）、`rocket eguidance` 以及四旋翼机模式
切换消融臂未找到有效改进，保持其实测工作点。
