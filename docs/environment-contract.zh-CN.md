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

设计主张及其验证条件见 [benchmark_report.zh-CN.md 附录 A](../benchmark_report.zh-CN.md#附录-a基础设施与可复验性)。

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

## 真实度预设

五个大气任务接受 `aero="legacy" | "realistic"`（默认 `"legacy"`）。
`"legacy"` 即已发布基准：海平面密度、不可压升力面、理想传感器、无限舵速、
一阶 Dryden，逐位复现。`"realistic"` 在不改动冻结 URDF/YAML 资产的前提下
启用整套高保真模型：

- **ISA 大气**（`environments/common/atmosphere.py`）：0–20 km 内的密度、压力、
  温度与声速。动压及所有体阻力项使用逐副本当地密度；4 km 高度的导弹因此看到
  `rho ≈ 0.819 kg/m^3`，而非 1.225。realistic 预设再叠加 +15 K、1% 比湿的
  非标准日（`rho = p/(R T (1 + 0.6078 q))`），密度约降 5%、声速升高；
  legacy/标准表路径保持标准日。
- **压缩性与展向效率**：`M<0.9` 的 Prandtl-Glauert 升力线斜率增长、跨声速
  混合带、`M>1.2` 的线性超声速 `4/sqrt(M^2-1)` 衰减、波阻乘子，以及 Oswald
  诱导阻力因子（机翼 0.85、尾翼 0.90、导弹尾舵 0.85；栅格翼保持 `e=1`）。
- **展向条带**：realistic 下每个升力面沿展向分为若干站点（固定翼/复合翼 6 个，
  导弹与栅格舵 5 个）。每个站点承担 `1/S` 面积，独立计算自身空速（含刚体转动
  `omega x r`）、动压（来流 + 按几何解算的旋翼滑流）、当地马赫数、舵偏与
  Beddoes-Leishman 失速状态；力与站点的 `r x F` 力矩求和回父面。由此得到物理
  滚转/俯仰阻尼（条带极限下 `C_l_p = -C_l_alpha/6`）以及展向失速/滑流渐进。
  单站点即逐位复现 legacy 的单点模型。
- **寄生阻力与非定常升力**：固定翼机身/起落架等效平板面积 `0.010 m^2`
  （每体轴）；升力/阻力/力矩按对流时间 `c/V`（10–200 ms）做一阶
  Wagner 型滞后；并加入 Beddoes-Leishman 型动态失速修正：滞后后缘分离点
  与由分离速率驱动的涡升力状态。修正以准定常极曲线的增量形式叠加，因此
  稳态附着流与失速后系数保持不变，只增加俯仰上/下扫的滞回与动态升力过冲。
  涡还带来阻力增量（`cn_v sin(alpha)`）与更强的低头力矩增量，二者在分离
  状态恒定时同样为零。
- **作动器**：逐舵面速率限幅（固定翼 250–300 deg/s、导弹 800 deg/s、
  栅格舵 200 deg/s）、1% 死区、一阶滞后、0–2 个物理步的指令传输延迟、
  机械回差与正反向速率不对称；仍未建模铰链力矩/负载依赖偏转。
- **传感器**（`environments/common/sensors.py`）：逐副本陀螺噪声与零偏随机游走、
  姿态/速度/位置噪声、导弹相对状态与 LOS 的导引头噪声，以及一步策略传输延迟
  （40 Hz 的 27 g quadx 关闭延迟，否则会破坏其稳定性）。σ 按连续时间噪声密度
  处理并以 `sqrt(dt_ref/dt)` 缩放，使 30/40/60 Hz 任务的噪声带宽可比；终端
  bootstrap 观测也走与策略相同的传感器链路。
- **推进**：螺旋桨推力按 `(rho/rho0)^(2/3)` 缩放；定距桨按叶素理论的二次
  推力曲线 `C_T(J) = C_T0 (1 - (J/J0)^2)`（`J = V_axial/(n D)`，`J0 = pitch/D`）
  随轴向来流掉推力，即空速 `V` 下保留静态推力的
  `1 - (V/(pitch * n_max * throttle))^2`（在随油门变化的桨距速度处归零）。
  三个电动平台带逐副本电池包（串联节数、容量、内阻、航电负载）：端电压随
  瞬时电流下垂并随荷电状态下降，可用推力按 `(V/V_full)^2` 缩放，电量按动量
  理论轴功率除以品质因数做库仑积分。两者都是回合内时变（长航时会掉电、掉
  推力）。复合翼滑流改为按实际旋翼/翼面几何计算（动量理论轴向发展
  `q = (T/A)/4 (1 + x/sqrt(x^2+R^2))^2`，外加强斯径向扩散），不再使用
  按主翼面积覆盖比例集总的近似；火箭/导弹加入环境压力喷管推力修正；
  导弹在 4 s 助推中烧掉 30% 推进剂质量。
- **气动弹性**：固定翼/复合翼升力面带一阶集中扭转模态（8 Hz、15% 阻尼，
  弹性轴在 35% 弦长处），由当地俯仰力矩与升力驱动。刚度由 100 m/s 的线性
  发散速度反推，因此任务包线内弹性扭转小于 0.2°，超过发散速度后发散
  （20° 限幅作为数值/结构止挡）。展向弯曲未建模。
- **机翼地效**：每个升力面按站点高度 h 与全展长 b 用升力线镜像涡解修正：
  `phi = (16h/b)^2/(1+(16h/b)^2)`（自由空间 1，贴地 0），升力线斜率变为
  `cl_alpha/(1+cl_alpha phi/(pi AR))`，诱导阻力乘 `phi`，并加入
  `-0.1(1-phi)cl` 的低头力矩增量。修正按展向站点计算，因此近地的带坡度机翼
  也会得到物理滚转力矩；与多旋翼的 Cheeseman-Bennett 推力因子相互独立。
- **多旋翼细节**：线性旋翼阻力、旋翼陀螺力矩，以及逐旋翼按各自世界高度
  计算的 Cheeseman-Bennett 地效因子（俯仰/滚转下仍然正确），不再使用机体质心
  高度。
- **导弹目标与导引头**：运动学目标的横向加速度上限 15 g，导引头观测带噪声；
  30% 推进剂燃烧同时缩放 base link 质量与惯性张量（质心移动仍未建模）。
  弹体导引头视场半角 60 deg、作用距离 20 km：视场外或超距时相对状态/LOS/
  接近速度整块置零并丢失锁定（传感器噪声无法重新注入信息）。测量块还经过
  一阶跟踪滤波（tau = 20 ms），并带天线罩斜率偏差（0.02 rad / rad 迎角或
  侧滑角）：视在 LOS 朝机头方向旋转，形成经典的寄生 LOS 速率回路。目标还带有点质量
  能量模型：抛物线阻力 `q S CD0 + (n m g)^2/(q S pi AR e)`、随密度衰减的
  可用推力与比例速度保持，因此持续高过载机动会掉速，而不是永远保持 250 m/s。
- **接触求解**：所有 realistic 大气任务都启用 Genesis 接触约束求解器，接触面
  之间会交换法向、摩擦与恢复冲量，而不再只置接触标志；legacy 路径保持仅检测
  以复现已发布基准。火箭着陆台因此可以真实支撑、倾倒并提供摩擦耗散；v2 成功
  判据要求连续稳定接触 0.5 s，而不是单次触地。
- **湍流与风切变**：精确 MIL-F-8785C Dryden 滤波器（纵向一阶、横向/垂向
  二阶）、随高度变化的尺度长度，以及幂律平均风剖面（`alpha=0.14`，参考
  高度 10 m）。realistic 任务每个控制步都按逐副本空速与高度重建滤波器
  （纯张量、可图形捕获），使 `L = L(h)`、`σ = σ(h)` 与带宽 `V/L` 跟随爬升或
  加速中的飞行器，而不是固定在构造时的参考条件。
- **域随机化**：`param_mismatch` 在构造时施加一次静态倍率，并把未扰动的
  SI 标称写入 `controller_design`；`domain_randomize` 每个副本每回合独立
  均匀采样（`False`/`{}` 关闭，字典在默认表上合并）。除 thrust/mass 外，
  realistic 默认表新增：
  - `mass`：缩放所有结构 link **及其惯量**（燃料/推进剂 link 保留各自的
    物理消耗模型），因此 ±10% 是总质量变化，而不是旧实现只改 body link 的
    ~1.3%；
  - `cl_alpha` 0.92–1.08、`cd0` 0.85–1.15、`stall` 0.90–1.10、
    `oswald` 0.95–1.05，四条独立气动随机轴；
  - `actuator_deadband` 0.5–2.0、`actuator_backlash` 0.5–2.0、
    `actuator_asymmetry` 0.7–1.3、`actuator_delay` 0–2 个物理步；
  - `sensor` 0.8–1.2（全部噪声 σ 与零偏游走）；
  - `cg`/`cg_y`/`cg_z`：沿机体 `+x/+y/+z` 的刚性质心偏移（满量程 =
    载具特征长度的 10%），无人机 0.9–1.1、导弹 0.95–1.05、火箭 0.98–1.02；
  - `inertia` 与 `inertia_x/y/z`：整体与逐对角分量惯量缩放，0.9–1.1。
  解析后的范围表写入 `cfg["domain_randomize"]`；每次 reset 后实际施加的
  逐副本倍率通过 `extras["dr_scales"]` 暴露，便于失败归因。legacy 基准表
  保持不变。

训练入口自 2026-11-10 起传入 `aero="realistic"`，新 run 会把它写入
`config.json` / `run.json`，`eval.py` 会原样恢复。每个 run 还会记录
`realism_version`（`"legacy"` 或当前预设版本）；修改任何 realistic 常量或
模型时需同步提升 `plant.py` 中的 `REALISM_VERSION`。需要复现已发布基准数值时
显式传 `aero="legacy"`。两种模式都仍未建模：机翼弯曲/着陆腿压溃（一阶扭转
已建模）、无人机
电池热效应/老化与燃油耗尽、雷诺数效应、舵机铰链力矩、涡升力以外的非定常阻力/
力矩滞回、相邻旋翼尾流干扰、站点间的展向诱导速度耦合（每个站点用父面的有限翼展
斜率，而非完整升力线求解）、起落架悬挂/滚动阻力（冻结 URDF 没有机轮或悬挂几何，
接触仍是平面摩擦接触）、导弹推进剂质心迁移、火箭推进剂晃动与着陆腿柔性，
以及平地以外的地形与障碍物。

数值验证套件见
[tests/test_fidelity_mechanisms.py](../tests/test_fidelity_mechanisms.py)：
逐项对照标准大气表、Prandtl-Glauert / 线性超声速斜率、Oswald 诱导阻力、
舵机速率/死区、非定常对流滞后、动态失速升力/阻力/力矩滞回与稳态极曲线保持、
Cheeseman-Bennett 地效、几何动量理论滑流、旋翼陀螺力矩、叶素前进比推力
曲线及其在 plant 层的落地、电池开路电压/库仑积分与能量记账、
展向条带与单点模型的等价比对、展向滑流剖面的中点积分、解析条带滚转阻尼、
升力线机翼地效因子及其自由空间/h→0 极限、plant 层条带高度接线、
气动弹性扭转衰减、发散趋势与超临界有界性、
高温/湿度密度关系、天线罩斜率 LOS 偏差与导引头滤波阶跃响应、
传感器噪声/零偏/延迟与传感器缩放随机化、质量/惯量一致性、独立气动系数、
3D 质心随机化、舵机延迟/回差/正反速率不对称动力学、MIL-F-8785C Dryden
功率谱形状与时变滤波器稳态方差、风切变、导引头视场丢锁、导弹推进剂消耗与
目标过载限制、
目标推力/阻力能量平衡，以及火箭与飞行任务的接触解。

## 安装

```bash
python -m pip install -r requirements.txt
```

精确版本锁定在 [requirements.txt](../requirements.txt)；参考环境为 Linux 上的
Python 3.10.20、PyTorch 2.11.0（CUDA 12.8）、`genesis-world` 1.4.4、
`rsl-rl-lib` 5.3.0、NumPy 2.2.6 与 SciPy 1.15.3（RTX 3090，驱动 580.173.02）。

六个 Genesis 任务在 GPU 后端需要 CUDA 版 PyTorch；四个张量动力学任务以及所有
经典基线也可以在纯 CPU 的 PyTorch 上运行。本套件不依赖 `gymnasium` / `pettingzoo`。
