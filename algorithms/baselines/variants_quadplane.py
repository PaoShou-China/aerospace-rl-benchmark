"""Quadplane-VTOL waypoints 任务的经典（非学习）基线变体（A/B 臂）。

本文件为 QuadplaneWaypointsEnv 提供两个额外的经典控制器，与默认的
``hybrid_cascade``（baselines.QuadplaneHybridBaseline）并列：

    se3_pusher  Lee SE(3) 几何控制全程四旋翼模式飞行整个航点任务，
                前推电机按机体 x 期望地速误差补充前向速度
    l1_wing     机翼模式 L1 侧向导引（Park-Deyst-Howe 2004）+ 纵向
                高度/空速级联；四旋翼通道仅在低速（失速保护）时辅助

两个控制器只读取 obs['policy']（构造时可读取环境公开常数），不触碰
env 内部缓冲；所有常数以类属性硬编码（不提供 CLI/配置）。内部全程
float64 批量张量、env.device，返回 (N, 8) float32 动作，每维 [-1, 1]。

动作约定（environments/vtol/env.py，_scale_actions）：
    a0-a2  机体角速度指令 (p, q, r) rad/s，动作 raw*π，内环速率 PID 执行
    a3     四旋翼总距 [0, 1]，动作 (raw+1)/2；电机推力 = 14.715 N * pwm²
    a4     副翼差动，正 → 右压坡度（与 fixedwing 同映射同符号）
    a5     平尾+垂尾，正 → 低头力矩（含垂尾反向耦合，视作扰动）
    a6     主翼（deflection_limit = 0，死通道，恒 0）
    a7     前推油门 [0, 1]，推力 = 58.86 N * throttle²，沿机体 +x
欧拉约定（x 前、y 左、z 上）：正 roll = 右坡度，正 pitch = 低头。

机体/动力常数冻结自 environments/common/plant.py quadplane 分支：
    total_mass = 2.35 kg（重力 23.05 N）
    4 x 四旋翼电机 14.715 N @ pwm=1（总距 1 → 总推力 58.86 N，
        悬停总距 = sqrt(23.05/4/14.715) ≈ 0.626）
    前推电机 58.86 N @ throttle=1（沿机体 +x）
    转动惯量（实测）I = (0.51, 0.31, 0.80) kg·m²
    内环速率 PID：kp = [0.9, 0.8, 4.0]，ki = [0.005, 0.005, 0.02]，
        kd = 0，lim = ±2.0（PID 单位）
    混矩线性化（悬停 pwm 0.626 处 dT/dpwm = 18.41 N）：
        ±1 PID 单位 → ±3.50 N·m roll / ±3.50 N·m pitch / ±0.176 N·m yaw
    机翼气动（quadplane.yaml）：主翼 1.6x0.3 m + 副翼 2x(0.3x0.3) m，
        升力线斜率 ≈ 4.25 /rad，零升迎角 -2°，失速迎角 +14°，
        2.35 kg / 0.66 m² 翼载的气动失速空速 ≈ 7 m/s（α≈14°）
"""

import math

import torch

from environments.common.dynamics import COMPUTE_DTYPE

from algorithms.baselines.baselines import Baseline, _vec, _quat_to_R, _vee, _wrap, _rot_to_rpy


# --------------------------------------------------------------------------- #
# quadplane: 全程四旋翼 Lee SE(3) 几何控制 + 前推速度补充
# --------------------------------------------------------------------------- #
class QuadplaneSE3PusherBaseline(Baseline):
    """SE(3) 几何跟踪（Lee, Leok & McClamroch 2010）适配四旋翼机 + 前推。

    方法
        位置外环 PD（带速度上限与加速度限幅）产生世界系期望合力 F_des；
        总距取其在机体推力轴上的投影 f = F_des·b3（推力沿机体 +z），
        经二次推力律 collective = sqrt(f / 58.86 N) 映射到 a3；
        期望姿态 R_d 由 b3_d = F_des 方向 + 偏航参考（航点方位角）构成，
        姿态误差 e_R = 1/2 vee(R_d' R - R' R_d)，力矩律
        M = -kR e_R - kOm w + w×Jw。力矩经混元逆映射回 PID 单位
        （tau = M / [3.50, 3.50, 0.176] N·m），再经内环速率 P 逆
        （w_cmd = w + tau / kp）映射到 a0-a2 —— 与 QuadXSE3Baseline 的
        mixer-inversion 思路相同，但常数全部换成四旋翼机自身的
        （质量 2.35 kg、总推力 58.86 N、惯量与混矩增益取自
        environments/common/plant.py quadplane 分支）。

        前推 a7：期望地速在机体 x 轴上的投影 v_d·b1 与实际前向空速
        vb[0] 之差（正值部分）比例饱和映射为油门，用于补充前向速度、
        减轻倾转负担；减速方向由姿态（后倾）负责，前推不参与制动。
        a4-a6 恒 0（本方法不使用气动面）。

    整定说明（smoke 调参，seed 7 / 64 envs）
        V_MAX = 17 m/s（30 s / 4 航点预算下 15 m/s 余量不足，实测超时偏多）；
        A_MAX = 6.5 m/s² 限制加速度指令范数——不加限幅时大速度设定差
        （~15 m/s）会产生 ~45 m/s² 的水平加速度指令，把机体拉到近 80°
        倾角，机翼大迎角失速且姿态饱和；6.5 m/s² 对应 ~33° 最大倾角，
        并保留约 V²/(2*A_MAX) ≈ 22 m 的减速距离（捕获半径 15 m 容纳过冲）。
        KR/KOM 按各轴期望带宽整定：roll/pitch wn≈3 rad/s、ζ≈0.9，
        yaw 受物理偏航力矩限制（0.176 N·m/单位）单独降到 wn≈0.87。
        前推增益 K_PUSH = 0.06/(m/s)：10 m/s 速度亏缺 ≈ 0.6 油门
        （21 N），覆盖过渡段加速需求。
        实测（64 envs，1200 步上限）：success_rate ≈ 0.86，
        waypoints_reached ≈ 3.9/4，失败模式以超时为主（无坠机）。
    """

    name = "se3_pusher"

    # ---- 机体/动力常数（plant.py quadplane 分支，冻结） ----
    MASS = 2.35  # kg
    G = 9.81  # m/s²，世界系 -z 向下
    TOTAL_THRUST = 58.86  # N，collective=1 时 4 电机总推力
    F_MIN = 0.2  # N，推力下限（pwm 下限 0.05 附近）
    INERTIA = (0.51, 0.31, 0.80)  # kg·m²，实测 URDF 惯量
    KP_RATE = (0.9, 0.8, 4.0)  # 内环速率 PID P 增益
    MIXER_TAU = (3.50, 3.50, 0.1758)  # N·m / PID 单位（悬停线性化）

    # ---- SE(3) 增益 ----
    KX, KV = 3.0, 3.0  # 位置 PD [1/s², 1/s]
    V_MAX = 17.0  # 巡航速度上限 [m/s]
    A_MAX = 6.5  # 加速度指令范数上限 [m/s²]
    KR = (4.6, 2.8, 0.6)  # 姿态 P [N·m/rad]（yaw 受权限限制单独降）
    KOM = (2.75, 1.67, 0.8)  # 姿态 D [N·m·s/rad]
    RATE_MAX = math.pi * 0.7  # 角速度指令限幅 [rad/s]

    # ---- 前推 ----
    K_PUSH = 0.06  # 前向速度误差 → 油门 [1/(m/s)]
    PUSH_TRIM = 0.20  # 油门基数（保持电机在线）
    PUSH_MAX = 0.90  # 油门上限（0.9² * 58.86 ≈ 47.7 N）

    # tunable gain space for online gain scheduling
    TUNABLE_PARAMS = {
        "KX": dict(low=1.0, high=6.0, default=3.0),
        "KV": dict(low=1.0, high=6.0, default=3.0),
        "V_MAX": dict(low=8.0, high=25.0, default=17.0),
        "A_MAX": dict(low=3.0, high=10.0, default=6.5),
        "K_PUSH": dict(low=0.02, high=0.15, default=0.06, log=True),
    }

    def __init__(self, env, design=None):
        super().__init__(env, design)
        for _k in self.TUNABLE_PARAMS:
            setattr(self, _k, self._param(_k))
        d = self.device
        self.e3 = torch.tensor((0.0, 0.0, 1.0), device=d, dtype=COMPUTE_DTYPE)
        self.kr = torch.tensor(self.KR, device=d, dtype=COMPUTE_DTYPE)
        self.kom = torch.tensor(self.KOM, device=d, dtype=COMPUTE_DTYPE)
        self.J = torch.tensor(self.INERTIA, device=d, dtype=COMPUTE_DTYPE)
        self.kp = torch.tensor(self.KP_RATE, device=d, dtype=COMPUTE_DTYPE)
        self.mix = torch.tensor(self.MIXER_TAU, device=d, dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 31:43].reshape(-1, 4, 3)
        R = _quat_to_R(q)  # body -> world
        b3 = R[:, :, 2]  # 机体推力轴（+z）在世界系
        b1 = R[:, :, 0]  # 机体 +x（前推方向）在世界系
        v_w = torch.einsum("nij,nj->ni", R, vb)
        target = p + torch.einsum("nij,nj->ni", R, delta[:, 0])

        # ---- 位置外环：PD -> 速度设定（范数上限）-> 加速度指令（范数上限）----
        e_x = p - target
        v_d = -self._gain("KX")[:, None] / self._gain("KV")[:, None] * e_x
        v_d = v_d * (
            self._gain("V_MAX")[:, None]
            / v_d.norm(dim=-1, keepdim=True).clamp_min(self._gain("V_MAX")[:, None])
        )
        a_cmd = self._gain("KV")[:, None] * (v_d - v_w)
        a_cmd = a_cmd * (self.A_MAX / a_cmd.norm(dim=-1).clamp_min(self.A_MAX))[:, None]
        F_des = self.MASS * (self.G * self.e3 + a_cmd)

        # ---- 总距：推力轴投影 + 二次推力律 ----
        f = (F_des * b3).sum(-1).clamp(self.F_MIN, self.TOTAL_THRUST)
        collective = torch.sqrt(f / self.TOTAL_THRUST)

        # ---- 期望姿态：b3_d = F_des 方向，偏航 = 航点方位角 ----
        n2 = F_des.norm(dim=-1)
        b3_d = torch.where((n2 > 1e-6)[:, None], F_des / n2[:, None].clamp_min(1e-6), b3)
        psi_d = torch.atan2(target[:, 1] - p[:, 1], target[:, 0] - p[:, 0])
        c = torch.stack((torch.cos(psi_d), torch.sin(psi_d), torch.zeros_like(psi_d)), -1)
        b1_d = c - (c * b3_d).sum(-1, keepdim=True) * b3_d
        e2 = torch.tensor((0.0, 1.0, 0.0), device=o.device, dtype=COMPUTE_DTYPE).expand_as(b3_d)
        # 航向投影退化（b3_d ≈ ±c）时回退到 e2
        b1_d = torch.where(
            (b1_d * b1_d).sum(-1, keepdim=True) < 1e-8,
            e2 - (e2 * b3_d).sum(-1, keepdim=True) * b3_d,
            b1_d,
        )
        b1_d = b1_d / b1_d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Rd = torch.stack((b1_d, torch.linalg.cross(b3_d, b1_d, dim=-1), b3_d), -1)

        # ---- 姿态 PD + 陀螺项 -> 力矩 ----
        e_R = _vee(Rd.transpose(-1, -2) @ R - R.transpose(-1, -2) @ Rd)
        M = -self.kr * e_R - self.kom * wb + torch.linalg.cross(wb, self.J * wb, dim=-1)

        # ---- 力矩 -> PID 单位（混元逆）-> 角速度指令（内环 P 逆）----
        tau = M / self.mix
        w_cmd = (wb + tau / self.kp).clamp(-self.RATE_MAX, self.RATE_MAX)

        # ---- 前推：机体 x 期望地速误差（仅正向）比例饱和 ----
        v_fwd_d = (v_d * b1).sum(-1)
        err_fwd = (v_fwd_d - vb[:, 0]).clamp_min(0.0)
        thr7 = (self.PUSH_TRIM + self.K_PUSH * err_fwd).clamp(0.0, self.PUSH_MAX)

        zero = torch.zeros_like(thr7)
        return (
            torch.cat(
                (
                    w_cmd / self.RATE_MAX,  # a0-a2: 机体角速度 / π
                    (2.0 * collective - 1.0)[:, None],  # a3: 总距 [0,1]
                    zero[:, None],
                    zero[:, None],
                    zero[:, None],  # a4-a6: 气动面（恒 0）
                    (2.0 * thr7 - 1.0)[:, None],  # a7: 前推油门 [0,1]
                ),
                -1,
            )
            .clamp(-1.0, 1.0)
            .to(torch.float32)
        )


# --------------------------------------------------------------------------- #
# quadplane: 机翼模式 L1 侧向导引 + 纵向级联，四旋翼低速辅助
# --------------------------------------------------------------------------- #
class QuadplaneL1WingBaseline(Baseline):
    """机翼模式 L1 导引（Park, Deyst & Howe 2004）+ 四旋翼失速保护。

    方法
        侧向：L1 距离调度 L1 = clamp(k*d + L0, L_MIN, L_MAX)，
        侧向加速度指令 a = 2V²sin(η)/L1（η = 航迹角到航点视线角，
        |η|>90° 时取满强度转弯），坡度指令 bank = -atan(a/g)
        （本环境正 roll = 右坡度使航向顺时针减小，目标在左需负坡度），
        坡度受气动失速边界 bank_lim = acos((v_stall/V)²) 与绝对上限
        约束；副翼 a4 = k_roll·(bank - roll) - kd_roll·p。近场（d_h<3 m）
        死区 + 100 m 出界回中护盾（与 FixedwingAutopilotBaseline 同式）。

        纵向：高度误差 -> 爬升率设定 vz_d = clamp(K_ALT·h_err)，爬升率
        误差 -> 俯仰指令 pitch_cmd = trim - K_CLIMB·(vz_d - vz)，平尾
        a5 = k_pitch·(pitch_cmd - pitch) - kd_pitch·(q - q_coord) + 积分
        （协调转弯补偿 q_coord = r·tan(roll)，积分消除配平残差）；
        空速由前推油门保持：thr = trim + K_V·(V_ref - V) + 爬升前馈。

        四旋翼辅助：a0-a2 姿态内环（与 hybrid 相同的机体增益）全程
        有效——低速时气动力矩趋于零，四旋翼差动矩自然成为唯一姿态
        权限；a3 总距仅在低速时介入：权重 w_quad = clamp((STALL+BAND-V)
        /2BAND)（V < STALL-BAND 全辅助，> STALL+BAND 退到怠速 A3_IDLE），
        总距支撑力 F_q = m(g + KZ·h_err - KDZ·vz)/cos(roll)（坡度补偿）
        经二次律映射，另有 ALT_FLOOR 以下强制辅助的安全底层；V_REF =
        15 m/s ≈ 气动失速空速（~7 m/s）的 2 倍裕度。

    整定说明（smoke 调参，seed 7 / 64 envs）
        L1 调度 0.4*d+8、[12, 30]：恒定 L1 会在捕获球外形成极限环
        绕点（r ≈ L1/(2sinη)），随距离收缩使平衡点落入 15 m 捕获半径；
        上限 30 对应 15 m/s、50° 坡度时转弯半径 ≈ V²/(g·tan50°) ≈ 18 m，
        75 m 航点反向不会出 120 m 边界。STALL_SPEED = 11.5 ± 1.5 过渡
        带：机翼 2 倍失速裕度接管前，总距托底高度、前推负责加速，
        避免"低-高-低"模式的 porpoising。前推配平 0.40（15 m/s 配平
        阻力 ~12 N 的二次律油门）。爬升需求按空速展开（见 actions），
        俯仰积分限幅 0.9 覆盖机翼配平力矩。
        实测（64 envs，1200 步上限）：success_rate ≈ 0.48，
        waypoints_reached ≈ 3.1/4，失败模式以超时为主（机翼爬升率
        ~3-4 m/s 上限，高空航点耗时；无坠机/出界）。
    """

    name = "l1_wing"

    # ---- 动力常数（plant.py quadplane 分支，冻结） ----
    MASS = 2.35  # kg
    TOTAL_THRUST = 58.86  # N，collective=1 时 4 电机总推力

    # ---- L1 侧向导引 ----
    L1_GAIN = 0.4  # L1 距离调度斜率 [-]
    L1_OFFSET = 8.0  # L1 距离调度基数 [m]
    L1_MIN = 12.0  # L1 下限 [m]
    L1_MAX = 30.0  # L1 上限 [m]
    MAX_BANK = math.radians(50.0)  # 坡度绝对上限
    V_STALL_GUARD = 10.0  # 气动坡度边界 v_stall [m/s]
    K_ROLL = 3.0  # 副翼坡度 P [1/s]
    KD_ROLL = 0.5  # 副翼滚转阻尼 [-]
    PN_SHIELD = 100.0  # 出界护盾起始水平距离 [m]
    K_SHIELD = 2.0  # 护盾坡度增益 [1/s]

    # ---- 纵向级联 ----
    K_ALT = 0.45  # 高度误差 -> 爬升率 [1/s]
    VZ_MIN = -3.0  # 下滑率下限 [m/s]
    VZ_MAX = 5.0  # 爬升率上限 [m/s]
    K_CLIMB = 0.07  # 爬升率误差 -> 俯仰 [rad/(m/s)]
    PITCH_TRIM = math.radians(-2.0)  # 巡航配平俯仰（负 = 微抬头）
    PITCH_UP_MAX = math.radians(25.0)  # 满速抬头限幅
    PITCH_DOWN_MAX = math.radians(10.0)  # 低头限幅
    PITCH_UP_LOW = math.radians(8.0)  # 低速抬头限幅（失速保护）
    PITCH_UP_RAMP = 5.0  # 抬头权限随空速展开的过渡带 [m/s]
    K_PITCH = 4.0  # 平尾俯仰 P [1/s]
    KD_PITCH = 0.45  # 俯仰角速率阻尼 [-]
    K_PITCH_I = 3.0  # 俯仰积分 [1/s²]
    I_PITCH_MAX = 0.9  # 俯仰积分限幅 [平尾单位]（配平余量需覆盖
    # 机翼俯仰力矩，实测巡航配平 a5≈-0.55）

    # ---- 前推空速保持 ----
    V_REF = 15.0  # 机翼巡航空速 [m/s]
    THR_TRIM = 0.40  # 油门配平（≈15 m/s 配平阻力）
    K_V = 0.06  # 空速误差 -> 油门 [1/(m/s)]
    K_CLIMB_THR = 0.015  # 爬升前馈 [1/(m/s)]
    THR7_MAX = 0.95  # 油门上限

    # ---- 四旋翼低速辅助 ----
    STALL_SPEED = 11.5  # 四旋翼辅助中心空速 [m/s]
    STALL_BAND = 1.5  # 混合半带宽 [m/s]
    KZ_Q = 0.6  # 辅助总距高度 P [1/s²]
    KDZ_Q = 1.5  # 辅助总距垂直阻尼 [1/s]
    A_QMAX = 3.5  # 辅助垂向加速度限幅 [m/s²]
    A3_IDLE = 0.25  # 机翼模式总距怠速（保留差动权限）
    ALT_FLOOR = 12.0  # 强制辅助的安全高度底层 [m]
    Z_HOLD = 15.0  # 辅助段保持高度（= 出生高度）[m]
    CLIMB_GATE_RAMP = 3.0  # 爬升需求随空速展开的过渡带 [m/s]

    # ---- 姿态内环（与 QuadplaneHybridBaseline 同机体增益） ----
    KP_ATT = 6.0  # roll/pitch 姿态 P [1/s]
    KD_ATT = 0.6  # roll/pitch 角速率阻尼 [-]
    KP_YAW = 1.0  # yaw 姿态 P [1/s]
    KD_YAW = 0.3  # yaw 角速率阻尼 [-]
    RATE_MAX = math.pi * 0.7  # 角速度指令限幅 [rad/s]

    def __init__(self, env):
        super().__init__(env)
        self.g = 9.81
        self.dt = 1.0 / env.agent_hz
        d = self.device
        self.k_att = torch.tensor(
            (self.KP_ATT, self.KP_ATT, self.KP_YAW), device=d, dtype=COMPUTE_DTYPE
        )
        self.kd_att = torch.tensor(
            (self.KD_ATT, self.KD_ATT, self.KD_YAW), device=d, dtype=COMPUTE_DTYPE
        )
        self.i_pitch = torch.zeros(self.num_envs, device=d, dtype=COMPUTE_DTYPE)

    def reset_idx(self, mask):
        self.i_pitch.masked_fill_(mask, 0.0)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        delta = o[:, 31:43].reshape(-1, 4, 3)
        R = _quat_to_R(q)
        roll, pitch, yaw = _rot_to_rpy(R)
        v_w = torch.einsum("nij,nj->ni", R, vb)
        target = p + torch.einsum("nij,nj->ni", R, delta[:, 0])
        V = vb.norm(dim=-1)  # 静风：空速 = 地速
        vz_w = v_w[:, 2]
        h_err = target[:, 2] - p[:, 2]  # 高度误差（上正）

        # ---- L1 侧向导引 ----
        d_h_vec = target[:, :2] - p[:, :2]
        d_h = d_h_vec.norm(dim=-1)
        l1_eff = (self.L1_GAIN * d_h + self.L1_OFFSET).clamp(self.L1_MIN, self.L1_MAX)
        los = torch.atan2(d_h_vec[:, 1], d_h_vec[:, 0])
        course = torch.where(
            v_w[:, :2].norm(dim=-1) > 2.0,
            torch.atan2(v_w[:, 1], v_w[:, 0]),
            torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        )  # 低速航向回退
        eta = _wrap(los - course)
        # |η| > 90° 时 sin(η) 在最需要转弯处反而减小：取满强度转弯
        eta_c = eta.clamp(-math.pi / 2, math.pi / 2)
        # 近场死区：航点近顶时视线方位是噪声
        a_lat = 2.0 * V * V * torch.sin(eta_c) / l1_eff * (d_h > 3.0).to(V.dtype)
        bank_l1 = -torch.atan(a_lat / self.g)
        # 气动坡度边界：n_max = (V/v_stall)²，速度越低坡度权限越小
        bank_lim = torch.acos(((self.V_STALL_GUARD / V) ** 2).clamp(0.0, 1.0)).clamp_max(
            self.MAX_BANK
        )
        # 出界护盾：接近边界且向外飞时压坡度回中
        pn = p.norm(dim=-1)
        v_toward = -(p * v_w).sum(-1) / pn.clamp_min(1e-6)
        w_shield = torch.sigmoid((pn - self.PN_SHIELD) / 4.0) * torch.sigmoid(
            (v_toward - 0.5) / 1.0
        )
        eta_home = _wrap(torch.atan2(-p[:, 1], -p[:, 0]) - course)
        bank_cmd = ((1.0 - w_shield) * bank_l1 + w_shield * (-self.K_SHIELD * eta_home)).clamp(
            -bank_lim, bank_lim
        )
        a4 = self.K_ROLL * _wrap(bank_cmd - roll) - self.KD_ROLL * wb[:, 0]

        # ---- 纵向：高度 -> 爬升率 -> 俯仰 ----
        # 爬升需求随空速展开：低速段以平飞加速为先——若带着 level 姿态
        # 用四旋翼硬爬，飞行路线角远大于俯仰角，机翼迎角深度为负成为
        # 气动刹车（V≈7 m/s 时阻力可达 30+ N，吞掉全部前推），速度崩溃
        # 后 L1 权限（∝V²）也随之消失（实测主导失败模式）。
        gate = ((V - self.STALL_SPEED) / self.CLIMB_GATE_RAMP).clamp(0.0, 1.0)
        vz_d = gate * (self.K_ALT * h_err).clamp(self.VZ_MIN, self.VZ_MAX)
        # 抬头权限随空速展开（低速大迎角 = 失速）
        pitch_up = self.PITCH_UP_LOW + (self.PITCH_UP_MAX - self.PITCH_UP_LOW) * (
            (V - self.STALL_SPEED) / self.PITCH_UP_RAMP
        ).clamp(0.0, 1.0)
        pitch_cmd = (
            (self.PITCH_TRIM - self.K_CLIMB * (vz_d - vz_w))
            .clamp_min(-pitch_up)
            .clamp_max(self.PITCH_DOWN_MAX)
        )
        i_pitch = (self.i_pitch + self.K_PITCH_I * (pitch_cmd - pitch) * self.dt).clamp(
            -self.I_PITCH_MAX, self.I_PITCH_MAX
        )
        self.i_pitch.copy_(i_pitch)
        # 协调转弯：稳态坡度转弯需要持续的机体俯仰角速率 r·tan(roll)
        q_coord = wb[:, 2] * torch.tan(roll.clamp(-1.45, 1.45))
        a5 = self.K_PITCH * (pitch_cmd - pitch) - self.KD_PITCH * (wb[:, 1] - q_coord) + i_pitch

        # ---- 前推：空速保持 + 爬升前馈 ----
        thr7 = (
            self.THR_TRIM + self.K_V * (self.V_REF - V) + self.K_CLIMB_THR * vz_d.clamp_min(0.0)
        ).clamp(0.0, self.THR7_MAX)

        # ---- 四旋翼低速辅助 ----
        w_low = ((self.STALL_SPEED + self.STALL_BAND - V) / (2.0 * self.STALL_BAND)).clamp(0.0, 1.0)
        # 安全底层：低高度强制辅助（与空速无关）
        w_safe = ((self.ALT_FLOOR - p[:, 2]) / 5.0).clamp(0.0, 1.0)
        w_quad = torch.maximum(w_low, w_safe)
        # 辅助总距只守过渡段高度带（出生高度），不追航点高度——爬升由
        # 机翼在获得空速后接管（见上方 gate）
        a_z = (self.KZ_Q * (self.Z_HOLD - p[:, 2]) - self.KDZ_Q * vz_w).clamp(
            -self.A_QMAX, self.A_QMAX
        )
        F_q = self.MASS * (self.g + a_z) / torch.cos(roll).clamp_min(0.5)
        col_full = torch.sqrt((F_q / self.TOTAL_THRUST).clamp(0.0, 1.0))
        collective = w_quad * col_full + (1.0 - w_quad) * self.A3_IDLE

        # ---- 姿态内环（四旋翼差动矩 + 气动面共同作用） ----
        # 近场 yaw 死区（航点近顶时方位角噪声）
        e_yaw = _wrap(los - yaw) * torch.sigmoid((d_h - 5.0) / 2.0)
        e_att = torch.stack((_wrap(bank_cmd - roll), _wrap(pitch_cmd - pitch), e_yaw), -1)
        w_cmd = (self.k_att * e_att - self.kd_att * wb).clamp(-self.RATE_MAX, self.RATE_MAX)

        zero = torch.zeros_like(a4)
        return (
            torch.cat(
                (
                    w_cmd / self.RATE_MAX,  # a0-a2: 机体角速度 / π
                    (2.0 * collective - 1.0)[:, None],  # a3: 总距 [0,1]
                    a4[:, None],
                    a5[:, None],
                    zero[:, None],  # a4/a5: 副翼、平尾+垂尾
                    (2.0 * thr7 - 1.0)[:, None],  # a7: 前推油门 [0,1]
                ),
                -1,
            )
            .clamp(-1.0, 1.0)
            .to(torch.float32)
        )
