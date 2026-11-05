"""quadx 第三个经典 baseline：滑模位置外环 + Euler 姿态内环 (SMC)。

与既有两个 quadx baseline 的区分：
- `se3_geometric`（默认）：SO(3) 流形上的几何控制（Lee 2010），四转子力矩分配。
- `cascaded_pid`：世界系线性 PID → 期望加速度 → Euler 姿态级联。
- 本类（smc）：滑模变结构控制（Runcharoon & Srichatrapimuk 2000 一族的
  quadrotor SMC 位置外环）。滑模面 s = e_v + λ e_p（e_p = p − p_ref），
  指数趋近律 d s/dt = −k sat(s/Φ)（边界层 sat 消颤），解出期望加速度
  a_des = −k sat(s/Φ) − λ e_v − g e3；姿态/油门通道与 cascaded_pid 同接口
  （期望 roll/pitch 由 heading 系投影给出），但外环结构（趋近律 + 滑模面）
  是独立的经典方法族。

只读 obs['policy']；批量 torch、float64 内部、输出 (N,4) float32 ∈ [-1,1]；
唯一内部状态为航点切换时清零（SMC 外环本身无积分器，无 reset_idx 状态）。
"""

import math

import torch

from environments.common.dynamics import COMPUTE_DTYPE

from algorithms.baselines.baselines import (
    Baseline,
    _quat_to_R,
    _rot_to_rpy,
    _vec,
    _wrap,
    _quadx_current_target,
)


class QuadXSMCBaseline(Baseline):
    """Sliding-mode position control + Euler attitude cascade (non-learning).

    Sliding surface (world frame):
        s = e_v + LAMBDA * e_p,   e_p = p - p_ref
    Exponential reaching law with a boundary layer (no chattering):
        a_des = -K_REACH * sat(s / PHI) - LAMBDA * e_v + G * e3
    where sat is the vector-wise elementwise saturation (boundary layer PHI
    [m/s]).  The specific-force command is m * (a_des + g e3) with the norm
    clamped at F_MAX; desired roll/pitch from the heading-frame projection
    (identical interface algebra to the cascaded-PID arm, fixed yaw
    reference), attitude P/PD -> body-rate commands executed by the plant's
    native 120 Hz rate PID.
    """

    name = "smc"

    # 冻结的 cf2x 机体常数（与 QuadXSE3Baseline 同参考模型）
    MASS = 0.027
    G = 9.81
    TOTAL_THRUST = 2.0
    THR_MAX = 0.8
    PWM_GAIN = 1.0

    # 滑模外环
    LAMBDA = 2.5  # 滑模面斜率 [1/s]（e_v 权重）
    K_REACH = 4.0  # 趋近律增益 [1/s^2]（m/s^2 per m/s）
    PHI = 0.4  # 边界层 [m/s]
    F_MAX_FRAC = 0.95  # 总推力上限占额（留裕度防积分性下坠）
    # 姿态内环（与 cascaded_pid 同接口、同参考增益）
    KP_ATT = 4.0
    KD_ATT = 0.35
    KP_YAW = 2.0
    KD_YAW = 0.20
    YAW_REF = 0.0
    RATE_MAX = math.pi

    def __init__(self, env, design=None):
        super().__init__(env, design=design)
        d, n = self.device, env.num_envs
        self.e3 = torch.tensor((0.0, 0.0, 1.0), device=d, dtype=COMPUTE_DTYPE)
        self.k_att = torch.tensor(
            (self.KP_ATT, self.KP_ATT, self.KP_YAW), device=d, dtype=COMPUTE_DTYPE
        )
        self.kd_att = torch.tensor(
            (self.KD_ATT, self.KD_ATT, self.KD_YAW), device=d, dtype=COMPUTE_DTYPE
        )
        self.f_max = self.TOTAL_THRUST * self.THR_MAX**2 * self.F_MAX_FRAC
        self.sat_force = torch.zeros(n, device=d, dtype=COMPUTE_DTYPE)
        self.sat_rate = torch.zeros(n, device=d, dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def actions(self, obs):
        o = _vec(obs).to(COMPUTE_DTYPE)
        wb, q, vb, p = o[:, :3], o[:, 3:7], o[:, 7:10], o[:, 10:13]
        R = _quat_to_R(q)
        roll, pitch, yaw = _rot_to_rpy(R)
        v_w = torch.einsum("nij,nj->ni", R, vb)
        e_p = p - _quadx_current_target(o, R)
        e_v = v_w  # 目标静止：滑模面 s = v + λ e_p（e_v 即世界系速度）

        # 指数趋近律 + 边界层 sat 消颤 → 期望加速度（含重力补偿）
        s = e_v + self.LAMBDA * e_p
        a_des = -self.K_REACH * torch.tanh(s / self.PHI) - self.LAMBDA * e_v
        a_cmd = a_des + self.G * self.e3

        # 总推力与期望 roll/pitch（heading 系投影，接口与 cascaded_pid 一致）
        f_cmd = self.MASS * a_cmd.norm(dim=-1)
        self.sat_force += (f_cmd > self.f_max).to(COMPUTE_DTYPE)
        f = f_cmd.clamp_max(self.f_max)
        throttle = (torch.sqrt(f / self.TOTAL_THRUST) / self.PWM_GAIN).clamp(0.0, self.THR_MAX)
        u_x, u_y, u_z = a_cmd[:, 0], a_cmd[:, 1], a_cmd[:, 2]
        pitch_d = torch.atan2(u_x, u_z)
        roll_d = torch.atan2(-u_y, torch.sqrt(u_x * u_x + u_z * u_z))

        # 姿态 P/PD -> 角速率指令（由 plant 120 Hz 速率 PID 执行）
        e_att = torch.stack(
            (
                _wrap(roll_d - roll),
                _wrap(pitch_d - pitch),
                _wrap(torch.full_like(yaw, self.YAW_REF) - yaw),
            ),
            -1,
        )
        w_cmd = self.k_att * e_att - self.kd_att * wb
        self.sat_rate += (w_cmd.abs() > self.RATE_MAX).to(COMPUTE_DTYPE).sum(-1)
        w_cmd = w_cmd.clamp(-self.RATE_MAX, self.RATE_MAX)
        return (
            torch.cat((w_cmd / self.RATE_MAX, (2.0 * throttle / self.THR_MAX - 1.0)[:, None]), -1)
            .clamp(-1.0, 1.0)
            .to(torch.float32)
        )
