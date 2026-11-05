"""Batched SI-unit model, ported from PyFlyt's noise-free QuadX dynamics.

PyFlyt source commit: 9095d41f6fc5f07fe711346b747b4f7fd475b424.
Motor order: (+x,-y), (-x,+y), (+x,+y), (-x,-y).
"""

from dataclasses import dataclass
import torch
from pathlib import Path
import yaml

# Hot-path compute dtype, fixed to float32: consumer GPUs (e.g. RTX 3090)
# run FP64 at 1/64 of FP32 throughput.  Measured 2026-11-04: quadx baseline
# success and episode return match the old fp64 path to ~1e-7.
COMPUTE_DTYPE = torch.float32


@dataclass(frozen=True)
class QuadParams:
    """SI-unit quadrotor physical parameters (noise-free reference values).

    Holds mass, thrust/torque coefficients, motor time constant and body drag
    constants.  ``from_reference`` rebuilds the values from the canonical
    cf2x reference asset so they stay in sync with the URDF/YAML used by
    the plant.
    """
    mass: float = 0.027
    kf: float = 3.16e-10
    km: float = 7.94e-12
    total_thrust: float = 2.0
    tau: float = 0.01
    drag_linear: float = 0.5 * 1.225 * 3.0 * 4e-4
    drag_angular: float = 1e-4

    @classmethod
    def from_reference(cls):
        """Build :class:`QuadParams` from the shipped cf2x reference YAML/URDF."""
        p = yaml.safe_load((Path(__file__).parent / "assets/reference_cf2x.yaml").read_text())
        from .consistency import asset_inertials

        m = p["motor_params"]
        d = p["drag_params"]
        return cls(
            mass=sum(r["mass"] for r in asset_inertials("quadx").values()),
            kf=m["thrust_coef"],
            km=m["torque_coef"],
            total_thrust=m["total_thrust"],
            tau=m["tau"],
            drag_linear=0.5 * 1.225 * d["drag_coef_xyz"] * d["drag_area_xyz"],
            drag_angular=d["drag_coef_pqr"],
        )

    @property
    def max_rpm(self):
        """Rad/s at unit Pwm: ``sqrt(total_thrust / (4 * kf))``."""
        return (self.total_thrust / (4 * self.kf)) ** 0.5

    @property
    def hover_pwm(self):
        """Static-float Pwm to cancel gravity: ``sqrt(mass * g / total_thrust)``."""
        return (self.mass * 9.81 / self.total_thrust) ** 0.5


def inverse_rotate(q, v):
    """World vector to body coordinates; quaternion order w,x,y,z.

    The double cross-product form is kept deliberately: a direct R(q)^T
    expansion was measured 5.8x slower in this launch-bound codebase
    (more small kernels), and torch.compile gave no further speedup.
    """
    xyz = -q[..., 1:]
    t = 2 * torch.linalg.cross(xyz, v)
    return v + q[..., :1] * t + torch.linalg.cross(xyz, t)


def euler(q):
    """Return roll, pitch, yaw (XYZ intrinsic, radians) from quaternion ``[w,x,y,z]``."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
            torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        -1,
    )


def quat_from_euler(rpy):
    """Roll-pitch-yaw (XYZ intrinsic, PyBullet convention) to quaternion [w,x,y,z]."""
    roll, pitch, yaw = rpy.unbind(-1)
    # Half-angle trig: cos(r/2), sin(r/2), etc.  Multiplying cos/sin by 0.5
    # instead (as this line once did) yields a scaled, non-unit quaternion
    # whose implied rotation is wrong for anything but small angles.
    cr, sr = (0.5 * roll).cos(), (0.5 * roll).sin()
    cp, sp = (0.5 * pitch).cos(), (0.5 * pitch).sin()
    cy, sy = (0.5 * yaw).cos(), (0.5 * yaw).sin()
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        -1,
    )


class QuadDynamics:
    """Noise-free batched quadrotor rigid-body dynamics (for reference/testing).

    Owns a per-replica throttle state of shape ``(n, 4)`` and computes the
    expected force/torque from commanded Pwm at each step.  Kept alongside
    :class:`TensorPlant` so unit tests can compare the noise-free analytic
    model against the Genesis-backed plant.
    """
    def __init__(self, n, device, dt=1 / 240, params=None):
        self.p = params or QuadParams.from_reference()
        if not 0 < dt <= self.p.tau:
            raise ValueError("Explicit Euler motor model requires 0 < dt <= tau")
        self.dt = dt
        self.throttle = torch.zeros((n, 4), device=device)
        from .consistency import asset_inertials

        ref = asset_inertials("quadx")
        self.arm = torch.tensor(
            [ref[f"prop{i}_link"]["local_com"].tolist() for i in range(1, 5)], device=device
        )
        self.spin = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)

    def reset(self, ids, pwm=0.0):
        """Set throttle to ``pwm`` for replicas indexed by ``ids``."""
        self.throttle[ids] = pwm

    def step(self, pwm, body_vel, body_ang, drag=True):
        """Run one explicit-Euler motor step and return expected force/torque.

        Parameters
        ----------
        pwm : (n, 4) tensor
            Commanded Pwm in [0, 1] per motor.
        body_vel : (n, 3) tensor
            Current body-frame velocity.
        body_ang : (n, 3) tensor
            Current body-frame angular velocity.
        drag : bool
            When True, apply linear body drag and angular drag.

        Returns
        -------
        force : (n, 3) tensor
            Body-frame force (z-axis is thrust).
        torque : (n, 3) tensor
            Body-frame torque.
        """
        # Same explicit Euler lag as PyFlyt; deliberately noise-free for parity.
        self.throttle += self.dt / self.p.tau * (pwm - self.throttle)
        rpm = self.throttle * self.p.max_rpm
        signed_rpm2 = rpm.abs() * rpm
        thrust = self.p.kf * signed_rpm2
        force = torch.zeros_like(body_vel)
        force[:, 2] = thrust.sum(-1)
        torque = torch.stack(
            (
                (thrust * self.arm[:, 1]).sum(-1),
                -(thrust * self.arm[:, 0]).sum(-1),
                (self.p.km * signed_rpm2 * self.spin).sum(-1),
            ),
            -1,
        )
        if drag:
            force -= self.p.drag_linear * body_vel.abs() * body_vel
            torque -= self.p.drag_angular * body_ang.abs() * body_ang
        return force, torque


class HoverController:
    """Independent cascaded PD baseline, not PyFlyt's original PID controller."""

    def __init__(self, dynamics):
        self.dynamics = dynamics
        p = dynamics.p
        thrust_column = torch.ones(4, device=dynamics.arm.device)
        self.alloc = torch.stack(
            (thrust_column, dynamics.arm[:, 1], -dynamics.arm[:, 0], dynamics.spin * p.km / p.kf)
        )
        self.inv = torch.linalg.inv(self.alloc)

    def __call__(self, pos, quat, vel, ang, target):
        rp = euler(quat)
        a = (3.0 * (target - pos) - 2.5 * vel).clamp(-3, 3)
        yaw = rp[:, 2]
        # Convert desired world acceleration to yaw-aligned roll/pitch.
        roll = (a[:, 0] * torch.sin(yaw) - a[:, 1] * torch.cos(yaw)) / 9.81
        pitch = (a[:, 0] * torch.cos(yaw) + a[:, 1] * torch.sin(yaw)) / 9.81
        desired = torch.stack((roll, pitch, torch.zeros_like(yaw)), -1)
        err = torch.atan2(torch.sin(desired - rp), torch.cos(desired - rp))
        torque = 0.0008 * err - 0.00018 * inverse_rotate(quat, ang)
        thrust = (
            self.dynamics.p.mass
            * (9.81 + a[:, 2])
            / (torch.cos(rp[:, 0]) * torch.cos(rp[:, 1])).clamp_min(0.5)
        )
        wrench = torch.cat((thrust[:, None], torque), -1)
        rotor_thrust = wrench @ self.inv.T
        return (rotor_thrust.clamp_min(0) / (self.dynamics.p.total_thrust / 4)).sqrt().clamp(0, 1)
