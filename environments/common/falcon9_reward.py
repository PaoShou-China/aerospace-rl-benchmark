"""Falcon 9 stage 1 landing reward function.

Ported from environments/rocket/env.py `rocket_shaping()` (v4 dense shaping,
per-policy-step), scaled to real scale (41.2 m stage, SI units, masses 25-50 t).

Keep the reward DESIGN PATTERN from the old Rocket — same 8 components, same
coefficient roles — but recalibrate thresholds for real Falcon 9:

| Component        | Old (1:1000 toy)     | New (real-scale Falcon 9)     |
|------------------|----------------------|-------------------------------|
| IC z range       | 400-450 m            | 3000-10000 m                  |
| IC vel_z         | randomize_drop=100 m/s (v3) | -40 to -100 m/s          |
| Success |vz|        | 0.02 m/s             | 2.0 m/s (landing burn)        |
| Success |omega|    | 0.02 rad/s           | 0.1 rad/s (body)              |
| Success tilt     | 0.1 rad (5.7°)       | 0.15 rad (8.6°)               |
| Fatal touch |v|   | 1.0 m/s              | 8 m/s (approx 30 km/h)        |
| Fatal touch spin | 0.35 rad/s           | 0.5 rad/s                     |
| Out of bounds xy | 200 m                | 200 m (OK)                    |
| Out of bounds z  | 500 m                | 15000 m                       |
| Max steps (30s)  | 30 * 40 Hz = 1200    | 40 * 50 Hz = 2000             |

Why these numbers?
  - Real Falcon 1 landing speeds ~2-4 m/s (FTS abort at ~10 m/s)
  - At entry-burn descent speeds (50-150 m/s), descent Δz per 50Hz step
    is 1-3 m; DESCENT_COEF = 0.1 → reward 0.1-0.3 per step, comparable to
    HOVER_COST 0.3 — shaping is meaningful.
  - min throttle TWR ≥ 1.08 always (can never hover steady); pulsing /
    aerodynamic descent is the only way. Reward must NOT heavily penalize
    throttle oscillation or climbing transients during hoverslam.

The reward function is called ONCE per POLICY step (50 Hz), using state
snapshots at the START and END of the 10 physics substeps. Contact info
is accumulated inside the substep loop.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Coefficients (recalibrated to real Falcon 9 scale)
# ---------------------------------------------------------------------------
# The hover cost (0.3/step at 50 Hz) → 15/s. Same role as v4: pure time
# pressure. Total hover cost over a 40s episode = 2000 * 0.3 = 600.
HOVER_COST = 0.3

# Hold reward: HOLD_COEF/(||p_xy|| + 0.1). Max at xy=0 is HOLD_COEF/0.1.
# At real scale, xy can be hundreds of metres; the term caps out at small
# xy. Keep 0.005 — negligible vs all other terms.
HOLD_COEF = 0.005

# Lateral progress: C * (d_xy_prev - d_xy) per step.
# Real Falcon 9 at entry burn speed (50-150 m/s) can cover 1-3 m per step
# (at 50 Hz). Want lateral shaping comparable to descent shaping (DESCENT_COEF).
# Old value 10 was calibrated for 1-2 m lateral/step in 1:1000 toy scale →
# reward 10-20/step, which dominated hover cost. At real scale, lateral
# velocities are similar (magnitude-wise) but the C=10 would give ~10/step
# and that's comparable to descent 0.1*1m = 0.1/step. So LATERAL_COEF=10 IS
# fine — my earlier 91748 number was a bug (computation error with
# prev.norm() instead of dxy_prev - dxy).
LATERAL_COEF = 10.0

# Descent progress: C * (z_prev - z). At 50 m/s descent: per 0.02s step,
# Δz = 1.0 m. DESCENT_COEF = 0.1 → reward 0.1/step, matches hover cost -0.3.
# So pure descent does NOT outrun time pressure — agent must add lateral
# shaping to be profitable. Keep at 0.1.
DESCENT_COEF = 0.1

# Deceleration: 0.1 * (dvz+1) * exp(-z) * sign(vz). At high altitude exp(-z)
# ≈ 0 so this term is essentially zero during entry burn; near the ground
# it rewards slowing down. Keep 0.1.
DECEL_COEF = 0.1

# Attitude penalties. Body yaw rate and tilt (rad). Keep 0.5.
YAW_COEF = 0.5
TILT_COEF = 0.5

# Contact bonus per policy step during pad contact; penalize high touchdown
# velocity. CONTACT_VEL_COEF = 1.0 → |vz| = 5 m/s → penalty -5/step.
CONTACT_BASE = 0.5
CONTACT_VEL_COEF = 1.0

# One-shot success bonus. 5000 >> 600 (hover cost over 40s). Dominance OK.
SUCCESS_BONUS = 5000.0

REWARD_KEYS = ('hover', 'hold', 'lateral', 'descent', 'decel', 'attitude',
               'contact', 'success')

# ---------------------------------------------------------------------------
# Thresholds (real-scale)
# ---------------------------------------------------------------------------
# Success thresholds — Falcon 9 lands at ~2-4 m/s.
SUCCESS_VZ_MAX = 2.0         # m/s world descent speed at success
SUCCESS_SPIN_MAX = 0.1       # rad/s body angular rate magnitude
SUCCESS_TILT_MAX = 0.15      # rad (8.6°) roll/pitch magnitude

# Fatal on-pad thresholds — anything faster than ~30 km/h touchdown = crash.
FATAL_PAD_SPEED = 8.0        # m/s speed at first pad contact
FATAL_PAD_SPIN = 0.5         # rad/s

# Bounds
BOUNDS_XY = 200.0            # m half-range
BOUNDS_Z = 15000.0           # m altitude upper bound
BOUNDS_Z_LOW = 0.0           # below ground = fatal

# Maximum simulation time
EPISODE_SECONDS = 40
POLICY_HZ = 50


# ---------------------------------------------------------------------------
# Compute body-frame tilt (roll/pitch combined) from world quaternion (wxyz)
# ---------------------------------------------------------------------------
def _body_tilt_from_quat(q_wxyz: Tensor) -> Tensor:
    """(N, 4) wxyz world quaternion → (N,) combined roll+pitch magnitude [rad]."""
    w, x, y, z = q_wxyz.unbind(-1)
    # Euler roll = atan2(2(wy + zx), 1 - 2(y² + z²))
    # Euler pitch = asin(2(xy - wz))
    sinp = 2 * (x * y - w * z)
    sinp = sinp.clamp(-1.0, 1.0)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    pitch = torch.asin(sinp)
    return torch.sqrt(roll * roll + pitch * pitch)


# ---------------------------------------------------------------------------
# Dense shaping (called once per POLICY step)
# ---------------------------------------------------------------------------
def falcon9_shaping(
    cfg: dict,
    pos_prev: Tensor, pos_now: Tensor,
    vel_z_now: Tensor, vel_z_prev: Tensor,
    quat_now: Tensor, body_omega_now: Tensor,
    contact_pad: Tensor,
    contact_vz: Tensor,
    newly_complete: Tensor,
) -> tuple[Tensor, dict]:
    """Per-policy-step dense reward for Falcon 9.

    Args:
        cfg: dict with keys hover/hold/lateral/descent/decel/yaw/tilt/
             contact_base/contact_vel/success
        pos_prev: (N, 3) world position at START of policy step
        pos_now:  (N, 3) world position at END of policy step
        vel_z_now: (N,) world z velocity at step end (m/s)
        vel_z_prev: (N,) world z velocity at step start (m/s)
        quat_now: (N, 4) wxyz world quaternion at step end
        body_omega_now: (N, 3) body-frame angular rate at step end (rad/s)
        contact_pad: (N,) bool — any pad contact during this policy step
        contact_vz: (N,) world descent speed at FIRST pad contact substep (m/s)
        newly_complete: (N,) bool — first success on this policy step

    Returns:
        total reward (N,), component dict with per-key (N,) tensors
    """
    N = pos_prev.shape[0]
    dev = pos_prev.device
    dxy_prev = pos_prev[:, :2].norm(dim=-1)
    dxy_now  = pos_now[:, :2].norm(dim=-1)
    dz = pos_prev[:, 2] - pos_now[:, 2]     # positive = descending
    dvz = vel_z_now - vel_z_prev

    z_clamped = pos_now[:, 2].clamp_min(0.0)
    vz_sign = torch.where(vel_z_now < 0, 1.0, -1.0)  # vz<0 descending → +1
    decel = (dvz + 1.0) * torch.exp(-z_clamped) * vz_sign

    tilt = _body_tilt_from_quat(quat_now)
    yaw_abs = body_omega_now[:, 2].abs()  # body-frame z = vehicle axis; yaw rate
    # body_omega: body-frame, so components are roll, pitch, yaw — standard
    # body omega convention gives [ω_roll, ω_pitch, ω_yaw] if x=right, y=forward, z=up
    # Falcon 9 uses body z up (vehicle axis), so yaw = ω_z

    comps = {
        'hover':   torch.full_like(dxy_now, -cfg['hover']),
        'hold':    cfg['hold'] / (dxy_now + 0.1),
        'lateral': cfg['lateral'] * (dxy_prev - dxy_now),
        'descent': cfg['descent'] * dz,
        'decel':   cfg['decel'] * decel,
        'attitude': (-cfg['yaw'] * yaw_abs
                     - cfg['tilt'] * tilt),
        'contact': torch.where(contact_pad,
                               cfg['contact_base'] - cfg['contact_vel'] * contact_vz.abs(),
                               torch.zeros_like(dxy_now)),
        'success': newly_complete.to(dxy_now.dtype) * cfg['success'],
    }
    total = sum(comps.values())
    return total, comps


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def default_reward_cfg() -> dict:
    return dict(
        hover=HOVER_COST, hold=HOLD_COEF, lateral=LATERAL_COEF,
        descent=DESCENT_COEF, decel=DECEL_COEF,
        yaw=YAW_COEF, tilt=TILT_COEF,
        contact_base=CONTACT_BASE, contact_vel=CONTACT_VEL_COEF,
        success=SUCCESS_BONUS,
    )


def landing_thresholds() -> dict:
    return dict(
        success_vz_max=SUCCESS_VZ_MAX,
        success_spin_max=SUCCESS_SPIN_MAX,
        success_tilt_max=SUCCESS_TILT_MAX,
        fatal_pad_speed=FATAL_PAD_SPEED,
        fatal_pad_spin=FATAL_PAD_SPIN,
        bounds_xy=BOUNDS_XY,
        bounds_z_high=BOUNDS_Z,
        bounds_z_low=BOUNDS_Z_LOW,
        episode_seconds=EPISODE_SECONDS,
        policy_hz=POLICY_HZ,
    )
