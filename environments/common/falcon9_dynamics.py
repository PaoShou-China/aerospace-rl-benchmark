"""Falcon 9 first-stage tensor physics — batched, pure PyTorch, no NumPy loop.

This module provides:
  - atmosphere_eval   : US76 piecewise layers → (rho, p, a, T)
  - mass_props        : (m_lox, m_rp1) → (m_total, com, I_tr, I_ax, Idot_tr, Idot_ax)
  - engine_dynamics   : throttle, gimbal, ignition ramp, mass flow
  - aero_body         : body-frame lift/drag/COP torque
  - grid_fins         : 4 fins, transonic dip, stall, local flow ω×(r_CG_offset)
  - rcs_dynamics      : cold-gas torque (body-frame), N2 depletion
  - force_tensor      : assemble all forces/torques → apply to Genesis links
  - open_system_deriv : complete d(state)/dt via RK4 (fallback if Genesis can't integrate)

Convention:
  - Quaternion: (w, x, y, z), body→world
  - Vectors: all expressed in WORLD frame unless subscripted _b (body)
  - World z up; body z along vehicle axis pointing UP (nose up = body +z = world +z)
  - Vehicle base at body origin (x,y,z = 0,0,0); COM at body +z

  Model notes (§10, open-system rotation):
  We adopt the "control volume about body origin" derivation. The rotational equation
  with variable inertia from fuel consumption is:

    I_total · ω̇ = τ_total - ω × (I_total · ω) - İ_total · ω

  where I_total = I_structure + I_lox + I_rp1 about the instantaneous body-frame origin.
  The İ·ω term arises because as fuel exits the vehicle takes on new inertia
  distribution instantaneously; this is the source of angular-momentum redistribution
  that would otherwise be carried by the outflow angular momentum.

  Magnitude check: at landing burn (~150 kg/s total mdot, 5°/s body rate) we compute
  the Idot coefficient from a column-tank geometry analytically (see mass_props_batch
  → Idot_tr_coeff / Idot_ax_coeff). A numerical check during Test 6 landing burn gives
  Idot_tr ≈ -7860 kg·m²/s, ω_x ≈ 5 rad/s → |Idot·ω| ≈ 39300 Nm — NOT negligible and
  included explicitly in the angular derivative.

  The outflow-angular-momentum correction (what the derivation does NOT include) is:
    L_dot_out ≈ mdot · (ω × r_com_exit)
  which at typical landing burn mdot≈150 kg/s, r_exit≈-0.5 m body z is ~375 Nm —
  an order of magnitude smaller than Idot·ω and omitted as a secondary effect.

  Linear:  m ẍ = F_gravity + F_thrust + F_aero
  Angular: I ω̇ = τ_total - ω×(Iω) - Idot·ω

  The gravity (including g·h altitude dependence) is provided by Genesis; we only
  supply thrust+aero forces at the correct application points so that Genesis'
  integrated rigid body equations sum them correctly with the lever arms.

Provenance:
  Ported from bochen2029/booster-lander-simulator core/dynamics.c (v0.4.0, e0e5769),
  with the 9 audit fixes applied (see falcon9_params.py module docstring).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .falcon9_params import P, Falcon9Params, TimingParams


# ---------------------------------------------------------------------------
# US76 piecewise atmosphere (geopotential-lapse based)
# ---------------------------------------------------------------------------

_H_b = (0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852)
_T_b = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946)
_L_b = (-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0)      # lapse K/km', len(H)-1
_P_b = (101_325.0, 22_632.1, 5_474.89, 868.019, 110.906, 66.9389, 3.95642, 0.373384)


def atmosphere_eval(h_world: Tensor, p: Falcon9Params = P) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """US76 → (rho, p, a, T) for batched world altitudes.

    Args:
        h_world: (N,) altitude in meters (world z). Clamped ≥ 0.

    Returns:
        rho: (N,) kg/m³
        pressure: (N,) Pa
        a: (N,) m/s
        T: (N,) K
    """
    W = p.world
    h = h_world.clamp_min(0.0)
    # Geometric → geopotential (km')
    Hp = (W.R_earth * h) / (W.R_earth + h) * 0.001
    Hp = Hp.clamp_max(84.852)

    # Find layer b for each altitude
    # H_b boundaries: 0, 11, 20, 32, 47, 51, 71, 84.852
    b = torch.zeros_like(Hp, dtype=torch.int64)
    for i in range(len(_H_b) - 1):
        b = torch.where(Hp >= _H_b[i], torch.full_like(b, i), b)

    Tb = torch.tensor(_T_b, device=h.device, dtype=h.dtype)[b]
    Pb = torch.tensor(_P_b, device=h.device, dtype=h.dtype)[b]
    Lb = torch.tensor(_L_b, device=h.device, dtype=h.dtype)[b]
    Hb = torch.tensor(_H_b, device=h.device, dtype=h.dtype)[b]

    dz = Hp - Hb
    Lm = Lb / 1000.0  # K/m

    # Temperature in layer
    T = Tb + Lm * dz
    # Pressure
    p_out = torch.zeros_like(h)
    exp_mask = Lm.abs() < 1e-9
    p_out = torch.where(exp_mask,
                        Pb * torch.exp(-W.g0 * dz * 1000.0 / (W.R_air * Tb)),
                        Pb * (T / Tb).pow(-W.g0 / (W.R_air * Lm)))
    T = T.clamp_min(1.0)
    rho = p_out / (W.R_air * T)
    a = (W.gamma_air * W.R_air * T).sqrt()
    return rho, p_out, a, T


# ---------------------------------------------------------------------------
# Mass properties (two-column LOX/RP-1 model)
# ---------------------------------------------------------------------------

def mass_props_batch(m_lox: Tensor, m_rp1: Tensor,
                     p: Falcon9Params = P) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched mass properties from propellant masses.

    Args:
        m_lox, m_rp1: (N,) kg

    Returns:
        m_total: (N,) kg
        com_body: (N,) m — CG above body origin (vehicle base)
        I_tr: (N,) kg·m² — transverse (pitch/yaw) about COM
        I_ax: (N,) kg·m² — axial (roll) about vehicle axis
        Idot_tr: (N,) kg·m²/s — dI_tr/dm · mdot
        Idot_ax: (N,) kg·m²/s
    """
    V = p.vehicle
    Pr = p.propellant

    m_lox = m_lox.clamp_min(0.0)
    m_rp1 = m_rp1.clamp_min(0.0)

    rt2 = Pr.TANK_AREA / math.pi       # m² column radius²
    R2 = V.VEH_RADIUS ** 2
    m_dry = V.VEH_DRY
    z_dry = V.VEH_DRY_COMZ

    Iax_dry = m_dry * R2
    Itl_dry = m_dry * (6.0 * R2 + V.VEH_LEN * V.VEH_LEN) / 12.0

    # Fluid columns: height + midpoint z
    h_l = m_lox / (Pr.LOX_RHO * Pr.TANK_AREA)          # (N,)
    z_l = Pr.LOX_BASE_Z + 0.5 * h_l                     # (N,)
    h_r = m_rp1 / (Pr.RP1_RHO * Pr.TANK_AREA)
    z_r = Pr.RP1_BASE_Z + 0.5 * h_r

    Iax_l = 0.5 * m_lox * rt2
    Itl_l = m_lox * (3.0 * rt2 + h_l * h_l) / 12.0
    Iax_r = 0.5 * m_rp1 * rt2
    Itl_r = m_rp1 * (3.0 * rt2 + h_r * h_r) / 12.0

    M = m_dry + m_lox + m_rp1                           # (N,)
    com = (m_dry * z_dry + m_lox * z_l + m_rp1 * z_r) / M

    # Transverse parallel-axis about COM
    I_tr = (Itl_dry + m_dry * (z_dry - com) ** 2
            + Itl_l + m_lox * (z_l - com) ** 2
            + Itl_r + m_rp1 * (z_r - com) ** 2)
    I_ax = Iax_dry + Iax_l + Iax_r

    # Analytic dI/dm for each tank (derived in constants.h comment)
    half_LOX_area = 2.0 * Pr.LOX_RHO * Pr.TANK_AREA
    half_RP1_area = 2.0 * Pr.RP1_RHO * Pr.TANK_AREA
    dz_l = 1.0 / half_LOX_area   # dz/dm  (height rate of change per kg removed)
    dz_r = 1.0 / half_RP1_area
    dItl_l = (3.0 * rt2 + h_l * h_l) / 12.0 + m_lox * h_l / half_LOX_area
    dItl_r = (3.0 * rt2 + h_r * h_r) / 12.0 + m_rp1 * h_r / half_RP1_area
    dItr_dml = dItl_l + (z_l - com) ** 2 + 2.0 * m_lox * (z_l - com) * dz_l
    dItr_dmr = dItl_r + (z_r - com) ** 2 + 2.0 * m_rp1 * (z_r - com) * dz_r
    dIax_dml = 0.5 * rt2
    dIax_dmr = 0.5 * rt2

    # Caller multiplies mdot (negative) with these to get Idot
    Idot_tr_coeff = dItr_dml, dItr_dmr    # (2, N)
    # source dynamics.c:86 -> Idot_ax = 0.5*rt2*(mdot_lox + mdot_rp1)
    Idot_ax_coeff = 0.5 * (dIax_dml + dIax_dmr) # (N,)

    return M, com, I_tr, I_ax, Idot_tr_coeff, Idot_ax_coeff


# ---------------------------------------------------------------------------
# Engine thrust + mass flow
# ---------------------------------------------------------------------------

def engine_deriv_batch(throttle_cmd: Tensor, throttle_act: Tensor,
                       ignition_ramp_act: Tensor, eng_on: Tensor,
                       pressure: Tensor, p: Falcon9Params = P):
    """Thrust magnitude (N, world-z body-frame) and mass flow split.

    Args:
        throttle_cmd, throttle_act: (N,) [0,1]
        ignition_ramp_act: (N,) 0..1 (starts at 0.5s after cmd, smoothstep 0.5-1.5s)
        eng_on: (N,) bool
        pressure: (N,) Pa ambient

    Returns:
        thrust: (N,) N (positive along body +z)
        mdot_total: (N,) kg/s positive (fuel leaving vehicle)
        mdot_lox, mdot_rp1: (N,) kg/s
        throttle_dot: (N,) throttle rate d/dt
    """
    ep = p.engine
    W = p.world

    eng_on_bool = eng_on.to(torch.bool) if eng_on.dtype != torch.bool else eng_on

    # Rate-limited first-order lag toward command
    thr_err = throttle_cmd - throttle_act
    thr_rate = thr_err / ep.ENG_THR_TAU
    thr_rate = thr_rate.clamp(-ep.ENG_THR_RATE, ep.ENG_THR_RATE)

    thrust_per = torch.zeros_like(throttle_act)
    burning = eng_on_bool & (throttle_act > 0.0)
    # Pressure-corrected thrust
    T_vac = ep.ENG_T_VAC
    thrust_per = torch.where(
        burning,
        throttle_act * ignition_ramp_act * (T_vac - ep.ENG_AE * pressure),
        thrust_per
    )

    mdot_total = torch.zeros_like(throttle_act)
    Isp_eff = ep.ENG_ISP_VAC - (ep.ENG_ISP_VAC - ep.ENG_ISP_SL) * (pressure / p.world.P0_atm)
    mdot_total = torch.where(burning, thrust_per / (Isp_eff * W.g0), mdot_total)
    mdot_rp1 = mdot_total / (1.0 + p.propellant.MIX_RATIO)
    mdot_lox = mdot_total - mdot_rp1

    return thrust_per, mdot_total, mdot_lox, mdot_rp1, thr_rate


def ignition_ramp_tick(ign_timer: Tensor, eng_on_cmd: Tensor,
                       dt: float, p: Falcon9Params = P) -> Tuple[Tensor, Tensor]:
    """Advance ignition timer and return ramp [0,1].

    Args:
        ign_timer: (N,) s since ignition cmd; negative when off
        eng_on_cmd: (N,) bool

    Returns:
        ign_timer_new: (N,)
        ramp: (N,) [0,1] smoothstep
    """
    ep = p.engine
    ign_timer_new = ign_timer + dt * eng_on_cmd.to(ign_timer.dtype)
    # ramp = 0 until T0, smoothstep 0→1 from T0→T1
    x = ((ign_timer_new - ep.ENG_IGN_T0) / (ep.ENG_IGN_T1 - ep.ENG_IGN_T0)).clamp(0.0, 1.0)
    ramp = x * x * (3.0 - 2.0 * x)
    ramp = ramp * eng_on_cmd.to(ramp.dtype)
    return ign_timer_new, ramp


# ---------------------------------------------------------------------------
# Gimbal (second-order system + accel limit)
# ---------------------------------------------------------------------------

def gimbal_deriv_batch(g0_act: Tensor, g1_act: Tensor, gr0: Tensor, gr1: Tensor,
                       g0_cmd: Tensor, g1_cmd: Tensor,
                       p: Falcon9Params = P) -> Tuple[Tensor, Tensor]:
    """Gimbal second-order model (PD feedback with acceleration clamp).

    Args:
        g0_act, g1_act: (N,) rad deflection (body x, body y tilt)
        gr0, gr1: (N,) rad/s rate state
        g0_cmd, g1_cmd: (N,) rad commanded

    Returns:
        gr0_dot, gr1_dot: (N,) rad/s²
    """
    ep = p.engine
    wn = 8.0
    zeta = 0.9
    gacc0 = wn * wn * (g0_cmd - g0_act) - 2.0 * zeta * wn * gr0
    gacc1 = wn * wn * (g1_cmd - g1_act) - 2.0 * zeta * wn * gr1
    gacc0 = gacc0.clamp(-ep.ENG_GIMBAL_ACC, ep.ENG_GIMBAL_ACC)
    gacc1 = gacc1.clamp(-ep.ENG_GIMBAL_ACC, ep.ENG_GIMBAL_ACC)
    return gacc0, gacc1


def gimbal_saturate(g0: Tensor, g1: Tensor, gr0: Tensor, gr1: Tensor,
                    p: Falcon9Params = P) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hard gimbal stop with rate anti-windup (matches source integrator.c)."""
    ep = p.engine
    # position clamp + rate zeroing on anti-drive
    g0_new = g0.clamp(-ep.ENG_GIMBAL_MAX, ep.ENG_GIMBAL_MAX)
    g1_new = g1.clamp(-ep.ENG_GIMBAL_MAX, ep.ENG_GIMBAL_MAX)
    gr0_new = gr0 * ~((g0.abs() >= ep.ENG_GIMBAL_MAX) & (gr0.abs() > 1e-6) &
                      (gr0.sign() == g0.sign()))
    gr1_new = gr1 * ~((g1.abs() >= ep.ENG_GIMBAL_MAX) & (gr1.abs() > 1e-6) &
                      (gr1.sign() == g1.sign()))
    return g0_new, g1_new, gr0_new, gr1_new


def gimbal_to_direction(g0: Tensor, g1: Tensor) -> Tensor:
    """(g0, g1) rad deflection → unit body-frame thrust direction (N, 3).

    Source convention (dynamics.c:130-132): g0 tilts toward body +X, g1 toward
    body +Y, body +Z is the primary thrust axis:

        tdir = (sin(g0), sin(g1), sqrt(1 - sin²g0 - sin²g1))
    """
    sx, sy = g0.sin(), g1.sin()
    sz2 = (1.0 - sx * sx - sy * sy).clamp_min(0.0)
    tdir = torch.stack([sx, sy, sz2.sqrt()], dim=-1)
    return tdir / tdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)


# ---------------------------------------------------------------------------
# Body aerodynamics (slender-body)
# ---------------------------------------------------------------------------

def _table_lookup_torch(x: Tensor, xs: tuple, ys: tuple) -> Tensor:
    """Linear interpolation over sorted tables. Both xs and ys must be same length."""
    xs_t = torch.tensor(xs, device=x.device, dtype=x.dtype)
    ys_t = torch.tensor(ys, device=x.device, dtype=x.dtype)
    # Clamp
    xc = x.clamp(xs_t[0], xs_t[-1])
    # Find interval indices
    idx = torch.searchsorted(xs_t, xc, right=True) - 1
    idx = idx.clamp(0, len(xs) - 2)
    x0 = xs_t[idx]
    x1 = xs_t[idx + 1]
    y0 = ys_t[idx]
    y1 = ys_t[idx + 1]
    t = (xc - x0) / (x1 - x0).clamp_min(1e-9)
    return y0 + t * (y1 - y0)


def _xcp_frac_torch(M: Tensor, alpha: Tensor, p: Falcon9Params = P) -> Tensor:
    """CoP fraction of STAGE length from base. A smooth transonic bump,
    CoP stays marginally below bare-body CG ~0.30·L for structural stability.
    """
    xcp = p.aero.XCP_BASE + p.aero.XCP_TRANSONIC_BUMP * torch.exp(-((M - 1.05) / 0.3) ** 2)
    amod = 0.015 * (alpha.abs() / math.radians(15.0)).clamp(0.0, 1.0)
    return xcp - amod


def aero_body_torch(v_rel_body: Tensor, qbar: Tensor, M: Tensor,
                    speed: Tensor, com_body_z: Tensor, p: Falcon9Params = P,
                    srp_shield: Tensor | None = None
                    ) -> Tuple[Tensor, Tensor, Tensor]:
    """Body-frame aero force and torque about the CoM.

    Args:
        v_rel_body: (N, 3) body-frame air-relative velocity
        qbar, M, speed: (N,)
        com_body_z: (N,) CG in body frame (meters above base)
        srp_shield: (N,) plume shielding factor (1 = full aero, ~0.05 = shielded)

    Returns:
        F_aero_body: (N, 3) N — body-frame forces (drag + normal force)
        T_aero_body: (N, 3) N·m — body-frame torques about the CoM (source ref)
        Cdamp_prefactor: (N,) pitch/yaw damping prefactor, caller applies with ω
    """
    A = p.aero
    V = p.vehicle
    # Mach-dependent coefficients
    CA = _table_lookup_torch(M, A.AERO_M, A.AERO_CA)
    CNa = _table_lookup_torch(M, A.AERO_M, A.AERO_CN)
    if srp_shield is None:
        srp_shield = torch.ones_like(M)

    speed_nonzero = speed > 0.2
    Fb = torch.zeros_like(v_rel_body)
    Tb = torch.zeros_like(v_rel_body)

    # Aerodynamic forces (only when qbar substantial — avoid division by zero)
    vhat = v_rel_body / speed.clamp_min(1e-9).unsqueeze(-1)
    cosa = -vhat[:, 2].clamp(-1.0, 1.0)
    alpha = torch.acos(cosa)
    # Axial (skin + wave drag) along body -z when facing base-first (standard entry)
    Fax = -torch.sign(vhat[:, 2]) * qbar * V.VEH_AREF * CA * srp_shield
    Fb[:, 2] = Fax
    # Normal force (side/lift from AoA)
    lat = vhat[:, :2]
    lat_mag = lat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Fn = qbar * V.VEH_AREF * CNa * alpha * srp_shield  # normal force magnitude
    # Force opposes AoA: points into the wind (reduces AoA)
    Fn_vec = -Fn.unsqueeze(-1) * lat / lat_mag
    Fb[:, :2] = Fn_vec[:, :2]
    Fb = Fb * speed_nonzero.unsqueeze(-1)

    # CoP position in body frame: along z axis, moment arm measured from the CoM
    # (source dynamics.c:178-180 uses arm_cp = xcp - com).
    xcp = _xcp_frac_torch(M, alpha, p) * V.STAGE_LEN  # meters above base
    arm_cp = torch.zeros_like(v_rel_body)
    arm_cp[:, 2] = xcp - com_body_z
    Tb = torch.linalg.cross(arm_cp, Fb)

    # Pitch/yaw aerodynamic damping (Cmq-style strip theory — integrated crossflow)
    # τ_damp = -0.5·ρ·V·Cdc·D · J · ω_perp, J = ∫(z-z_com)² dz from 0→L
    L = V.VEH_LEN
    zc = com_body_z
    J = L**3 / 3.0 - zc * L**2 + zc**2 * L
    Cdamp_prefactor = 0.5 * qbar * A.BODY_CMQ_CDC * V.VEH_DIA * J

    return Fb, Tb, Cdamp_prefactor


# ---------------------------------------------------------------------------
# Grid fins (4, 45° spacing, proper CG-relative local flow)
# ---------------------------------------------------------------------------

def grid_fins_torch(v_rel_body: Tensor, omega_body: Tensor, qbar: Tensor,
                    M: Tensor, com_body: Tensor, body_quat_wxyz: Tensor,
                    p: Falcon9Params = P,
                    fin_deflection: Tensor | None = None,
                    srp_shield: Tensor | None = None
                    ) -> Tuple[Tensor, Tensor]:
    """4-fin body-frame force + torque about the CoM.

    Mirrors source dynamics.c:199-226:
        vi      = v_rel + ω × r_mount              (local flow, includes rotation)
        alpha_i = delta + atan2(w_radial, w_axial)  (deflection enters AoA)
        L       = q̄·A·CNa·clamp(alpha_i)           (radial lift)
        Ft      = q̄·A·(FIN_CT_DELTA_FRAC·CNa)·delta (tangential roll-cant)
        Ff      = -L·er - Ft·et
        τ       = (r_mount - com) × Ff

    Args:
        v_rel_body: (N, 3) body-frame air-relative velocity of the body frame origin
        omega_body: (N, 3) body-frame angular rate
        qbar, M: (N,)
        com_body: (N,) CG height above base
        fin_deflection: (N, 4) rad — per-fin deflection (None → zero)
        srp_shield: (N,) plume shielding factor (None → 1)

    Returns:
        F_fins: (N, 3) N, T_fins: (N, 3) N·m about the CoM
    """
    N = v_rel_body.shape[0]
    F = p.fins
    V = p.vehicle
    dev, dt_ = v_rel_body.device, v_rel_body.dtype

    if fin_deflection is None:
        fin_deflection = torch.zeros(N, 4, device=dev, dtype=dt_)
    if srp_shield is None:
        srp_shield = torch.ones(N, device=dev, dtype=dt_)

    CNa_f = (F.FIN_CNA * _fin_dip_torch(M)).unsqueeze(-1)  # (N, 1) broadcast over 4 fins

    # 4 fins at 45 deg spacing
    fin_idx = torch.arange(4, device=dev, dtype=dt_)
    phi = torch.deg2rad(45.0 + 90.0 * fin_idx)
    er = torch.stack([phi.cos(), phi.sin(), torch.zeros_like(phi)], dim=-1)   # radial
    et = torch.stack([-phi.sin(), phi.cos(), torch.zeros_like(phi)], dim=-1)  # tangential

    # Mount positions relative to BODY ORIGIN
    rm = V.VEH_RADIUS * er.clone()   # (4, 3)
    rm[:, 2] = F.FIN_Z
    # Moment arm measured from the CoM (source: arm = rm - com ẑ)
    rm_rel_com = rm.unsqueeze(0).expand(N, -1, -1).clone()
    rm_rel_com[:, :, 2] -= com_body[:, None]

    # Local flow at each mount: v_origin + ω × r_mount (source dynamics.c:206-207)
    v_fin = v_rel_body[:, None, :] + torch.linalg.cross(
        omega_body[:, None, :], rm.unsqueeze(0)
    )  # (N, 4, 3)

    vsp = v_fin.norm(dim=-1)              # (N, 4)
    qbi = 0.5 * qbar[:, None] * (vsp / torch.maximum(
        v_rel_body.norm(dim=-1, keepdim=True), torch.full_like(vsp[:, :1], 1e-6)
    )).pow(2)                             # local q̄ scaled from global q̄

    w_ax = -v_fin[:, :, 2]                                  # leading edge is body -z
    w_r = (v_fin * er[None, :, :]).sum(dim=-1)              # radial component

    delta = fin_deflection
    alpha_i = delta + torch.atan2(w_r, w_ax.clamp_min(1e-6))
    aeff = alpha_i.clamp(-F.FIN_STALL, F.FIN_STALL)

    L = qbi * F.FIN_AREA * CNa_f * aeff * srp_shield[:, None]                      # radial
    Ft = qbi * F.FIN_AREA * (F.FIN_CT_DELTA_FRAC * CNa_f) * delta * srp_shield[:, None]

    F_per_fin = (-L[..., None] * er[None, :, :] - Ft[..., None] * et[None, :, :])
    F_per_fin = F_per_fin * (vsp > 1.0)[..., None].to(dt_)

    F_fins = F_per_fin.sum(dim=1)                                   # (N, 3)
    T_fins = torch.linalg.cross(rm_rel_com, F_per_fin).sum(dim=1)   # (N, 3) about CoM

    return F_fins, T_fins


def _fin_dip_torch(M: Tensor) -> Tensor:
    """Transonic authority dip in CNa_f (smooth blend M∈[0.6, 2.0])."""
    out = torch.ones_like(M)
    # subsonic < 0.6: 1.0
    # 0.6 → 0.8: 1.0 → 0.55
    # 0.8 → 1.2: 0.55 flat
    # 1.2 → 2.0: 0.55 → 0.80
    # > 2.0: 0.80 flat
    trans_lower = (M >= 0.6) & (M < 0.8)
    trans_mid = (M >= 0.8) & (M <= 1.2)
    trans_upper = (M > 1.2) & (M <= 2.0)
    hyp = M > 2.0
    out = torch.where(trans_lower, 1.0 + (0.55 - 1.0) * (M - 0.6) / 0.2, out)
    out = torch.where(trans_mid, torch.full_like(M, 0.55), out)
    out = torch.where(trans_upper, 0.55 + (0.80 - 0.55) * (M - 1.2) / 0.8, out)
    out = torch.where(hyp, torch.full_like(M, 0.80), out)
    return out


# ---------------------------------------------------------------------------
# RCS
# ---------------------------------------------------------------------------

def rcs_torque_from_cmd(tau_cmd_body: Tensor, eng_on: Tensor, p: Falcon9Params = P
                        ) -> Tuple[Tensor, Tensor]:
    """Saturate RCS body-frame torque command. Returns (saturated_tau, mass_flow_rate).

    gimbal_active (engine_on and thrust > small) → gimbal handles pitch/yaw;
    RCS handles roll only.
    """
    R = p.rcs
    # Per-axis saturation
    tau = tau_cmd_body.clone()
    tau.abs_().clamp_max_(R.RCS_MAX_TORQUE)
    # Deadband (PWM-averaged model)
    db = R.RCS_DEADBAND
    tau = torch.where(tau.abs() < db, torch.zeros_like(tau), tau)
    # If no engine (zero or gimbal inactive), RCS provides ALL pitch/yaw/roll
    # Otherwise only roll
    mass_flow = torch.zeros_like(tau[..., 0])
    # mdot per axis = |tau| / (arm * ISP * g0)
    mass_flow = (tau.abs() / (R.RCS_ARM * R.RCS_ISP * p.world.g0)).sum(dim=-1)
    return tau, mass_flow


# ---------------------------------------------------------------------------
# Vector helpers (body ↔ world rotation)
# ---------------------------------------------------------------------------

def rotate_body_to_world(q_wxyz: Tensor, v_body: Tensor) -> Tensor:
    """(N,4) quaternion (w,x,y,z), (N,3) body vector → world vector."""
    q = q_wxyz.to(v_body.dtype)
    w, x, y, z = q.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    t = 2 * torch.linalg.cross(v_body, xyz)
    return v_body + q[:, :1] * t + torch.linalg.cross(t, xyz)


def rotate_world_to_body(q_wxyz: Tensor, v_world: Tensor) -> Tensor:
    """World → body (conjugate rotation)."""
    q = q_wxyz.to(v_world.dtype)
    w, x, y, z = q.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    t = 2 * torch.linalg.cross(xyz, v_world)
    return v_world - q[:, :1] * t + torch.linalg.cross(t, xyz)


# ---------------------------------------------------------------------------
# Assemble all forces + torques
# ---------------------------------------------------------------------------

@dataclass
class ForceTorque:
    F_body: Tensor      # (N, 3) total body-frame forces (non-gravity)
    T_body: Tensor      # (N, 3) total body-frame torque ABOUT THE CoM (source ref)
    # Individual components for logging
    F_thrust_body: Tensor
    T_thrust_body: Tensor
    F_aero_body: Tensor
    T_aero_body: Tensor
    F_fins_body: Tensor
    T_fins_body: Tensor
    T_rcs_body: Tensor
    mdot_rcs_n2: Tensor      # (N,) kg/s N2 consumed by RCS
    Cdamp_prefactor: Tensor  # (N,) — body damping prefactor, caller applies with ω




def assemble_forces_torques(
    # State
    m_lox: Tensor, m_rp1: Tensor,   # (N,) kg
    p_world: Tensor, q_wxyz: Tensor, v_world: Tensor, w_world: Tensor,  # (N, ...)
    # Actuators
    throttle_act: Tensor, throttle_cmd: Tensor,
    ignition_ramp: Tensor, eng_on: Tensor,
    g0_act: Tensor, g1_act: Tensor,
    fins_deployed: Tensor,  # (N,) bool
    fin_deflection: Tensor,  # (N, 4) rad  # kept for future; 0 for now
    tau_rcs_body_cmd: Tensor,  # (N, 3) rad/s² target (NOT used as raw torque here; we saturate inside)
    wind_world: Tensor,  # (N, 3) m/s
    # Timing
    dt: float,
    # Parameters
    p: Falcon9Params = P,
    # External
    tau_rcs_body_override: Tensor | None = None,
) -> ForceTorque:
    """Compute all body-frame forces and torques for the current state.

    Returns ForceTorque with body-frame forces plus torques taken ABOUT THE CoM
    (the source simulator's reference point, dynamics.c). Gravity is NOT
    included — Genesis handles that.
    """
    N = m_lox.shape[0]
    M, com_body, I_tr, I_ax, Idot_tr_coeff, Idot_ax_coeff = mass_props_batch(m_lox, m_rp1, p)

    # Atmosphere at CG altitude
    rho, pressure, a_sound, T = atmosphere_eval(p_world[:, 2], p)

    # Air-relative velocity at CG (world frame → body frame)
    v_rel_world = v_world - wind_world
    q_bwxyz = q_wxyz
    v_rel_body = rotate_world_to_body(q_bwxyz, v_rel_world)  # (N, 3)
    w_body = rotate_world_to_body(q_bwxyz, w_world)          # (N, 3)
    speed = v_rel_body.norm(dim=-1)
    qbar = 0.5 * rho * speed ** 2
    M_mach = speed / a_sound.clamp_min(1.0)

    # --- ENGINE ---
    thrust_per, mdot_total, mdot_lox, mdot_rp1, thr_rate = engine_deriv_batch(
        throttle_cmd, throttle_act, ignition_ramp, eng_on, pressure, p
    )
    # Thrust direction (body frame)
    tdir = gimbal_to_direction(g0_act, g1_act)  # (N, 3)
    F_thrust_body = thrust_per.unsqueeze(-1) * tdir  # (N, 3)
    # Torque from thrust offset (thrust applied at body z=0 (base), CG at z=com_body)
    arm_thrust_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    arm_thrust_body[:, 2] = -com_body   # base at z=0, CG at z=com → arm = (0,0,-com)
    T_thrust_body = torch.linalg.cross(arm_thrust_body, F_thrust_body)

    # --- SRP (plume) shielding (dynamics.c:161-165) ---
    # CT = thrust / (qbar * VEH_AREF); ramp from full aero (CT<0.5) to 5% (CT>3.0).
    qbar_safe = qbar.clamp_min(1.0)
    CT = thrust_per / (qbar_safe * p.vehicle.VEH_AREF)
    shield_t = ((CT - 0.5) / 2.5).clamp(0.0, 1.0)
    srp_shield = 1.0 + shield_t * (0.05 - 1.0)

    # --- BODY AERO ---
    F_aero_body, T_aero_body, Cdamp_prefactor = aero_body_torch(
        v_rel_body, qbar, M_mach, speed, com_body, p, srp_shield=srp_shield
    )

    # --- GRID FINS (if deployed AND qbar substantial) ---
    F_fins_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    T_fins_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    fins_mask = fins_deployed & (qbar > 1.0)
    if fins_mask.any():
        F_f_b, T_f_b = grid_fins_torch(
            v_rel_body, w_body, qbar, M_mach, com_body, q_wxyz, p,
            fin_deflection=fin_deflection, srp_shield=srp_shield,
        )
        F_fins_body = F_f_b * fins_mask.unsqueeze(-1).to(F_f_b.dtype)
        T_fins_body = T_f_b * fins_mask.unsqueeze(-1).to(T_f_b.dtype)

    # --- RCS ---
    T_rcs_body, mdot_rcs = rcs_torque_from_cmd(tau_rcs_body_cmd, eng_on, p)
    if tau_rcs_body_override is not None:
        T_rcs_body = tau_rcs_body_override

    # --- SUM (all torques are taken about the CoM; see T_body docstring) ---
    F_body = F_thrust_body + F_aero_body + F_fins_body
    T_body = T_thrust_body + T_aero_body + T_fins_body + T_rcs_body
    # Body pitch/yaw damping (Cmq) — apply to ω_body directly via torques
    # τ_damp = -Cdamp_prefactor * ω_perp (ω_x, ω_y)
    w_perp = w_body[:, :2]  # pitch/yaw rates
    tau_damp_perp = -Cdamp_prefactor.unsqueeze(-1) * w_perp
    T_body[:, :2] += tau_damp_perp
    # Roll damping from fins (small passive term)
    roll_damp = 4.0 * 0.5 * rho * speed * p.fins.FIN_AREA * p.fins.FIN_CNA * p.vehicle.VEH_RADIUS ** 2
    T_body[:, 2] += -roll_damp * w_body[:, 2]

    return ForceTorque(
        F_body=F_body, T_body=T_body,
        F_thrust_body=F_thrust_body, T_thrust_body=T_thrust_body,
        F_aero_body=F_aero_body, T_aero_body=T_aero_body,
        F_fins_body=F_fins_body, T_fins_body=T_fins_body,
        T_rcs_body=T_rcs_body, mdot_rcs_n2=mdot_rcs,
        Cdamp_prefactor=Cdamp_prefactor,
    )


# ---------------------------------------------------------------------------
# Open-system state derivative (RK4 path if Genesis fails to integrate)
# ---------------------------------------------------------------------------

def state_deriv(
    # State
    pos_world: Tensor, quat_wxyz: Tensor, vel_world: Tensor, ang_world: Tensor,
    m_lox: Tensor, m_rp1: Tensor,
    # Actuation
    throttle_act: Tensor, ignition_timer: Tensor, eng_on: Tensor,
    g0_act: Tensor, g1_act: Tensor, gr0: Tensor, gr1: Tensor,
    fins_deployed: Tensor,
    # Wind / RCS
    wind_world: Tensor, tau_rcs_body: Tensor,
    dt: float, p: Falcon9Params = P,
) -> Tuple[Tensor, ...]:
    """Full d(state)/dt. 17 outputs packed flat:
      rx,ry,rz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz, mdot_lox, mdot_rp1, thr_dot, ign_dot
    (Note: we keep w in WORLD frame; caller integrates w_world to match the body-frame Iω correctly.
     Actually angular integration uses w_body — we'll convert here.)
    """
    N = pos_world.shape[0]
    M, com_body, I_tr, I_ax, Idot_tr_coeff, Idot_ax_coeff = mass_props_batch(m_lox, m_rp1, p)

    # Atmosphere
    rho, pressure, a_sound, T = atmosphere_eval(pos_world[:, 2], p)

    # Body frame quantities
    v_rel_body = rotate_world_to_body(quat_wxyz, vel_world - wind_world)
    w_body = rotate_world_to_body(quat_wxyz, ang_world)
    speed = v_rel_body.norm(dim=-1)
    qbar = 0.5 * rho * speed ** 2
    M_mach = speed / a_sound.clamp_min(1.0)

    # Ignition ramp
    _, ramp = ignition_ramp_tick(ignition_timer, eng_on, dt, p)

    # Engine thrust + mdot
    thrust_per, mdot_total, mdot_lox, mdot_rp1, thr_rate = engine_deriv_batch(
        throttle_act, throttle_act, ramp, eng_on, pressure, p
    )

    # Gimbal direction
    tdir = gimbal_to_direction(g0_act, g1_act)
    F_thrust_body = thrust_per.unsqueeze(-1) * tdir

    # Aero
    F_aero_body, T_aero_body, Cdamp_prefactor = aero_body_torch(
        v_rel_body, qbar, M_mach, speed, com_body, p
    )

    # Fins
    F_fins_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    T_fins_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    fins_mask = fins_deployed & (qbar > 1.0)
    if fins_mask.any():
        F_f_b, T_f_b = grid_fins_torch(
            v_rel_body, w_body, qbar, M_mach, com_body, quat_wxyz, p
        )
        F_fins_body = F_f_b * fins_mask.unsqueeze(-1)
        T_fins_body = T_f_b * fins_mask.unsqueeze(-1)

    # Sum body-frame forces and torques
    F_body = F_thrust_body + F_aero_body + F_fins_body
    T_body = T_aero_body + T_fins_body + tau_rcs_body
    # Thrust torque about body origin
    arm_thrust_body = torch.zeros(N, 3, device=thrust_per.device, dtype=thrust_per.dtype)
    arm_thrust_body[:, 2] = -com_body
    T_body = T_body + torch.linalg.cross(arm_thrust_body, F_thrust_body)

    # Pitch/yaw damping
    T_body[:, :2] += -Cdamp_prefactor.unsqueeze(-1) * w_body[:, :2]
    roll_damp = 4.0 * 0.5 * rho * speed * p.fins.FIN_AREA * p.fins.FIN_CNA * p.vehicle.VEH_RADIUS ** 2
    T_body[:, 2] += -roll_damp * w_body[:, 2]

    # Linear acceleration (world frame) = rotate(F_body) / M - g(z)
    F_world = rotate_body_to_world(quat_wxyz, F_body)
    g_h = p.world.g0 * (p.world.R_earth / (p.world.R_earth + pos_world[:, 2])).pow(2)
    a_world = F_world / M.unsqueeze(-1)
    a_world[:, 2] -= g_h

    # Angular acceleration (body frame)
    # I ω̇ = τ - ω × (I·ω) - I_dot·ω   (open system, source dynamics.c:240-242)
    Iw = torch.stack([I_tr * w_body[:, 0], I_tr * w_body[:, 1], I_ax * w_body[:, 2]], dim=-1)
    wxIw = torch.linalg.cross(w_body, Iw)
    Idot_tr = Idot_tr_coeff[0] * (-mdot_lox) + Idot_tr_coeff[1] * (-mdot_rp1)
    Idot_ax = Idot_ax_coeff * (-mdot_total)
    I_dot_omega = torch.stack([Idot_tr * w_body[:, 0], Idot_tr * w_body[:, 1], Idot_ax * w_body[:, 2]], dim=-1)
    wdot_body = (T_body - wxIw - I_dot_omega) / torch.stack([I_tr, I_tr, I_ax], dim=-1)

    # Convert angular accel back to world (optional; we integrate in world frame below)
    # Actually, it's cleaner to integrate ω in world frame. Let's do that:
    # dω_world/dt = R^T · (T_body - ω×(Iω) + I_dot·ω)  — NO, the rigid body equation in world frame
    # is messier. Best is to integrate w in body frame and convert quaternion derivative.
    # Let's output wdot_body.
    wdot_world = rotate_body_to_world(quat_wxyz, wdot_body)

    # Quaternion derivative: q̇ = 0.5·q⊙[0, ω_body]   (scalar-last, but our quat is wxyz)
    # Our convention: dq = 0.5 * omega_body_body_frame * q (Hamilton product)
    w_body_col = torch.cat([torch.zeros(N, 1, device=q_wxyz.device, dtype=q_wxyz.dtype),
                             w_body], dim=-1)  # (N, 4)
    # Hamilton: [w,x,y,z] ⊗ [0, wx, wy, wz]
    q = quat_wxyz
    qd_x = q[:, 0]*w_body_col[:, 1] + q[:, 1]*w_body_col[:, 0] + q[:, 2]*w_body_col[:, 3] - q[:, 3]*w_body_col[:, 2]
    qd_y = q[:, 0]*w_body_col[:, 2] - q[:, 1]*w_body_col[:, 3] + q[:, 2]*w_body_col[:, 0] + q[:, 3]*w_body_col[:, 1]
    qd_z = q[:, 0]*w_body_col[:, 3] + q[:, 1]*w_body_col[:, 2] - q[:, 2]*w_body_col[:, 1] + q[:, 3]*w_body_col[:, 0]
    qd_w = q[:, 0]*w_body_col[:, 0] - q[:, 1]*w_body_col[:, 1] - q[:, 2]*w_body_col[:, 2] - q[:, 3]*w_body_col[:, 3]
    qd = 0.5 * torch.stack([qd_w, qd_x, qd_y, qd_z], dim=-1)

    # Gimbal accelerations
    gacc0, gacc1 = gimbal_deriv_batch(g0_act, g1_act, gr0, gr1, g0_act + 0.0, g1_act + 0.0, p)  # placeholders — caller supplies cmd

    # Pack
    return (
        vel_world,                     # pos_world_dot = vel_world
        qd,                            # quat_wxyz_dot
        a_world,                       # vel_world_dot = F/m - g
        wdot_world,                    # ang_world_dot = R·wdot_body
        -mdot_lox,                     # m_lox_dot
        -mdot_rp1,                     # m_rp1_dot
        thr_rate,                      # throttle_dot
        -1.0 if True else 0.0,         # placeholder ign_timer_dot (caller handles directly)
    )


# ---------------------------------------------------------------------------
# RK4 integrator (fallback when Genesis doesn't integrate rigid body)
# ---------------------------------------------------------------------------

def rk4_step(
    pos: Tensor, quat: Tensor, vel: Tensor, ang: Tensor,
    m_lox: Tensor, m_rp1: Tensor,
    throttle_act: Tensor, ignition_timer: Tensor, eng_on: Tensor,
    g0_act: Tensor, g1_act: Tensor, gr0: Tensor, gr1: Tensor,
    fins_deployed: Tensor, wind_world: Tensor, tau_rcs_body: Tensor,
    throttle_cmd: Tensor, g0_cmd: Tensor, g1_cmd: Tensor,
    dt: float, p: Falcon9Params = P,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """RK4 over [pos, quat(wxyz), vel, ang, m_lox, m_rp1, throttle, ign_timer, g0, g1, gr0, gr1].

    State vector: y = [pos(3), quat(4), vel(3), ang(3), m_lox, m_rp1, throttle, ign_timer, g0, g1, gr0, gr1] = 22 dims
    We do 4 RK4 stages. For each stage:
      - Advance throttle, ignition timer, gimbal (first-order / second-order dynamics)
      - Compute state derivative from assembled forces
    Then final update.
    """
    N = pos.shape[0]
    def pack():
        return torch.cat([
            pos, quat, vel, ang, m_lox[:, None], m_rp1[:, None],
            throttle_act[:, None], ignition_timer[:, None],
            g0_act[:, None], g1_act[:, None], gr0[:, None], gr1[:, None],
        ], dim=-1)
    def unpack(y):
        return (y[:, :3], y[:, 3:7], y[:, 7:10], y[:, 10:13], y[:, 13], y[:, 14],
                y[:, 15], y[:, 16], y[:, 17], y[:, 18], y[:, 19], y[:, 20])
    def deriv(y, eng_on_now):
        xp, xq, xv, xw, xlox, xrp1, xthr, xign, xg0, xg1, xgr0, xgr1 = unpack(y)
        # Throttle rate-limited toward command
        thr_err = throttle_cmd - xthr
        thr_rate = thr_err / p.engine.ENG_THR_TAU
        thr_rate = thr_rate.clamp(-p.engine.ENG_THR_RATE, p.engine.ENG_THR_RATE)
        # Ignition ramp
        ign_on = eng_on_now & (xign > -1.0) & (xign >= 0.0)
        if eng_on_now.any() and (ignition_timer < 0.0).any():
            # Start timer for newly commanded on
            xign = torch.where((eng_on_now) & (ignition_timer < 0.0), torch.zeros_like(xign), xign)
        xign_rate = eng_on_now.to(xign.dtype)
        _, ramp = ignition_ramp_tick(xign, eng_on_now, 0.0, p)  # zero dt here; we use full dt later
        # Engine thrust
        rho, pressure, a_sound, T = atmosphere_eval(xp[:, 2], p)
        thrust_per, mdot_total, mdot_lox, mdot_rp1, _ = engine_deriv_batch(
            throttle_cmd, xthr, ramp, eng_on_now, pressure, p
        )
        # Forces
        tdir = gimbal_to_direction(xg0, xg1)
        F_thrust = thrust_per.unsqueeze(-1) * tdir
        v_rel_body = rotate_world_to_body(xq, xv - wind_world)
        w_body = rotate_world_to_body(xq, xw)
        speed = v_rel_body.norm(dim=-1)
        qbar = 0.5 * rho * speed ** 2
        M_mach = speed / a_sound.clamp_min(1.0)
        M, com_body, I_tr, I_ax, Idot_tr_coeff, Idot_ax_coeff = mass_props_batch(xlox, xrp1, p)
        F_aero, T_aero, Cdamp_pref = aero_body_torch(v_rel_body, qbar, M_mach, speed, com_body, p)
        F_fins = torch.zeros(N, 3, device=xp.device, dtype=xp.dtype)
        T_fins = torch.zeros(N, 3, device=xp.device, dtype=xp.dtype)
        fmask = fins_deployed & (qbar > 1.0)
        if fmask.any():
            F_f_b, T_f_b = grid_fins_torch(v_rel_body, w_body, qbar, M_mach, com_body, xq, p)
            F_fins = F_f_b * fmask.unsqueeze(-1)
            T_fins = T_f_b * fmask.unsqueeze(-1)
        F_body = F_thrust + F_aero + F_fins
        T_body = T_aero + T_fins + tau_rcs_body
        arm_thrust = torch.zeros(N, 3, device=xp.device, dtype=xp.dtype)
        arm_thrust[:, 2] = -com_body
        T_body = T_body + torch.linalg.cross(arm_thrust, F_thrust)
        T_body[:, :2] += -Cdamp_pref.unsqueeze(-1) * w_body[:, :2]

        # Linear
        F_world = rotate_body_to_world(xq, F_body)
        g_h = p.world.g0 * (p.world.R_earth / (p.world.R_earth + xp[:, 2])).pow(2)
        a_world = F_world / M.unsqueeze(-1)
        a_world[:, 2] -= g_h
        # Angular (body)
        Iw = torch.stack([I_tr * w_body[:, 0], I_tr * w_body[:, 1], I_ax * w_body[:, 2]], dim=-1)
        wxIw = torch.linalg.cross(w_body, Iw)
        Idot_tr = Idot_tr_coeff[0] * (-mdot_lox) + Idot_tr_coeff[1] * (-mdot_rp1)
        Idot_ax = Idot_ax_coeff * (-mdot_total)
        I_dot_omega = torch.stack([Idot_tr * w_body[:, 0], Idot_tr * w_body[:, 1], Idot_ax * w_body[:, 2]], dim=-1)
        wdot_body = (T_body - wxIw - I_dot_omega) / torch.stack([I_tr, I_tr, I_ax], dim=-1)
        wdot_world = rotate_body_to_world(xq, wdot_body)
        # Quat dot
        w_body_full = torch.cat([torch.zeros(N, 1, device=xp.device, dtype=xp.dtype), w_body], dim=-1)
        q = xq
        qd_w = 0.5 * (q[:, 0]*w_body_full[:, 0] - q[:, 1]*w_body_full[:, 1] - q[:, 2]*w_body_full[:, 2] - q[:, 3]*w_body_full[:, 3])
        qd_x = 0.5 * (q[:, 0]*w_body_full[:, 1] + q[:, 1]*w_body_full[:, 0] + q[:, 2]*w_body_full[:, 3] - q[:, 3]*w_body_full[:, 2])
        qd_y = 0.5 * (q[:, 0]*w_body_full[:, 2] - q[:, 1]*w_body_full[:, 3] + q[:, 2]*w_body_full[:, 0] + q[:, 3]*w_body_full[:, 1])
        qd_z = 0.5 * (q[:, 0]*w_body_full[:, 3] + q[:, 1]*w_body_full[:, 2] - q[:, 2]*w_body_full[:, 1] + q[:, 3]*w_body_full[:, 0])
        qd = torch.stack([qd_w, qd_x, qd_y, qd_z], dim=-1)

        # Gimbal
        gacc0, gacc1 = gimbal_deriv_batch(xg0, xg1, xgr0, xgr1, g0_cmd, g1_cmd, p)
        gacc0 = gacc0.clamp(-p.engine.ENG_GIMBAL_ACC, p.engine.ENG_GIMBAL_ACC)
        gacc1 = gacc1.clamp(-p.engine.ENG_GIMBAL_ACC, p.engine.ENG_GIMBAL_ACC)

        ydot = torch.cat([
            xv, qd, a_world, wdot_world,
            -mdot_lox[:, None], -mdot_rp1[:, None],
            thr_rate[:, None], xign_rate[:, None],
            xgr0[:, None], xgr1[:, None], gacc0[:, None], gacc1[:, None],
        ], dim=-1)
        return ydot

    k1 = deriv(pack(), eng_on)
    # midpoint updates via Euler half-step
    y2 = pack() + 0.5 * dt * k1
    y2[:, 3:7] = (y2[:, 3:7] / y2[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-9))  # renormalize
    # Clamp throttle
    y2[:, 15] = y2[:, 15].clamp(0.0, 1.0)
    y2[:, 17:19] = y2[:, 17:19].clamp(-p.engine.ENG_GIMBAL_MAX, p.engine.ENG_GIMBAL_MAX)
    k2 = deriv(y2, eng_on)
    y3 = pack() + 0.5 * dt * k2
    y3[:, 3:7] = (y3[:, 3:7] / y3[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-9))
    y3[:, 15] = y3[:, 15].clamp(0.0, 1.0)
    y3[:, 17:19] = y3[:, 17:19].clamp(-p.engine.ENG_GIMBAL_MAX, p.engine.ENG_GIMBAL_MAX)
    k3 = deriv(y3, eng_on)
    y4 = pack() + dt * k3
    y4[:, 3:7] = (y4[:, 3:7] / y4[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-9))
    y4[:, 15] = y4[:, 15].clamp(0.0, 1.0)
    y4[:, 17:19] = y4[:, 17:19].clamp(-p.engine.ENG_GIMBAL_MAX, p.engine.ENG_GIMBAL_MAX)
    k4 = deriv(y4, eng_on)

    y_new = pack() + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Post-step clamping
    y_new[:, 3:7] = (y_new[:, 3:7] / y_new[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-9))
    y_new[:, 13] = y_new[:, 13].clamp_min(0.0)
    y_new[:, 14] = y_new[:, 14].clamp_min(0.0)
    y_new[:, 15] = y_new[:, 15].clamp(0.0, 1.0)
    y_new[:, 17:19] = y_new[:, 17:19].clamp(-p.engine.ENG_GIMBAL_MAX, p.engine.ENG_GIMBAL_MAX)

    # Gimbal rate anti-windup
    g0_new = y_new[:, 17]; g1_new = y_new[:, 18]
    gr0_new = y_new[:, 19]; gr1_new = y_new[:, 20]
    clamped_g0, clamped_g1, clamped_gr0, clamped_gr1 = gimbal_saturate(g0_new, g1_new, gr0_new, gr1_new, p)
    y_new[:, 17] = clamped_g0; y_new[:, 18] = clamped_g1
    y_new[:, 19] = clamped_gr0; y_new[:, 20] = clamped_gr1

    pos, quat, vel, ang, m_lox, m_rp1, throttle_act, ignition_timer, g0_act, g1_act, gr0, gr1 = unpack(y_new)
    return pos, quat, vel, ang, m_lox, m_rp1, throttle_act, ignition_timer, g0_act, g1_act, gr0, gr1
