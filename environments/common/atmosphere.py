"""International Standard Atmosphere and compressibility corrections.

The legacy plant evaluates every aerodynamic force at sea-level density
(``rho = 1.225 kg/m^3``) with incompressible lift slopes.  That is adequate
for the fixed-wing / quadplane tasks, which stay below 100 m, but it is a
large modelling error for the missile (4 km spawn, ~Mach 1.8) and the rocket
(hundreds of metres of altitude, transonic terminal descent).  This module
supplies the two missing pieces of physics as pure torch functions:

- ISA density, pressure, temperature and speed of sound versus geometric
  altitude (troposphere + lower stratosphere, clamped to 0-20 km);
- Prandtl-Glauert / linear-supersonic lift-slope factors and an engineering
  wave-drag rise factor for lifting surfaces.

Both are opt-in through ``aero="realistic"``; the default benchmark path never
calls them and therefore stays bit-for-bit identical to the published results.

References
----------
- ISO 2533:1975 / U.S. Standard Atmosphere 1976 (ISA troposphere and
  lower-stratosphere layers).
- Prandtl-Glauert similarity rule for subsonic compressibility.
- Linear supersonic thin-airfoil theory, ``C_l_alpha = 4 / sqrt(M^2 - 1)``.
- Shevell, *Fundamentals of Flight*, for the qualitative wave-drag rise
  bracketed by the piecewise factor below (an engineering fit, not a
  coefficient database).
"""

import math

import torch

from .dynamics import COMPUTE_DTYPE

# Sea-level ISA constants.
RHO0 = 1.225  # kg/m^3
T0 = 288.15  # K
P0 = 101325.0  # Pa
LAPSE = 0.0065  # K/m (troposphere)
G0 = 9.80665  # m/s^2
R_AIR = 287.05287  # J/(kg K)
GAMMA = 1.4
SOUND_SPEED0 = math.sqrt(GAMMA * R_AIR * T0)  # 340.294 m/s

_H_TROPOPAUSE = 11000.0  # m
_H_CEILING = 20000.0  # m; above this the model is held constant
_T_TROPOPAUSE = T0 - LAPSE * _H_TROPOPAUSE
_P_TROPOPAUSE = P0 * (_T_TROPOPAUSE / T0) ** (G0 / (R_AIR * LAPSE))


def isa(altitude, temperature_offset=0.0, specific_humidity=0.0):
    """Return ISA ``(density, pressure, temperature, sound_speed)``.

    Parameters
    ----------
    altitude : tensor or float
        Geometric altitude [m], any shape.  Values outside ``[0, 20 km]`` are
        clamped (the benchmark never leaves that band).
    temperature_offset : float
        Off-standard-day temperature deviation [K] added to the ISA
        temperature.  Pressure stays at the ISA value for the altitude (the
        standard hot-day/density-altitude approximation), so a positive offset
        lowers the density by ``T/(T + offset)`` and raises the sound speed.
    specific_humidity : float
        Water-vapour mass mixing ratio [kg/kg].  Moist air is less dense at the
        same pressure and temperature: ``rho = p/(R T (1 + 0.6078 q))``.

    Returns
    -------
    tuple of four tensors
        Density [kg/m^3], pressure [Pa], temperature [K], speed of sound
        [m/s], each with the input shape and ``COMPUTE_DTYPE``.
    """
    h = torch.as_tensor(altitude, dtype=COMPUTE_DTYPE).clamp(0.0, _H_CEILING)
    # Troposphere (0-11 km): linear temperature lapse, polytropic pressure.
    t_tropo = T0 - LAPSE * h
    p_tropo = P0 * (t_tropo / T0).pow(G0 / (R_AIR * LAPSE))
    # Lower stratosphere (11-20 km): isothermal, exponential pressure.
    t_strat = torch.full_like(h, _T_TROPOPAUSE)
    p_strat = _P_TROPOPAUSE * torch.exp(
        -G0 * (h - _H_TROPOPAUSE) / (R_AIR * _T_TROPOPAUSE)
    )
    tropo = h <= _H_TROPOPAUSE
    temperature = torch.where(tropo, t_tropo, t_strat) + float(temperature_offset)
    pressure = torch.where(tropo, p_tropo, p_strat)
    # 0.6078 = R_dry/R_vapour - 1: the standard first-order moist-air factor.
    density = pressure / (R_AIR * temperature * (1.0 + 0.6078 * float(specific_humidity)))
    sound_speed = torch.sqrt(GAMMA * R_AIR * temperature)
    return density, pressure, temperature, sound_speed


def density_ratio(altitude):
    """ISA density divided by the legacy sea-level constant."""
    return isa(altitude)[0] / RHO0


def lift_slope_factor(mach, cl_alpha_2d=2.0 * math.pi):
    """Lift-curve-slope multiplier relative to a surface's incompressible slope.

    Subsonic: Prandtl-Glauert ``1/sqrt(1-M^2)``, capped at 2.0 so the
    transonic singularity cannot destabilise the plant.  Supersonic: linear
    thin-airfoil theory gives the *absolute* slope ``4/sqrt(M^2-1)`` per rad,
    which is converted to a multiplier by dividing by the surface's own
    ``cl_alpha_2d``.  This matters for low-aspect-ratio missile fins, whose
    baseline slope is already ~3/rad rather than 2*pi: at M=1.8 their slope
    should stay near its incompressible value instead of being cut in half.

    ``cl_alpha_2d`` may be a scalar or a per-surface tensor; all terms
    broadcast.  The band ``0.9 < M < 1.2`` is blended linearly through
    ``2.0 -> 1.0 ->`` the supersonic value.
    """
    m = torch.as_tensor(mach, dtype=COMPUTE_DTYPE).clamp_min(0.0)
    base = torch.as_tensor(cl_alpha_2d, dtype=COMPUTE_DTYPE).clamp_min(1e-3)
    pg = (1.0 / torch.sqrt((1.0 - m.square()).clamp_min(1e-4))).clamp_max(2.0)
    transonic = 2.0 - (m - 0.9).clamp(0.0, 0.1) / 0.1  # 2.0 at 0.9 -> 1.0 at 1.0
    supersonic_abs = (4.0 / torch.sqrt((m.square() - 1.0).clamp_min(1e-4))).clamp(0.5, 12.0)
    supersonic = (supersonic_abs / base).clamp(0.25, 2.0)
    supersonic_blend = 1.0 + (supersonic - 1.0) * (m - 1.0).clamp(0.0, 0.2) / 0.2
    return torch.where(
        m <= 0.9,
        pg,
        torch.where(m <= 1.0, transonic, torch.where(m < 1.2, supersonic_blend, supersonic)),
    )


def drag_factor(mach):
    """Wave-drag multiplier for surface ``C_d0`` and post-stall ``C_d90``.

    Piecewise engineering fit: flat below the drag-divergence Mach number
    (0.85), rising linearly to 1.5 at M=1.05, then decaying as
    ``(1.05/M)^0.3`` while staying at or above the incompressible value.
    At the low Mach numbers of the fixed-wing / quadplane tasks the factor is
    exactly 1, so enabling compressibility does not perturb their aerodynamics.
    """
    m = torch.as_tensor(mach, dtype=COMPUTE_DTYPE).clamp_min(0.0)
    rise = 1.0 + 2.5 * (m - 0.85).clamp(0.0, 0.20)
    decay = (1.5 * (1.05 / m.clamp_min(1.05)).pow(0.3)).clamp(1.0, 1.5)
    return torch.where(m <= 0.85, torch.ones_like(m), torch.where(m <= 1.05, rise, decay))
