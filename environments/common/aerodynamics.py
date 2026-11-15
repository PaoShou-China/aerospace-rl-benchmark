"""Batched PyFlyt lifting-surface model, including positive/negative stall.

Formula attribution: PyFlyt LiftingSurface (MIT), source commit documented in README.
"""

import math
import numpy as np
import torch

from .atmosphere import RHO0, drag_factor, lift_slope_factor
from .dynamics import COMPUTE_DTYPE

# First-order unsteady lift lag switch (realistic mode only).
_UNSTEADY_ENABLED = True
# Beddoes-Leishman-style dynamic-stall model (realistic mode only):
# lagged trailing-edge separation point plus a vortex-lift state that is
# driven by the separation-point RATE, so steady-state polars are unchanged.
_DYNAMIC_STALL_ENABLED = True
_DS_TF_SCALE = 2.0  # separation lag time constant ~ 2 c / V
_DS_TV_SCALE = 3.0  # vortex decay time constant ~ 3 c / V
_DS_KV = 0.5  # vortex-lift gain per unit separation-rate excursion
_DS_CM = 0.35  # nose-down moment per unit vortex lift (pitching-moment stall)
# Servo command-path non-idealities (realistic mode only): a 0-2 physics-step
# transport delay, mechanical backlash, asymmetric extension/retraction rates
# and a randomisable deadband.
_ACT_MAX_DELAY = 2
_ACT_BACKLASH = 0.005  # fraction of full command at scale 1.0
# Wing-in-ground-effect pitching-moment coefficient: the reduced downwash
# moves the centre of pressure slightly aft, modelled as a nose-down increment
# proportional to (1 - phi) * cl (first-order engineering estimate).
_GE_MOMENT_COEFF = 0.1
# Lumped wing torsional aeroelasticity (realistic fixed-wing/quadplane only).
# The elastic axis sits at 35 % chord behind the quarter-chord aerodynamic
# centre, which is the classical divergence layout; the torsional stiffness is
# derived from a target divergence speed so the trend is physical by
# construction.  Structural damping/ frequency are typical for a stiff UAV wing.
_AE_ELASTIC_AXIS = 0.35  # elastic-axis position, fraction of chord aft of LE
_AE_FREQ_HZ = 8.0  # first torsional natural frequency [Hz]
_AE_DAMPING = 0.15  # structural damping ratio
_AE_TWIST_LIMIT = 0.35  # rad; numerical/structural stop


def ground_effect(ground_height, span, slope, aspect):
    """Lifting-line ground-effect factors for a finite wing.

    The image-vortex system below the ground plane reduces the downwash by
    ``phi = (16 h/b)^2 / (1 + (16 h/b)^2)`` (McCormick's induced-drag factor,
    the classical lifting-line image solution).  With the induced angle
    scaling as ``phi``, the lift slope rises from its free-air value to
    ``cl_alpha / (1 + cl_alpha phi/(pi AR))`` and the induced drag falls by
    the same ``phi``.

    Args:
        ground_height: height of the wing (or station) above the ground [m].
        span: full wing span [m].
        slope: free-air finite-span lift slope [1/rad].
        aspect: wing aspect ratio.

    Returns:
        ``(lift_ratio, phi)`` broadcast to the input shape.  ``lift_ratio`` is
        1 in free air and ``1 + cl_alpha/(pi AR)`` in the h -> 0 limit;
        ``phi`` is 1 in free air and 0 at the ground.
    """
    h = torch.as_tensor(ground_height)
    b = torch.as_tensor(span, device=h.device, dtype=h.dtype).clamp_min(1e-3)
    ratio = 16.0 * (h.clamp_min(0.0) / b)
    phi = ratio.square() / (1.0 + ratio.square())
    cl2d = slope / (1.0 - slope / (math.pi * aspect)).clamp_min(1e-3)
    kappa = cl2d / (math.pi * aspect)
    return (1.0 + kappa) / (1.0 + kappa * phi), phi


class LiftingSurfaceBatch:
    """Batched PyFlyt-style lifting-surface model with positive/negative stall.

    Evaluates lift/drag/moment coefficients across a per-link (n_surfaces,) set
    of aerodynamic surfaces shared across ``n`` vehicle replicas.  ``actuation``
    is a first-order lag buffer driven by ``step``; velocity is supplied by
    the caller (the plant rotates it into body frame before calling).

    ``realistic=True`` enables three opt-in corrections that the legacy
    benchmark path leaves out: a per-replica air density (ISA), a finite-span
    Oswald efficiency ``e`` on the induced angle of attack, and
    Prandtl-Glauert / linear-supersonic lift-slope plus wave-drag factors.
    With ``realistic=False`` every arithmetic expression below is unchanged
    from the upstream port, so calm-air benchmark runs remain bit-identical.
    """
    def __init__(
        self,
        params,
        lifts,
        forwards,
        n,
        device,
        dt,
        realistic=False,
        oswald=None,
        rate_limit_deg_s=None,
        deadband=0.0,
        strips=1,
        aeroelastic=False,
        divergence_speed=100.0,
    ):
        self.dt = dt
        self.realistic = bool(realistic)
        # Spanwise strip resolution (realistic mode only).  Each lifting
        # surface is split into ``strips`` equal spanwise stations so the
        # local velocity, dynamic pressure, slipstream, Mach and stall state
        # are evaluated per station; forces and moments are summed.  One strip
        # reproduces the upstream single-point model exactly.
        self.strips = int(max(1, strips))
        self.n_surfaces = len(params)
        if self.strips > 1 and not self.realistic:
            raise ValueError("spanwise strips require realistic=True")
        n_strip = self.n_surfaces * self.strips
        strip_params = [p for p in params for _ in range(self.strips)]
        strip_lifts = [v for v in lifts for _ in range(self.strips)]
        strip_forwards = [v for v in forwards for _ in range(self.strips)]
        self.lift = torch.tensor(strip_lifts, device=device, dtype=COMPUTE_DTYPE)
        self.forward = torch.tensor(strip_forwards, device=device, dtype=COMPUTE_DTYPE)
        self.torque_axis = torch.linalg.cross(self.lift, self.forward)
        for key in strip_params[0]:
            setattr(
                self,
                key,
                torch.tensor([p[key] for p in strip_params], device=device, dtype=COMPUTE_DTYPE),
            )
        # Servo hardware is shared by the strips of one surface: its lag/rate/
        # backlash state stays at one column per surface.
        self.tau_surface = self.tau.view(self.n_surfaces, self.strips)[:, 0]
        if (self.tau_surface < dt).any():
            raise ValueError("Surface dt must not exceed actuator tau")
        self.aspect = self.span / self.chord
        # Each strip carries 1/strips of the surface area.
        self.area = self.span * self.chord / self.strips
        # Spanwise station offsets from the surface centre, along the surface
        # span axis (``torque_axis`` = lift x forward).
        station = (
            torch.arange(self.strips, device=device, dtype=COMPUTE_DTYPE)
            - 0.5 * (self.strips - 1)
        )
        span_surface = self.span.view(self.n_surfaces, self.strips)[:, :1]
        axis_surface = self.torque_axis.view(self.n_surfaces, self.strips, 3)[:, :1, :]
        self._strip_offset = (
            station[None, :, None] * (span_surface / self.strips)[:, :, None] * axis_surface
        ).reshape(n_strip, 3)

        # Precompute using the same NumPy float64 operations/order as upstream.
        def constant(fn):
            """Evaluate ``fn`` on each surface param dict and return a float64 tensor."""
            return torch.tensor([fn(p) for p in strip_params], device=device, dtype=COMPUTE_DTYPE)

        self.slope = constant(
            lambda p: p["Cl_alpha_2D"]
            * (
                (p["span"] / p["chord"])
                / (
                    (p["span"] / p["chord"])
                    + 2 * ((p["span"] / p["chord"]) + 4) / ((p["span"] / p["chord"]) + 2)
                )
            )
        )
        self.aero_tau = constant(
            lambda p: 1
            - (
                np.arccos(2 * p["flap_to_chord"] - 1)
                - np.sin(np.arccos(2 * p["flap_to_chord"] - 1))
            )
            / np.pi
        )
        self.alpha0 = constant(lambda p: np.deg2rad(p["alpha_0_base"]))
        self.stall_p = constant(lambda p: np.deg2rad(p["alpha_stall_P_base"]))
        self.stall_n = constant(lambda p: np.deg2rad(p["alpha_stall_N_base"]))
        # Oswald span efficiency.  The frozen upstream YAMLs cannot carry new
        # keys, so the plant supplies per-surface values explicitly; the
        # legacy path keeps e = 1 and reproduces the upstream induced-drag
        # term exactly.
        if self.realistic:
            values = [1.0] * self.n_surfaces if oswald is None else [float(v) for v in oswald]
            if len(values) != self.n_surfaces:
                raise ValueError("oswald must have one value per surface")
            values = [v for v in values for _ in range(self.strips)]
            self.oswald = torch.tensor(
                [max(v, 1e-3) for v in values], device=device, dtype=COMPUTE_DTYPE
            )
        else:
            self.oswald = torch.ones_like(self.slope)
        # Servo rate limit expressed in normalized command units per second
        # (rate_deg_s / deflection_limit_deg); zero/absent means unlimited.
        if self.realistic:
            rates = (
                [0.0] * self.n_surfaces
                if rate_limit_deg_s is None
                else [float(v) for v in rate_limit_deg_s]
            )
            if len(rates) != self.n_surfaces:
                raise ValueError("rate_limit_deg_s must have one value per surface")
            limit = self.deflection_limit.view(self.n_surfaces, self.strips)[:, 0].clamp_min(1e-6)
            rate_tensor = torch.tensor(rates, device=device, dtype=COMPUTE_DTYPE)
            norm = rate_tensor / limit
            self.rate_limit = torch.where(
                rate_tensor > 0.0,
                norm,
                torch.full_like(norm, float("inf")),
            )
            self.deadband = float(min(max(deadband, 0.0), 0.5))
        else:
            self.rate_limit = None
            self.deadband = 0.0
        # Lumped torsional elastic DOF per surface (realistic only).  The
        # stiffness is calibrated at the end of __init__, once the coefficient
        # scales exist, from the full linearised elastic-axis moment slope.
        self.aeroelastic = bool(aeroelastic) and self.realistic
        self._divergence_speed = float(divergence_speed)
        self.actuation = torch.zeros(
            (n, self.n_surfaces), device=device, dtype=COMPUTE_DTYPE
        )
        # First-order unsteady-aerodynamics state (realistic mode only).
        self._aero_state = torch.zeros((n, n_strip, 3), device=device, dtype=COMPUTE_DTYPE)
        self._aero_ready = torch.zeros(n, device=device, dtype=torch.bool)
        # Dynamic-stall states: trailing-edge separation point and vortex lift.
        self._fsep = torch.ones((n, n_strip), device=device, dtype=COMPUTE_DTYPE)
        self._vortex = torch.zeros((n, n_strip), device=device, dtype=COMPUTE_DTYPE)
        # Per-replica domain-randomization scales (1.0 = nominal).  Lift slope,
        # zero-lift drag, stall margin and Oswald efficiency are independent
        # axes so a policy can see realistic combinations (e.g. early stall
        # with a nominal slope) instead of a single coupled "aero" factor.
        self.slope_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.cd0_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.stall_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.oswald_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.tau_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.rate_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        # Servo command-path states (realistic only; allocated for both modes
        # so the setters are always callable).
        self.deadband_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.backlash_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.rate_asym_scale = torch.ones((n, 1), device=device, dtype=COMPUTE_DTYPE)
        self.delay_steps = torch.zeros((n,), device=device, dtype=torch.long)
        self._backlash_state = torch.zeros(
            (n, self.n_surfaces), device=device, dtype=COMPUTE_DTYPE
        )
        self._delay_ring = torch.zeros(
            (n, self.n_surfaces, _ACT_MAX_DELAY + 1), device=device, dtype=COMPUTE_DTYPE
        )
        self._delay_head = torch.zeros((n,), device=device, dtype=torch.long)
        if self.aeroelastic:
            # Elastic-axis moment slope at zero lift:
            #   dM/dalpha = q S c [ (x_ea - 0.25) cl_alpha + cm_alpha ]
            # The pitching moment itself carries a negative cm_alpha for the
            # upstream polar, so omitting it overstated the divergence speed.
            # Calibrate k from the actual model slope so that the linearised
            # divergence speed is exactly ``divergence_speed``.
            span_surface = self.span.view(self.n_surfaces, self.strips)[:, 0]
            chord_surface = self.chord.view(self.n_surfaces, self.strips)[:, 0]
            slope_surface = self.slope.view(self.n_surfaces, self.strips)[:, 0]
            eps = 1e-3
            a0_flat = self.alpha0.view(1, n_strip)
            cmd0 = torch.zeros((1, n_strip), device=device, dtype=COMPUTE_DTYPE)
            cm_plus = self.coefficients(a0_flat + eps, cmd0)[2][0]
            cm_minus = self.coefficients(a0_flat - eps, cmd0)[2][0]
            cm_alpha = ((cm_plus - cm_minus) / (2.0 * eps)).view(
                self.n_surfaces, self.strips
            )[:, 0]
            net = ((_AE_ELASTIC_AXIS - 0.25) * slope_surface + cm_alpha).clamp_min(1e-6)
            q_div = 0.5 * RHO0 * self._divergence_speed ** 2
            # q S c * slope: the area S = span*chord and the moment arm in
            # chord fractions contributes one more chord factor.
            self._twist_k = q_div * span_surface * chord_surface.square() * net
            omega = 2.0 * math.pi * _AE_FREQ_HZ
            self._twist_i = self._twist_k / omega**2
            self._twist_c = 2.0 * _AE_DAMPING * self._twist_k / omega
            self._twist = torch.zeros(
                (n, self.n_surfaces), device=device, dtype=COMPUTE_DTYPE
            )
            self._twist_rate = torch.zeros_like(self._twist)

    def strip_offsets(self):
        """Body-frame offset of every spanwise station from its surface centre.

        Returns ``(n_surfaces, strips, 3)`` in the same surface-major order as
        the flat strip axis used internally.
        """
        return self._strip_offset.view(self.n_surfaces, self.strips, 3)

    def reset(self, ids):
        """Zero actuation buffer for replicas indexed by ``ids``."""
        self.actuation[ids] = 0
        self._aero_ready[ids] = False
        self._fsep[ids] = 1.0
        self._vortex[ids] = 0.0
        self._backlash_state[ids] = 0.0
        self._delay_ring[ids] = 0.0
        self._delay_head[ids] = 0
        if self.aeroelastic:
            self._twist[ids] = 0.0
            self._twist_rate[ids] = 0.0

    def set_aero_scale(self, scale):
        """Scale lift slope and zero-lift drag by a float or per-replica tensor."""
        if isinstance(scale, (int, float)):
            self.slope_scale.fill_(float(scale))
            self.cd0_scale.fill_(float(scale))
        else:
            value = scale.to(self.slope_scale.device).to(COMPUTE_DTYPE).reshape(-1, 1)
            self.slope_scale.copy_(value)
            self.cd0_scale.copy_(value)

    def _set_component(self, buffer, scale):
        if isinstance(scale, (int, float)):
            buffer.fill_(float(scale))
        else:
            buffer.copy_(
                scale.to(buffer.device).to(COMPUTE_DTYPE).reshape(-1, 1).clamp_min(1e-3)
            )

    def set_cl_alpha_scale(self, scale):
        """Scale the lift-curve slope only."""
        self._set_component(self.slope_scale, scale)

    def set_cd0_scale(self, scale):
        """Scale the zero-lift / parasite drag only."""
        self._set_component(self.cd0_scale, scale)

    def set_stall_scale(self, scale):
        """Scale the positive/negative stall margins about the zero-lift angle."""
        self._set_component(self.stall_scale, scale)

    def set_oswald_scale(self, scale):
        """Scale the Oswald span efficiency (clamped to a physical range)."""
        self._set_component(self.oswald_scale, scale)

    def set_actuator_scale(self, scale):
        """Scale servo lag/rate by a float or per-replica tensor (larger = slower)."""
        if isinstance(scale, (int, float)):
            value = max(float(scale), 1e-3)
            self.tau_scale.fill_(value)
            self.rate_scale.fill_(1.0 / value)
        else:
            value = (
                scale.to(self.tau_scale.device).to(COMPUTE_DTYPE).reshape(-1, 1).clamp_min(1e-3)
            )
            self.tau_scale.copy_(value)
            self.rate_scale.copy_(1.0 / value)

    def set_actuator_deadband(self, scale):
        """Scale the servo deadband fraction."""
        self._set_component(self.deadband_scale, scale)

    def set_actuator_backlash(self, scale):
        """Scale the mechanical backlash fraction."""
        self._set_component(self.backlash_scale, scale)

    def set_actuator_asymmetry(self, scale):
        """Scale the retraction rate relative to the extension rate."""
        self._set_component(self.rate_asym_scale, scale)

    def set_actuator_delay(self, steps):
        """Set the command transport delay in physics steps (0-2)."""
        if isinstance(steps, (int, float)):
            value = torch.full(
                (self.delay_steps.numel(),), float(steps),
                device=self.delay_steps.device, dtype=COMPUTE_DTYPE,
            )
        else:
            value = steps.to(self.delay_steps.device).to(COMPUTE_DTYPE).reshape(-1)
        self.delay_steps.copy_(
            value.clamp(0.0, float(_ACT_MAX_DELAY)).round().to(torch.long)
        )

    def coefficients(self, alpha, actuation, mach=None):
        """Compute linear or post-stall lift/drag/moment coefficients.

        Parameters
        ----------
        alpha : (n, n_surfaces) tensor
            Angle of attack in radians.
        actuation : (n, n_surfaces) tensor
            Flap deflection command in [-1, 1].
        mach : (n, n_surfaces) tensor, optional
            Surface-local Mach number.  Only used when ``realistic=True``;
            it scales the lift slope and the zero-lift / post-stall drag.

        Returns
        -------
        tuple of three (n, n_surfaces) tensors
            ``(cl, cd, cm)``.
        """
        delta = torch.deg2rad(actuation * self.deflection_limit)
        slope = self.slope * self.slope_scale
        cd0 = self.Cd_0 * self.cd0_scale
        stall_p, stall_n = self._stall_angles()
        cd90 = -0.0426 * delta**2 + 0.21 * delta + 1.98
        if self.realistic and mach is not None:
            slope = slope * lift_slope_factor(mach, self.Cl_alpha_2D)
            wave = drag_factor(mach)
            cd0 = cd0 * wave
            cd90 = cd90 * wave
        dcl = slope * self.aero_tau * self.eta * delta
        clmaxp = slope * (stall_p - self.alpha0) + self.flap_to_chord * dcl
        clmaxn = slope * (stall_n - self.alpha0) + self.flap_to_chord * dcl
        a0 = self.alpha0 - dcl / slope
        ap = a0 + clmaxp / slope
        an = a0 + clmaxn / slope
        linear = (alpha > an) & (alpha < ap)
        cl = slope * (alpha - a0)
        ai = cl / (math.pi * self.aspect)
        effective = alpha - a0 - ai
        ct = cd0 * torch.cos(effective)
        cn = (cl + ct * torch.sin(effective)) / torch.cos(effective)
        cd = cn * torch.sin(effective) + ct * torch.cos(effective)
        cm = -cn * (0.25 - 0.175 * (1 - 2 * effective / math.pi))
        # np.interp clips outside the stall-to-90-degree interval.
        aipos = (
            clmaxp / (math.pi * self.aspect) * (1 - ((alpha - ap) / (math.pi / 2 - ap)).clamp(0, 1))
        )
        aineg = (
            clmaxn
            / (math.pi * self.aspect)
            * ((alpha + math.pi / 2) / (an + math.pi / 2)).clamp(0, 1)
        )
        ai_stall = torch.where(alpha > 0, aipos, aineg)
        effective_stall = alpha - a0 - ai_stall
        sn, cs = torch.sin(effective_stall), torch.cos(effective_stall)
        cns = cd90 * sn * (1 / (0.56 + 0.44 * sn.abs()) - 0.41 * (1 - torch.exp(-17 / self.aspect)))
        cts = 0.5 * cd0 * cs
        cls = cns * cs - cts * sn
        cds = cns * sn + cts * cs
        cms = -cns * (0.25 - 0.175 * (1 - 2 * effective_stall.abs() / math.pi))
        if self.realistic:
            # Oswald span efficiency as an explicit induced-drag increment on
            # top of the e=1 baseline (the upstream angle convention makes a
            # direct alpha_i/e substitution reduce drag instead of raising it).
            oswald = (self.oswald * self.oswald_scale).clamp(0.3, 1.0)
            k = (1.0 / oswald - 1.0) / (math.pi * self.aspect)
            cd = cd + k * cl.square()
            cds = cds + k * cls.square()
        return tuple(torch.where(linear, a, b) for a, b in [(cl, cls), (cd, cds), (cm, cms)])

    def _stall_angles(self):
        """Positive/negative stall angles after the stall-margin scale."""
        return (
            self.alpha0 + (self.stall_p - self.alpha0) * self.stall_scale,
            self.alpha0 - (self.alpha0 - self.stall_n) * self.stall_scale,
        )

    def _attached_lift(self, alpha, actuation, mach=None):
        """Potential (fully attached) lift and flap-shifted zero-lift angle."""
        delta = torch.deg2rad(actuation * self.deflection_limit)
        slope = self.slope * self.slope_scale
        if self.realistic and mach is not None:
            slope = slope * lift_slope_factor(mach, self.Cl_alpha_2D)
        dcl = slope * self.aero_tau * self.eta * delta
        a0 = self.alpha0 - dcl / slope
        return slope * (alpha - a0), alpha - a0

    def _dynamic_stall(self, alpha, actuation, mach, speed, cl_qs, cd_qs, cm_qs, mask=None):
        """Beddoes-Leishman-style dynamic-stall correction to (cl, cd, cm).

        A trailing-edge separation point ``f`` follows the static separation
        curve through a first-order lag of ``2 c / V`` (Kirchhoff lift
        ``cl = cl_pot ((1 + sqrt(f))/2)^2``), and a vortex-lift state driven by
        the separation-point rate adds the leading-edge vortex lift seen while
        the flow sheds.  The correction is blended in only around and beyond
        stall, so the attached-flow polar is exactly the quasi-steady one and
        steady post-stall flight relaxes back to it (the vortex term vanishes
        when df/dt = 0).

        The vortex also produces drag (its normal force projected on the
        chord, ``cn_v sin(alpha_eff)``) and a stronger nose-down moment than
        the steady polar; both vanish in steady flow for the same reason.
        """
        cl_pot, alpha_eff = self._attached_lift(alpha, actuation, mach)
        stall_p, stall_n = self._stall_angles()
        stall = torch.where(
            alpha_eff >= 0.0,
            stall_p - self.alpha0,
            self.alpha0 - stall_n,
        ).clamp_min(1e-3)
        ratio = alpha_eff.abs() / stall
        # Separation onset at 90% of the static stall angle: below it the flow
        # is treated as fully attached, so the dynamic increment is exactly
        # zero for normal flight and the published polar is untouched.
        f_stat = torch.where(
            ratio <= 0.9,
            torch.ones_like(ratio),
            torch.where(
                ratio <= 1.0,
                1.0 - 0.05 * ((ratio - 0.9) / 0.1),
                0.95 * torch.exp(-2.0 * (ratio - 1.0)),
            ),
        )
        tf = (_DS_TF_SCALE * self.chord[None, :] / speed.clamp_min(1.0)).clamp(0.02, 0.5)
        tv = (_DS_TV_SCALE * self.chord[None, :] / speed.clamp_min(1.0)).clamp(0.02, 0.6)
        fresh = (~self._aero_ready)[:, None]
        f_prev = torch.where(fresh, f_stat, self._fsep)
        f_new = (f_prev + self.dt / tf * (f_stat - f_prev)).clamp(0.0, 1.0)
        dfdt = (f_new - f_prev) / self.dt
        vortex_target = (-_DS_KV * dfdt * tf * cl_pot).clamp(-2.0, 2.0)
        vortex = self._vortex + self.dt / tv * (vortex_target - self._vortex)
        vortex = torch.where(fresh, torch.zeros_like(vortex), vortex).clamp(-2.0, 2.0)
        cl_k = cl_pot * ((1.0 + f_new.sqrt()) * 0.5).square()
        cl_k_static = cl_pot * ((1.0 + f_stat.sqrt()) * 0.5).square()
        # Apply the dynamic effect as an INCREMENT over the quasi-steady polar:
        # with f_dyn == f_stat and zero vortex rate the increment vanishes, so
        # steady attached and post-stall flight keep the published coefficients
        # exactly.  Pitch-up (f_dyn > f_stat) adds lift, pitch-down subtracts it.
        cl = cl_qs + (cl_k - cl_k_static) + vortex
        # Vortex normal force projected on the chord: the shedding vortex
        # carries the same drag penalty a separated wake would.
        cd = cd_qs + vortex * torch.sin(alpha_eff)
        cm = cm_qs - _DS_CM * vortex
        if mask is None:
            self._fsep.copy_(f_new)
            self._vortex.copy_(vortex)
        else:
            self._fsep.copy_(torch.where(mask[:, None], f_new, self._fsep))
            self._vortex.copy_(torch.where(mask[:, None], vortex, self._vortex))
        return cl, cd, cm

    def step(self, cmd, velocity, density=None, sound_speed=None, q_boost=None,
             body_rate=None, ground_height=None, mask=None):
        """Run one actuator lag step and return body-frame forces/moments.

        Parameters
        ----------
        cmd : (n, n_surfaces) tensor
            Flap deflection command in [-1, 1].
        velocity : (n, n_surfaces, 3) tensor
            Body-frame air-relative velocity at each surface's link.
        density : (n,) tensor, optional
            Per-replica air density [kg/m^3].  ``None`` uses the legacy
            sea-level constant; passing a tensor changes the dynamic pressure
            for every surface.
        sound_speed : (n,) tensor, optional
            Per-replica speed of sound [m/s]; with ``realistic=True`` it turns
            surface speed into Mach number for the compressibility factors.
        q_boost : (n, n_surfaces) or (n, n_surfaces*strips) tensor, optional
            Extra dynamic pressure [Pa] from a propeller slipstream.  With
            spanwise strips the caller supplies one value per strip
            (surface-major order, the flat strip axis).
        body_rate : (n, 3) tensor, optional
            Body-frame angular velocity [rad/s].  With spanwise strips each
            station's local velocity includes ``omega x r``, which is what
            produces roll/pitch damping from the spanwise lift distribution.
        ground_height : (n, k) tensor, optional
            Height of each station above the ground plane [m].  Enables the
            wing-in-ground-effect correction (see :func:`ground_effect_ratio`).
        mask : (n,) bool tensor, optional
            Active replicas.  Inactive replicas keep every persistent state
            (servo lag/backlash/delay, unsteady/dynamic-stall/aeroelastic
            states) frozen, so a partial reset or warmup cannot advance a
            neighbouring episode.  ``None`` advances all replicas.

        Returns
        -------
        force : (n, n_surfaces, 3) tensor
            Body-frame force per surface.
        torque : (n, n_surfaces, 3) tensor
            Body-frame moment per surface.
        """
        velocity = velocity.to(COMPUTE_DTYPE)
        if mask is None:
            mask = torch.ones(cmd.shape[0], dtype=torch.bool, device=cmd.device)
        else:
            mask = mask.to(device=cmd.device, dtype=torch.bool).reshape(-1)
        if self.strips > 1:
            # Replicate each surface to its spanwise stations and add the
            # rigid-body rotation contribution at each station.
            velocity = velocity.repeat_interleave(self.strips, dim=1)
            if body_rate is not None:
                velocity = velocity + torch.linalg.cross(
                    body_rate[:, None, :], self._strip_offset[None, :, :]
                )
        if self.realistic:
            target = cmd.clamp(-1.0, 1.0)
            deadband = self.deadband * self.deadband_scale
            if self.deadband > 0.0:
                magnitude = (target.abs() - deadband).clamp_min(0.0) / (1.0 - deadband)
                target = torch.sign(target) * magnitude
            # Mechanical backlash: the linkage only moves once the command
            # leaves the current lash band.
            backlash = _ACT_BACKLASH * self.backlash_scale
            difference = target - self._backlash_state
            backlash_new = (
                self._backlash_state
                + torch.sign(difference) * (difference.abs() - 0.5 * backlash).clamp_min(0.0)
            ).clamp(-1.0, 1.0)
            self._backlash_state.copy_(
                torch.where(mask[:, None], backlash_new, self._backlash_state)
            )
            target = backlash_new
            # Pure transport delay in the command path (0-2 physics steps).
            head = self._delay_head
            ring = self._delay_ring
            ring_new = ring.clone()
            ring_new.scatter_(
                2,
                head[:, None, None].expand(-1, ring.shape[1], 1),
                target.unsqueeze(-1),
            )
            index = (
                head[:, None, None] - self.delay_steps[:, None, None]
            ).remainder(ring.shape[2]).expand(-1, ring.shape[1], 1)
            target = ring_new.gather(2, index).squeeze(-1)
            self._delay_ring.copy_(
                torch.where(mask[:, None, None], ring_new, ring)
            )
            self._delay_head.copy_(
                torch.where(mask, (head + 1).remainder(ring.shape[2]), head)
            )
            tau = (self.tau_surface[None, :] * self.tau_scale).clamp_min(1.05 * self.dt)
            delta = self.actuation + self.dt / tau * (target - self.actuation)
            max_step = self.rate_limit[None, :] * self.rate_scale * self.dt
            # Asymmetric servo: the retraction rate is scaled independently.
            down_step = max_step * self.rate_asym_scale
            delta = torch.maximum(
                torch.minimum(delta, self.actuation + max_step), self.actuation - down_step
            )
            self.actuation.copy_(torch.where(mask[:, None], delta, self.actuation))
        else:
            # Keep the in-place update form under graph capture/compilers, but
            # leave inactive replicas' actuation untouched.
            tau = (self.tau_surface[None, :] * self.tau_scale).clamp_min(1.05 * self.dt)
            delta = self.actuation + self.dt / tau * (cmd - self.actuation)
            self.actuation.copy_(torch.where(mask[:, None], delta, self.actuation))
        alpha = torch.atan2(-(velocity * self.lift).sum(-1), (velocity * self.forward).sum(-1))
        if self.aeroelastic:
            # Elastic twist (small-angle) adds to the geometric angle of attack.
            alpha = alpha + self._twist.repeat_interleave(self.strips, dim=1)
        speed = velocity.norm(dim=-1)
        mach = None
        if self.realistic and sound_speed is not None:
            mach = speed / sound_speed[:, None].clamp_min(1.0)
        # One physical servo drives all strips of a surface; the aerodynamic
        # evaluation runs per station.
        actuation = self.actuation
        if self.strips > 1:
            actuation = actuation.repeat_interleave(self.strips, dim=1)
        cl, cd, cm = self.coefficients(alpha, actuation, mach)
        if self.realistic and _DYNAMIC_STALL_ENABLED:
            cl, cd, cm = self._dynamic_stall(
                alpha, actuation, mach, speed, cl, cd, cm, mask=mask
            )
        if self.realistic and _UNSTEADY_ENABLED:
            # First-order Wagner-style lift lag: the circulatory coefficients
            # approach their quasi-steady value over one convective time
            # c/V (clamped to 10-200 ms).  This smooths stall entry and the
            # immediate response to a flap step without changing the steady
            # polar.
            tau = (self.chord[None, :] / speed.clamp_min(1.0)).clamp(0.01, 0.2)
            current = torch.stack((cl, cd, cm), -1)
            blended = self._aero_state + (self.dt / tau)[..., None] * (current - self._aero_state)
            fresh = (~self._aero_ready)[:, None, None]
            blended = torch.where(fresh, current, blended)
            self._aero_state.copy_(
                torch.where(mask[:, None, None], blended, self._aero_state)
            )
            self._aero_ready.copy_(self._aero_ready | mask)
            cl, cd, cm = self._aero_state.unbind(-1)
        if self.realistic and ground_height is not None:
            # Quasi-steady wing-in-ground-effect correction applied after the
            # unsteady blend: the reduced downwash raises the lift slope,
            # removes (1 - phi) of the induced drag and adds a small nose-down
            # moment.  Stations close to the ground are corrected strongly,
            # which also gives the physical roll moment from a banked wing.
            cl0 = cl
            lift_ratio, phi = ground_effect(
                ground_height, self.span[None, :], self.slope[None, :],
                self.aspect[None, :],
            )
            cl = cl0 * lift_ratio
            oswald = (self.oswald * self.oswald_scale).clamp(0.3, 1.0)
            cd = (cd - (1.0 - phi) * cl0.square() / (
                math.pi * self.aspect[None, :] * oswald
            )).clamp_min(0.0)
            cm = cm - _GE_MOMENT_COEFF * (1.0 - phi) * cl0
        if density is None:
            qa = 0.5 * RHO0 * velocity.square().sum(-1) * self.area
        else:
            qa = 0.5 * density[:, None] * velocity.square().sum(-1) * self.area
        if q_boost is not None:
            qa = qa + q_boost * self.area
        if self.aeroelastic:
            # Aerodynamic moment about the elastic axis: the quarter-chord
            # moment plus the lift acting at (x_ea - 0.25c) aft of the
            # quarter-chord aerodynamic centre.  ``_AE_ELASTIC_AXIS`` is the
            # absolute elastic-axis position aft of the leading edge (0.35),
            # so the lift arm is 0.10c, not 0.35c.
            moment = (
                qa
                * self.chord[None, :]
                * (cm + (_AE_ELASTIC_AXIS - 0.25) * cl)
            ).reshape(qa.shape[0], self.n_surfaces, self.strips).sum(2)
            accel = (
                moment - self._twist_k * self._twist - self._twist_c * self._twist_rate
            ) / self._twist_i
            # Semi-implicit damping keeps the explicit 240 Hz integration stable.
            rate = ((self._twist_rate + self.dt * accel) / (
                1.0 + self.dt * self._twist_c / self._twist_i
            )).clamp(-50.0, 50.0)
            twist = (self._twist + self.dt * rate).clamp(
                -_AE_TWIST_LIMIT, _AE_TWIST_LIMIT
            )
            self._twist_rate.copy_(
                torch.where(mask[:, None], rate, self._twist_rate)
            )
            self._twist.copy_(
                torch.where(mask[:, None], twist, self._twist)
            )
        lift, drag = cl * qa, cd * qa
        normal = lift * torch.cos(alpha) + drag * torch.sin(alpha)
        parallel = lift * torch.sin(alpha) - drag * torch.cos(alpha)
        force = normal[..., None] * self.lift + parallel[..., None] * self.forward
        torque = (qa * cm * self.chord)[..., None] * self.torque_axis
        if self.strips > 1:
            # Moment of each station's force about the surface centre, then
            # sum the stations back onto their parent surface.
            torque = torque + torch.linalg.cross(self._strip_offset[None, :, :], force)
            n = force.shape[0]
            force = force.reshape(n, self.n_surfaces, self.strips, 3).sum(2)
            torque = torque.reshape(n, self.n_surfaces, self.strips, 3).sum(2)
        return force, torque
