"""Falcon 9 first-stage parameters.

Provenance tags:
  [official]  SpaceX Falcon 9 Users Guide, 2025-05-09 revision
  [community]  Open literature, NASA Spaceflight, FAA, etc.
  [estimate]   Engineering estimate when no public source exists
  [chosen]     Numerical choice (controller gains, damping coefficients, etc.)
  [derived]    Computed from other parameters (e.g. mdot = T/(Isp g0))
  [mirror]     Copied from bochen2029/booster-lander-simulator constants.h (v0.4.0, e0e5769)
               then audited and corrected per the 9 audit findings below

Source repository reference:
  https://github.com/bochen2029-pixel/booster-lander-simulator
  commit e0e57693a172594e9db965712fdd6b4469f0823d
  LICENSE: MIT (per repository README)

Audit fixes applied to source constants.h values:
  AUDIT-1  Minimum throttle: source ENG_THR_MIN=0.40 → changed to 0.57 [derived from 2025
           users guide 108,300 lbf vacuum (min) / 190,000 lbf vacuum (max) ≈ 0.57; upstream
           0.40 was unverified engineering estimate]
  AUDIT-2  Grid-fin local velocity: source uses r_mount_relative_to_BODY_ORIGIN; we use
           r_mount_relative_to_CG. Fixed in tensor flow in falcon9_dynamics.py (v_f = v_CG + ω × r_mount2CG)
  AUDIT-3  Minimum throttle as discrete idle state, NOT zero thrust. Engine shutdown is a separate
           discrete state (engine_on=False, thrust=0). Cannot exploit sub-throttle via rapid on/off.
  AUDIT-4  Contact substep: fin deflection IS updated (contrary to source, which skipped fins in
           contact_substep). We unify actuation update and contact integration into one loop.
  AUDIT-5  First-touch forced shutdown: NOT carried. Engine state is independent of contact phase.
  AUDIT-6  Variable-mass rotation equation: uses controlled-volume derivation (see model docstring).
           I_dot*ω term is retained but the source lacked outflow-angular-momentum correction.
           Simplified here as I_total·ωdot = τ_total - ω × (I_total·ω) + I_dot·ω (see §10 model notes).
  AUDIT-7  Body aero CoP at zcp ≈ 0.28·L_STAGE from base, marginally below CG (~0.30·L_STAGE) →
           bare body slightly unstable, fins needed for trim. Matches source but independently verified.
  AUDIT-8  RCS: explicit 6-nozzle cold-gas system at booster attach ring, 2 axis sets (pitch/yaw 2 nozzles
           each + roll 2 nozzles). Cannot provide arbitrary 3-axis torque from TVC alone.
  AUDIT-9  Slosh: disabled. Not claimed to model liquid dynamics. Stated as approximation.
"""

from dataclasses import dataclass, field
import math


# =============================================================================
# World / environment
# =============================================================================
@dataclass(frozen=True)
class WorldParams:
    g0: float = 9.80665          # m/s² [official]
    R_earth: float = 6_356_766.0  # m, geopotential [official]
    R_air: float = 287.053        # J/(kg·K) [official]
    gamma_air: float = 1.4        # [official]
    rho0: float = 1.225           # kg/m³ sea level [official]
    P0_atm: float = 101_325.0     # Pa [official]


# =============================================================================
# Integration (strategy: 1000 Hz physics, 50 Hz policy — 20 substeps per policy tick)
# dt=0.001 over 0.002 because Falcon 9 (29 t, r=1.83 m) edge-contacts the pad
# at ~10× the toy rocket's impact impulse. 0.002s produces per-substep impulses
# that exceed Genesis' implicit contact solver conditioning (→ Invalid constraint
# forces / NaN). 100/200 rigid iterations also bump solver robustness.
# =============================================================================
@dataclass(frozen=True)
class TimingParams:
    physics_dt: float = 0.001      # s, 1000 Hz [chosen]
    policy_dt: float = 0.020       # s, 50 Hz [chosen]
    substeps_per_policy: int = 20  # = 0.020 / 0.001
    max_episode_seconds: float = 300.0  # s [chosen, up from 30s]


# =============================================================================
# Vehicle structure (Falcon 9 first stage + interstage, real scale)
# =============================================================================
@dataclass(frozen=True)
class VehicleParams:
    VEH_LEN: float = 47.7           # m total [official, Falcon 9 Users Guide 2025 §2.3]
    STAGE_LEN: float = 41.2         # m first stage [official]
    INTERSTAGE_LEN: float = 6.5     # m [community]
    VEH_DIA: float = 3.66           # m [official]
    VEH_RADIUS: float = 1.83        # m, derived from DIA
    VEH_AREF: float = 10.52         # m², π·D²/4 [derived]
    VEH_DRY: float = 25_600.0       # kg incl. legs + grid fins [community, estimate 24-27t]
    VEH_DRY_COMZ: float = 12.4      # m above base [estimate, pinned by source but unverified]
    # Base link origin convention: body-frame z=0 at base (bottom of booster + leg attach ring),
    # body-frame z positive UPWARD along vehicle axis.
    # Engine (Merlin-1D single center) gimbal pivot is at body z = 0 (base plane).
    # CG body-frame z = VEH_DRY_COMZ initially, shifts with fuel.


# =============================================================================
# Propellant (two-column model: LOX above RP-1)
# =============================================================================
@dataclass(frozen=True)
class PropellantParams:
    LOX_MAX: float = 287_400.0      # kg [community, source of truth on Wikipedia]
    RP1_MAX: float = 123_500.0      # kg [community]
    LOX_RHO: float = 1_220.0        # kg/m³, subcooled [estimate]
    RP1_RHO: float = 833.0          # kg/m³ [estimate]
    TANK_AREA: float = 9.9          # m² column cross-section [derived: Aref minus wall]
    LOX_BASE_Z: float = 16.0        # m, LOX column base above vehicle base [chosen, matches source]
    RP1_BASE_Z: float = 1.6         # m, RP-1 column base above vehicle base [chosen, matches source]
    MIX_RATIO: float = 2.33         # LOX:RP1 by mass [community]
    RCS_N2: float = 300.0           # kg cold-gas budget [estimate]


# =============================================================================
# Engine (Merlin-1D, single center, vacuum-optimized sea-level-rated)
# =============================================================================
@dataclass(frozen=True)
class EngineParams:
    ENG_T_SL: float = 845_000.0     # N, sea-level 100% [official, Users Guide 2025: 185,000 lbf / eng]
    ENG_ISP_SL: float = 282.0       # s [official]
    ENG_ISP_VAC: float = 311.0      # s [official]
    ENG_AE: float = 0.859           # m², nozzle exit area [derived: (T_vac - T_sl)/P0_atm]
    ENG_T_VAC: float = 932_000.0    # N, vacuum 100% [derived: mdot * g0 * ISP_vac]
    # AUDIT-1: min throttle NOT 0.40. Users Guide gives range 108,300-190,000 lbf vacuum
    # → 0.569 ≈ 0.57 vacuum, ~0.55 sea-level. Use conservative 0.57 as default min.
    ENG_THR_MIN: float = 0.57       # min throttle fraction [derived from official range]
    ENG_THR_MAX: float = 1.0
    # TVC: single Merlin 2-axis gimbal (NOT octaweb cluster in this model)
    ENG_GIMBAL_MAX: float = math.radians(5.0)        # rad, ±5 deg [community]
    ENG_GIMBAL_RATE: float = math.radians(15.0)      # rad/s [chosen]
    ENG_GIMBAL_ACC: float = math.radians(60.0)       # rad/s² [chosen]
    # Engine nozzle position in body frame (body origin = base_link bottom)
    # Used to compute gimbal lever-arm torque r_nozzle × F_thrust when applying
    # composite force to the Genesis single-body.
    ENG_NOZZLE_Z_BODY: float = -0.5                 # m, below body origin [chosen]
    # Throttle dynamics
    ENG_THR_RATE: float = 2.0       # throttle/s slew limit [chosen]
    ENG_THR_TAU: float = 0.10       # s first-order lag [chosen]
    # Ignition / shutdown
    ENG_IGN_GREEN: float = 0.30     # s: TEA-TEB flash [chosen]
    ENG_IGN_T0: float = 0.50        # s: thrust starts rising [chosen]
    ENG_IGN_T1: float = 1.50        # s: reaches commanded [chosen]
    ENG_SHUT_TAU: float = 0.15      # s shutdown decay [chosen]
    # Relight budget
    ENG_RELIGHTS_MAX: int = 2       # [estimate, matches source sim.c default]
    # Number of engines (center single for landing; 3-cluster for entry burn)
    ENG_CLUSTER_SIZES: tuple = (1, 3)  # [chosen]


# =============================================================================
# Grid fins (4, body-fixed at 45° spacing)
# =============================================================================
@dataclass(frozen=True)
class GridFinParams:
    FIN_COUNT: int = 4
    FIN_Z: float = 45.0             # m, body-z of fin mount [estimate — above interstage]
    FIN_RADIUS: float = 1.83        # m (VEH_RADIUS) [derived]
    FIN_AREA: float = 2.4           # m² per fin [chosen, matches source]
    FIN_DEFL_MAX: float = math.radians(20.0)  # rad max deflection [community]
    FIN_RATE: float = math.radians(20.0)       # rad/s [chosen]
    FIN_STALL: float = math.radians(25.0)      # rad stall AoA [chosen]
    FIN_CNA: float = 3.0            # /rad subsonic normal-force slope [chosen]
    # Roll-cant coefficient as fraction of CNa_f — small tangential force from deflection
    FIN_CT_DELTA_FRAC: float = 0.35  # [chosen]
    # Fins deploy altitude threshold (low qbar stow, high qbar deploy)
    FIN_DEPLOY_H: float = 25_000.0   # m, deploy below this [source scenario default]


# =============================================================================
# RCS (cold-gas, six-nozzle for pitch/yaw/roll)
# =============================================================================
@dataclass(frozen=True)
class RCSParams:
    RCS_FORCE: float = 500.0        # N per nozzle [estimate]
    RCS_ISP: float = 68.0           # s [estimate]
    RCS_Z: float = 40.5             # m body-z of attach ring [estimate, below grid fins]
    RCS_ARM: float = 1.83           # m = VEH_RADIUS, moment arm [derived]
    RCS_DEADBAND: float = 30.0      # Nm deadband (PWM-averaged model) [chosen]
    # Torque per axis = 2 nozzles * F * arm
    RCS_MAX_TORQUE: float = 2 * 500.0 * 1.83  # Nm [derived]


# =============================================================================
# Body aerodynamics
# =============================================================================
@dataclass(frozen=True)
class AeroParams:
    # Mach-dependent coefficients (matches source AERO_M/AERO_CA/AERO_CN tables, independently
    # verified against slender-body data)
    AERO_M: tuple = (0.0, 0.6, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0, 8.0)
    AERO_CA: tuple = (0.85, 0.88, 1.10, 1.40, 1.25, 1.10, 0.95, 0.92, 0.90)
    AERO_CN: tuple = (2.0, 2.1, 2.4, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0)
    # Pitch/yaw damping coefficient: Cdc = 0.6 gives ζ~0.1 (moderate) [chosen]
    BODY_CMQ_CDC: float = 0.6
    # CoP as fraction of stage length from base (subsonic 0.28L → marginally below CG ≈ 0.30L)
    # CoP = xcp_frac(M,α) * STAGE_LEN, source has smooth transonic bump.
    XCP_BASE: float = 0.28         # [chosen, from source audit D-005]
    XCP_TRANSONIC_BUMP: float = 0.01  # peak at M=1.05 [chosen]


# =============================================================================
# Landing legs (4, deployed ≈ 18 m footprint diameter)
# =============================================================================
@dataclass(frozen=True)
class LegParams:
    LEG_COUNT: int = 4
    LEG_SPAN: float = 18.0          # m footprint diameter [official]
    LEG_RADIUS: float = 9.0         # m half-span
    LEG_DEPLOY_H: float = 250.0     # m, deploy below [source scenario]
    LEG_DEPLOY_T: float = 1.2       # m sweep time [chosen]
    # Contact model (handled by Genesis in our architecture; values here for reference)
    LEG_K: float = 3.0e6            # N/m spring [chosen]
    LEG_C: float = 1.2e5            # Ns/m damper [chosen]
    LEG_MU_S: float = 0.6
    LEG_MU_K: float = 0.45
    LEG_CRUSH_F: float = 4.0e5      # N plateau [chosen]
    LEG_CRUSH_S: float = 0.40       # m stroke [chosen]
    LEG_FOOT_Z_BASE: float = 0.0   # feet at body z=0 (base plane) when stowed
    LEG_FOOT_Z_DEPLOY: float = -1.0 # feet 1 m below body z=0 when fully deployed


# =============================================================================
# Pad geometry
# =============================================================================
@dataclass(frozen=True)
class PadParams:
    PAD_RADIUS: float = 26.0        # m RTLS pad [chosen]
    PAD_Z: float = 0.0              # m deck height, flat


# =============================================================================
# Terminal verdict thresholds (diagnostic, NOT part of dynamics)
# =============================================================================
@dataclass(frozen=True)
class VerdictParams:
    TD_V_PERFECT: float = 2.0       # m/s impact speed
    TD_V_GOOD: float = 4.0
    TD_V_HARD: float = 6.0
    TD_LAT_PERFECT: float = 2.0     # m lateral offset
    TD_LAT_GOOD: float = 10.0
    TD_TILT_PERFECT: float = math.radians(1.0)
    TD_TILT_GOOD: float = math.radians(3.0)
    QBAR_MAX: float = 80_000.0      # Pa struct limit
    TD_V_TARGET: float = 1.5        # m/s desired touchdown


# =============================================================================
# Scenario presets (from booster-lander-simulator/core/scenario.c, v0.4.0)
#
# Five canonical Falcon 9 landing scenarios calibrated against real flight
# trajectories. Each entry gives the pre-dispersion initial state plus the
# seeded-dispersion sigmas applied per run. Heights / velocities come from
# Falcon 9 flight telemetry and entry-burn / landing-burn sequencing; the
# propellant column reserves enough for one landing burn (~2500 kg) plus
# margin, matching the source simulator's loadout convention.
#
# h0        world-z altitude [m], body-origin (CoM is raised by mp.com above)
# vz0       initial world-z velocity [m/s]  (negative = descending)
# lateral   horizontal offset from pad [m]
# prop0     total propellant mass [kg]  = m_lox + m_rp1 (split by MIX_RATIO)
# tilt0     initial body tilt from vertical [rad] (azimuth randomised)
# u_ref     reference wind speed [m/s] at deck height (power-law profile)
# w20       gust strength [kt] Dryden 20 kt reference
# fins      initial grid-fins deployed flag (1=deployed, 0=stowed)
#
# Dispersion sigma (per run, seeded):
#   h0     ± 6%   (altitude reading noise + variable coast from entry burn)
#   vz0    ± 12 m/s  (terminal scenario)  or ± 20 m/s  (aero/entry)
#   lx,ly  ± 30 / 150 / 250 m  per scenario's physical divert ceiling
#   vx,vy  ± 5 m/s
#   tilt   ± 2 deg
#   ωx,ωy,ωz ± 0.02 rad/s
# =============================================================================
@dataclass(frozen=True)
class ScenarioDef:
    name: str
    h0: float           # m
    vz0: float          # m/s (negative = descending)
    lateral: float      # m
    prop0: float        # kg total (split LOX/RP1 by MIX_RATIO)
    tilt0: float        # rad
    u_ref: float        # m/s reference wind at deck
    w20: float          # kt Dryden gust strength
    fins: int           # 0 stowed, 1 deployed
    # Per-scenario dispersion sigmas (see scenario.c for exact values)
    h0_sigma_frac: float = 0.06
    vz0_sigma: float = 12.0
    lat_sigma: float = 30.0
    velxy_sigma: float = 5.0
    tilt_sigma: float = math.radians(2.0)
    angvel_sigma: float = 0.02


SCENARIO_PRESETS: dict[str, ScenarioDef] = {
    # ---- RL training workhorse: the final-approach / landing-burn segment ----
    # 2 km altitude, post-entry-burn, subsonic terminal descent already slowed
    # from ~250 m/s by aerodynamic drag; grid-fins still stowed (passive aero
    # decelerating, TVC handles attitude once the landing burn lights).
    "terminal": ScenarioDef(
        name="terminal", h0=2000.0, vz0=-180.0, lateral=0.0, prop0=8500.0,
        tilt0=math.radians(3.0), u_ref=3.0, w20=15, fins=0,
        h0_sigma_frac=0.06, vz0_sigma=12.0, lat_sigma=30.0,
    ),

    # ---- Entry-burn completion near Mach 1, high qbar, fins deployed ----
    # 12 km altitude, supersonic / transonic, lateral divert from a 500 m
    # cross-range offset is entirely grid-fin authority (aerodynamic guidance).
    "aero_offset": ScenarioDef(
        name="aero_offset", h0=12000.0, vz0=-330.0, lateral=500.0, prop0=10000.0,
        tilt0=math.radians(5.0), u_ref=6.0, w20=30, fins=1,
        h0_sigma_frac=0.06, vz0_sigma=20.0, lat_sigma=150.0,
    ),

    # ---- Entry burn ignition: high Mach, high altitude, full prop reserve ----
    # 62 km altitude, Mach ~5, 3 km lateral offset (typical boostership
    # downrange), 30 t propellant for entry-burn + landing-burn budget.
    "entry": ScenarioDef(
        name="entry", h0=62000.0, vz0=-1500.0, lateral=3000.0, prop0=30000.0,
        tilt0=math.radians(6.0), u_ref=8.0, w20=30, fins=1,
        h0_sigma_frac=0.06, vz0_sigma=20.0, lat_sigma=250.0,
    ),

    # ---- ASDS-night droneship recovery: larger lateral, heavier wind ----
    "asds_night": ScenarioDef(
        name="asds_night", h0=12000.0, vz0=-330.0, lateral=800.0, prop0=10000.0,
        tilt0=math.radians(5.0), u_ref=8.0, w20=30, fins=1,
        h0_sigma_frac=0.06, vz0_sigma=20.0, lat_sigma=250.0,
    ),

    # ---- Chaos: harshest parameters for robustness evaluation ----
    "chaos": ScenarioDef(
        name="chaos", h0=9000.0, vz0=-250.0, lateral=600.0, prop0=9000.0,
        tilt0=math.radians(8.0), u_ref=12.0, w20=45, fins=1,
        h0_sigma_frac=0.06, vz0_sigma=20.0, lat_sigma=250.0,
    ),
}


# =============================================================================
# Aggregate container
# =============================================================================
@dataclass(frozen=True)
class Falcon9Params:
    world: WorldParams = field(default_factory=WorldParams)
    timing: TimingParams = field(default_factory=TimingParams)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    propellant: PropellantParams = field(default_factory=PropellantParams)
    engine: EngineParams = field(default_factory=EngineParams)
    fins: GridFinParams = field(default_factory=GridFinParams)
    rcs: RCSParams = field(default_factory=RCSParams)
    aero: AeroParams = field(default_factory=AeroParams)
    legs: LegParams = field(default_factory=LegParams)
    pad: PadParams = field(default_factory=PadParams)
    verdict: VerdictParams = field(default_factory=VerdictParams)
    # Default scenario preset name (used when env is constructed without an
    # explicit initial state). "terminal" = final 2 km, most common RL setup.
    default_scenario: str = "terminal"

    # Useful derived accessors
    @property
    def VEH_AREF(self) -> float: return self.vehicle.VEH_AREF

    @property
    def TOTAL_PROPELLANT_MAX(self) -> float:
        return self.propellant.LOX_MAX + self.propellant.RP1_MAX

    @property
    def VEH_MASS_GROSS(self) -> float:
        return self.vehicle.VEH_DRY + self.TOTAL_PROPELLANT_MAX

    def engine_thrust(self, throttle_act: float, p_amb: float) -> float:
        """Presssure-corrected vacuum thrust times throttle. Returns 0 for throttle<=0.

        thrust = throttle * (T_vac - A_e * p_amb)
        T_vac = mdot_max * g0 * ISP_vac (derived from sea-level: T_sl, ISP_sl)
        """
        ep = self.engine
        if throttle_act <= 0.0:
            return 0.0
        T_vac = ep.ENG_T_VAC
        A_e = ep.ENG_AE
        return throttle_act * (T_vac - A_e * p_amb)


# Default singleton for convenience
P = Falcon9Params()
