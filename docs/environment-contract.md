# Environment Contract and Wind Disturbance

**English** · [简体中文](environment-contract.zh-CN.md)

## Contract

- `obs_buf` / `rew_buf` are `float32`, internal physics state is `float64`.
- `extras['time_outs']` marks pure time-limit truncations (for rsl-rl
  time-limit bootstrapping); `extras['terminal_observations']` carries the
  pre-reset observation.
- `completed_episodes` / `completed_successes` / `completed_returns`
  accumulate on device; `complete` is the per-replica success flag.
- Actions arrive on `env.device`; implicit host/device copies raise.
- Physics, contacts, reward, reset and RNG are masked tensor kernels, so
  inactive replicas are frozen and draw no random numbers. The only per-step
  host sync is `bool(done.any())`, which gates episode-boundary bookkeeping.

Design claims and their verification conditions are written up in
[benchmark_report.md §Appendix A](../benchmark_report.md#appendix-a-infrastructure-and-reproducibility).

## Wind disturbance

Five atmospheric tasks (QuadX-Waypoints, Fixedwing-Waypoints, Quadplane-VTOL,
Rocket-Landing, Missile-Intercept) accept optional wind:

- `steady_wind` (tuple, m/s) — constant world-frame wind velocity;
- `dryden` (0–1 scale of the severity presets) + `severity`
  (`light | moderate | severe`) — Dryden gusts from
  [environments/common/disturbances.py](../environments/common/disturbances.py),
  advanced once per control step.

Aerodynamic terms always evaluate air-relative velocity `v_air = v - wind`,
so calm air (`steady_wind=None, dryden=0`) reproduces the benchmark
bit-for-bit. `scripts/eval.py` restores each run's recorded `env_kwargs`, so a
checkpoint trained in wind is evaluated in the same wind.

| Task | Training-entry wind |
|---|---|
| quadx, fixedwing, quadplane | pure Dryden gusts, `dryden=0.3, severity="moderate"` |
| rocket | `dryden=0.0` (same caliber as the calm-air baseline) |
| missile, quadx_track | calm air |
| satellite, transfer, docking, rendezvous | vacuum |

## Realistic fidelity preset

The five atmospheric plants expose `aero="legacy" | "realistic"` (default
`"legacy"`). `"legacy"` is the published benchmark: sea-level density,
incompressible lifting surfaces, ideal sensors, unlimited servo rate and
first-order Dryden gusts, bit-for-bit. `"realistic"` enables the full
high-fidelity stack below without touching the frozen URDF/YAML assets:

- **ISA atmosphere** (`environments/common/atmosphere.py`): density, pressure,
  temperature and speed of sound versus altitude (0–20 km). Dynamic pressure
  and every body-drag term use the per-replica local density; the missile at
  4 km therefore sees `rho ~ 0.819 kg/m^3` instead of 1.225. The realistic
  preset adds a +15 K, 1 % specific-humidity off-standard day
  (`rho = p/(R T (1 + 0.6078 q))`), which lowers the density by ~5 % and
  raises the speed of sound; the legacy/table path keeps the standard day.
- **Compressibility and span efficiency**: Prandtl-Glauert lift-slope growth
  below `M=0.9`, a blended transonic band, linear-supersonic
  `4/sqrt(M^2-1)` decay above `M=1.2`, a wave-drag multiplier, and Oswald
  induced-drag factors (0.85 wings, 0.90 tails, 0.85 missile fins; grid fins
  keep `e=1`).
- **Spanwise strips**: every realistic lifting surface is split into spanwise
  stations (6 on the fixed-wing/quadplane surfaces, 5 on missile and grid
  fins). Each station carries `1/S` of the surface area and sees its own
  air-relative velocity including the rigid-body rotation `omega x r`, its own
  dynamic pressure (freestream plus the geometrically resolved rotor
  slipstream), local Mach number, flap deflection and Beddoes-Leishman stall
  state. Forces and the `r x F` station moments are summed back onto the
  parent surface, which gives physical roll/pitch damping
  (`C_l_p = -C_l_alpha/6` in the strip limit) and spanwise stall/slipstream
  progression. One station reproduces the legacy single-point model exactly.
- **Parasite drag and unsteady lift**: a fixed-wing fuselage/gear equivalent
  flat-plate area of `0.010 m^2` per body axis; a first-order Wagner-style
  lift/drag/moment lag over one convective time `c/V` (10–200 ms); and a
  Beddoes-Leishman-style dynamic-stall correction with a lagged trailing-edge
  separation point and a rate-driven vortex-lift state. The correction is
  applied as an increment over the quasi-steady polar, so steady attached and
  post-stall coefficients are unchanged and only pitch-up/pitch-down
  hysteresis is added. The vortex also contributes drag (`cn_v sin(alpha)`)
  and a strong nose-down pitching-moment increment, both zero when the
  separation state is steady.
- **Actuators**: per-surface servo rate limits (fixed-wing 250–300 deg/s,
  missile 800 deg/s, grid fins 200 deg/s), a 1% deadband, a first-order lag,
  a 0–2 physics-step command transport delay, mechanical backlash and
  asymmetric extension/retraction rates. No hinge-moment/load-dependent
  deflection model.
- **Sensors** (`environments/common/sensors.py`): per-replica gyro noise and
  bias random walk, attitude/velocity/position noise, missile seeker noise on
  the relative-state/LOS block, and a one-policy-step transport delay
  (disabled for the 40 Hz 27 g quadx task, where it destabilises the plant).
  Sigmas are continuous-time densities rescaled by `sqrt(dt_ref/dt)`, so the
  30/40/60 Hz tasks have comparable noise bandwidth, and the terminal
  bootstrap observation uses the same sensor pipeline as the policy.
- **Propulsion**: propeller thrust scales with `(rho/rho0)^(2/3)`; fixed-pitch
  propellers lose thrust with axial advance through the quadratic
  blade-element fit `C_T(J) = C_T0 (1 - (J/J0)^2)`, `J = V_axial/(n D)`,
  `J0 = pitch/D`, so at airspeed `V` a rotor keeps
  `1 - (V/(pitch * n_max * throttle))^2` of its static thrust (zero at the
  throttle-dependent pitch speed). The three electric
  vehicles carry a per-replica battery pack (series cells, capacity, internal
  resistance, avionics load): terminal voltage sags with the instantaneous
  current and follows the state of charge, available thrust scales with
  `(V/V_full)^2`, and charge is Coulomb-counted from momentum-theory shaft
  power divided by a figure of merit. Both effects are intra-episode (a long
  flight drains and weakens the pack); the quadplane slipstream is computed
  from the actual rotor and surface geometry
  (momentum-theory axial development `q = (T/A)/4 (1 + x/sqrt(x^2+R^2))^2`
  with Gaussian radial spreading), not a lumped wing-area coverage fraction;
  the rocket and missile gain the ambient-pressure nozzle-thrust correction;
  the missile burns 30% propellant mass over its 4 s boost.
- **Aeroelasticity**: the fixed-wing/quadplane lifting surfaces carry a lumped
  first torsional mode (8 Hz, 15 % damping) about a 35 %-chord elastic axis,
  driven by the local pitching moment and lift. The stiffness is derived from
  a 100 m/s linear divergence speed, so the elastic twist stays below 0.2 deg
  in the task envelope and grows without bound beyond it (clamped at 20 deg as
  a numerical/structural stop). Spanwise bending is not modelled.
- **Wing-in-ground effect**: every lifting surface is corrected with the
  lifting-line image solution at its station height `h` and full span `b`:
  `phi = (16h/b)^2 / (1 + (16h/b)^2)` (1 in free air, 0 at the ground), lift
  slope `cl_alpha / (1 + cl_alpha phi/(pi AR))`, induced drag scaled by
  `phi`, and a nose-down moment increment `-0.1 (1-phi) cl`. Because the
  correction is evaluated per spanwise station, a banked wing near the ground
  also gets the physical rolling moment. This is separate from the multirotor
  Cheeseman-Bennett thrust factor.
- **Multirotor detail**: a linear rotor-drag term, rotor gyroscopic body
  moment, and a Cheeseman-Bennett ground-effect factor evaluated per rotor
  from its own world height (correct under pitch/bank), not from body height.
- **Missile target and seeker**: the kinematic target's lateral acceleration
  is capped at 15 g, its seeker observations are noisy, and the 30% propellant
  burn scales both the base-link mass and inertia tensor (COM shift is not
  modelled). The body-fixed seeker has a 60 deg half-angle and 20 km range:
  outside it the target relative-state/LOS/crossing-speed block is zeroed and
  the lock flag drops (the noise model cannot re-introduce information). The
  measured block also passes through a first-order tracking filter
  (tau = 20 ms) and carries a radome-slope bias (0.02 rad per rad of AoA /
  sideslip), which rotates the apparent LOS toward the nose and feeds the
  classical parasitic LOS-rate loop. The
  target also carries a point-mass energy model: parabolic drag
  `q S CD0 + (n m g)^2/(q S pi AR e)`, density-lapsed installed thrust and a
  proportional speed hold, so a sustained high-g manoeuvre bleeds speed
  instead of holding 250 m/s forever.
- **Contact resolution**: every realistic atmospheric task runs the Genesis
  contact-constraint solver, so a touching surface exchanges normal, friction
  and restitution impulses instead of only raising a flag. The legacy path
  keeps detection-only contacts for the published benchmark. On the rocket,
  the pad can now support, tip and frictionally dissipate the booster, and v2
  success requires 0.5 s of continuous settled pad contact rather than a
  single touch.
- **Turbulence and shear**: exact MIL-F-8785C Dryden filters (first-order
  longitudinal, second-order lateral/vertical), altitude-dependent scale
  lengths, and a power-law mean-wind profile with `alpha=0.14` referenced to
  10 m. Realistic tasks rebuild the filters every control step from the
  per-replica airspeed and altitude (tensor-only, graph-capture safe), so
  `L = L(h)`, `sigma = sigma(h)` and the bandwidth `V/L` follow a climbing or
  accelerating vehicle instead of the construction-time reference condition.
- **Domain randomization**: `param_mismatch` applies a static multiplier at
  construction and stores the unperturbed SI nominal in `controller_design`;
  `domain_randomize` samples an independent uniform factor per replica per
  episode (`False`/`{}` disables, a dict merges over the defaults).  Besides
  thrust/mass, the realism-only default table adds:
  - `mass`: every structural link **and its inertia** (fuel/propellant links
    keep their physical depletion model), so the requested ±10% is the total
    mass change, not the old body-link-only ~1.3%;
  - `cl_alpha` 0.92–1.08, `cd0` 0.85–1.15, `stall` 0.90–1.10, `oswald`
    0.95–1.05 as independent aerodynamic axes;
  - `actuator_deadband` 0.5–2.0, `actuator_backlash` 0.5–2.0,
    `actuator_asymmetry` 0.7–1.3 and `actuator_delay` 0–2 physics steps;
  - `sensor` 0.8–1.2 (all noise sigmas and the bias walk);
  - `cg`/`cg_y`/`cg_z`: rigid body-`+x/+y/+z` CG displacement (full scale =
    10% of the vehicle's characteristic length), 0.9–1.1 on the UAVs,
    0.95–1.05 on the missile and 0.98–1.02 on the booster;
  - `inertia` plus `inertia_x/y/z`: uniform and per-diagonal inertia scaling,
    0.9–1.1.
  The resolved range table is written to `cfg["domain_randomize"]`, and the
  applied per-replica factors are exposed as `extras["dr_scales"]` after every
  reset for failure attribution.  The legacy benchmark table is unchanged.

The training entries pass `aero="realistic"` as of 2026-11-10, so new runs
record it in `config.json` / `run.json` and `eval.py` restores it. Every run
also records `realism_version` (`"legacy"` or the current preset version);
bump `REALISM_VERSION` in `plant.py` whenever a realistic constant or model
changes. Pass `aero="legacy"` explicitly to reproduce the published benchmark
numbers.
Known omissions on both paths: wing bending and landing-leg crush (the lumped
torsional wing mode is modelled),
battery thermal/aging effects and fuel depletion for UAVs, Reynolds-number
effects, servo hinge moments, unsteady drag/moment hysteresis beyond the
modelled vortex lift, rotor-wake interaction between neighbouring propellers,
spanwise induced-velocity coupling between stations (each station uses its
parent surface's finite-span slope rather than a full lifting-line solve),
landing-gear suspension/rolling resistance (the frozen URDF assets carry no
wheel or suspension geometry, so contacts stay flat-plane friction contacts),
missile propellant COM migration, rocket propellant slosh and landing-leg
flexibility, and terrain/obstacles beyond the flat ground plane.

The numerical verification suite is
[tests/test_fidelity_mechanisms.py](../tests/test_fidelity_mechanisms.py): it
checks ISA against the standard-atmosphere table, Prandtl-Glauert / linear
supersonic slopes, Oswald induced drag, servo rate/deadband, the unsteady
convective lag, dynamic-stall lift/drag/moment hysteresis and steady-polar
preservation, Cheeseman-Bennett ground effect, geometric momentum-theory slipstream,
rotor gyroscopic torque, the blade-element advance-ratio thrust curve and its
plant-level application, battery open-circuit voltage, Coulomb-counting and
energy bookkeeping, spanwise-strip equivalence with the single-point model,
midpoint quadrature of spanwise slipstream profiles, the analytic strip roll
damping, the lifting-line wing-in-ground-effect factors and their free-air /
h -> 0 limits, plant-level strip-height wiring, aeroelastic twist decay,
divergence trend and above-critical boundedness, the hot-day/humidity density
relations, the radome-slope LOS bias and seeker-filter step response, sensor noise/bias/latency and sensor-scale
randomization, mass/inertia consistency, independent aero coefficients,
3-D CG randomization, actuator delay/backlash/asymmetry dynamics, the
MIL-F-8785C Dryden PSD shape and the time-varying filter stationarity, wind
shear, seeker field-of-view lock loss, missile propellant depletion and target
load limit, the target thrust/drag energy balance, and the contact solution
for rocket and flight tasks.

## Installation

```bash
python -m pip install -r requirements.txt
```

Exact versions are pinned in [requirements.txt](../requirements.txt); the
reference environment is Python 3.10.20 with PyTorch 2.11.0 (CUDA 12.8),
`genesis-world` 1.4.4, `rsl-rl-lib` 5.3.0, NumPy 2.2.6 and SciPy 1.15.3 on
Linux (RTX 3090, driver 580.173.02).

A CUDA build of PyTorch is required for the six Genesis tasks on the GPU
backend; the four tensor-dynamics tasks and every classical baseline also run
on CPU-only PyTorch. `gymnasium` / `pettingzoo` are not dependencies of this
suite.
