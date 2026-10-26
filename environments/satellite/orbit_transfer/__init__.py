"""Low-thrust perigee-raising orbital transfer task (electric propulsion).

Description
    A 100 kg medium-Earth-orbit satellite with three orthogonal thruster
    pairs (10 mN per axis, Isp = 4000 s electric propulsion) fires for 166
    control steps of 300 s (~13.8 h, ~5 revolutions of the a = 10,000 km,
    e = 0.15 orbit) to raise the pericenter radius of 8500 km while keeping
    the apocenter radius of 11,500 km fixed and propellant use low. The
    initial mean anomaly is resampled uniformly on [-pi, pi] at every
    reset; the initial orbit otherwise is fixed (i = 60 deg, RAAN = 120
    deg, argument of perigee = 240 deg). The reachable delta-v budget is
    ~15 m/s, so meaningful perigee raising requires phasing thrust near
    perigee. Dynamics are two-body point-mass gravity (WGS84
    mu = 3.986004418e14 m^3/s^2) plus constant thrust over each control
    step, integrated with fixed-step RK4 at 10 s substeps, 30 substeps per
    control step (the reference uses Orekit adaptive Dormand-Prince
    8(5,3); per-revolution phase error here stays far below reward
    granularity). The propellant drain is the Tsiolkovsky rate scaled by
    the commanded thrust, mdot = -F_max ||u||_1 / (Isp g0), so each axis
    burns in proportion to its |u| component. Reset states come from a
    fixed 8-iteration Newton solve for the eccentric anomaly. Fully
    tensorized: all replicas advance on one device, no engine, no gym.

Action Space (3,)
    Thrust-direction commands in [-1, 1]^3, one per body axis; each axis
    delivers up to 10 mN of continuous thrust, giving a three-axis
    acceleration of ~3e-4 m/s^2 at full burn.

Observation Space (8,)
    float32, unbounded raw state, 'policy' TensorDict group:
    [elapsed time (1), position in GCRF (3), velocity in GCRF (3),
    remaining mass (1)]. Returned unnormalized, exactly as the reference.

Reward
    Dense potential-based shaping every step:
    r_t = Phi(s_{t+1}) - Phi(s_t)
    with Phi = 1e-4 * (-|ra - ra0| + (rp - rp0) + 2e5 * (m - mass0)),
    radii in meters and mass in kg. Phi is zero at every reset, so the
    episode sum equals the reference terminal reward exactly,
    R = 1e-4 * (-|dra| + drp + 2e5 * dm) -- i.e. +0.1 per km of perigee
    radius raised, -0.1 per km of |apogee radius drift|, -20 per kg of
    propellant spent (the reference's Table I weights). R is recorded at
    the 166-step limit as the success criterion and in the Episode
    metrics, but is not paid as a separate sparse grant.

Episode End
    Truncation: 166 control steps (49,800 s of mission time), the only
    reachable terminal. Success is R > 0 at that final state (the
    reference's is_success=True at any done is vacuous under
    timeout-only termination), so completed_successes counts missions.
    time_outs is all-False: the fixed horizon is a true terminal, which
    disables the rsl_rl gamma * V timeout bootstrap. Defensive
    terminations (perigee below the WGS84 radius, non-positive orbital
    energy, mass <= 1 kg) guard the metrics but are unreachable at the
    reference 10 mN thrust.

Difficulty
    None. The initial orbit is fixed at e0=0.15 (ra_ref/rp_ref/vel_ref
    derived accordingly). 2026-10-24: raised from the reference e0=0.1 --
    the baseline sat at 100% success, so the initial orbit was made more
    eccentric (rp0 = 8500 km, ra0 = 11,500 km).

References
    - gym-satellite-trajectory
      (github.com/zampanteymedio/gym-satellite-trajectory),
      PerigeeRaisingEnv, reproduced parameter-for-parameter.
    - Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital
      Transfer using Dynamics-Agnostic Reinforcement Learning"
      (arXiv:2211.08272, IEEE).
    - Vallado, D. A. Fundamentals of Astrodynamics and Applications --
      Kepler-to-Cartesian conversion (Newton-Raphson eccentric-anomaly
      solve) and osculating-orbit recovery used here.
    - Method documentation: docs/method/env-orbit-transfer.md.
"""

from .env import LowThrustTransferEnv
