"""Low-thrust perigee-raising orbital transfer task (electric propulsion).

Description
    A 100 kg medium-Earth-orbit satellite with three orthogonal thruster
    pairs (10 mN per axis, Isp = 4000 s electric propulsion) fires for 166
    control steps of 300 s (~13.8 h, ~5 revolutions of the a = 10,000 km,
    e = 0.1 orbit) to raise the pericenter radius of 9000 km while keeping
    the apocenter radius of 11,000 km fixed and propellant use low. The
    initial mean anomaly is resampled uniformly on [-pi, pi] at every
    reset; the initial orbit otherwise is fixed (i = 60 deg, RAAN = 120
    deg, argument of perigee = 240 deg). The reachable delta-v budget is
    ~15 m/s, so meaningful perigee raising requires phasing thrust near
    perigee. Dynamics are two-body point-mass gravity (WGS84
    mu = 3.986004418e14 m^3/s^2) plus constant thrust over each control
    step, integrated with fixed-step RK4 at 10 s substeps (the reference
    uses Orekit adaptive Dormand-Prince 8(5,3); per-revolution phase error
    here stays far below reward granularity). Fully tensorized: all
    replicas advance on one device, no engine, no gym. The reward is
    sparse and terminal only.

Action Space (3,)
    Thrust-direction commands in [-1, 1]^3, one per body axis; each axis
    delivers up to 10 mN of continuous thrust, giving a three-axis
    acceleration of ~3e-4 m/s^2 at full burn.

Observation Space (8,)
    float32, unbounded raw state, 'policy' TensorDict group:
    [elapsed time (1), position in GCRF (3), velocity in GCRF (3),
    remaining mass (1)]. Returned unnormalized, exactly as the reference.

Reward
    Terminal only, granted once when the 166-step limit is reached:
    R = 1e-4 * (-|dra| + drp + 2e5 * dm)
    with radii in meters and mass in kg -- i.e. +0.1 per km of perigee
    radius raised, -0.1 per km of |apogee radius drift|, -20 per kg of
    propellant spent (the reference's Table I weights).

Episode End
    Truncation: 166 control steps (49,800 s of mission time).
    There are no early terminations or failures. is_success is True at
    any done, mirroring the reference semantics, so completed_successes
    counts episodes rather than mission achievements.

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
"""

from .env import LowThrustTransferEnv
