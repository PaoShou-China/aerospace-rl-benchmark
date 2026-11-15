"""Unit tests for the opt-in realistic aerodynamics path.

These are pure-torch checks: no Genesis scene is constructed, so they run on
CPU in the normal test suite.  The legacy benchmark path is covered by the
bit-identity comparison documented in the commit; here we pin the physics
corrections themselves (ISA, compressibility, Oswald, second-order Dryden).
"""

import torch

from environments.common.aerodynamics import LiftingSurfaceBatch
from environments.common.atmosphere import (
    RHO0,
    drag_factor,
    isa,
    lift_slope_factor,
)
from environments.common.disturbances import WindDisturbance
from environments.common.sensors import SensorModel


def _surface_params(**overrides):
    p = dict(
        Cl_alpha_2D=6.283,
        chord=0.3,
        span=1.6,
        flap_to_chord=0.3,
        eta=0.65,
        alpha_0_base=-2.0,
        alpha_stall_P_base=14.0,
        alpha_stall_N_base=-9.0,
        Cd_0=0.01,
        deflection_limit=30.0,
        tau=0.05,
    )
    p.update(overrides)
    return p


def test_isa_reference_values():
    rho0, p0, t0, a0 = (float(x) for x in isa(torch.tensor(0.0)))
    assert abs(rho0 - RHO0) < 1e-3
    assert abs(p0 - 101325.0) < 2.0
    assert abs(t0 - 288.15) < 1e-3
    assert abs(a0 - 340.294) < 0.05
    # Standard-atmosphere checkpoints.
    assert abs(float(isa(torch.tensor(4000.0))[0]) - 0.8191) < 2e-3
    assert abs(float(isa(torch.tensor(11000.0))[0]) - 0.3639) < 2e-3


def test_compressibility_factors():
    assert abs(float(lift_slope_factor(torch.tensor(0.0))) - 1.0) < 1e-6
    assert float(lift_slope_factor(torch.tensor(0.8))) > 1.5
    assert float(lift_slope_factor(torch.tensor(1.8))) < 0.6
    # A low-aspect-ratio fin whose incompressible slope is already ~3/rad
    # keeps that slope at M=1.8 (linear supersonic theory gives ~2.98/rad).
    assert 0.85 < float(lift_slope_factor(torch.tensor(1.8), 3.0)) < 1.15
    assert abs(float(drag_factor(torch.tensor(0.2))) - 1.0) < 1e-6
    assert float(drag_factor(torch.tensor(1.0))) > 1.3


def test_surface_dynamic_pressure_scales_with_density():
    params = _surface_params()
    surf = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    cmd = torch.zeros(1, 1)
    vel = torch.tensor([[[20.0, 0.0, 2.0]]])
    f0, _ = surf.step(cmd, vel)
    density = torch.tensor([float(isa(torch.tensor(4000.0))[0])])
    f1, _ = surf.step(cmd, vel, density=density)
    ratio = density[0] / RHO0
    assert torch.allclose(f1, f0 * ratio, rtol=1e-5, atol=1e-6)


def test_oswald_increases_induced_drag_only_when_underway():
    params = _surface_params()
    legacy = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    realistic = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[0.8],
    )
    alpha = torch.full((1, 1), 0.10)
    cmd = torch.zeros(1, 1)
    cl_l, cd_l, _ = legacy.coefficients(alpha, cmd)
    cl_r, cd_r, _ = realistic.coefficients(alpha, cmd)
    assert torch.allclose(cl_r, cl_l)
    assert float(cd_r) > float(cd_l)


def test_compressibility_increases_lift_slope():
    params = _surface_params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0],
    )
    alpha = torch.full((1, 1), 0.10)
    cmd = torch.zeros(1, 1)
    cl_slow, _, _ = surf.coefficients(alpha, cmd, torch.full((1, 1), 50.0 / 340.294))
    cl_fast, _, _ = surf.coefficients(alpha, cmd, torch.full((1, 1), 250.0 / 340.294))
    assert float(cl_fast) > 1.4 * float(cl_slow)


def test_dryden_second_order_stationary_std():
    n = 16
    wind = WindDisturbance(
        n, device="cpu", seed=5, dryden=1.0, severity="moderate",
        dt=1 / 240, model="dryden",
    )
    vals = []
    for _ in range(30000):
        vals.append(wind.step())
    v = torch.stack(vals)[5000:]
    std = v.std(dim=0).mean(dim=0)
    sigma = wind._sigma
    assert torch.allclose(std, sigma, rtol=0.2, atol=0.1), (std, sigma)


def test_wind_shear_power_law():
    wind = WindDisturbance(
        3, device="cpu", seed=7, dryden=0.0, steady_wind=(0.0, 5.0, 0.0),
        shear_exponent=0.14, shear_reference_height=10.0,
    )
    alt = torch.tensor([10.0, 40.0, 80.0])
    out = wind.step(altitude=alt)
    expected = 5.0 * (alt / 10.0) ** 0.14
    assert torch.allclose(out[:, 1], expected, rtol=1e-5)
    assert torch.allclose(out[:, 0], torch.zeros(3), atol=1e-6)


def test_actuator_rate_limit_binds():
    params = _surface_params()
    legacy = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    realistic = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], rate_limit_deg_s=[300.0], deadband=0.0,
    )
    cmd = torch.ones(1, 1)
    vel = torch.zeros(1, 1, 3)
    legacy.step(cmd, vel)
    realistic.step(cmd, vel)
    # 300 deg/s over 1/240 s = 1.25 deg of a 30 deg full scale.
    expected = (300.0 / 30.0) / 240.0
    assert abs(float(realistic.actuation) - expected) < 1e-6
    assert float(legacy.actuation) > float(realistic.actuation)


def test_actuator_and_aero_scales():
    params = _surface_params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 2, "cpu", 1 / 240,
        realistic=True, oswald=[1.0],
    )
    cmd = torch.zeros(2, 1)
    alpha = torch.full((2, 1), 0.1)
    base_cl, base_cd, _ = surf.coefficients(alpha, cmd)
    surf.set_aero_scale(torch.tensor([1.1, 0.9]))
    scaled_cl, scaled_cd, _ = surf.coefficients(alpha, cmd)
    assert torch.allclose(scaled_cl[0], base_cl[0] * 1.1, rtol=1e-5)
    assert torch.allclose(scaled_cl[1], base_cl[1] * 0.9, rtol=1e-5)
    assert scaled_cd[0] > base_cd[0]
    surf.set_actuator_scale(torch.tensor([2.0, 0.5]))
    assert torch.allclose(surf.tau_scale[:, 0], torch.tensor([2.0, 0.5]))


def test_sensor_noise_latency_and_reset():
    sensor = SensorModel(
        4, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=1,
    )
    obs = torch.zeros(4, 13)
    obs[:, 6] = 1.0  # unit quaternion w
    first = sensor.apply(obs, torch.ones(4, dtype=torch.bool))
    second = sensor.apply(obs, torch.ones(4, dtype=torch.bool))
    assert torch.allclose(second[:, :3], first[:, :3])
    assert not torch.allclose(first[:, :3], obs[:, :3])
    sensor.reset(torch.tensor([True, False, False, False]))
    third = sensor.apply(obs, torch.ones(4, dtype=torch.bool))
    assert torch.isfinite(third).all()


def test_sensor_zero_noise_identity():
    sensor = SensorModel(
        3, "cpu", 5, gyro_sigma=0.0, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=0,
    )
    obs = torch.randn(3, 13)
    obs[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    out = sensor.apply(obs, torch.ones(3, dtype=torch.bool))
    assert torch.allclose(out, obs)


def test_dryden_masked_step_keeps_shapes_and_finiteness():
    wind = WindDisturbance(4, device="cpu", seed=9, dryden=1.0, dt=1 / 240, model="dryden")
    full = wind.step()
    mask = torch.tensor([True, False, True, False])
    sub = wind.step(mask)
    assert full.shape == (4, 3)
    assert sub.shape == (2, 3)
    assert torch.isfinite(full).all() and torch.isfinite(sub).all()
    wind.reset(mask)
    assert torch.equal(wind._state2[:, mask], torch.zeros(3, 2, 2))
