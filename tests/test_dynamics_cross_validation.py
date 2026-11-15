"""Independent cross-validation probes for the plant dynamics.

These tests target invariants that a sign/dimensional/frame error is likely
to break but the behavioural task tests can miss:

* the aeroelastic stiffness is calibrated from the full elastic-axis moment
  slope (not just the lift term) and the calibrated divergence speed behaves
  as documented below/above critical;
* a masked surface step freezes every persistent aerodynamic state of an
  inactive replica (servo delay/backlash, unsteady, dynamic stall, twist);
* the missile weave load factor uses the path acceleration phase;
* the aim quaternion is valid for a 180-degree (antipodal) aim;
* the Falcon-9 US76 atmosphere, variable-inertia derivative and RCS torque
  sign match independent references;
* the standalone QuadDynamics reference model loads the shipped cf2x asset.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from environments.common.aerodynamics import LiftingSurfaceBatch
from environments.common.dynamics import QuadDynamics, QuadParams
from environments.common.falcon9_dynamics import (
    atmosphere_eval,
    mass_props_batch,
    rcs_torque_from_cmd,
)
from environments.common.falcon9_params import P as F9
from environments.missile.env import MissileInterceptEnv, _aim_quat


def _surface_params():
    return dict(
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


# ---------------------------------------------------------------------------
# Aeroelastic calibration
# ---------------------------------------------------------------------------
def test_aeroelastic_stiffness_includes_chord_and_pitching_moment_slope():
    """k = q_div S c [(x_ea - 0.25) cl_alpha + cm_alpha].

    The old implementation used q_div * S * (x_ea - 0.25) * cl_alpha
    (one chord factor missing) and a 0.35c lift arm in the moment, two errors
    that nearly cancelled and hid each other from the behavioural test.
    """
    surf = LiftingSurfaceBatch(
        [_surface_params()], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=6,
        aeroelastic=True, divergence_speed=100.0,
    )
    span = float(surf.span[0])
    chord = float(surf.chord[0])
    slope = float(surf.slope[0])
    eps = 1e-3
    alpha0 = float(surf.alpha0[0])
    cmd = torch.zeros(1, 6)
    cm_p = float(surf.coefficients(torch.full((1, 6), alpha0 + eps), cmd)[2][0, 0])
    cm_m = float(surf.coefficients(torch.full((1, 6), alpha0 - eps), cmd)[2][0, 0])
    cm_alpha = (cm_p - cm_m) / (2 * eps)
    net = 0.10 * slope + cm_alpha
    assert net > 0.0
    expected = 0.5 * 1.225 * 100.0 ** 2 * span * chord ** 2 * net
    assert abs(float(surf._twist_k[0]) - expected) < 1e-4 * expected


def test_aeroelastic_is_stable_below_the_calibrated_divergence_speed():
    """Below 100 m/s the calibrated mode must settle, not grow.

    The previous (double-error) model diverged at ~90 m/s because its
    effective divergence speed was ~97 m/s, so this is a regression guard for
    the calibration above.
    """
    surf = LiftingSurfaceBatch(
        [_surface_params()], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=6,
        aeroelastic=True, divergence_speed=100.0,
    )
    cmd = torch.zeros(1, 1)
    a = math.radians(5.0)
    velocity = torch.tensor([[[90.0 * math.cos(a), 0.0, -90.0 * math.sin(a)]]])
    surf.reset(torch.tensor([True]))
    for _ in range(4000):
        surf.step(cmd, velocity)
    assert abs(float(surf._twist[0, 0])) < 0.1
    assert abs(float(surf._twist_rate[0, 0])) < 1e-3


# ---------------------------------------------------------------------------
# Masked surface-state isolation
# ---------------------------------------------------------------------------
def test_masked_surface_step_freezes_every_inactive_state():
    surf = LiftingSurfaceBatch(
        [_surface_params(), _surface_params()],
        [[0, 0, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 0]],
        2, "cpu", 1 / 240,
        realistic=True, oswald=[1.0, 1.0],
        rate_limit_deg_s=[300.0, 300.0], deadband=0.01, strips=2,
        aeroelastic=True, divergence_speed=100.0,
    )
    surf.set_actuator_delay(torch.tensor([1.0, 1.0]))
    cmd = torch.tensor([[0.4, -0.3], [0.2, -0.1]])
    velocity = torch.tensor(
        [[[20.0, 0.0, -6.0], [20.0, 0.0, -6.0]],
         [[20.0, 0.0, -6.0], [20.0, 0.0, -6.0]]]
    )
    mask_all = torch.tensor([True, True])
    for _ in range(5):
        surf.step(cmd, velocity, mask=mask_all)
    names = (
        "actuation", "_backlash_state", "_delay_ring", "_delay_head",
        "_aero_state", "_aero_ready", "_fsep", "_vortex", "_twist", "_twist_rate",
    )
    before = {name: getattr(surf, name).clone() for name in names}
    surf.step(cmd, velocity, mask=torch.tensor([True, False]))
    for name in names:
        now = getattr(surf, name)
        assert torch.equal(now[1], before[name][1]), name


# ---------------------------------------------------------------------------
# Missile target / launch geometry
# ---------------------------------------------------------------------------
def test_missile_weave_load_factor_uses_acceleration_phase():
    env = MissileInterceptEnv.__new__(MissileInterceptEnv)
    env.t_type = torch.tensor([1])
    env.t_a = torch.tensor([7.5 * 9.81])
    env.t_omega = torch.tensor([2 * math.pi * 0.75])
    env.t_radius = torch.tensor([1.0])
    env.sim_time = torch.tensor([0.0])
    # At phase pi/2 the lateral velocity is zero and the lateral acceleration
    # is maximal; the lift demand is sqrt(1 + (a/g)^2), not 1.
    env.t_phase = torch.tensor([math.pi / 2])
    load = float(env._target_load_factor(torch.tensor([250.0])))
    assert abs(load - math.sqrt(1 + 7.5 ** 2)) < 1e-4
    # At phase 0 the acceleration (hence the manoeuvre load) is zero.
    env.t_phase = torch.tensor([0.0])
    assert abs(float(env._target_load_factor(torch.tensor([250.0]))) - 1.0) < 1e-6


def test_missile_aim_quaternion_handles_antipodal_aim():
    from environments.common.falcon9_dynamics import rotate_body_to_world

    for aim in (
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    ):
        q = _aim_quat(aim, torch.zeros(1))
        assert abs(float(q.norm()) - 1.0) < 1e-6
        rotated = rotate_body_to_world(q, torch.tensor([[1.0, 0.0, 0.0]]))
        assert torch.allclose(rotated, aim, atol=1e-6)


# ---------------------------------------------------------------------------
# Falcon 9 public-data model
# ---------------------------------------------------------------------------
def test_falcon9_us76_atmosphere_matches_standard_table():
    # US Standard Atmosphere 1976 density checkpoints [kg/m^3].
    for altitude, rho_ref in (
        (0.0, 1.2250),
        (10_000.0, 0.4135),
        (20_000.0, 0.08803),
        (30_000.0, 0.01841),
        (40_000.0, 0.003996),
        (50_000.0, 0.001027),
    ):
        rho = float(atmosphere_eval(torch.tensor([altitude]))[0])
        assert abs(rho - rho_ref) / rho_ref < 0.015, (altitude, rho, rho_ref)


def test_falcon9_variable_inertia_derivative_matches_finite_difference():
    m_lox = torch.tensor([F9.propellant.LOX_MAX], dtype=torch.float64)
    m_rp1 = torch.tensor([F9.propellant.RP1_MAX], dtype=torch.float64)
    _, _, inertia, _, (d_lox, d_rp1), _ = mass_props_batch(m_lox, m_rp1)
    for which, deriv in ((0, d_lox), (1, d_rp1)):
        dm = 1.0
        a, b = m_lox.clone(), m_rp1.clone()
        if which == 0:
            a -= dm
        else:
            b -= dm
        _, _, inertia2, _, _, _ = mass_props_batch(a, b)
        fd = float((inertia2 - inertia) / (-dm))
        assert abs(fd - float(deriv)) < 1e-3 * abs(fd), (which, fd, float(deriv))


def test_falcon9_rcs_torque_preserves_command_sign():
    cmd = torch.tensor([[-900.0, 900.0, -100.0]])
    tau, mdot = rcs_torque_from_cmd(cmd, torch.zeros(1, dtype=torch.bool), F9)
    assert float(tau[0, 0]) < 0
    assert float(tau[0, 1]) > 0
    assert float(tau[0, 2]) < 0
    assert torch.all(tau.abs() <= F9.rcs.RCS_MAX_TORQUE + 1e-6)
    assert float(mdot[0]) > 0
    # Engine lit: gimbal owns pitch/yaw, RCS is roll-only.
    tau_on, _ = rcs_torque_from_cmd(cmd, torch.ones(1, dtype=torch.bool), F9)
    assert float(tau_on[0, 0]) < 0
    assert float(tau_on[0, 1]) == 0.0
    assert float(tau_on[0, 2]) == 0.0


def test_falcon9_v1_baseline_uses_the_real_scale_fuel_mass():
    """The v1 fuel link is in kg; v2's 410.9 kg constant is 1000x too small."""
    from algorithms.baselines.baselines import RocketPDGBaseline
    from algorithms.baselines.variants_fw_rocket import RocketEGuidanceBaseline

    class _FakeEnv:
        device = "cpu"
        num_envs = 1
        num_actions = 7
        controller_design = {}
        f9_real = True

    for cls in (RocketPDGBaseline, RocketEGuidanceBaseline):
        baseline = cls(_FakeEnv())
        assert baseline.M_DRY == 25600.0
        assert baseline.FUEL_FULL == 410900.0
        assert baseline.T_MAX == 845000.0


def test_rocket_design_mass_is_the_spawn_load_not_full_capacity():
    from environments.common.plant import TensorPlant

    plant = TensorPlant("rocket", 1, "cpu", rocket_benchmark="v2", warmup_iterations=0)
    try:
        expected = 138.2 + 0.05 * 410.9
        assert abs(plant.nominal_param("mass") - expected) < 0.1
    finally:
        plant.close()


# ---------------------------------------------------------------------------
# Standalone reference model
# ---------------------------------------------------------------------------
def test_quad_reference_model_loads_shipped_asset():
    params = QuadParams.from_reference()
    assert abs(params.mass - 0.027) < 1e-6
    dynamics = QuadDynamics(1, "cpu", dt=1 / 240, params=params)
    arm = dynamics.arm
    assert torch.allclose(
        arm,
        torch.tensor(
            [
                [0.028, -0.028, 0.0],
                [-0.028, 0.028, 0.0],
                [0.028, 0.028, 0.0],
                [-0.028, -0.028, 0.0],
            ]
        ),
        atol=1e-6,
    )
