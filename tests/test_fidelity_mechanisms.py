"""Quantitative verification of the realistic-fidelity mechanisms.

Reference values are taken from the standard atmosphere (ISO 2533 / US
Standard Atmosphere 1976), MIL-F-8785C Dryden turbulence, Prandtl-Glauert /
linear supersonic thin-airfoil theory, momentum-theory propeller scaling and
the Cheeseman-Bennett ground-effect factor.  Each test either checks a closed
form directly or measures the implementation against it.
"""

import math

import pytest
import torch

from environments.common.aerodynamics import LiftingSurfaceBatch
from environments.common.atmosphere import isa, lift_slope_factor
from environments.common.disturbances import WindDisturbance
from environments.common.plant import ground_effect_factor, rotor_gyro_torque
from environments.common.sensors import SensorModel
from environments.fixedwing import FixedwingWaypointsEnv
from environments.missile import MissileInterceptEnv
from environments.missile import env as missile_env
from environments.rocket import RocketLandingEnv

RHO0 = 1.225


def _params():
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


def test_isa_against_standard_atmosphere_table():
    # ISO 2533 / US Standard Atmosphere 1976 checkpoints.
    table = {
        0.0: (1.2250, 288.150, 340.294),
        1000.0: (1.1116, 281.650, 336.434),
        4000.0: (0.8191, 262.150, 324.579),
        11000.0: (0.3639, 216.650, 295.069),
        20000.0: (0.0880, 216.650, 295.069),
    }
    for h, (rho_ref, t_ref, a_ref) in table.items():
        rho, _, temp, sound = (float(x) for x in isa(torch.tensor(h)))
        assert rho == pytest.approx(rho_ref, rel=2e-3), h
        assert temp == pytest.approx(t_ref, abs=0.2), h
        assert sound == pytest.approx(a_ref, rel=1e-3), h


def test_prandtl_glauert_and_supersonic_lift_slope():
    for m in (0.3, 0.6, 0.8):
        expected = 1.0 / math.sqrt(1.0 - m * m)
        assert float(lift_slope_factor(torch.tensor(m))) == pytest.approx(expected, rel=1e-5)
    # Linear supersonic thin-airfoil theory: 4/sqrt(M^2-1) per rad.
    m = 1.8
    expected_2pi = (4.0 / math.sqrt(m * m - 1.0)) / (2.0 * math.pi)
    assert float(lift_slope_factor(torch.tensor(m))) == pytest.approx(expected_2pi, rel=1e-5)
    expected_low_ar = (4.0 / math.sqrt(m * m - 1.0)) / 3.0
    assert float(lift_slope_factor(torch.tensor(m), 3.0)) == pytest.approx(
        expected_low_ar, rel=1e-5
    )


def test_oswald_increment_matches_standard_induced_drag():
    params = _params()
    legacy = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    realistic = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[0.8],
    )
    alpha = torch.full((1, 1), 0.10)
    cmd = torch.zeros(1, 1)
    cl, cd_legacy, _ = legacy.coefficients(alpha, cmd)
    cl_r, cd_realistic, _ = realistic.coefficients(alpha, cmd)
    assert torch.allclose(cl, cl_r)
    aspect = params["span"] / params["chord"]
    expected = (1.0 / 0.8 - 1.0) * cl.square() / (math.pi * aspect)
    assert torch.allclose(cd_realistic - cd_legacy, expected, rtol=1e-5)


def test_actuator_rate_limit_and_deadband():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], rate_limit_deg_s=[300.0], deadband=0.01,
    )
    vel = torch.zeros(1, 1, 3)
    surf.step(torch.ones(1, 1), vel)
    assert float(surf.actuation) == pytest.approx((300.0 / 30.0) / 240.0, abs=1e-6)
    # A command inside the 1% deadband must not move the surface.
    surf.reset(torch.tensor([True]))
    surf.step(torch.full((1, 1), 0.005), vel)
    assert float(surf.actuation) == pytest.approx(0.0, abs=1e-9)


def test_unsteady_lift_lag_uses_convective_time():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0],
    )
    speed = 20.0
    cmd = torch.zeros(1, 1)
    alpha1, alpha2 = 0.10, 0.20

    def velocity(alpha):
        return torch.tensor(
            [[[speed * math.cos(alpha), 0.0, -speed * math.sin(alpha)]]]
        )

    cl1, _, _ = surf.coefficients(torch.full((1, 1), alpha1), cmd)
    surf.step(cmd, velocity(alpha1))  # initialises the state at cl1
    cl2, _, _ = surf.coefficients(torch.full((1, 1), alpha2), cmd)
    surf.step(cmd, velocity(alpha2))
    tau = params["chord"] / speed
    expected = cl1 + (1.0 / 240.0) / tau * (cl2 - cl1)
    assert float(tau) == pytest.approx(0.015)
    assert torch.allclose(surf._aero_state[..., 0], expected, rtol=1e-5)


def test_surface_reset_clears_unsteady_and_actuation():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 2, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], rate_limit_deg_s=[300.0],
    )
    surf.step(torch.ones(2, 1), torch.tensor([[[20.0, 0.0, 1.0]], [[20.0, 0.0, 1.0]]]))
    mask = torch.tensor([True, False])
    surf.reset(mask)
    assert torch.equal(surf.actuation[0], torch.zeros(1))
    assert surf.actuation[1].abs().sum() > 0
    assert not bool(surf._aero_ready[0])
    assert bool(surf._aero_ready[1])


def test_ground_effect_matches_cheeseman_bennett():
    radius = 0.15
    # At z=R and z=2R the image-system factor is 16/15 and 64/63.
    assert float(ground_effect_factor(torch.tensor(radius), torch.tensor(radius))) == pytest.approx(
        16.0 / 15.0, rel=1e-5
    )
    assert float(
        ground_effect_factor(torch.tensor(radius), torch.tensor(2.0 * radius))
    ) == pytest.approx(64.0 / 63.0, rel=1e-5)


def test_rotor_gyro_torque_sign():
    h = torch.tensor([0.01])
    wb = torch.tensor([[1.0, 0.0, 0.0]])
    torque = rotor_gyro_torque(h, wb)
    # H along +z, body rate along +x -> torque along +y.
    assert torch.allclose(torque, torch.tensor([[0.0, 0.01, 0.0]]), atol=1e-9)


def test_sensor_noise_bias_and_latency_statistics():
    n = 256
    sensor = SensorModel(
        n, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=1e-4, latency=1,
    )
    obs = torch.zeros(n, 13)
    obs[:, 6] = 1.0
    vals = []
    for _ in range(2000):
        out = sensor.apply(obs, torch.ones(n, dtype=torch.bool))
        vals.append(out[:, 0].clone())
    vals = torch.stack(vals)
    # Gyro noise sigma 0.01 plus a bias random walk of 1e-4 per step.
    expected = math.sqrt(0.01**2 + (1e-4 * math.sqrt(2000)) ** 2)
    assert float(vals.std()) == pytest.approx(expected, rel=0.2)
    # Latency: a fresh sensor fed the same frame twice must repeat the first
    # noisy sample on the second call.
    fresh = SensorModel(
        n, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=1,
    )
    first = fresh.apply(obs, torch.ones(n, dtype=torch.bool))
    second = fresh.apply(obs, torch.ones(n, dtype=torch.bool))
    assert torch.allclose(first[:, :3], second[:, :3])


def test_dryden_psd_matches_mil_f_8785c_shape():
    n = 6
    dt = 1 / 240
    wind = WindDisturbance(n, device="cpu", seed=5, dryden=1.0, dt=dt, model="dryden")
    vals = []
    for _ in range(30000):
        vals.append(wind.step())
    x = torch.stack(vals)[2000:].permute(1, 0, 2)
    psd = (torch.fft.rfft(x, dim=1).abs() ** 2).mean(0)
    freqs = torch.fft.rfftfreq(x.shape[1], d=dt)
    lengths = (200.0, 200.0, 50.0)
    speed = wind.V
    for axis in range(3):
        tau = lengths[axis] / speed
        omega = 2.0 * math.pi * freqs
        if axis == 0:
            reference = 1.0 / (1.0 + (tau * omega) ** 2)
        else:
            reference = (1.0 + 3.0 * (tau * omega) ** 2) / (1.0 + (tau * omega) ** 2) ** 2
        band = (freqs > 0.01) & (freqs < 5.0)
        corr = torch.corrcoef(
            torch.stack((torch.log(psd[band, axis] + 1e-30), torch.log(reference[band] + 1e-30)))
        )[0, 1]
        assert float(corr) > 0.95, (axis, float(corr))


def test_wind_shear_power_law():
    wind = WindDisturbance(
        3, device="cpu", seed=7, dryden=0.0, steady_wind=(0.0, 5.0, 0.0),
        shear_exponent=0.14, shear_reference_height=10.0,
    )
    alt = torch.tensor([10.0, 40.0, 80.0])
    out = wind.step(altitude=alt)
    assert torch.allclose(out[:, 1], 5.0 * (alt / 10.0) ** 0.14, rtol=1e-5)


def test_missile_propellant_mass_depletion():
    env = MissileInterceptEnv(
        num_envs=1, backend="cpu", seed=7, reward_variant="terminal_gauss", aero="realistic"
    )
    try:
        m0 = env.plant.drone.get_links_mass(
            links_idx_local=[env.plant._body_link_id]
        ).clone()
        i0 = env.plant.drone.get_links_inertia(
            links_idx_local=[env.plant._body_link_id]
        ).clone()
        act = torch.zeros(1, 3)
        for _ in range(150):  # 2.5 s at 60 Hz
            env.step(act)
        fuel = float(env.plant.missile_fuel[0])
        m1 = env.plant.drone.get_links_mass(
            links_idx_local=[env.plant._body_link_id]
        ).clone()
        assert fuel == pytest.approx(1.0 - 2.5 / 4.0, abs=0.03)
        ratio = float(m1[0, 0] / m0[0, 0])
        assert ratio == pytest.approx(0.7 + 0.3 * fuel, abs=0.01)
        i1 = env.plant.drone.get_links_inertia(
            links_idx_local=[env.plant._body_link_id]
        )
        inertia_ratio = (i1[0, 0].diagonal() / i0[0, 0].diagonal()).mean()
        assert float(inertia_ratio) == pytest.approx(ratio, rel=0.02)
    finally:
        env.close()


def test_missile_target_load_limit():
    original = missile_env.A_LAT
    missile_env.A_LAT = 30.0
    env = MissileInterceptEnv(num_envs=8, backend="cpu", seed=7, aero="realistic")
    try:
        limit = 15.0 * 9.81
        assert float(env.t_a.max()) <= limit * 1.001
        assert float(env.t_radius.min()) >= (250.0**2 / limit) * 0.999
    finally:
        env.close()
        missile_env.A_LAT = original


def test_sensor_noise_scales_with_control_rate():
    n = 256
    slow = SensorModel(
        n, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=0, dt=1.0 / 30.0,
    )
    fast = SensorModel(
        n, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=0, dt=1.0 / 60.0,
    )
    obs = torch.zeros(n, 13)
    obs[:, 6] = 1.0
    a, b = [], []
    for _ in range(500):
        a.append(slow.apply(obs, torch.ones(n, dtype=torch.bool))[:, 0])
        b.append(fast.apply(obs, torch.ones(n, dtype=torch.bool))[:, 0])
    ratio = torch.stack(a).std() / torch.stack(b).std()
    # Rate noise is a density: the 30 Hz sample carries less per-sample
    # variance than the 60 Hz sample (1/sqrt(2)), while the integrated angle
    # random walk over one second is unchanged.
    assert float(ratio) == pytest.approx(1.0 / math.sqrt(2.0), rel=0.1)


def test_sensor_api_guards():
    with pytest.raises(ValueError):
        SensorModel(1, "cpu", 3, latency=2)
    sensor = SensorModel(1, "cpu", 3, latency=0)
    with pytest.raises(ValueError):
        sensor.apply(torch.zeros(1, 13), torch.ones(1, dtype=torch.bool), los_slice=(0, 2))


def test_realism_version_in_cfg():
    env = FixedwingWaypointsEnv(num_envs=1, backend="cpu", seed=7, aero="realistic")
    try:
        assert env.cfg["realism_version"] == "1.8"
    finally:
        env.close()
    env = FixedwingWaypointsEnv(num_envs=1, backend="cpu", seed=7, aero="legacy")
    try:
        assert env.cfg["realism_version"] == "legacy"
    finally:
        env.close()


def test_terminal_observation_uses_sensor_pipeline():
    env = FixedwingWaypointsEnv(num_envs=2, backend="cpu", seed=7, aero="realistic")
    try:
        obs = env.get_observations()
        env.step(torch.zeros(2, 4))
        terminal = env.extras["terminal_observations"]["policy"]
        assert torch.allclose(terminal, env.obs_buf)
        assert not torch.allclose(terminal, env._observation)
    finally:
        env.close()


def test_actuator_scale_floor_keeps_explicit_euler_stable():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], rate_limit_deg_s=[300.0],
    )
    surf.set_actuator_scale(0.0)
    assert float(surf.tau_scale) == pytest.approx(1e-3)
    surf.step(torch.ones(1, 1), torch.zeros(1, 1, 3))
    assert torch.isfinite(surf.actuation).all()


def test_rocket_contact_solver_rests_on_pad():
    env = RocketLandingEnv(num_envs=1, backend="cpu", seed=7, aero="realistic")
    try:
        p = torch.zeros(1, 3)
        p[0, 2] = 2.60  # base centre; the 4.77 m body bottom sits just above the pad
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        zero = torch.zeros(1, 3)
        env.plant.set_state(p, q, zero, zero)
        act = torch.zeros(1, 7)
        for _ in range(80):
            env.step(act)
        p_end, _, v_end, _ = env.plant.state()
        assert bool(env.plant.pad_contact[0])
        assert not bool(env.plant.ground_contact[0])
        assert abs(float(v_end[0, 2])) < 0.01
        assert 2.55 < float(p_end[0, 2]) < 2.61
        assert int(env.touch_dwell[0]) >= env._touch_dwell_steps
        assert bool(env.complete[0])
    finally:
        env.close()


def test_legacy_mode_disables_new_mechanisms():
    env = FixedwingWaypointsEnv(num_envs=2, backend="cpu", seed=7, aero="legacy")
    try:
        assert env._sensor is None
        assert env.plant.contact_solver is False
        assert env.plant.surfaces.realistic is False
        assert env.plant.surfaces.rate_limit is None
    finally:
        env.close()


def test_contact_solver_enabled_for_realistic_flight_tasks():
    from environments.vtol import QuadplaneWaypointsEnv

    for cls in (FixedwingWaypointsEnv, QuadplaneWaypointsEnv):
        env = cls(num_envs=1, backend="cpu", seed=7, aero="realistic")
        try:
            assert env.plant.contact_solver is True
        finally:
            env.close()


def test_seeker_fov_loss_zeroes_the_target_block():
    original = missile_env.SEEKER_HALF_ANGLE
    missile_env.SEEKER_HALF_ANGLE = math.radians(0.001)
    env = MissileInterceptEnv(num_envs=2, backend="cpu", seed=7, aero="realistic")
    try:
        assert not bool(env.seeker_lock.any())
        assert float(env.obs_buf[:, 23:34].abs().sum()) == 0.0
        env.step(torch.zeros(2, 3))
        assert not bool(env.seeker_lock.any())
        assert float(env.obs_buf[:, 23:34].abs().sum()) == 0.0
        # A wide field of view re-acquires the target and repopulates the block.
        missile_env.SEEKER_HALF_ANGLE = original
        env.reset()
        assert bool(env.seeker_lock.all())
        assert float(env.obs_buf[:, 23:26].abs().sum()) > 0.0
    finally:
        env.close()
        missile_env.SEEKER_HALF_ANGLE = original


def test_sensor_cg_and_inertia_domain_randomization():
    env = FixedwingWaypointsEnv(
        num_envs=2, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        com0 = plant.drone.get_links_COM().clone()
        plant.perturb_cg(1.1, axis=0)
        plant.perturb_cg(1.05, axis=1)
        com1 = plant.drone.get_links_COM()
        # CG randomisation is a rigid body offset (link COMs are mostly zero),
        # so the base-link COM moves by the full 3-D shift.
        delta = (com1 - com0)[:, plant._body_link_id]
        cg_scale = torch.tensor([[1.1, 1.05, 1.0]]).expand(2, 3)
        shift = torch.zeros(2, 3)
        for axis in range(3):
            shift = shift + (
                (cg_scale[:, axis] - 1.0)[:, None]
                * plant._cg_shift_ref[:, None]
                * plant._cg_axes_local[axis][:, plant._body_link_id]
            )
        assert torch.allclose(delta, shift, atol=1e-6)
        assert float(delta.norm()) > 1e-3
        inertia0 = plant.drone.get_links_inertia().clone()
        plant.perturb_inertia(0.8)
        inertia1 = plant.drone.get_links_inertia()
        d0 = inertia0.diagonal(dim1=2, dim2=3)
        d1 = inertia1.diagonal(dim1=2, dim2=3)
        mask = d0 > 1e-9
        assert torch.allclose(d1[mask], d0[mask] * 0.8, rtol=1e-5)
        # Per-axis inertia scaling only touches one diagonal component.
        plant.perturb_inertia(1.2, axis=1)
        d2 = plant.drone.get_links_inertia().diagonal(dim1=2, dim2=3)
        assert torch.allclose(d2[..., 1][mask[..., 1]], d1[..., 1][mask[..., 1]] * 1.2)
        assert torch.allclose(d2[..., 0][mask[..., 0]], d1[..., 0][mask[..., 0]])
        # Mass DR scales every structural link and the inertia consistently.
        m0 = plant.drone.get_links_mass().sum(dim=1)
        plant.perturb_mass(1.1)
        m1 = plant.drone.get_links_mass().sum(dim=1)
        assert torch.allclose(m1 / m0, torch.full((2,), 1.1), rtol=1e-5)
        env._apply_perturbation("sensor", 2.0)
        assert torch.allclose(env._sensor.sensor_scale, torch.full((2, 1), 2.0))
        assert plant.nominal_param("cg") == 1.0
        assert plant.nominal_param("inertia") == 1.0
        assert plant.nominal_param("sensor") == 1.0
    finally:
        env.close()


def test_aero_and_actuator_component_randomization():
    env = FixedwingWaypointsEnv(
        num_envs=2, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        surf = env.plant.surfaces
        env._apply_perturbation("cl_alpha", 1.2)
        env._apply_perturbation("cd0", 0.9)
        env._apply_perturbation("stall", 0.95)
        env._apply_perturbation("oswald", 1.05)
        env._apply_perturbation("actuator_deadband", 2.0)
        env._apply_perturbation("actuator_backlash", 1.5)
        env._apply_perturbation("actuator_asymmetry", 0.8)
        env._apply_perturbation("actuator_delay", 2.0)
        assert torch.allclose(surf.slope_scale, torch.full((2, 1), 1.2))
        assert torch.allclose(surf.cd0_scale, torch.full((2, 1), 0.9))
        assert torch.allclose(surf.stall_scale, torch.full((2, 1), 0.95))
        assert torch.allclose(surf.oswald_scale, torch.full((2, 1), 1.05))
        assert torch.allclose(surf.deadband_scale, torch.full((2, 1), 2.0))
        assert torch.allclose(surf.backlash_scale, torch.full((2, 1), 1.5))
        assert torch.allclose(surf.rate_asym_scale, torch.full((2, 1), 0.8))
        assert torch.equal(surf.delay_steps, torch.full((2,), 2, dtype=torch.long))
    finally:
        env.close()


def test_actuator_delay_backlash_and_asymmetry_dynamics():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], rate_limit_deg_s=[300.0],
    )
    vel = torch.zeros(1, 1, 3)
    surf.set_actuator_delay(2.0)
    surf.step(torch.zeros(1, 1), vel)
    surf.step(torch.zeros(1, 1), vel)
    assert float(surf.actuation) == 0.0
    surf.step(torch.ones(1, 1), vel)  # delayed command still zero
    assert float(surf.actuation) == 0.0
    surf.step(torch.ones(1, 1), vel)  # still the two-step-old zero
    assert float(surf.actuation) == 0.0
    surf.step(torch.ones(1, 1), vel)  # command now reaches the servo
    assert float(surf.actuation) > 0.0
    # Backlash: a command reversal smaller than the lash band does not move.
    surf.reset(torch.tensor([True]))
    surf.set_actuator_delay(0.0)
    surf.set_actuator_backlash(4.0)  # 2% of full command
    for _ in range(200):
        surf.step(torch.ones(1, 1), vel)
    held = float(surf.actuation)
    for _ in range(20):
        surf.step(torch.full((1, 1), 0.99), vel)
    assert float(surf.actuation) == pytest.approx(held, abs=1e-6)
    # Asymmetric rate: retraction is slower than extension.
    surf.reset(torch.tensor([True]))
    surf.set_actuator_backlash(0.0)
    surf.set_actuator_asymmetry(0.5)
    up = 0.0
    for _ in range(200):
        surf.step(torch.ones(1, 1), vel)
        up = float(surf.actuation)
    for _ in range(1):
        surf.step(torch.full((1, 1), -1.0), vel)
    fall = up - float(surf.actuation)
    surf.reset(torch.tensor([True]))
    surf.set_actuator_asymmetry(1.0)
    for _ in range(200):
        surf.step(torch.ones(1, 1), vel)
    for _ in range(1):
        surf.step(torch.full((1, 1), -1.0), vel)
    fall_sym = up - float(surf.actuation)
    assert fall < fall_sym * 0.7


def test_sensor_scale_multiplies_noise():
    n = 128
    sensor = SensorModel(
        n, "cpu", 3, gyro_sigma=0.01, attitude_sigma=0.0, velocity_sigma=0.0,
        position_sigma=0.0, bias_step=0.0, latency=0,
    )
    obs = torch.zeros(n, 13)
    obs[:, 6] = 1.0
    sensor.set_scale(0.0)
    quiet = sensor.apply(obs, torch.ones(n, dtype=torch.bool))
    assert torch.allclose(quiet[:, :3], torch.zeros(n, 3))
    sensor.set_scale(2.0)
    samples = torch.stack(
        [sensor.apply(obs, torch.ones(n, dtype=torch.bool))[:, 0] for _ in range(200)]
    )
    assert float(samples.std()) == pytest.approx(0.02, rel=0.1)


def test_slipstream_geometry_matches_momentum_theory():
    from environments.common.plant import slipstream_dynamic_pressure

    radius = 0.15
    prop = torch.tensor([[[0.0, 0.0, 0.5]]])
    thrust = torch.tensor([[10.0]])
    below = torch.tensor([[[0.0, 0.0, 0.0]]])
    q_below = float(slipstream_dynamic_pressure(prop, below, thrust, radius)[0, 0])
    x = 0.5
    expected = (
        10.0 / (math.pi * radius**2) * 0.25
        * (1.0 + x / math.sqrt(x * x + radius * radius)) ** 2
    )
    assert q_below == pytest.approx(expected, rel=1e-5)
    # A surface outside the slipstream core sees a much smaller boost ...
    offset = torch.tensor([[[0.6, 0.0, 0.0]]])
    q_offset = float(slipstream_dynamic_pressure(prop, offset, thrust, radius)[0, 0])
    assert q_offset < 0.01 * q_below
    # ... and one upstream of the rotor sees none at all.
    above = torch.tensor([[[0.0, 0.0, 1.0]]])
    assert float(slipstream_dynamic_pressure(prop, above, thrust, radius)[0, 0]) == 0.0


def test_dynamic_stall_preserves_steady_polar_and_adds_hysteresis():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0],
    )
    quasi = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    speed = 20.0
    cmd = torch.zeros(1, 1)

    def velocity(deg):
        a = math.radians(deg)
        return torch.tensor([[[speed * math.cos(a), 0.0, -speed * math.sin(a)]]])

    # Steady high-alpha flight must relax to the published post-stall polar.
    surf.reset(torch.tensor([True]))
    for _ in range(1200):
        surf.step(cmd, velocity(20.0))
    cl_quasi, _, _ = quasi.coefficients(torch.full((1, 1), math.radians(20.0)), cmd)
    assert float(surf._aero_state[0, 0, 0]) == pytest.approx(float(cl_quasi), abs=1e-3)

    # A 2 -> 22 -> 2 deg sweep: lift is higher on the way up (delayed stall)
    # and the dynamic peak exceeds the quasi-steady maximum.
    surf.reset(torch.tensor([True]))
    up, down = [], []
    for i in range(180):
        surf.step(cmd, velocity(2.0 + 20.0 * i / 179))
        up.append(float(surf._aero_state[0, 0, 0]))
    for i in range(180):
        surf.step(cmd, velocity(22.0 - 20.0 * i / 179))
        down.append(float(surf._aero_state[0, 0, 0]))
    up15 = up[int(round((15.0 - 2.0) / 20.0 * 179))]
    down15 = down[int(round((22.0 - 15.0) / 20.0 * 179))]
    assert up15 > down15 + 0.08
    assert max(up) > 1.5 * float(cl_quasi)


def test_dryden_time_varying_filters_track_flight_condition():
    n = 2
    wind = WindDisturbance(
        n, device="cpu", seed=5, dryden=1.0, dt=1 / 240, model="dryden",
        reference_altitude=40.0, time_varying=True,
    )
    altitude = torch.tensor([30.0, 300.0])
    airspeed = torch.tensor([20.0, 60.0])
    out = wind.step(altitude=altitude, airspeed=airspeed)
    assert torch.isfinite(out).all()
    assert torch.allclose(wind._airspeed, airspeed)
    assert torch.allclose(wind._altitude, altitude)
    # Every rebuilt filter must have stationary output std == its sigma.
    A, B, C = wind._A, wind._B, wind._C
    q = B.unsqueeze(-1) * B.unsqueeze(-2)
    kron = torch.einsum("nsij,nskl->nsikjl", A, A).reshape(n, 3, 4, 4)
    lhs = torch.eye(4)[None, None] - kron
    p = torch.linalg.solve(lhs, q.reshape(n, 3, 4, 1)).squeeze(-1).reshape(n, 3, 2, 2)
    std = torch.einsum("nsi,nsij,nsj->ns", C, p, C).clamp_min(0.0).sqrt()
    sigma, lengths = wind._flight_conditions()
    assert torch.allclose(std, sigma, rtol=3e-3)
    # MIL-F-8785C: u/v turbulence intensity grows as altitude falls, while
    # sigma_w stays at its reference value.
    assert float(sigma[0, 0]) > float(sigma[1, 0])
    assert float(sigma[0, 2]) == pytest.approx(float(sigma[1, 2]), rel=1e-5)
    assert float(lengths[0, 2]) < float(lengths[1, 2])


def test_spanwise_strips_match_single_point_and_integrate_wash():
    params = _params()
    single = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=1,
    )
    strips = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=6,
    )
    assert strips.strips == 6 and strips.n_surfaces == 1
    cmd = torch.zeros(1, 1)
    velocity = torch.tensor([[[20.0, 0.0, -0.6]]])

    # Uniform flow: the six stations carry 1/6 of the area each and their
    # identical contributions must sum back to the single-point result.
    f1, t1 = single.step(cmd, velocity)
    f6, t6 = strips.step(cmd, velocity)
    assert torch.allclose(f6, f1, atol=1e-6)
    assert torch.allclose(t6, t1, atol=1e-6)

    # Spanwise station offsets from the surface centre.
    offsets = strips.strip_offsets()[0, :, 1]
    assert float(offsets[0]) == pytest.approx(-5 / 6 * params["span"] / 2, rel=1e-6)
    assert float(offsets[-1]) == pytest.approx(5 / 6 * params["span"] / 2, rel=1e-6)

    # Midpoint quadrature of a spanwise slipstream profile: the strip sum is
    # exact for a linear q(s), and a full-period sine adds nothing.
    n = 6
    s = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    q0, q1 = 30.0, 12.0
    linear = (q0 + q1 * s).to(torch.float32)[None]
    f_lin, _ = strips.step(cmd, velocity, q_boost=linear)
    f_mean, _ = strips.step(cmd, velocity, q_boost=torch.full((1, n), q0 + 0.5 * q1))
    assert torch.allclose(f_lin, f_mean, atol=1e-6)
    sine = (q0 + q1 * torch.sin(2 * math.pi * s)).to(torch.float32)[None]
    f_sine, _ = strips.step(cmd, velocity, q_boost=sine)
    f_zero, _ = strips.step(cmd, velocity, q_boost=torch.full((1, n), q0))
    assert torch.allclose(f_sine, f_zero, atol=1e-5)

    # Closed-form force increase for a uniform q boost: the local lift and
    # drag both scale with q, and the strip sum equals the analytic integral
    # cl * S * q (plus the drag projection onto the surface normal).
    alpha = torch.atan2(torch.tensor([[0.6]]), torch.tensor([[20.0]]))
    cl, cd, _ = single.coefficients(alpha, cmd)
    alpha_v = float(alpha[0, 0])
    delta = float(f_mean[0, 0, 2] - f1[0, 0, 2])
    expected = (
        float(cl[0, 0]) * math.cos(alpha_v) + float(cd[0, 0]) * math.sin(alpha_v)
    ) * params["span"] * params["chord"] * (q0 + 0.5 * q1)
    assert delta == pytest.approx(expected, rel=1e-4)


def test_spanwise_strips_add_roll_damping_with_analytic_slope():
    params = _params()
    strips = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=6,
    )
    single = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=1,
    )
    cmd = torch.zeros(1, 1)
    velocity = torch.tensor([[[20.0, 0.0, 0.0]]])
    rate = torch.tensor([[0.2, 0.0, 0.0]])  # +p about body x
    _, torque = strips.step(cmd, velocity, body_rate=rate)
    _, torque_single = single.step(cmd, velocity, body_rate=rate)
    # A positive roll rate must produce a negative (damping) roll moment.
    assert float(torque[0, 0, 0]) < 0.0
    # The single-point model cannot see the spanwise velocity gradient.
    assert abs(float(torque_single[0, 0, 0])) < 1e-3
    # Small-rate analytic slope, tau_x = -q S cl_alpha p b^2 / 12V, with the
    # six-station midpoint quadrature (2.8% low against b^2/12).
    q = 0.5 * RHO0 * 20.0**2
    slope = float(strips.slope[0])
    analytic = -q * params["span"] * params["chord"] * slope * 0.2 * params["span"] ** 2 / (
        12.0 * 20.0
    )
    assert float(torque[0, 0, 0]) == pytest.approx(analytic, rel=0.05)
    assert analytic < 0.0


def test_spanwise_strips_keep_the_legacy_path_single_point():
    params = _params()
    legacy = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    assert legacy.strips == 1
    assert legacy.n_surfaces == 1
    assert float(legacy._strip_offset.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        LiftingSurfaceBatch(
            [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240, strips=6
        )


def test_ground_effect_matches_lifting_line_image_model():
    from environments.common.aerodynamics import ground_effect

    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=1,
    )
    span = torch.tensor([params["span"]])
    slope = surf.slope
    aspect = surf.aspect
    # Free air: phi -> 1 and the lift ratio -> 1.
    ratio_hi, phi_hi = ground_effect(torch.tensor([[1e3]]), span, slope, aspect)
    assert float(ratio_hi[0, 0]) == pytest.approx(1.0, rel=1e-6)
    assert float(phi_hi[0, 0]) == pytest.approx(1.0, rel=1e-6)
    # At the ground: phi -> 0 and the lift slope rises to cl_alpha/(1+...)^-1.
    ratio_lo, phi_lo = ground_effect(torch.tensor([[0.0]]), span, slope, aspect)
    assert float(phi_lo[0, 0]) == 0.0
    cl2d = float(slope[0]) / (1.0 - float(slope[0]) / (math.pi * float(aspect[0])))
    expected = 1.0 + cl2d / (math.pi * float(aspect[0]))
    assert float(ratio_lo[0, 0]) == pytest.approx(expected, rel=1e-5)
    # Monotone in height.
    heights = torch.linspace(0.01, 5.0, 40).unsqueeze(-1)
    ratios, phis = ground_effect(heights, span, slope, aspect)
    assert (ratios[1:] <= ratios[:-1] + 1e-6).all()
    assert (phis[1:] >= phis[:-1] - 1e-6).all()


def test_ground_effect_raises_lift_cuts_drag_and_pitches_down():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=1,
    )
    cmd = torch.zeros(1, 1)
    velocity = torch.tensor([[[20.0, 0.0, -2.0]]])  # alpha ~ 5.7 deg
    f_free, t_free = surf.step(cmd, velocity, ground_height=torch.tensor([[1e3]]))
    f_ground, t_ground = surf.step(cmd, velocity, ground_height=torch.tensor([[0.05]]))
    v_hat = velocity / velocity.norm(dim=-1, keepdim=True)
    drag_free = -float((f_free * v_hat).sum())
    drag_ground = -float((f_ground * v_hat).sum())
    # Wing close to the ground: more lift, less induced drag, nose-down moment.
    assert float(f_ground[0, 0, 2]) > float(f_free[0, 0, 2])
    assert drag_ground < drag_free
    assert float(t_ground[0, 0, 1]) < float(t_free[0, 0, 1])
    # The free-air case must be unchanged by passing a large height.
    f_none, t_none = surf.step(cmd, velocity)
    assert torch.allclose(f_free, f_none, atol=1e-5)
    assert torch.allclose(t_free, t_none, atol=1e-6)
    # Closed-form lift change at fixed alpha.
    cl0, _, _ = surf.coefficients(torch.atan2(torch.tensor([[2.0]]), torch.tensor([[20.0]])), cmd)
    from environments.common.aerodynamics import ground_effect

    ratio, _ = ground_effect(
        torch.tensor([[0.05]]), surf.span[None, :], surf.slope[None, :], surf.aspect[None, :]
    )
    q = 0.5 * RHO0 * (20.0**2 + 2.0**2)
    area = params["span"] * params["chord"]
    expected = float(cl0[0, 0]) * (float(ratio[0, 0]) - 1.0) * q * area
    assert float(f_ground[0, 0, 2] - f_free[0, 0, 2]) == pytest.approx(
        expected * math.cos(math.atan2(2.0, 20.0)), rel=0.05
    )


def test_fixedwing_wing_ground_effect_is_wired_at_plant_level():
    env = FixedwingWaypointsEnv(
        num_envs=1, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        mask = torch.ones(1, dtype=torch.bool)
        command = torch.zeros(1, 4)
        wing = plant.surface_ids[-1]

        def wing_lift(height):
            plant.zero(mask)
            plant.set_state(
                torch.tensor([[0.0, 0.0, height]]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[20.0, 0.0, 0.0]]),
                torch.zeros(1, 3),
            )
            plant.physics_step(command, mask)
            return float(plant.force[0, wing, 2])

        high = wing_lift(30.0)
        low = wing_lift(0.25)
        assert low > high * 1.02
    finally:
        env.close()


def test_dynamic_stall_adds_drag_and_moment_hysteresis():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0],
    )
    quasi = LiftingSurfaceBatch([params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240)
    speed = 20.0
    cmd = torch.zeros(1, 1)

    def velocity(deg):
        a = math.radians(deg)
        return torch.tensor([[[speed * math.cos(a), 0.0, -speed * math.sin(a)]]])

    def quasi_coefficients(deg):
        alpha = torch.full((1, 1), math.radians(deg))
        _, cd, cm = quasi.coefficients(alpha, cmd)
        return float(cd[0, 0]), float(cm[0, 0])

    # Steady post-stall flight relaxes to the published polar: the vortex (and
    # therefore its drag and moment increments) vanishes at df/dt = 0.
    surf.reset(torch.tensor([True]))
    for _ in range(1200):
        surf.step(cmd, velocity(20.0))
    cd_q, cm_q = quasi_coefficients(20.0)
    assert float(surf._aero_state[0, 0, 1]) == pytest.approx(cd_q, abs=2e-3)
    assert float(surf._aero_state[0, 0, 2]) == pytest.approx(cm_q, abs=2e-3)

    # 2 -> 22 -> 2 deg sweep: the vortex adds drag and nose-down moment on the
    # up-sweep, and both drop again on the way down.
    surf.reset(torch.tensor([True]))
    alphas = [2.0 + 20.0 * i / 179 for i in range(180)]
    up = []
    for a in alphas:
        surf.step(cmd, velocity(a))
        up.append((float(surf._aero_state[0, 0, 1]), float(surf._aero_state[0, 0, 2])))
    down = []
    for a in alphas[::-1]:
        surf.step(cmd, velocity(a))
        down.append((float(surf._aero_state[0, 0, 1]), float(surf._aero_state[0, 0, 2])))
    down.reverse()
    i15 = int(round((15.0 - 2.0) / 20.0 * 179))
    assert up[i15][0] > down[i15][0] + 0.01
    assert up[i15][1] < down[i15][1] - 0.01
    # Peak dynamic-drag increment over the same-alpha quasi-steady value.
    dcd = [u[0] - quasi_coefficients(a)[0] for u, a in zip(up, alphas)]
    dcm = [u[1] - quasi_coefficients(a)[1] for u, a in zip(up, alphas)]
    assert max(dcd) > 0.05
    assert min(dcm) < -0.02


def test_aeroelastic_twist_decays_and_diverges_above_critical_speed():
    params = _params()
    surf = LiftingSurfaceBatch(
        [params], [[0, 0, 1]], [[1, 0, 0]], 1, "cpu", 1 / 240,
        realistic=True, oswald=[1.0], strips=6, aeroelastic=True, divergence_speed=100.0,
    )
    assert surf.aeroelastic
    cmd = torch.zeros(1, 1)

    def velocity(speed, deg=5.0):
        a = math.radians(deg)
        return torch.tensor([[[speed * math.cos(a), 0.0, -speed * math.sin(a)]]])

    # No aerodynamics: the torsional DOF decays at the structural rate.  The
    # energy (phase-independent) must fall monotonically.
    surf.reset(torch.tensor([True]))
    surf._twist[0, 0] = 0.02
    zero = torch.zeros(1, 1, 3)

    def energy():
        theta = float(surf._twist[0, 0])
        rate = float(surf._twist_rate[0, 0])
        return 0.5 * float(surf._twist_k[0]) * theta**2 + 0.5 * float(surf._twist_i[0]) * rate**2

    energy_0 = energy()
    for _ in range(24):  # 0.1 s
        surf.step(cmd, zero)
    assert energy() < 0.5 * energy_0
    for _ in range(456):  # 2 s total
        surf.step(cmd, zero)
    assert abs(float(surf._twist[0, 0])) < 1e-3
    assert abs(float(surf._twist_rate[0, 0])) < 1e-2
    assert energy() < 1e-3 * energy_0

    # Below the divergence speed the twist settles to a finite equilibrium.
    surf.reset(torch.tensor([True]))
    for _ in range(3000):
        surf.step(cmd, velocity(70.0))
    below = float(surf._twist[0, 0])
    assert 0.02 < below < 0.2
    assert abs(float(surf._twist_rate[0, 0])) < 1e-3

    # Above it the twist grows and then oscillates around the bounded stop;
    # the integration stays finite (energy bounded).
    surf.reset(torch.tensor([True]))
    peak = 0.0
    for _ in range(4800):  # 20 s
        surf.step(cmd, velocity(120.0))
        peak = max(peak, abs(float(surf._twist[0, 0])))
    assert peak > 0.1
    assert peak <= 0.35 + 1e-6
    assert torch.isfinite(surf._twist).all()


def test_missile_radome_slope_and_seeker_tracking_lag():
    from environments.missile.env import (
        RADOME_SLOPE,
        SEEKER_FILTER_TAU,
        first_order_lag,
        radome_biased_los,
    )

    r = torch.tensor([[1000.0, 0.0, 0.0]])
    lam = torch.tensor([[0.0, 0.002, 0.0]])
    # Positive angle of attack: the apparent LOS rotates nose-up by slope*alpha.
    v = torch.tensor([[200.0, 0.0, 20.0]])
    r_b, lam_b = radome_biased_los(r, lam, v)
    aoa = math.atan2(-20.0, 200.0)
    assert math.atan2(float(r_b[0, 2]), float(r_b[0, 0])) == pytest.approx(
        RADOME_SLOPE * aoa, rel=1e-5
    )
    assert abs(float(lam_b[0, 1]) - 0.002) < 1e-6
    # Zero AoA/sideslip: the measurement is unbiased.
    r0, lam0 = radome_biased_los(r, lam, torch.tensor([[200.0, 0.0, 0.0]]))
    assert torch.allclose(r0, r, atol=1e-5)
    assert torch.allclose(lam0, lam, atol=1e-8)
    # Sideslip rotates about body z with the opposite sign.
    rb, _ = radome_biased_los(r, lam, torch.tensor([[200.0, 20.0, 0.0]]))
    beta = math.atan2(20.0, 200.0)
    assert math.atan2(float(rb[0, 1]), float(rb[0, 0])) == pytest.approx(
        -RADOME_SLOPE * beta, rel=1e-5
    )
    # Tracking filter: backward-Euler step, within 3% of the continuous
    # first-order response after 10 control steps.
    state = torch.zeros(1, 1)
    measured = torch.ones(1, 1)
    for _ in range(10):
        state = first_order_lag(state, measured, 1 / 120, SEEKER_FILTER_TAU)
    expected = 1.0 - (1.0 - (1 / 120) / (SEEKER_FILTER_TAU + 1 / 120)) ** 10
    assert float(state) == pytest.approx(expected, rel=1e-6)
    exact = 1.0 - math.exp(-(10 / 120) / SEEKER_FILTER_TAU)
    assert abs(float(state) - exact) < 0.03


def test_isa_hot_day_and_humidity_density():
    from environments.common.atmosphere import GAMMA, R_AIR, isa

    rho0, p0, t0, a0 = (float(x) for x in isa(torch.tensor(100.0)))
    rho_hot, p_hot, t_hot, a_hot = (
        float(x) for x in isa(torch.tensor(100.0), temperature_offset=15.0)
    )
    # Pressure stays at the ISA value; the hot day lowers the density and
    # raises the speed of sound.
    assert p_hot == pytest.approx(p0, rel=1e-6)
    assert t_hot == pytest.approx(t0 + 15.0, abs=0.01)
    assert rho_hot == pytest.approx(p0 / (R_AIR * t_hot), rel=1e-5)
    assert rho_hot < rho0
    assert a_hot == pytest.approx(math.sqrt(GAMMA * R_AIR * t_hot), rel=1e-6)
    # 1 % specific humidity reduces the density by 1/(1 + 0.6078 q).
    rho_hum = float(isa(torch.tensor(100.0), specific_humidity=0.01)[0])
    assert rho_hum == pytest.approx(rho0 / (1.0 + 0.6078 * 0.01), rel=1e-5)
    # The realistic plant runs the off-standard day; legacy stays at sea level.
    env = FixedwingWaypointsEnv(
        num_envs=1, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        plant.zero(torch.ones(1, dtype=torch.bool))
        plant.set_state(
            torch.tensor([[0.0, 0.0, 100.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
        )
        plant.physics_step(torch.zeros(1, 4), torch.ones(1, dtype=torch.bool))
        assert float(plant.air_density[0]) == pytest.approx(
            rho_hot / (1.0 + 0.6078 * 0.01), rel=1e-5
        )
    finally:
        env.close()


def test_target_energy_model_matches_thrust_drag_balance():
    env = MissileInterceptEnv(
        num_envs=2, backend="cpu", seed=7, aero="realistic",
        reward_variant="terminal_gauss", domain_randomize=False,
    )
    try:
        # Straight, level, trimmed flight: the speed-hold exactly cancels drag.
        env.t_type[:] = 0
        env.t_a[:] = 0.0
        v0 = torch.full((2,), 250.0)
        active = torch.tensor([True, True])
        assert torch.allclose(env._target_speed_step(v0, active), v0, atol=1e-3)
        # A sustained 7.5 g barrel exceeds the available thrust at 4 km, so the
        # target loses energy instead of holding V_T forever.
        env.t_type[1] = 2
        env.t_radius[1] = 250.0**2 / (7.5 * 9.81)
        env.t_omega_b[1] = 250.0 / float(env.t_radius[1])
        env.t_speed[:] = 250.0
        for _ in range(90):  # 1.5 s, inside the 2 s unrecoverable-miss window
            env.step(torch.zeros(2, 3))
        assert float(env.t_speed[0]) == pytest.approx(250.0, abs=0.5)
        assert float(env.t_speed[1]) < 248.5
        assert torch.isfinite(env.t_speed).all()
    finally:
        env.close()


def test_propeller_advance_ratio_matches_blade_element_curve():
    from environments.common.plant import prop_advance_factor

    pitch_speed = 30.0
    throttle = torch.tensor([[0.5, 0.8, 1.0]])
    # Static thrust is unchanged at every throttle setting (no inflow).
    assert torch.allclose(
        prop_advance_factor(torch.zeros(1, 3), pitch_speed, throttle),
        torch.ones(1, 3),
    )
    # Quadratic in the advance ratio J = V / (n D), per rotor.
    v = torch.tensor([[3.0, 6.0, 9.0]])
    assert torch.allclose(
        prop_advance_factor(v, pitch_speed, throttle),
        1.0 - (v / (pitch_speed * throttle)).square(),
        rtol=1e-6,
    )
    # Monotone decay along the thrust axis, zero at each pitch speed.
    speeds = torch.linspace(0.0, 40.0, 81)[:, None].expand(-1, 3)
    factors = prop_advance_factor(speeds, pitch_speed, throttle)
    assert (factors[1:] <= factors[:-1] + 1e-12).all()
    assert float(
        prop_advance_factor(torch.tensor([[15.0]]), pitch_speed, torch.tensor([[0.5]]))
    ) == 0.0
    assert float(
        prop_advance_factor(torch.tensor([[30.0]]), pitch_speed, torch.tensor([[1.0]]))
    ) == 0.0
    # A tailwind along the thrust axis never exceeds the static value.
    assert float(
        prop_advance_factor(torch.tensor([[-5.0]]), pitch_speed, torch.tensor([[0.5]]))
    ) == 1.0


def test_fixedwing_advance_ratio_and_battery_thrust_at_plant_level():
    from environments.common.plant import battery_open_circuit_voltage

    env = FixedwingWaypointsEnv(
        num_envs=1, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        mask = torch.ones(1, dtype=torch.bool)
        command = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # full throttle

        def one_step(speed):
            """Reset the plant, hold the given body speed, run one substep."""
            plant.zero(mask)
            plant.set_state(
                torch.tensor([[0.0, 0.0, 100.0]]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[speed, 0.0, 0.0]]),
                torch.zeros(1, 3),
            )
            plant.physics_step(command, mask)
            throttle = float(plant.throttle[0, 0])
            rho = float(plant.air_density[0])
            thrust = float(plant.force[0, plant.links["motor_link"], 0])
            return thrust, throttle, rho

        # Static, fresh pack: advance and voltage factors are exactly one, so
        # the legacy quadratic throttle law survives unchanged.
        thrust, throttle, rho = one_step(0.0)
        legacy = 18.0 * throttle**2 * (rho / RHO0) ** (2.0 / 3.0)
        assert thrust == pytest.approx(legacy, rel=1e-5)
        assert float(plant.battery_charge[0]) == pytest.approx(1.0, abs=1e-5)

        # Forward flight: thrust falls by the blade-element factor at the
        # throttle actually reached in that substep.
        thrust, throttle, rho = one_step(6.0)
        legacy = 18.0 * throttle**2 * (rho / RHO0) ** (2.0 / 3.0)
        factor = 1.0 - (6.0 / (plant._prop_pitch_speed[0] * throttle)) ** 2
        assert thrust == pytest.approx(legacy * factor, rel=1e-4)

        # The pack starts at the full-charge open-circuit voltage and the
        # voltage factor equals one before any load has been applied.
        cells = plant._battery_cells
        assert float(plant.battery_voltage[0]) == pytest.approx(cells * 4.2, rel=1e-6)
        plant.battery_current.zero_()
        plant.battery_charge.fill_(1.0)
        assert float(plant._battery_thrust_factor()[0, 0]) == pytest.approx(1.0, rel=1e-6)
        # Half charge: the open-circuit voltage falls along the linear cell
        # curve and the thrust factor with its square.
        plant.battery_charge.fill_(0.5)
        v_half = battery_open_circuit_voltage(torch.tensor([0.5]), cells)
        expected = (v_half / (cells * 4.2)).square()
        assert float(plant._battery_thrust_factor()[0, 0]) == pytest.approx(
            float(expected[0]), rel=1e-6
        )
    finally:
        env.close()


def test_battery_coulomb_counting_and_voltage_sag():
    from environments.common.plant import battery_open_circuit_voltage

    env = FixedwingWaypointsEnv(
        num_envs=1, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        mask = torch.ones(1, dtype=torch.bool)
        command = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        plant.zero(mask)
        plant.set_state(
            torch.tensor([[0.0, 0.0, 100.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
        )
        charge_0 = float(plant.battery_charge[0])
        voltage_0 = float(plant.battery_voltage[0])
        coulombs = 0.0
        joules = 0.0
        for _ in range(120):  # 0.5 s at full throttle
            plant.physics_step(command, mask)
            current = float(plant.battery_current[0])
            voltage = float(plant.battery_voltage[0])
            coulombs += current * plant.physics_dt
            joules += current * voltage * plant.physics_dt
        charge_1 = float(plant.battery_charge[0])
        assert charge_1 < charge_0
        # Coulomb counting: the charge deficit times capacity equals the
        # integral of the recorded current.
        assert (charge_0 - charge_1) * plant._battery_capacity_c == pytest.approx(
            coulombs, rel=2e-4
        )
        assert float(plant.battery_energy[0]) == pytest.approx(joules, rel=2e-4)
        # The loaded terminal voltage sits below the open-circuit voltage and
        # the pack sags further as it discharges.
        v_oc = float(battery_open_circuit_voltage(plant.battery_charge, plant._battery_cells)[0])
        assert float(plant.battery_voltage[0]) < v_oc
        assert float(plant.battery_voltage[0]) < voltage_0
        # At full throttle the delivered thrust stays below the no-loss
        # static reference because the advance ratio and the sagging pack
        # both remove thrust.
        throttle = float(plant.throttle[0, 0])
        rho = float(plant.air_density[0])
        static = 18.0 * throttle**2 * (rho / RHO0) ** (2.0 / 3.0)
        assert float(plant._elec_thrust[0, 0]) < 0.95 * static
        assert float(plant._battery_thrust_factor()[0, 0]) < 0.995
    finally:
        env.close()


def test_quadplane_propulsion_channels_and_legacy_path():
    from environments.vtol import QuadplaneWaypointsEnv

    env = QuadplaneWaypointsEnv(
        num_envs=1, backend="cpu", seed=7, aero="realistic", domain_randomize=False
    )
    try:
        plant = env.plant
        # Four lift rotors + one pusher, each with its own disk area.
        assert plant._prop_pitch_speed.shape == (5,)
        assert plant._prop_area.shape == (5,)
        assert plant._elec_thrust.shape == (1, 5)
        assert plant._battery_cells == 4
        assert float(plant.battery_charge[0]) == pytest.approx(1.0, abs=1e-5)
        # A fresh, unloaded pack delivers exactly the nominal thrust.
        plant.battery_current.zero_()
        plant.battery_charge.fill_(1.0)
        assert float(plant._battery_thrust_factor()[0, 0]) == pytest.approx(1.0, rel=1e-6)
    finally:
        env.close()

    env = FixedwingWaypointsEnv(num_envs=1, backend="cpu", seed=7, aero="legacy")
    try:
        assert env.plant._prop_pitch_speed is None
        assert env.plant._prop_area is None
        assert env.plant.battery_charge is None
        assert env.plant._elec_thrust is None
    finally:
        env.close()
