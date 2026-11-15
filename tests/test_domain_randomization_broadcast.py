"""Regression tests for batched-parameter (broadcast) handling.

Domain randomization is ON by default for every task, so the satellite plants
overwrite their live design constants (``mass`` / ``f_max`` / ``isp`` /
``rotor_h_dyn`` / ``inertia``) with per-env ``(num_envs, ...)`` tensors.  Two
classes of bug used to follow from that:

* a controller reading the LIVE attribute (``float(env.mass)``) crashed;
* a per-env ``(num_envs,)`` plant constant multiplied straight into an
  ``(num_envs, 3)`` action/acceleration vector raised (or, when the replica
  count matched the trailing axis, silently produced a ``(num_envs,
  num_envs)`` block);
* ``apply_domain_randomization`` resampled EVERY replica whenever a single
  replica reset, so one episode's reset re-randomized its neighbours.

The tests below pin the fixes: controllers resolve their scalar design model
from ``env.controller_design`` (with the ``h_dyn`` -> ``rotor_h_dyn`` alias),
partial resets only touch the resetting replicas, and the wind filter honours
mask/index ``env_ids``.
"""

from __future__ import annotations

import torch
import pytest

from environments import make
from environments.common.disturbances import WindDisturbance
from environments.common.spaces import Box
from algorithms.baselines.baselines import BASELINE_VARIANTS, build_baseline

SAT_TASKS = ("docking", "rendezvous", "transfer", "satellite")


def _broadcast_params(env):
    """Perturbable params the plant currently holds as per-env tensors."""
    out = {}
    for name in env.PERTURBABLE_PARAMS:
        value = getattr(env, name, None)
        if torch.is_tensor(value) and value.shape[:1] == (env.num_envs,):
            out[name] = value
    return out


@pytest.mark.parametrize("task", SAT_TASKS)
def test_domain_randomization_broadcasts_plant_constants(task):
    env = make(task, num_envs=4, backend="cpu", seed=7)
    try:
        params = _broadcast_params(env)
        assert params, f"{task}: default domain randomization did not perturb anything"
        # the nominal model is preserved separately for the controller
        for name in params:
            assert name in env.controller_design, f"{task}: {name} missing from controller_design"
    finally:
        env.close()


@pytest.mark.parametrize("task", SAT_TASKS)
def test_partial_reset_only_resamples_the_reset_replicas(task):
    env = make(task, num_envs=4, backend="cpu", seed=7)
    try:
        params = _broadcast_params(env)
        assert params
        before = {name: value.clone() for name, value in params.items()}
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[2] = True
        env.reset_idx(mask)
        changed = False
        for name, old in before.items():
            new = getattr(env, name)
            assert torch.allclose(new[~mask], old[~mask]), (
                f"{task}: {name} was re-randomized on a replica that did not reset"
            )
            changed |= not torch.allclose(new[mask], old[mask])
        assert changed, f"{task}: the resetting replica kept its old plant realization"
    finally:
        env.close()


@pytest.mark.parametrize("task", SAT_TASKS)
def test_baselines_actions_under_default_domain_randomization(task):
    env = make(task, num_envs=3, backend="cpu", seed=7)
    try:
        assert _broadcast_params(env)
        obs = env.get_observations()
        variants = [None] + sorted(BASELINE_VARIANTS.get(task, {}))
        for variant in variants:
            for design in (None, env.controller_design):
                ctrl = build_baseline(task, env, variant, design=design)
                action = ctrl.actions(obs)
                assert action.shape == (env.num_envs, env.num_actions), (task, variant)
                assert torch.isfinite(action).all(), (task, variant)
                assert action.abs().max() <= 1.0 + 1e-6, (task, variant)
    finally:
        env.close()


def test_wind_step_honours_env_ids():
    wind = WindDisturbance(
        8, device="cpu", seed=3, steady_wind=[1.0, 0.0, 0.0], dryden=1.0, dt=1.0 / 240
    )
    assert wind.step().shape == (8, 3)
    assert wind.step(torch.tensor([1, 5])).shape == (2, 3)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[0] = True
    assert wind.step(mask).shape == (1, 3)
    # replicas outside the mask keep a frozen filter state
    mask[1] = True
    frozen = wind._state[:, ~mask].clone()
    wind.step(mask)
    assert torch.allclose(wind._state[:, ~mask], frozen)


def test_wind_masked_step_keeps_other_replica_noise_streams():
    """Per-replica counter RNG: ``step(env_ids)`` must not shift other streams.

    The old implementation drew every gust from ONE shared CPU generator, so
    how many samples a replica consumed depended on how many OTHER replicas had
    been stepped (and in which order) -- replica i's gust history changed the
    moment a neighbour reset.  Streams are now keyed per replica, so a masked
    step hands every selected replica exactly the sample it would have received
    from a full step, and leaves the unselected counters untouched.
    """
    n = 8
    full = WindDisturbance(n, device="cpu", seed=3, dryden=1.0, dt=1.0 / 240).step()
    partial = WindDisturbance(n, device="cpu", seed=3, dryden=1.0, dt=1.0 / 240)
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    sel = partial.step(mask)
    assert torch.allclose(sel, full[mask]), "masked step consumed a different sample"

    # a replica that was skipped twice still gets its FIRST sample when it is
    # finally stepped, regardless of how many masked steps the others took
    other = WindDisturbance(n, device="cpu", seed=3, dryden=1.0, dt=1.0 / 240)
    for _ in range(4):
        other.step(mask)
    first = other.step(torch.tensor([0]))
    assert torch.allclose(first[0], full[0]), "skipped replica's stream shifted"


@pytest.mark.parametrize(
    "task", ("quadx", "quadx_track", "quadplane", "fixedwing", "missile", "rocket")
)
def test_atmospheric_controller_design_records_physical_nominal(task):
    """``param_mismatch`` must store SI nominals, not the 1.0 scale placeholder.

    ``PERTURBABLE_PARAMS`` maps ``thrust``/``mass`` to scale factors (1.0).  The
    bookkeeping used to copy that placeholder into ``env.controller_design``,
    so a controller resolving ``_design("mass")`` would have received 1.0 kg
    (and 1 N for thrust) instead of the vehicle's real nominal.
    """
    env = make(
        task,
        num_envs=4,
        backend="cpu",
        seed=7,
        domain_randomize=False,
        param_mismatch={"thrust": 0.1, "mass": 0.1, "aero": 0.1, "actuator": 0.1},
    )
    try:
        design = dict(env.controller_design)
    finally:
        env.close()
    ref = make(task, num_envs=4, backend="cpu", seed=7, domain_randomize=False)
    try:
        # Every requested parameter must be recorded, and nothing outside the
        # env's declared perturbable set may appear.  Realism-only params
        # (sensor/cg/inertia) are absent from this legacy mismatch spec.
        assert {"thrust", "mass", "aero", "actuator"} <= set(design)
        assert set(design) <= set(ref.PERTURBABLE_PARAMS), task
        for name, value in design.items():
            assert isinstance(value, float), (task, name, type(value))
            assert value > 0.0, (task, name, value)
            if name in ("aero", "actuator", "sensor", "cg", "inertia"):
                # These are multiplicative scale factors with no physical
                # nominal, so 1.0 is the correct design value.
                assert value == pytest.approx(1.0)
                continue
            assert value != pytest.approx(1.0), f"{task}: {name} kept the 1.0 scale placeholder"
            if task == "rocket" and name == "mass":
                # The booster is built fully fuelled (549.1 kg) and reset() fills
                # it to the 5 % spawn load (158.7 kg), so the nominal captured at
                # construction legitimately differs from a fresh env's
                # post-reset mass.  Any physical mass is fine; 1.0 kg was not.
                continue
            # the recorded value is the PRE-perturbation physical nominal
            assert value == pytest.approx(ref.plant.nominal_param(name)), (task, name)
    finally:
        ref.close()


def test_docking_action_bounds_match_normalized_policy():
    """``rendezvous_docking`` actions are normalized; +-f_max belongs elsewhere."""
    env = make("docking", num_envs=4, backend="cpu", seed=7)
    try:
        ones = torch.ones(3, dtype=env.action_low.dtype, device=env.device)
        assert torch.allclose(env.action_low, -ones)
        assert torch.allclose(env.action_high, ones)
        assert env.clip_actions == 1.0
        # nominal physical authority is kept separately (live f_max is per-env
        # under the default domain randomization)
        assert torch.allclose(
            env.thrust_bounds_N,
            torch.full(
                (3,), env.controller_design["f_max"], dtype=env.thrust_bounds_N.dtype, device=env.device
            ),
        )
        physical = env.physical_action_space
        assert physical.shape == (env.num_actions,)
        assert torch.allclose(torch.as_tensor(physical.low), env.action_low)
        assert torch.allclose(torch.as_tensor(physical.high), env.action_high)
    finally:
        env.close()


def test_box_contains_accepts_scalar_sample():
    box = Box(low=-1.0, high=1.0, shape=(1,))
    assert box.contains(0.5) is True
    assert box.contains(2.0) is False
    wide = Box(low=-1.0, high=1.0, shape=(3,))
    assert wide.contains(0.5) is False  # a scalar is not a 3-D sample
    assert wide.contains(torch.zeros(3)) is True


def test_docking_lqr_accumulator_scales_per_env_accel():
    """eval_docking_lqr._EpAcc: a_max is per-env under default DR."""
    from environments.satellite.rendezvous_docking.eval_docking_lqr import _EpAcc

    env = make("docking", num_envs=5, backend="cpu", seed=7)
    try:
        ctrl = build_baseline("docking", env)
        # DR rewrites f_max / mass as per-env tensors -> a_max is (num_envs,)
        a_max = env.f_max / env.mass
        assert torch.is_tensor(a_max) and a_max.shape == (env.num_envs,)
        obs = env.get_observations()
        acc = _EpAcc(env.num_envs, env.device, env.range_fail, 0.8 * env.range_fail)
        for _ in range(3):
            acc.step(ctrl.actions(obs), env.dt, a_max)
        # a flat (num_envs,) bound would have produced a (num_envs, num_envs)
        # cost block instead of one number per replica
        assert acc.t["cost"].shape == (env.num_envs,)
        assert torch.isfinite(acc.t["cost"]).all()
    finally:
        env.close()


def test_steering_compare_handles_domain_randomization():
    """scripts/satellite_steering_compare: rotor_h_dyn is per-env under DR."""
    import scripts.satellite_steering_compare as sc
    from environments.satellite.attitude_control import SatelliteAttitudeEnv

    env = SatelliteAttitudeEnv(num_envs=4, backend="cpu", seed=7, num_cmgs=4)
    try:
        ref = build_baseline("satellite", env, "quat_pd_mp_pinv_cmg")
        arm = build_baseline("satellite", env, "quat_pd_damped_cmg")
        assert torch.is_tensor(env.rotor_h_dyn)  # default DR -> per-env tensor
        res = sc.run_arm(
            "damped",
            env,
            lambda obs: arm.actions(obs),
            ("random_target", None, None),
            seed=7,
            ref=ref,
            ctrl=arm,
        )
        # reported on the nominal design model: h * n_cmg * action_scale
        # = 20 * 4 * 3 = 240 (a live per-env tensor cannot be a scalar report)
        assert res["peak_torque_authority"] == pytest.approx(240.0)
        assert res["torque_error_fraction_of_authority"] > 0.0
        assert 0.0 <= res["success_rate"] <= 1.0
    finally:
        env.close()
