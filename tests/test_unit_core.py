"""Unit tests for the pure-Python core modules (no Genesis dependency).

Covers:
  * TensorRandom: uniformity, determinism, mask isolation
  * Domain-randomization specs (parse, resolve, static, per-episode)
  * Baseline._design chain (env.live vs controller_design vs explicit)
  * ResidualVecEnv: authority bounds, zero-policy identity, phase gate
  * Returns-to-go (terminal/truncated/bootstrap semantics)
  * delta_action_targets / phase_weights (BC helpers)
  * build_cfg: distribution selection, default PPO config integrity
  * _warmup_cosine_decay boundary values
"""
from __future__ import annotations

import math

import pytest
import torch

from algorithms.train_utils import _warmup_cosine_decay, build_cfg
from algorithms.warmstart.bc_warmstart import (
    BC_DEFAULTS,
    delta_action_targets,
    phase_weights,
    returns_to_go,
)
from environments.common.domain_randomization import (
    apply_domain_randomization,
    apply_static_perturbation,
    as_env_mask,
    parse_perturbation_spec,
    resolve_domain_randomize,
    sample_scales,
)
from environments.common.plant import TensorRandom


# --------------------------------------------------------------------------- #
# TensorRandom
# --------------------------------------------------------------------------- #
class TestTensorRandom:
    def test_determinism(self):
        a = TensorRandom(4, "cpu", 7)
        b = TensorRandom(4, "cpu", 7)
        assert torch.equal(a.uniform(3, torch.ones(4, dtype=torch.bool)),
                           b.uniform(3, torch.ones(4, dtype=torch.bool)))

    def test_mask_isolation(self):
        """Disabled replicas must NOT consume random numbers or advance the
        counter, and they must not influence active replicas' bitstreams."""
        r1 = TensorRandom(2, "cpu", 7)
        r2 = TensorRandom(2, "cpu", 7)
        # r1: both active -> 2*3=6 numbers consumed
        a1 = r1.uniform(3, torch.tensor([True, True]))
        # r2: only replica 0 active
        a2 = r2.uniform(3, torch.tensor([True, False]))
        # replica 0 of r1 and r2 should match (same seed, same mask bool)
        assert torch.equal(a1[0], a2[0])
        # replica 1 of r1 advanced its counter; r2 replica 1 did not
        assert r1.counter[1].item() == 3
        assert r2.counter[1].item() == 0

    def test_mask_disabled_never_advances(self):
        r = TensorRandom(1, "cpu", 11)
        orig = r.counter.clone()
        r.uniform(5, torch.tensor([False]))
        assert torch.equal(r.counter, orig)

    def test_uniform_range(self):
        r = TensorRandom(3, "cpu", 11)
        u = r.uniform(10, torch.ones(3, dtype=torch.bool))
        assert (u >= 0).all() and (u < 1).all()

    def test_normal_gaussian_shape(self):
        r = TensorRandom(4, "cpu", 5)
        mean = 10.0
        n = r.normal(torch.ones(4, dtype=torch.bool), mean)
        assert n.shape == (4,)
        assert torch.isfinite(n).all()


# --------------------------------------------------------------------------- #
# Euler <-> quaternion helpers
# --------------------------------------------------------------------------- #
class TestQuaternionHelpers:
    """Guard the half-angle convention used by spawn attitude randomization."""

    def test_quat_from_euler_matches_scipy_and_is_unit(self):
        from scipy.spatial.transform import Rotation

        from environments.common.dynamics import euler, quat_from_euler

        rpy = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.35, 0.0, 0.0],
                [0.0, 0.0, math.pi],
                [0.3, -0.2, 2.5],
                [1.2, 0.7, -2.0],
            ],
            dtype=torch.float64,
        )
        q = quat_from_euler(rpy)
        # Unit norm is the property the old ``0.5 * cos(r)`` form broke: even
        # the identity attitude came out as (0.125, 0, 0, 0).
        assert torch.allclose(q.norm(dim=-1), torch.ones(len(rpy), dtype=torch.float64))
        # scipy returns xyzw, the repo helper returns wxyz.
        ref = Rotation.from_euler("xyz", rpy.numpy()).as_quat()
        assert torch.allclose(q, torch.from_numpy(ref[:, [3, 0, 1, 2]]), atol=1e-12)
        # Round-trip through the repo's own inverse (pitch stays inside +-90 deg).
        assert torch.allclose(euler(q), rpy, atol=1e-6)

    def test_falcon9_rotation_and_tilt_helpers(self):
        from environments.common.dynamics import euler, inverse_rotate, quat_from_euler
        from environments.common.falcon9_dynamics import (
            rotate_body_to_world,
            rotate_world_to_body,
        )
        from environments.common.falcon9_reward import _body_tilt_from_quat

        rpy = torch.tensor(
            [[0.3, -0.4, 2.2], [0.7, 0.0, 0.0], [0.0, 0.9, 0.0], [1.2, -1.1, -2.0]],
            dtype=torch.float64,
        )
        q = quat_from_euler(rpy)
        v = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        # body->world must equal the validated dynamics helper applied with q^-1
        # (the old implementation silently applied the conjugate rotation).
        conj = torch.cat((q[:, :1], -q[:, 1:]), -1)
        fwd = rotate_body_to_world(q, v)
        assert torch.allclose(fwd, inverse_rotate(conj, v), atol=1e-12)
        # world->body must be the exact inverse (the old implementation also
        # scaled the vector magnitude by 1 + 2|q_v|^2).
        assert torch.allclose(rotate_world_to_body(q, fwd), v, atol=1e-12)
        # Combined tilt must match |roll|,|pitch| from the validated inverse.
        tilt = _body_tilt_from_quat(q)
        expected = euler(q)[:, :2].pow(2).sum(-1).sqrt()
        assert torch.allclose(tilt, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Domain randomization specs
# --------------------------------------------------------------------------- #
class TestDomainRandomization:
    def test_parse_fractional(self):
        assert parse_perturbation_spec({"mass": 0.2}) == {"mass": (0.8, 1.2)}

    def test_parse_range(self):
        assert parse_perturbation_spec({"mass": (0.85, 1.15)}) == {"mass": (0.85, 1.15)}

    def test_parse_dict(self):
        assert parse_perturbation_spec({"mass": {"low": 0.9, "high": 1.1}}) == {"mass": (0.9, 1.1)}

    def test_parse_unknown(self):
        with pytest.raises(TypeError):
            parse_perturbation_spec({"mass": "bad"})

    def test_parse_empty(self):
        assert parse_perturbation_spec(None) == {}
        assert parse_perturbation_spec({}) == {}

    def test_resolve_none_disables(self):
        assert resolve_domain_randomize(None, None) is None

    def test_resolve_false_disables(self):
        assert resolve_domain_randomize(False, {"mass": (0.8, 1.2)}) is None

    def test_resolve_true_default(self):
        out = resolve_domain_randomize(True, {"mass": (0.8, 1.2)})
        assert out == {"mass": (0.8, 1.2)}

    def test_resolve_merge(self):
        base = {"mass": (0.8, 1.2), "thrust": (0.9, 1.1)}
        merged = resolve_domain_randomize({"mass": (0.95, 1.05)}, base)
        assert merged == {"mass": (0.95, 1.05), "thrust": (0.9, 1.1)}

    def test_resolve_unknown(self):
        with pytest.raises(TypeError):
            resolve_domain_randomize(42, None)

    def test_sample_scales_deterministic(self):
        spec = {"mass": (0.8, 1.2), "thrust": (0.9, 1.1)}
        a = sample_scales(spec, 4, "cpu", 7, 0)
        b = sample_scales(spec, 4, "cpu", 7, 0)
        for k in spec:
            assert torch.equal(a[k], b[k])
            assert (a[k] >= spec[k][0]).all() and (a[k] <= spec[k][1]).all()

    def test_sample_scales_counter_changes(self):
        spec = {"mass": (0.8, 1.2)}
        a = sample_scales(spec, 4, "cpu", 7, 0)
        b = sample_scales(spec, 4, "cpu", 7, 1)
        assert not torch.equal(a["mass"], b["mass"])

    def test_as_env_mask_index_tensor(self):
        class E: num_envs = 4; device = "cpu"
        m = as_env_mask(E(), torch.tensor([0, 2]))
        assert m.tolist() == [True, False, True, False]

    def test_as_env_mask_bool_tensor(self):
        class E: num_envs = 3; device = "cpu"
        m = as_env_mask(E(), torch.tensor([False, True, False]))
        assert m.tolist() == [False, True, False]

    def test_apply_static_captures_design_before_perturb(self):
        class E:
            PERTURBABLE_PARAMS = {"mass": 1.0, "thrust": 100.0}
            def __init__(self):
                self.mass = 1.0
                self.thrust = 100.0
                self._applied = {}
            def _apply_perturbation(self, name, scale):
                self._applied[name] = float(scale)

        e = E()
        apply_static_perturbation(e, {"mass": 0.2, "thrust": 1.1})
        assert e._applied == {"mass": 1.2, "thrust": 1.1}
        # controller_design captured NOMINAL (before any _apply)
        assert e.controller_design == {"mass": 1.0, "thrust": 100.0}
        assert e._static_scales == {"mass": 1.2, "thrust": 1.1}

    def test_apply_static_unknown_param(self):
        class E: PERTURBABLE_PARAMS = {"mass": 1.0}
        with pytest.raises(KeyError):
            apply_static_perturbation(E(), {"bogus": 0.2})

    def test_apply_per_episode_masked(self):
        class E:
            PERTURBABLE_PARAMS = {"mass": 1.0}
            def __init__(self):
                self.num_envs = 4
                self.device = torch.device("cpu")
                self._static_scales = {}
                self._dr_scales = {}
            def _apply_perturbation(self, name, scales):
                self._dr_scales[name] = scales.clone()

        e = E()
        # first: randomize all, then resample only indices [0, 2]
        apply_domain_randomization(
            e, {"mass": (0.8, 1.2)}, seed=1, counter=0, mask=None
        )
        prev = e._dr_scales["mass"].clone()
        apply_domain_randomization(
            e, {"mass": (0.8, 1.2)}, seed=2, counter=1,
            mask=torch.tensor([True, False, True, False]),
        )
        cur = e._dr_scales["mass"]
        assert torch.not_equal(cur[1], prev[1]).sum() == 0  # unchanged
        assert torch.not_equal(cur[3], prev[3]).sum() == 0
        assert torch.not_equal(cur[0], prev[0]).sum() == 1  # resampled
        assert torch.not_equal(cur[2], prev[2]).sum() == 1


# --------------------------------------------------------------------------- #
# returns_to_go
# --------------------------------------------------------------------------- #
class TestReturnsToGo:
    def test_gamma_required_raises(self):
        """Docstring says `gamma` must be provided; confirm ValueError."""
        rews = torch.ones(3, 2)
        dones = torch.zeros(3, 2, dtype=torch.bool); dones[-1] = True
        with pytest.raises(ValueError, match="gamma is required"):
            returns_to_go(rews, dones, gamma=None)

    def test_basic_episode_boundaries(self):
        rews = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        dones = torch.tensor([[False, True], [False, True], [True, False]])
        r = returns_to_go(rews, dones, 0.5)
        # env0: acc step 2 = 1.0; step 1 = 1 + 0.5*1 = 1.5; step 0 = 1 + 0.75 = 1.75
        assert r[2, 0].item() == pytest.approx(1.0)
        assert r[1, 0].item() == pytest.approx(1.5)
        assert r[0, 0].item() == pytest.approx(1.75)
        # env1: done at step 0 and step 2 -> each has zero terminal
        assert r[:, 1].tolist() == pytest.approx([2.0, 2.0, 2.0])

    def test_terminated_truncated_different_terminal_value(self):
        """Bootstrap-less truncation propagates an 'invalid' flag backwards
        inside one episode; bootstrap-supplied truncation keeps the target
        valid and adds gamma * V. Termination carries zero terminal value."""
        T, N = 3, 2
        rews = torch.ones(T, N)
        # env0: plain trunc at last step -> invalid from step 1 backward
        # env1: trunc at last step WITH bootstrap -> valid, target has + 0.99*5
        dones = torch.tensor([[False, False], [False, False], [True, True]])
        terminated = torch.zeros_like(dones)
        truncated = dones  # all trunc, none term
        boot = torch.full((T, N), 5.0)
        out_b1, v_b1 = returns_to_go(
            rews, dones, 0.99,
            terminated=terminated, truncated=truncated, bootstrap_values=boot,
        )
        out_none, v_none = returns_to_go(
            rews, dones, 0.99,
            terminated=terminated, truncated=truncated,
        )
        # env0: no bootstrap -> invalid propagation back from trunc boundary
        # truncated w/o bootstrap -> invalid_tail at t=2 -> valid[1] = valid[2] = False
        assert not v_none[1, 0].item()
        # env1: bootstrap supplied -> valid all the way
        assert v_b1[1, 1].item()
        assert out_b1[2, 1].item() == pytest.approx(1.0 + 0.99 * 5.0)
        # Termination (unlike truncation) is ALWAYS a valid boundary with zero
        # terminal value, so v_t[2, 0] stays True
        term_only = torch.zeros_like(dones)
        term_only[2, 0] = True
        out_t, v_t = returns_to_go(
            rews, dones, 0.99, terminated=term_only, truncated=dones & ~term_only,
        )
        assert v_t[2, 0].item()  # true terminal: valid with zero value
        assert out_t[2, 0].item() == pytest.approx(1.0)  # only immediate rews[2,0]


# --------------------------------------------------------------------------- #
# BC helper tests
# --------------------------------------------------------------------------- #
class TestBcHelpers:
    def test_bc_defaults_present(self):
        for key in ("steps", "epochs", "batch_size", "lr", "fit_critic"):
            assert key in BC_DEFAULTS

    def test_delta_action_resets_on_episode_boundary(self):
        acts = torch.tensor([[0.5, -0.3], [0.6, -0.2], [0.1, 0.9], [0.2, 0.7]], dtype=torch.float32)
        t_in_ep = torch.tensor([0, 1, 0, 1])
        out = delta_action_targets(acts, t_in_ep, scale=2.0)
        # row 0 (t=0): prev zero -> (0.5, -0.3)/2
        assert out[0].tolist() == pytest.approx([0.25, -0.15])
        # row 2 (t=0): prev zero -> (0.1, 0.9)/2
        assert out[2].tolist() == pytest.approx([0.05, 0.45])
        # row 1: prev = row 0 -> (0.6-0.5)/2, (-0.2+0.3)/2
        assert out[1].tolist() == pytest.approx([0.05, 0.05])

    def test_delta_action_clamps(self):
        acts = torch.tensor([[1.0, -1.0]])
        prev = torch.tensor([[-1.0, 1.0]])
        out = delta_action_targets(acts, torch.zeros(1), scale=1.0)
        assert (out.abs() <= 1.001).all()

    def test_phase_weights_normalize(self):
        obs = torch.randn(10, 5)
        w = phase_weights(obs, {"index": [0], "near": 0.1, "weight": 3.0})
        assert w.shape == (10,)
        assert w.mean().item() == pytest.approx(1.0)

    def test_phase_weights_no_normalize(self):
        obs = torch.randn(5, 3)
        w = phase_weights(obs, {"near": 10.0, "weight": 3.0, "normalize": False})
        assert (w == 3.0).all()


# --------------------------------------------------------------------------- #
# _warmup_cosine_decay
# --------------------------------------------------------------------------- #
class TestWarmupCosine:
    def test_monotonic_from_peak(self):
        init, peak, end = 1e-4, 1e-3, 1e-5
        warmup = 20
        decay = 100
        vals = [_warmup_cosine_decay(s, init, peak, end, warmup, decay)
                for s in range(0, 110, 5)]
        # during warmup rising; after peak monotonic decay
        assert vals[0] == pytest.approx(init)
        assert vals[warmup // 5] == pytest.approx(peak, rel=0.05)
        # at/after decay steps -> end
        assert vals[-1] == pytest.approx(end, rel=0.05)


# --------------------------------------------------------------------------- #
# build_cfg
# --------------------------------------------------------------------------- #
class TestBuildCfg:
    def test_beta_distribution_default(self):
        cfg = build_cfg()
        assert cfg["actor"]["distribution_cfg"]["class_name"] == "BetaDistribution"

    def test_gaussian_choice(self):
        cfg = build_cfg(distribution="gaussian")
        assert cfg["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"

    def test_unknown_distribution(self):
        with pytest.raises(ValueError):
            build_cfg(distribution="cauchy")

    def test_kl_anchor_class(self):
        cfg = build_cfg(kl_anchor_coef=0.1)
        assert cfg["algorithm"]["class_name"] == "algorithms.ppo.kl_anchor:KLAnchorPPO"

    def test_shared_defaults(self):
        cfg = build_cfg()
        assert cfg["algorithm"]["learning_rate"] == 1e-3
        assert cfg["num_steps_per_env"] == 32
        assert cfg["actor"]["hidden_dims"] == [128, 128]


# --------------------------------------------------------------------------- #
# ResidualVecEnv (pure: authority, channel validation, zero-init)
# --------------------------------------------------------------------------- #
class TestResidualVecEnv:
    def _make_env(self, a_dim=3):
        class FakeEnv:
            device = torch.device("cpu")
            num_envs = 4
            num_actions = a_dim
            clip_actions = 1.0
            def get_observations(self):
                return {"policy": torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32)}
            def step(self, a):
                return {"policy": torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32)}, \
                       torch.zeros(self.num_envs), \
                       torch.zeros(self.num_envs, dtype=torch.bool), \
                       torch.zeros(self.num_envs, dtype=torch.bool), \
                       {}
            def __getattr__(self, k): raise AttributeError(k)
        return FakeEnv()

    def test_zero_policy_gives_exact_nominal(self):
        from algorithms.residual.residual_env import ResidualVecEnv

        class Nominal:
            name = "mock"
            def reset_idx(self, mask): pass
            def actions(self, obs):
                return torch.full((4, 3), 0.5, device="cpu", dtype=torch.float32)

        env = self._make_env()
        wrapper = ResidualVecEnv(env, Nominal(), residual_limit=0.3)
        obs, rew, term, trunc, _ = wrapper.step(torch.zeros(4, 3))
        applied = wrapper.last_applied
        assert torch.equal(applied, torch.full_like(applied, 0.5))

    def test_authority_half_range(self):
        from algorithms.residual.residual_env import ResidualVecEnv

        class Nominal:
            name = "mock"
            def reset_idx(self, mask): pass
            def actions(self, obs):
                return torch.zeros((4, 2), device="cpu", dtype=torch.float32)

        env = self._make_env(a_dim=2)
        ResidualVecEnv(env, Nominal(), residual_limit=1.0)
        with pytest.raises(ValueError):
            ResidualVecEnv(env, Nominal(), residual_limit=1.01)

    def test_zero_init_policy_output(self):
        from algorithms.residual.residual_env import zero_init_policy_output

        class _Dist:
            def __init__(self):
                self.log_std_param = torch.tensor([0.0])
        class _Actor:
            def __init__(self):
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)
                )
                self.distribution = _Dist()
        class _Alg:
            def __init__(self):
                self.actor = _Actor()
                self.critic = _Actor()
        class _Runner:
            def __init__(self):
                self.alg = _Alg()
        out = zero_init_policy_output(_Runner())
        assert out["zero_init_actor_output"] is True

    def test_phase_gate_smoke(self):
        from algorithms.residual.residual_env import ResidualVecEnv

        class Nominal:
            name = "mock"
            def reset_idx(self, mask): pass
            def actions(self, obs):
                return torch.zeros((4, 2), device="cpu", dtype=torch.float32)

        w = ResidualVecEnv(
            self._make_env(a_dim=2), Nominal(), residual_limit=0.5,
            residual_rho=None,
        )
        w.phase_gate = {"index": [0], "near": 1.0, "soft": 0.1, "gain": 2.0}
        w._obs = {"policy": torch.zeros(4, 3)}
        scale = w._authority_scale(torch.float32)
        assert torch.allclose(scale, torch.full_like(scale, 3.0), atol=1e-3)

    def test_set_authority_fraction_invalid(self):
        from algorithms.residual.residual_env import ResidualVecEnv

        class Nominal:
            name = "mock"
            def reset_idx(self, mask): pass
            def actions(self, obs):
                return torch.zeros((4, 2), device="cpu", dtype=torch.float32)

        w = ResidualVecEnv(self._make_env(a_dim=2), Nominal(), residual_limit=0.3)
        with pytest.raises(ValueError):
            w.set_authority_fraction(-0.1)
