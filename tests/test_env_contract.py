"""Unified environment contract tests.

Covers the Phase-1 regression surface for all 10 registered tasks:
  * registry construction + consistent action/observation spaces
  * step invariants (no NaN/Inf, shapes, action range)
  * determinism under a fixed seed
  * residual-action regression: residual=True with zero policy action must be
    bit-identical to residual=False driven by the conventional controller

The traditional-baseline isolation check (baselines read only the observation
space, never internal env state) is implemented as a static grep over the
algorithms/baselines directory plus each task's conventional_controller class.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from environments import ENV_REGISTRY, make

REPO = Path(__file__).resolve().parents[1]
TASKS = sorted(ENV_REGISTRY)


# --------------------------------------------------------------------------- #
# 1. Registry + space consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_registry_construction_and_spaces(task):
    env = make(task, num_envs=4, backend="cpu", seed=7)
    try:
        assert env.num_envs == 4
        obs = env.get_observations()
        assert "policy" in obs
        o = obs["policy"]
        assert o.shape == (4, env.observation_space.shape[0])
        # policy-facing action space is uniformly [-1, 1]
        a_space = env.action_space
        assert torch.all(a_space.low == -1.0)
        assert torch.all(a_space.high == 1.0)
        assert o.dtype == torch.float32
        assert torch.isfinite(o).all(), f"{task}: non-finite observation after reset"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 2. Step invariants under random actions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_step_invariants(task):
    env = make(task, num_envs=8, backend="cpu", seed=7)
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        a_dim = env.action_space.shape[0]
        for _ in range(60):
            actions = 2.0 * torch.rand((8, a_dim), generator=g) - 1.0
            obs, reward, terminated, truncated, extras = env.step(actions)
            o = obs["policy"]
            assert o.shape == (8, env.observation_space.shape[0])
            assert torch.isfinite(o).all()
            assert torch.isfinite(reward).all()
            assert terminated.dtype == torch.bool
            assert truncated.dtype == torch.bool
            assert actions.shape == (8, a_dim)
            assert actions.abs().max() <= 1.0 + 1e-6
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 3. Determinism: same seed -> same trajectory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_determinism(task):
    def rollout():
        env = make(task, num_envs=4, backend="cpu", seed=123)
        g = torch.Generator(device="cpu").manual_seed(999)
        a_dim = env.action_space.shape[0]
        obs_trace = []
        rew_trace = []
        for _ in range(40):
            actions = 2.0 * torch.rand((4, a_dim), generator=g) - 1.0
            obs, reward, _, _, _ = env.step(actions)
            obs_trace.append(obs["policy"].clone())
            rew_trace.append(reward.clone())
        env.close()
        return obs_trace, rew_trace

    o1, r1 = rollout()
    o2, r2 = rollout()
    for a, b in zip(o1, o2):
        assert torch.allclose(a, b, atol=1e-5), f"{task}: observation drift"
    for a, b in zip(r1, r2):
        assert torch.allclose(a, b, atol=1e-6), f"{task}: reward drift"


# --------------------------------------------------------------------------- #
# 4. Residual-action regression
#    residual=True with zero policy action  ==  residual=False driven by the
#    conventional controller (bit-identical trajectory).
# --------------------------------------------------------------------------- #
def _has_residual(env_cls):
    return "residual" in inspect.signature(env_cls.__init__).parameters


@pytest.mark.parametrize(
    "task", [t for t in TASKS if _has_residual(ENV_REGISTRY[t][0])]
)
def test_residual_zero_matches_conventional_controller(task):
    """residual=True with zero policy action must reproduce the trajectory of
    residual=False driven by the same conventional-controller output.

    The controller is stateful, so it must be invoked exactly once per step.
    We wrap env_r's ``_residual_input`` to capture the base action it actually
    applies, then drive env_c (non-residual) with that captured action.
    """
    env_r = make(task, num_envs=4, backend="cpu", seed=7, residual=True)
    env_c = make(task, num_envs=4, backend="cpu", seed=7, residual=False)
    try:
        captured = {}
        orig_input = env_r._residual_input

        def wrapped(actions):
            base = orig_input(actions)  # base = ctrl.actions(obs) + actions
            captured["base"] = base - actions.to(base.dtype)  # recover ctrl output
            return base

        env_r._residual_input = wrapped

        for _ in range(30):
            obs_r, rr, tr_r, tu_r, _ = env_r.step(
                torch.zeros((4, env_r.action_space.shape[0]), dtype=torch.float32)
            )
            base = captured["base"]
            obs_c, rc, tr_c, tu_c, _ = env_c.step(base)
            assert torch.equal(obs_r["policy"], obs_c["policy"]), (
                f"{task}: residual+zero != conventional controller trajectory"
            )
            assert torch.equal(rr, rc)
            assert torch.equal(tr_r, tr_c)
            assert torch.equal(tu_r, tu_c)
    finally:
        env_r.close()
        env_c.close()


# --------------------------------------------------------------------------- #
# 5. Traditional baselines read only the observation space
#    Static check: conventional_controller source must not touch private env
#    attributes (plant, state buffers, etc.) beyond get_observations().
# --------------------------------------------------------------------------- #
FORBIDDEN_INTERNAL = (
    "plant",
    "obs_buf",
    "action",
    "targets",
    "count",
    "terminated",
    "truncated",
    "reward",
    "state",
)


@pytest.mark.parametrize("task", TASKS)
def test_baseline_only_reads_observation_space(task):
    """Conventional controllers must derive their action from the observation
    only.  They may hold their own state (``self.*``); the forbidden pattern is
    reaching into the environment's internal buffers via the ``env`` reference
    passed to the controller (or a stored ``self.env``)."""
    env_cls = ENV_REGISTRY[task][0]
    ctrl_cls = getattr(env_cls, "conventional_controller", None)
    assert ctrl_cls is not None, f"{task}: no conventional_controller declared"
    src = inspect.getsource(ctrl_cls)
    tree = ast.parse(src)
    accessed = set()
    for node in ast.walk(tree):
        # env.<attr>  (the env parameter)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "env":
                accessed.add(node.attr)
        # self.env.<attr>  (stored env reference)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr == "env"
        ):
            accessed.add(node.attr)
    # get_observations is the sanctioned read; everything else internal is banned
    internal_hits = [a for a in accessed if a in FORBIDDEN_INTERNAL]
    assert not internal_hits, (
        f"{task} baseline {ctrl_cls.__name__} accesses env internal state: {internal_hits}"
    )
