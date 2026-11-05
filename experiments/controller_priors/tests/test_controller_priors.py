"""Tests for the controller-prior experiment chain (residual + BC data)."""

import copy
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]  # repository root
_CP = _ROOT / "experiments" / "controller_priors"  # controller-prior package
sys.path.insert(0, str(_ROOT))

from algorithms.baselines.baselines import build_baseline  # noqa: E402
from algorithms.residual.residual_env import (  # noqa: E402
    ResidualVecEnv,
    zero_init_policy_output,
    success_gated_authority_curriculum,
)
from algorithms.residual.action_space import (  # noqa: E402
    DeltaActionVecEnv,
    ActionRatePenaltyVecEnv,
)
from algorithms.ppo.kl_anchor import KLAnchorPPO  # noqa: E402
from algorithms.train_utils import RslVecEnv, build_cfg, _save_runner  # noqa: E402
from algorithms.warmstart.bc_warmstart import (  # noqa: E402
    fit_behavior_cloning,
    returns_to_go,
    phase_weights,
    delta_action_targets,
)
from algorithms.warmstart.expert_dataset import (  # noqa: E402
    collect_expert_dataset,
    flatten_split,
    split_dataset,
    merge_datasets,
    save_expert_dataset,
    load_expert_dataset,
    select_representative_episodes,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from scripts.first_episode import FirstEpisodes  # noqa: E402
from environments.satellite.rendezvous_docking import SpaceDockingEnv  # noqa: E402

# `scripts` is already bound to the repository-level package; add the
# controller-prior tree afterwards for its `analysis` package, and load the
# pure ranking helper by file to avoid the package-name collision.
sys.path.insert(0, str(_CP))
from analysis.aggregate import bootstrap_ci  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cp_select_checkpoint", _CP / "scripts" / "select_checkpoint.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
rank_candidates = _mod.rank_candidates


def make_env(n=4, seed=7, **kw):
    return SpaceDockingEnv(num_envs=n, backend="cpu", seed=seed, **kw)


def make_controller(env, **kw):
    return build_baseline("docking", env, **kw)


class SpyController:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.observe_calls = 0
        self.resets = 0
        self.last_obs_digest = None
        self.name = getattr(inner, "name", "spy")

    def actions(self, obs):
        self.calls += 1
        o = obs["policy"] if hasattr(obs, "keys") else obs
        self.last_obs_digest = float(o.sum())
        return self.inner.actions(obs)

    def reset_idx(self, mask):
        self.resets += 1
        return self.inner.reset_idx(mask)

    def observe_applied_action(self, action):
        self.observe_calls += 1
        return self.inner.observe_applied_action(action)


def _rollout_direct(env, ctrl, steps):
    obs = env.get_observations()
    ctrl.reset_idx(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    out = []
    for _ in range(steps):
        with torch.no_grad():
            a = ctrl.actions(obs)
        obs, _, term, trunc, _ = env.step(a)
        ctrl.reset_idx(term | trunc)
        out.append(obs["policy"].clone())
    return out


def test_residual_action_scaling_and_channels():
    env = make_env()
    ctrl = make_controller(env)
    w = ResidualVecEnv(env, ctrl, residual_rho=0.05, task="docking")
    assert torch.allclose(w.residual_limit, torch.full((3,), 0.1, dtype=torch.float32))
    assert w.physics_limits()["residual_limit_thrust_N"] == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ResidualVecEnv(env, ctrl, residual_limit=0.1, residual_rho=0.1)
    with pytest.raises(ValueError):
        ResidualVecEnv(env, ctrl, residual_rho=0.6)
    env2 = make_env()
    ctrl2 = make_controller(env2)
    w2 = ResidualVecEnv(env2, ctrl2, residual_limit=[0.05, 0.1, 0.2], task="docking")
    assert w2.residual_rho.tolist() == pytest.approx([0.025, 0.05, 0.1])
    assert w2.physics_limits()["residual_limit_thrust_N"] == pytest.approx([0.5, 1.0, 2.0])


def test_zero_residual_reproduces_controller_trajectory():
    n, steps = 4, 40
    env_a = make_env(n, seed=11)
    ctrl_a = make_controller(env_a)
    env_b = make_env(n, seed=11)
    ctrl_b = make_controller(env_b)
    wrapped = ResidualVecEnv(env_b, ctrl_b, residual_rho=0.2, task="docking")
    reference = _rollout_direct(env_a, ctrl_a, steps)
    zero = torch.zeros(n, 3)
    obs = wrapped.get_observations()
    for t in range(steps):
        obs, _, _, _, _ = wrapped.step(zero)
        assert torch.equal(obs["policy"], reference[t]), f"diverged at step {t}"
    assert wrapped.stats_snapshot()["mean_abs_residual"] == 0.0


def test_controller_advanced_once_and_observation_timing():
    env = make_env(6, seed=3)
    spy = SpyController(make_controller(env))
    w = ResidualVecEnv(env, spy, residual_rho=0.1, task="docking")
    w.get_observations()
    for _ in range(5):
        cur = w.get_observations()["policy"].sum().item()
        w.step(torch.zeros(6, 3))
        assert spy.last_obs_digest == pytest.approx(cur)
    assert spy.calls == 5
    assert spy.observe_calls == 5
    assert spy.resets == 6  # constructor + five per-step resets
    assert int(spy.inner.steps_left.min()) <= 495


def test_actual_applied_action_bookkeeping():
    env = make_env(8, seed=5)
    ctrl = make_controller(env)
    w = ResidualVecEnv(env, ctrl, residual_rho=0.2, task="docking")
    ctrl.reset_idx(torch.ones(8, dtype=torch.bool, device=env.device))
    u = torch.ones(8, 3)  # large positive residual, saturating on some channels
    w.step(u)
    applied = w.last_applied.to(torch.float64)
    # The controller integrates its NOMINAL design model, not the randomized
    # per-env plant constants (env.f_max is a (num_envs,) tensor under DR).
    design = env.controller_design
    expected = (
        applied.norm(dim=-1) * float(design["f_max"]) * float(design["dt"])
        / (float(design["isp"]) * float(design["g0"]))
    )
    assert torch.allclose(ctrl.fuel_est, expected, rtol=1e-6, atol=1e-12)
    assert w.stats_snapshot()["mean_abs_residual"] > 0


def test_partial_reset_resets_controller_state():
    env = make_env(6, seed=9)
    ctrl = make_controller(env)
    w = ResidualVecEnv(env, ctrl, residual_rho=0.05, task="docking")
    done_seen = None
    for _ in range(400):
        obs, _, term, trunc, _ = w.step(torch.zeros(6, 3))
        done = term | trunc
        if bool(done.any()):
            done_seen = done
            break
    assert done_seen is not None
    assert torch.all(ctrl.steps_left[done_seen] == ctrl.max_steps)
    assert torch.all(ctrl.fuel_est[done_seen] == 0)
    assert torch.all(ctrl.steps_left[~done_seen] < ctrl.max_steps)


def test_terminal_observation_and_timeout():
    env = make_env(2, seed=13)
    ctrl = make_controller(env)
    obs = env.get_observations()
    ctrl.reset_idx(torch.ones(2, dtype=torch.bool, device=env.device))
    zero = torch.zeros(2, 3)
    extras = None
    for _ in range(env.max_steps):
        obs, _, term, trunc, extras = env.step(zero)
        ctrl.reset_idx(term | trunc)
        if bool((term | trunc).any()):
            break
    done = term | trunc
    assert bool(done.any())
    term_obs = extras["terminal_observations"]["policy"]
    assert torch.isfinite(term_obs).all()
    # terminal observation is the PRE-reset state, not the next episode's init
    for i in range(2):
        assert torch.equal(term_obs[i], extras["terminal_observations"]["policy"][i])
    if bool(trunc.any()):
        assert torch.isfinite(extras["time_outs"].to(torch.float32)).all()
    assert not torch.allclose(term_obs[done], obs["policy"][done])


def test_dataset_episode_split_and_no_leakage():
    env = make_env(4, seed=17)
    ds = collect_expert_dataset(
        env, episodes_per_replica=2, controller=make_controller(env), seed=17
    )
    splits = split_dataset(ds, seed=17)
    ids = {k: {(ep["replica"], ep["episode_id"]) for ep in v} for k, v in splits.items()}
    assert ids["train"] and ids["val"] and not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    for name, eps in splits.items():
        pairs = flatten_split(ds, eps)
        assert pairs["done"][-1].item() is True
        # every flattened sequence ends at its true episode boundary
        firsts = torch.nonzero(pairs["t_in_episode"] == 0).flatten().tolist()
        lasts = torch.nonzero(pairs["done"]).flatten().tolist()
        assert len(firsts) == len(lasts) == len(eps)
        for a, b in zip(firsts, lasts):
            assert pairs["episode_id"][b] == pairs["episode_id"][a]
        assert float(pairs["rew"].sum()) == pytest.approx(
            sum(ep["episode_return"] for ep in eps), rel=1e-6
        )


def test_returns_to_go_boundary_semantics():
    rews = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[False], [False], [True]])
    term = torch.tensor([[False], [False], [False]])
    trunc = torch.tensor([[False], [False], [True]])
    legacy = returns_to_go(rews, dones, 0.9)
    assert legacy[:, 0].tolist() == pytest.approx([2.71, 1.9, 1.0])
    targets, valid = returns_to_go(rews, dones, 0.9, terminated=term, truncated=trunc)
    assert valid[:, 0].tolist() == [False, False, False]
    boot = torch.tensor([[10.0], [10.0], [10.0]])
    targets_b, valid_b = returns_to_go(
        rews, dones, 0.9, terminated=term, truncated=trunc, bootstrap_values=boot
    )
    assert valid_b.all()
    assert targets_b[2, 0].item() == pytest.approx(1.0 + 0.9 * 10.0)
    # success is terminal, not a timeout: zero terminal value is valid
    succ = torch.tensor([[False], [False], [True]])
    _, valid_s = returns_to_go(
        rews, dones, 0.9, terminated=succ, truncated=torch.zeros(3, 1, dtype=torch.bool)
    )
    assert valid_s.all()


def test_normalizer_checkpoint_roundtrip():
    env = make_env(4, seed=19)
    cfg = build_cfg(learning_rate=3e-4, gamma=0.999)
    runner = OnPolicyRunner(
        RslVecEnv(env), copy.deepcopy(cfg), log_dir="/tmp/cp_test_tb", device="cpu"
    )
    obs = torch.rand(64, 1, 6)
    acts = torch.rand(64, 1, 3) * 2 - 1
    fit_behavior_cloning(
        runner,
        obs,
        acts,
        torch.zeros(64, 1),
        torch.ones(64, 1, dtype=torch.bool),
        epochs=3,
        batch_size=32,
        lr=1e-3,
        gamma=0.999,
        fit_critic=False,
        log=lambda *a, **k: None,
        normalizer_mode="fit_once",
    )
    probe = torch.rand(10, 1, 6)
    before = runner.alg.actor({"policy": probe.reshape(10, 6)}).clone()
    path = "/tmp/cp_test_bc.pt"
    _save_runner(runner, path)
    runner2 = OnPolicyRunner(
        RslVecEnv(make_env(4, seed=19)),
        copy.deepcopy(cfg),
        log_dir="/tmp/cp_test_tb2",
        device="cpu",
    )
    runner2.load(path)
    after = runner2.alg.actor({"policy": probe.reshape(10, 6)})
    assert torch.allclose(before, after, atol=1e-6)


def test_frozen_bc_base_zero_residual_reproduces_policy_trajectory():
    """Frozen-BC residual arm: an untrained (zero) residual must replay the base
    policy bit-for-bit, exactly like the analytical-controller case."""
    from algorithms.residual.residual_env import PolicyBaseController

    n, steps = 4, 30
    env_ref = make_env(n, seed=33)
    env_res = make_env(n, seed=33)
    cfg = build_cfg(learning_rate=3e-4, gamma=0.999, init_std=0.5)
    runner = OnPolicyRunner(
        RslVecEnv(env_ref),
        copy.deepcopy(cfg),
        log_dir="/tmp/cp_test_frozen_tb",
        device="cpu",
    )
    base = PolicyBaseController(
        runner.alg.actor.eval(),
        device="cpu",
        action_bound=float(env_ref.clip_actions),
        name="frozen_bc",
    )
    wrapped = ResidualVecEnv(env_res, base, residual_rho=0.05, task="docking")
    reference = _rollout_direct(env_ref, base, steps)
    zero = torch.zeros(n, 3)
    obs = wrapped.get_observations()
    for t in range(steps):
        obs, _, _, _, _ = wrapped.step(zero)
        assert torch.equal(obs["policy"], reference[t]), f"diverged at step {t}"
    assert wrapped.stats_snapshot()["mean_abs_residual"] == 0.0


def test_residual_checkpoint_rebuild():
    env = make_env(4, seed=21)
    cfg = build_cfg(learning_rate=3e-4, gamma=0.999, init_std=0.5)
    ctrl = make_controller(env)
    wrapped = ResidualVecEnv(env, ctrl, residual_rho=0.05, task="docking")
    runner = OnPolicyRunner(
        RslVecEnv(wrapped), copy.deepcopy(cfg), log_dir="/tmp/cp_test_tb3", device="cpu"
    )
    zero_init_policy_output(runner)
    probe = runner.alg.actor({"policy": wrapped.get_observations()["policy"]})
    assert float(probe.abs().max()) == pytest.approx(0.0, abs=1e-6)
    path = "/tmp/cp_test_res.pt"
    _save_runner(runner, path)
    runner2 = OnPolicyRunner(
        RslVecEnv(make_env(4, seed=21)),
        copy.deepcopy(cfg),
        log_dir="/tmp/cp_test_tb4",
        device="cpu",
    )
    runner2.load(path)
    out2 = runner2.alg.actor({"policy": torch.zeros(4, 6)})
    assert float(out2.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_model_mismatch_design_is_pinned_not_truth():
    nominal = make_env(2, seed=23)
    shifted = make_env(2, seed=23, plant_kwargs={"mass": 550.0, "f_max": 9.0, "isp": 250.0})
    c_nom = make_controller(nominal)
    c_shift_pinned = make_controller(shifted, design={"mass": 500.0, "f_max": 10.0, "isp": 300.0})
    c_shift_lazy = make_controller(shifted)
    assert c_shift_pinned.mass == 500.0 and c_shift_pinned.f_max == 10.0
    assert c_shift_lazy.mass == 550.0 and c_shift_lazy.f_max == 9.0  # would leak the truth
    assert c_nom.mass == c_shift_pinned.mass


def test_satellite_controller_actions_with_scalar_gains():
    """PD gains are design scalars; thresholds must still broadcast per env."""
    from environments.satellite.attitude_control import SatelliteAttitudeEnv

    env = SatelliteAttitudeEnv(num_envs=4, backend="cpu", seed=47)
    ctrl = build_baseline("satellite", env, design={"kp": 3.0, "kd": 4.0})
    obs = env.get_observations()
    with torch.no_grad():
        act = ctrl.actions(obs)
    assert act.shape == (4, env.num_actions)
    assert torch.isfinite(act).all()


def test_nonfinite_policy_action_raises():
    env = make_env(2, seed=25)
    w = ResidualVecEnv(env, make_controller(env), residual_rho=0.05, task="docking")
    bad = torch.zeros(2, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        w.step(bad)


def test_first_episode_accounting():
    env = make_env(6, seed=27)
    ctrl = make_controller(env)
    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    ctrl.reset_idx(torch.ones(6, dtype=torch.bool, device=env.device))
    for t in range(600):
        with torch.no_grad():
            a = ctrl.actions(obs)
        obs, _, term, trunc, _ = env.step(a)
        ctrl.reset_idx(term | trunc)
        cohort.update(env)
        if (t + 1) % 32 == 0 and cohort.finished():
            break
    rep = cohort.report()
    assert rep["protocol"] == "first_episode_per_replica"
    assert rep["complete"] and rep["episodes_finished"] == 6
    assert rep["successes"] == int(env.completed_successes.sum())


# --------------------------------------------------------------------------
# improvement-ablation mechanisms
# --------------------------------------------------------------------------
def test_phase_weights_upweights_terminal_pairs_and_normalizes():
    obs = torch.zeros(4, 6)
    obs[0, :3] = 10.0  # far
    obs[1, :3] = 0.1  # near
    obs[2, :3] = 0.2  # near
    obs[3, :3] = 8.0  # far
    w = phase_weights(obs, {"index": [0, 1, 2], "near": 1.0, "weight": 3.0, "far_weight": 1.0})
    assert w[1] > w[0] and w[2] > w[3]
    assert float(w.mean()) == pytest.approx(1.0)
    # relative weighting only: without normalization the terminal pairs are 3x
    w_raw = phase_weights(
        obs, {"index": [0, 1, 2], "near": 1.0, "weight": 3.0, "far_weight": 1.0, "normalize": False}
    )
    assert w_raw.tolist() == pytest.approx([1.0, 3.0, 3.0, 1.0])


def test_weighted_bc_summary_and_finite_loss():
    env = make_env(4, seed=29)
    cfg = build_cfg(learning_rate=3e-4, gamma=0.999)
    runner = OnPolicyRunner(
        RslVecEnv(env), copy.deepcopy(cfg), log_dir="/tmp/cp_test_tb5", device="cpu"
    )
    obs = torch.rand(32, 1, 6)
    acts = torch.rand(32, 1, 3) * 2 - 1
    weights = phase_weights(obs, {"index": [0, 1, 2], "near": 0.5})
    summary = fit_behavior_cloning(
        runner,
        obs,
        acts,
        torch.zeros(32, 1),
        torch.ones(32, 1, dtype=torch.bool),
        epochs=2,
        batch_size=16,
        lr=1e-3,
        gamma=0.999,
        fit_critic=False,
        log=lambda *a, **k: None,
        sample_weights=weights,
    )
    assert summary["sample_weighting"] is True
    assert summary["sample_weight_mean"] == pytest.approx(1.0)
    assert summary["final_actor_mse"] == pytest.approx(summary["final_train_actor_mse"])


def _fake_dataset(T, episode_id, replica=0, obs_dim=6, act_dim=3):
    data = {
        "obs": torch.zeros(T, 1, obs_dim),
        "act": torch.zeros(T, 1, act_dim),
        "rew": torch.zeros(T, 1, dtype=torch.float64),
        "terminated": torch.zeros(T, 1, dtype=torch.bool),
        "truncated": torch.zeros(T, 1, dtype=torch.bool),
        "done": torch.zeros(T, 1, dtype=torch.bool),
        "success": torch.zeros(T, 1, dtype=torch.bool),
        "episode_id": torch.full((T, 1), episode_id, dtype=torch.long),
        "t_in_episode": torch.arange(T).reshape(T, 1),
        "terminal_obs": torch.zeros(T, 1, obs_dim),
    }
    ep = dict(
        replica=replica,
        episode_id=episode_id,
        start=0,
        end=T - 1,
        complete=True,
        terminated=False,
        truncated=False,
        success=False,
        cut=False,
        steps=T,
        episode_return=0.0,
        initial_obs=[0.0] * obs_dim,
        terminal_obs=[0.0] * obs_dim,
    )
    return dict(
        data=data,
        episodes=[ep],
        meta=dict(
            controller_name="fake",
            collection=dict(
                steps_used=T,
                transitions=T,
                episodes_done=1,
                complete_episodes=1,
                collection_wall_seconds=1.0,
            ),
            dagger=None,
        ),
    )


def _multi_replica_dataset(T=2000, replicas=8, obs_dim=16, act_dim=4):
    done = torch.zeros(T, replicas, dtype=torch.bool)
    done[-1] = True
    data = {
        "obs": torch.rand(T, replicas, obs_dim) * 0.5,
        "act": torch.rand(T, replicas, act_dim) * 0.5,
        "rew": torch.zeros(T, replicas, dtype=torch.float64),
        "terminated": torch.zeros(T, replicas, dtype=torch.bool),
        "truncated": done.clone(),
        "done": done.clone(),
        "success": torch.zeros(T, replicas, dtype=torch.bool),
        "episode_id": torch.zeros(T, replicas, dtype=torch.long),
        "t_in_episode": torch.arange(T).reshape(T, 1).expand(T, replicas),
    }
    episodes = [
        dict(replica=i, episode_id=0, start=0, end=T - 1, complete=True,
             terminated=False, truncated=True, success=False, cut=False, steps=T,
             episode_return=0.0, initial_obs=[0.0] * obs_dim,
             terminal_obs=[0.0] * obs_dim)
        for i in range(replicas)
    ]
    meta = dict(controller_name="fake",
                collection=dict(steps_used=T, transitions=T * replicas,
                                episodes_done=replicas, complete_episodes=replicas,
                                collection_wall_seconds=1.0))
    return dict(data=data, episodes=episodes, meta=meta)


def test_dataset_compact_roundtrip_restores_training_dtypes(tmp_path):
    ds = _multi_replica_dataset()
    saved = save_expert_dataset(ds, tmp_path / "d.pt")
    # compact on disk: half-precision states/actions, no per-step terminal obs
    for k in ("obs", "act"):
        assert saved["data"][k].dtype == torch.float16
    assert saved["data"]["rew"].dtype == torch.float32
    assert saved["data"]["episode_id"].dtype == torch.int16
    assert saved["data"]["t_in_episode"].dtype == torch.int16
    assert "terminal_obs" not in saved["data"]
    assert saved["meta"]["storage"]["within_budget"] is True

    loaded = load_expert_dataset(tmp_path / "d.pt")
    for k in ("obs", "act"):
        assert loaded["data"][k].dtype == torch.float32
        assert torch.allclose(loaded["data"][k], ds["data"][k], atol=2e-3)
    assert loaded["data"]["rew"].dtype == torch.float64
    assert loaded["data"]["episode_id"].dtype == torch.long
    assert loaded["data"]["t_in_episode"].dtype == torch.long
    assert len(loaded["episodes"]) == len(ds["episodes"])
    assert loaded["meta"]["storage"]["encoding"].startswith("float16")


def test_dataset_size_cap_drops_whole_replicas(tmp_path):
    ds = _multi_replica_dataset()
    saved = save_expert_dataset(ds, tmp_path / "capped.pt", max_mb=0.4)
    size_mb = (tmp_path / "capped.pt").stat().st_size / 2 ** 20
    assert size_mb <= 0.4
    keep = int(saved["data"]["obs"].shape[1])
    assert keep < int(ds["data"]["obs"].shape[1])
    assert saved["meta"]["storage"]["within_budget"] is True
    assert saved["meta"]["storage"]["replicas_written"] == keep
    assert saved["meta"]["collection"]["envs"] == keep
    # whole replicas (= whole episodes) only, so the split still works
    assert len(saved["episodes"]) == keep
    assert all(ep["replica"] < keep for ep in saved["episodes"])
    splits = split_dataset(saved, seed=7)
    assert all(len(v) > 0 for v in splits.values())


def _sequence_dataset(n_episodes=200, steps=5, obs_dim=2):
    """One replica holding ``n_episodes`` consecutive complete episodes with
    distinct initial observations (a stand-in for a long env sweep)."""
    T = n_episodes * steps
    obs = torch.zeros(T, 1, obs_dim)
    episode_id = torch.zeros(T, 1, dtype=torch.long)
    t_in_episode = torch.zeros(T, 1, dtype=torch.long)
    done = torch.zeros(T, 1, dtype=torch.bool)
    episodes = []
    for i in range(n_episodes):
        s, e = i * steps, (i + 1) * steps - 1
        episode_id[s:e + 1] = i
        t_in_episode[s:e + 1] = torch.arange(steps).unsqueeze(-1)
        done[e] = True
        obs[s:e + 1] = float(i)  # tags each row with its source episode
        episodes.append(dict(replica=0, episode_id=i, start=s, end=e, complete=True,
                             terminated=False, truncated=True, success=False, cut=False,
                             steps=steps, episode_return=0.0,
                             initial_obs=[i / n_episodes, 1.0 - i / n_episodes],
                             terminal_obs=[0.0] * obs_dim))
    data = dict(obs=obs, act=torch.zeros(T, 1, 1),
                rew=torch.zeros(T, 1, dtype=torch.float64),
                terminated=torch.zeros(T, 1, dtype=torch.bool),
                truncated=done.clone(), done=done,
                success=torch.zeros(T, 1, dtype=torch.bool),
                episode_id=episode_id, t_in_episode=t_in_episode)
    meta = dict(controller_name="fake",
                collection=dict(steps_used=T, transitions=T, episodes_done=n_episodes,
                                complete_episodes=n_episodes, collection_wall_seconds=1.0))
    return dict(data=data, episodes=episodes, meta=meta)


def test_representative_selection_covers_initial_conditions():
    ds = _sequence_dataset()
    sel = select_representative_episodes(ds, 12)
    assert len(sel["episodes"]) == 12
    assert tuple(sel["data"]["obs"].shape) == (5, 12, 2)
    assert sel["meta"]["selection"] == dict(
        method="farthest_point", requested=12, candidates=200, kept=12,
        feature="initial policy observation (z-scored)")
    # farthest-point keeps the extremes of the initial-condition space
    first = [ep["initial_obs"][0] for ep in sel["episodes"]]
    assert min(first) <= 0.02 and max(first) >= 0.98
    # repacked column j holds exactly the selected episode (whole, re-indexed)
    for j, ep in enumerate(sel["episodes"]):
        source = int(round(ep["initial_obs"][0] * 200))
        assert int(sel["data"]["obs"][0, j, 0]) == source
        assert int(sel["data"]["episode_id"][0, j]) == j
        assert int(sel["data"]["t_in_episode"][4, j]) == 4
        assert bool(sel["data"]["done"][4, j])
    pairs = flatten_split(sel, sel["episodes"])
    assert pairs["obs"].shape[0] == 60
    assert all(len(v) > 0 for v in split_dataset(sel, seed=7).values())


def test_representative_selection_noop_when_pool_is_small():
    ds = _sequence_dataset(n_episodes=20)
    assert select_representative_episodes(ds, 50) is ds
    assert select_representative_episodes(ds, 0) is ds


def test_merge_datasets_offsets_and_unique_ids():
    a, b, c = _fake_dataset(3, 0), _fake_dataset(2, 0), _fake_dataset(4, 0)
    merged = merge_datasets([a, b, c])
    assert merged["data"]["obs"].shape[0] == 9
    assert [ep["start"] for ep in merged["episodes"]] == [0, 3, 5]
    assert [ep["end"] for ep in merged["episodes"]] == [2, 4, 8]
    assert [ep["episode_id"] for ep in merged["episodes"]] == [0, 1, 2]
    # the per-step id column must be renumbered to match the episode table
    step_ids = merged["data"]["episode_id"][:, 0]
    assert [int(step_ids[s]) for s in (0, 3, 5)] == [0, 1, 2]
    assert merged["meta"]["collection"]["steps_used"] == 9
    assert merged["meta"]["merged"] is True
    with pytest.raises(ValueError):
        merge_datasets([a, _fake_dataset(2, 0, obs_dim=4)])


def test_delta_action_wrapper_bounded_increments_and_reset():
    env = make_env(2, seed=31)
    w = DeltaActionVecEnv(env, delta_scale=0.25)
    obs = w.get_observations()
    obs, _, _, _, _ = w.step(torch.zeros(2, 3))
    assert torch.equal(w.last_applied, torch.zeros(2, 3))
    obs, _, _, _, _ = w.step(torch.ones(2, 3))
    assert torch.allclose(w.last_applied, torch.full((2, 3), 0.25))
    # saturates at the env action bound instead of growing without limit
    for _ in range(10):
        w.step(torch.ones(2, 3))
    assert float(w.last_applied.max()) == pytest.approx(1.0)
    w.reset()
    assert torch.equal(w.prev_action, torch.zeros(2, 3))
    with pytest.raises(ValueError):
        w.step(torch.full((2, 3), float("nan")))


def test_delta_action_targets_invert_wrapper_map_and_reset_per_episode():
    acts = torch.tensor(
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.35, 0.0, 0.0]]
    )
    t_in_episode = torch.tensor([0, 1, 2, 0, 1])
    targets = delta_action_targets(acts, t_in_episode, 0.25)
    # first step of each episode advances from the zero previous command
    assert torch.allclose(targets[0], torch.tensor([0.8, 0.0, 0.0]))
    assert torch.allclose(targets[1], torch.tensor([0.8, 0.0, 0.0]))
    assert torch.allclose(targets[2], torch.tensor([-1.0, 0.0, 0.0]))  # clipped
    assert torch.allclose(targets[3], torch.tensor([1.0, 0.0, 0.0]))  # clipped
    # replaying d_t through the wrapper reproduces the expert commands
    prev, applied = torch.zeros(3), []
    for i, t in enumerate([0, 1]):
        if int(t_in_episode[i]):
            prev = prev + 0.25 * targets[i]
        else:
            prev = torch.zeros(3) + 0.25 * targets[i]
        applied.append(prev.clone())
    assert torch.allclose(torch.stack(applied)[:, 0], torch.tensor([0.2, 0.4]))
    with pytest.raises(ValueError):
        delta_action_targets(acts, torch.tensor([0, 1, 2, 0]), 0.25)


def test_action_rate_penalty_only_changes_training_reward():
    env_a = make_env(2, seed=33)
    env_b = make_env(2, seed=33)
    w = ActionRatePenaltyVecEnv(env_a, coef=0.01)
    action = torch.tensor([[0.5, -0.5, 0.2], [0.1, 0.1, 0.1]])
    _, rew_wrapped, term_a, trunc_a, _ = w.step(action)
    _, rew_plain, term_b, trunc_b, _ = env_b.step(action)
    penalty = 0.01 * (action - torch.zeros(2, 3)).pow(2).sum(dim=-1)
    assert torch.allclose(rew_wrapped, rew_plain - penalty, atol=1e-6)
    assert torch.equal(term_a, term_b) and torch.equal(trunc_a, trunc_b)
    assert torch.equal(w.last_env_reward, rew_plain)
    # the same action twice has zero rate penalty (state evolves, so only the
    # penalty term is asserted here)
    w.step(action)
    assert torch.equal(w.last_penalty, torch.zeros(2))


def test_per_channel_rho_and_phase_gate_authority():
    env = make_env(2, seed=35)
    ctrl = make_controller(env)
    w = ResidualVecEnv(
        env,
        ctrl,
        residual_rho={
            "channels": [0.05, 0.05, 0.1],
            "gate": {"index": [0, 1, 2], "near": 20.0, "soft": 5.0, "gain": 1.0},
        },
        task="docking",
    )
    assert w.residual_limit.tolist() == pytest.approx([0.1, 0.1, 0.2])
    assert w.physics_limits()["residual_limit_thrust_N"] == pytest.approx([1.0, 1.0, 2.0])
    obs = w.get_observations()
    far = {"policy": obs["policy"].clone()}
    far["policy"][:, :3] = 100.0
    w._obs = far
    scale_far = w._authority_scale(torch.float32)  # (num_envs, num_actions)
    assert float(scale_far[0, 0]) == pytest.approx(1.0, abs=1e-2)
    near = {"policy": obs["policy"].clone()}
    near["policy"][:, :3] = 0.0
    w._obs = near
    scale_near = w._authority_scale(torch.float32)
    assert float(scale_near[0, 0]) > 1.5  # near-target authority is doubled
    assert w.stats_snapshot()["phase_gate"]["near"] == 20.0


def test_authority_curriculum_raises_only_after_success():
    env = make_env(4, seed=37)
    w = ResidualVecEnv(env, make_controller(env), residual_rho=0.1, task="docking")
    hook = success_gated_authority_curriculum(
        start=0.01,
        end=1.0,
        step=0.1,
        success_threshold=0.9,
        min_episodes=8,
        log=lambda *a, **k: None,
    )
    env.completed_episodes.fill_(10)
    env.completed_successes.fill_(10)
    info = hook(None, w, 1, 10)
    assert info["gate_met"] is True
    assert info["authority_fraction"] == pytest.approx(0.11)
    assert w.authority_fraction == pytest.approx(0.11)
    env.completed_successes.zero_()
    info2 = hook(None, w, 2, 10)
    assert info2["gate_met"] is False
    assert info2["authority_fraction"] == pytest.approx(0.11)
    # the residual authority really is scaled down at the start of training
    w.set_authority_fraction(0.01)
    w.get_observations()
    w.step(torch.ones(4, 3))
    st = w.stats_snapshot()
    assert st["authority_fraction"] == pytest.approx(0.01)
    assert float(w.last_residual.abs().max()) < 0.05


def test_dagger_labels_are_controller_actions_and_beta_one_is_pure_expert():
    def collect(env, policy, beta):
        return collect_expert_dataset(
            env,
            episodes_per_replica=1,
            controller=None,
            policy=policy,
            beta=beta,
            seed=41,
            max_steps=200,
        )

    env_a = make_env(2, seed=41)
    env_b = make_env(2, seed=41)

    def zero(obs):
        return torch.zeros(2, 3, device=obs["policy"].device)

    expert = collect(env_a, None, 0.0)
    beta_one = collect(env_b, zero, 1.0)
    assert torch.equal(expert["data"]["obs"], beta_one["data"]["obs"])
    assert torch.equal(expert["data"]["act"], beta_one["data"]["act"])
    assert beta_one["meta"]["dagger"]["beta"] == 1.0
    assert expert["meta"]["dagger"] is None


def test_checkpoint_ranking_prefers_success_then_error_then_return():
    rows = [
        {
            "name": "a",
            "success_rate": 0.5,
            "complete": True,
            "terminal_range_m": 0.5,
            "mean_episode_return": 10,
        },
        {
            "name": "b",
            "success_rate": 0.8,
            "complete": False,
            "terminal_range_m": 1.0,
            "mean_episode_return": 20,
        },
        {
            "name": "c",
            "success_rate": 0.8,
            "complete": True,
            "terminal_range_m": 0.2,
            "mean_episode_return": 5,
        },
        {
            "name": "d",
            "success_rate": 0.8,
            "complete": True,
            "terminal_range_m": 0.2,
            "mean_episode_return": 30,
        },
    ]
    ranked = rank_candidates(rows, "terminal_range_m")
    assert [r["name"] for r in ranked] == ["d", "c", "b", "a"]
    # a missing terminal metric never wins a tie on success+completeness
    ranked_none = rank_candidates(
        [{**rows[2], "terminal_range_m": None}, rows[3]], "terminal_range_m"
    )
    assert ranked_none[0]["name"] == "d"


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    vals = [0.2, 0.5, 0.8, 0.6, 0.9]
    lo, hi = bootstrap_ci(vals, seed=7)
    lo2, hi2 = bootstrap_ci(vals, seed=7)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= statistics_mean(vals) <= hi
    assert bootstrap_ci([], seed=7) == (None, None)
    assert bootstrap_ci([0.5], seed=7) == (0.5, 0.5)


def statistics_mean(values):
    return sum(values) / len(values)


def test_aggregate_reads_stage_best_eval_as_fallback(tmp_path):
    """Selection arms evaluate the chosen checkpoint and only write
    ``*_stage_best.json``; the aggregator must not mark them unevaluated."""
    import json

    from analysis.aggregate import collect

    run = tmp_path / "docking" / "bc_ppo" / "improve_selection" / "seed_7__dag"
    (run / "eval").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"updates": 250}))
    for mode in ("det", "stoch"):
        (run / "eval" / f"{mode}_stage_best.json").write_text(
            json.dumps(
                {
                    "stage": "best",
                    "updates": None,
                    "success_rate": 1.0,
                    "complete": True,
                    "initial_condition_digest": "abc123",
                }
            )
        )
    rows, _, _ = collect(tmp_path, tag="dag")
    modes = {r["mode"]: r for r in rows}
    assert set(modes) == {"det", "stoch"}
    for r in modes.values():
        assert r["stage"] == "best"
        assert r["success_rate"] == 1.0
        assert r["complete"] is True
    assert not any(r["mode"] == "not_evaluated" for r in rows)


def test_kl_anchor_ppo_wraps_ppo_and_decays_beta():
    env = make_env(4, seed=39)
    cfg = build_cfg(
        learning_rate=3e-4,
        gamma=0.999,
        init_std=0.5,
        num_steps_per_env=8,
        num_learning_epochs=1,
        num_mini_batches=1,
        kl_anchor_coef=0.05,
        kl_anchor_decay_updates=4,
        lr_schedule="fixed",
    )
    assert cfg["algorithm"]["class_name"] == "algorithms.ppo.kl_anchor:KLAnchorPPO"
    runner = OnPolicyRunner(
        RslVecEnv(env), copy.deepcopy(cfg), log_dir="/tmp/cp_test_tb6", device="cpu"
    )
    assert isinstance(runner.alg, KLAnchorPPO)
    runner.alg.set_bc_anchor(coef=0.05, decay_updates=4)
    assert runner.alg.bc_anchor is not None
    assert not any(p.requires_grad for p in runner.alg.bc_anchor.parameters())
    runner.learn(2)
    assert runner.alg._anchor_calls == 2
    assert runner.alg._anchor_coef() == pytest.approx(0.05 * (1 - 2 / 4))
