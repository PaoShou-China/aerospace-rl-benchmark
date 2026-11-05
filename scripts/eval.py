"""Deterministic evaluation of a trained RSL-RL checkpoint on the batched GPU envs.

Usage: python eval.py quadx|quadx_track|fixedwing|rocket|missile|quadplane|
       satellite|transfer|docking|rendezvous
Defaults: 1024 envs on GPU, seed 7, frozen deterministic policy (mean action,
no exploration noise). The env configuration used at training time
(env_kwargs) is restored from results/<task>/config.json.

Unbiased episode accounting: the loop runs until EVERY replica has completed
at least one full episode (safety-capped by --max-steps), so no in-flight
episode is dropped and a zero-success run cannot be an artifact of a too-short
window (fixed-wing episodes last up to 3600 steps).
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

import torch  # noqa: E402
from scripts.first_episode import FirstEpisodes  # noqa: E402

from algorithms.train_utils import RslVecEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from environments.multirotor import QuadXWaypointsEnv, QuadXTrackEnv  # noqa: E402
from environments.fixedwing import FixedwingWaypointsEnv  # noqa: E402
from environments.rocket import RocketLandingEnv  # noqa: E402
from environments.missile import MissileInterceptEnv  # noqa: E402
from environments.vtol.env import QuadplaneWaypointsEnv  # noqa: E402
from environments.satellite.attitude_control import SatelliteAttitudeEnv  # noqa: E402
from environments.satellite.orbit_transfer import LowThrustTransferEnv  # noqa: E402
from environments.satellite.rendezvous_docking import (  # noqa: E402
    SpaceDockingEnv,
    SpaceRendezvousEnv,
)

ENVS = {
    "quadx": QuadXWaypointsEnv,
    "quadx_track": QuadXTrackEnv,
    "fixedwing": FixedwingWaypointsEnv,
    "rocket": RocketLandingEnv,
    "missile": MissileInterceptEnv,
    "quadplane": QuadplaneWaypointsEnv,
    "satellite": SatelliteAttitudeEnv,
    "transfer": LowThrustTransferEnv,
    "docking": SpaceDockingEnv,
    "rendezvous": SpaceRendezvousEnv,
}


def _eval_wrapper(cfg, vehicle):
    """Wrapper recorded in config.json, or None for a plain env-action run.

    A gain-scheduling run's actor expects ``base_obs + n_gains`` and emits gain
    settings, not env actions, so evaluation MUST rebuild the same wrapper as
    training or the checkpoint cannot even be loaded.
    """
    meta = cfg.get("experiment_meta") or {}
    if meta.get("method") != "gain_scheduling":
        return None
    from experiments.gainsched.train_gainsched import make_gain_sched_wrapper

    return make_gain_sched_wrapper(
        vehicle,
        float(meta.get("gain_scale_frac", 1.0)),
        float(meta.get("smoothing", 0.0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vehicle", choices=list(ENVS))
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="graded check: fixed small validation cohort (64 envs, same seed "
        "and same completion criterion) writing eval_quick.json; use it during "
        "training and run the full default evaluation once at the end",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="safety cap on env steps while every replica finishes one episode",
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--performance", action="store_true",
                        help="use a frozen CUDA policy snapshot and supported environment optimizations")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="training dir holding checkpoint.pt (default results/<vehicle>)",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="report filename written into --out (default eval.json; --quick "
        "defaults to eval_quick.json so the graded check never overwrites a "
        "full report)",
    )
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 64 if args.quick else 1024
    if args.report_name is None:
        args.report_name = "eval_quick.json" if args.quick else "eval.json"

    out = args.out or Path(__file__).resolve().parent.parent / "results" / args.vehicle
    ckpt = out / "checkpoint.pt"
    assert ckpt.exists(), f"missing {ckpt}"
    cfg = json.loads((out / "config.json").read_text())
    # Per-dim exploration std is dumped as a JSON list (train_rocket's _JsonStd
    # keeps json.dumps working). rsl_rl builds the distribution as
    # `init_std * torch.ones(output_dim)`, which needs tensor semantics: a plain
    # list raises "only integer tensors of a single element can be converted to
    # an index" when the runner is reconstructed. Restore the tensor form so a
    # saved config is reloadable exactly as it was at training time.
    init_std = cfg.get("actor", {}).get("distribution_cfg", {}).get("init_std")
    if isinstance(init_std, list):
        cfg["actor"]["distribution_cfg"]["init_std"] = torch.tensor(init_std)

    torch.manual_seed(args.seed)
    _setup_t0 = time.perf_counter()
    # restore the exact env construction used at training time
    env_kwargs = dict(cfg.get("env_kwargs") or {})
    env = ENVS[args.vehicle](
        num_envs=args.num_envs, backend=args.backend, seed=args.seed, **env_kwargs
    )
    wrapper = _eval_wrapper(cfg, args.vehicle)
    if wrapper is not None:
        env = wrapper(env)
    residual_cfg = cfg.get("residual") or {}
    if residual_cfg.get("controller"):
        # Residual-RL run: rebuild the SAME wrapper used at training -- the
        # checkpoint's actor emits the residual variable u_t, the classical
        # base controller composes a_t = a0 + rho*tanh(u_t) inside the wrapper.
        from algorithms.baselines.baselines import build_baseline
        from algorithms.residual.residual_env import ResidualVecEnv

        controller = build_baseline(
            args.vehicle,
            env,
            residual_cfg.get("variant"),
            design=residual_cfg.get("controller_design"),
        )
        env = ResidualVecEnv(
            env,
            controller,
            residual_rho=float(cfg.get("residual_rho", 0.05)),
            task=args.vehicle,
        )
    if cfg.get("algo") == "flash_sac":
        raise RuntimeError(
            "flash_sac checkpoints are no longer supported (algorithm removed); "
            f"re-train with PPO to evaluate {ckpt}"
        )
    runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir="/tmp/eval_tb", device=str(env.device))
    runner.load(str(ckpt))
    actor = runner.alg.actor.eval()
    policy = actor
    performance_paths = []
    if args.performance:
        from algorithms.inference import InferencePolicy
        from algorithms.performance import enable_environment

        policy = InferencePolicy(actor)
        performance_paths.append(policy.backend + "_inference")
        if torch.device(env.device).type == "cuda" and type(env) in ENVS.values():
            performance_paths += enable_environment(env, compile_step=False, compile_aerodynamics=False)

    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    setup_seconds = time.perf_counter() - _setup_t0
    start = time.perf_counter()
    steps = 0
    with torch.no_grad():
        while steps < args.max_steps:
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            steps += 1
            cohort.update(env)
            # Completion is checked every 32 steps: the per-step stats above
            # stay device-side, and bool(seen.all()) is the only host sync.
            if steps % 32 == 0 and cohort.finished():
                break
    wall = time.perf_counter() - start

    report = {
        "task": args.vehicle,
        "checkpoint": str(ckpt),
        "env_steps": steps,
        "num_envs": args.num_envs,
        "env_kwargs": env_kwargs,
        "quick": bool(args.quick),
        "setup_seconds": setup_seconds,
        "eval_wall_seconds": wall,
        "performance_paths": performance_paths,
    }
    if cfg.get("experiment_meta"):
        report["experiment_meta"] = cfg["experiment_meta"]
    if hasattr(env, "stats_snapshot"):
        report["gain_schedule_stats"] = env.stats_snapshot()
    report.update(cohort.report())
    report["benchmark"] = env.cfg.get("benchmark")
    # per-first-episode mean task metrics (touch quality etc.), from the cohort
    n = max(report["episodes_finished"], 1)
    report["task_metrics"] = {k: v / n for k, v in report["metric_counts"].items()}
    # global counters (quadx/fixedwing) cover every episode in the window
    m = max(report["episodes_completed_total"], 1)
    report["task_metrics"].update({k: v / m for k, v in report["global_metric_counts"].items()})
    print(json.dumps(report, indent=2), flush=True)
    (out / args.report_name).write_text(json.dumps(report, indent=2))
    env.close()
    if not report["complete"]:
        raise RuntimeError("Incomplete first-episode cohort; partial report saved")


if __name__ == "__main__":
    main()
