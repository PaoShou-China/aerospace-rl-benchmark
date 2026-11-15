"""First-episode deterministic / stochastic evaluation for controller-prior arms.

Protocol (reused from scripts/first_episode.py):
- each replica contributes exactly its FIRST complete episode; the loop runs until
  every replica has finished one (safety-capped by --max-steps);
- terminal observations are read from extras BEFORE the automatic reset;
- incomplete cohorts are saved as partial reports and explicitly marked;
- scalar global metrics keep their run-wide meaning (FirstEpisodes.report);
- the initial observation digest is stored so paired initialization across arms
  can be verified in the analysis.

For the residual arm the controller lives outside the env (ResidualVecEnv).  The
same raw policy observation is used for every arm (no nominal-action augmentation),
so the comparison stays on one observation contract.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from algorithms.train_utils import RslVecEnv  # noqa: E402
from algorithms.residual.residual_env import ResidualVecEnv  # noqa: E402
from algorithms.residual.action_space import DeltaActionVecEnv  # noqa: E402
from scripts.first_episode import FirstEpisodes  # noqa: E402
from analysis.task_metrics import (  # noqa: E402
    init_accumulators,
    accumulate,
    finalize,
    PROFILE_NOTE,
)
from scripts.common import (  # noqa: E402
    run_dir,
    make_env,
    make_controller,
    load_frozen_bc_policy,
    write_json,
    task_config,
    digest_tensor,
)
from scripts import conditions  # noqa: E402

LEARNED = (
    "bc_only",
    "ppo_scratch",
    "bc_ppo",
    "residual_ppo",
    "frozen_bc_residual_ppo",
    "bc_dagger",
    "bc_dagger_ppo",
)


def _load_eval_config(out, method, task, cond, cfg):
    ec = None
    p = Path(out) / "eval_config.json"
    if p.exists():
        ec = json.loads(p.read_text())
    ec = dict(ec or {})
    env_kwargs = ec.get("eval_env_kwargs")
    if env_kwargs is None:
        env_kwargs = conditions.eval_env_kwargs(task, cond, cfg)
    return {
        "eval_env_kwargs": dict(env_kwargs or {}),
        "controller_design": ec.get("controller_design", cond.get("controller_design")),
        "controller_kwargs": ec.get("controller_kwargs", cond.get("controller_kwargs") or {}),
        "baseline_variant": ec.get("baseline_variant", cond.get("baseline_variant")),
        "residual_rho": ec.get("residual_rho", cond["residual_rho"]),
        "residual_squash": ec.get("residual_squash", cond["residual_squash"]),
        "action_mode": ec.get("action_mode", "absolute"),
        "delta_scale": ec.get("delta_scale", 0.25),
        "phase_gate": ec.get("phase_gate"),
        "bc_checkpoint": ec.get("bc_checkpoint"),
        "authority_fraction": ec.get("authority_fraction", 1.0),
        "action_semantics": ec.get("action_semantics", cfg.get("action_semantics")),
        "dump_replicas": ec.get("eval", {}).get(
            "dump_replicas", cfg["eval"].get("dump_replicas", [])
        ),
        "max_steps": ec.get("eval", {}).get("max_steps", cfg["eval"].get("max_steps", 3000)),
    }


def _make_eval_env(task, cfg, ev, *, num_envs, backend, eval_seed, method, cond):
    env = make_env(
        task,
        cfg,
        num_envs=num_envs,
        seed=eval_seed,
        backend=backend,
        env_kwargs=ev["eval_env_kwargs"],
    )
    if ev.get("action_mode") == "delta":
        if method in ("residual_ppo", "frozen_bc_residual_ppo"):
            raise ValueError(
                "delta actions compose with the raw env; residual arms "
                "already provide bounded increments (mirrors train_arm)"
            )
        # delta parameterization is part of the action semantics, not an
        # auxiliary loss: evaluation must use the identical wrapper.
        env = DeltaActionVecEnv(env, delta_scale=float(ev.get("delta_scale", 0.25)))
    if method in ("residual_ppo", "frozen_bc_residual_ppo"):
        if method == "residual_ppo":
            cond_for_controller = dict(cond)
            cond_for_controller["controller_design"] = ev["controller_design"]
            cond_for_controller["controller_kwargs"] = ev["controller_kwargs"]
            cond_for_controller["baseline_variant"] = ev["baseline_variant"]
            controller = make_controller(task, env, cond_for_controller)
        else:
            bc_ckpt = ev.get("bc_checkpoint")
            if not bc_ckpt:
                raise FileNotFoundError(
                    "frozen_bc_residual_ppo evaluation needs the BC base "
                    "checkpoint recorded in eval_config.json"
                )
            controller = load_frozen_bc_policy(bc_ckpt, env, name=f"frozen_bc:{task}")
        env = ResidualVecEnv(
            env,
            controller,
            variant=ev["baseline_variant"],
            residual_rho=ev["residual_rho"],
            residual_squash=ev["residual_squash"],
            task=task,
            controller_design=ev["controller_design"],
            teacher_identity=getattr(controller, "name", type(controller).__name__),
            name="residual_eval",
        )
        if ev.get("authority_fraction") is not None:
            env.set_authority_fraction(float(ev["authority_fraction"]))
        return env, controller
    controller = None
    if method == "traditional":
        cond_for_controller = dict(cond)
        cond_for_controller["controller_design"] = ev["controller_design"]
        cond_for_controller["controller_kwargs"] = ev["controller_kwargs"]
        cond_for_controller["baseline_variant"] = ev["baseline_variant"]
        controller = make_controller(task, env, cond_for_controller)
    return env, controller


def _load_actor(env, checkpoint):
    cfg_path = Path(checkpoint).parent / "config.json"
    if not cfg_path.exists():
        # selection/best_checkpoint.pt lives one level below the seed dir, whose
        # config.json describes the network shape it was copied from
        cfg_path = Path(checkpoint).parent.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json next to {checkpoint}")
    cfg = json.loads(cfg_path.read_text())
    init_std = cfg.get("actor", {}).get("distribution_cfg", {}).get("init_std")
    if isinstance(init_std, list):
        cfg["actor"]["distribution_cfg"]["init_std"] = torch.tensor(init_std)
    runner = OnPolicyRunner(
        RslVecEnv(env), copy.deepcopy(cfg), log_dir="/tmp/cp_eval_tb", device=str(env.device)
    )
    runner.load(str(checkpoint))
    actor = runner.alg.actor.eval()
    return actor, cfg


def _rollout(
    task, cfg, env, *, method, actor, controller, deterministic, eval_seed, max_steps, dump_replicas
):
    n = int(env.num_envs)
    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    init_digest = digest_tensor(obs["policy"])
    acc = init_accumulators(task, env, n)
    dt_control = (1.0 / float(env.agent_hz)) if hasattr(env, "agent_hz") else float(env.dt)
    first_steps = torch.zeros(n, device=env.device, dtype=torch.long)
    first_terminal = torch.zeros(n, device=env.device, dtype=torch.bool)
    perturb_sq = torch.zeros((), device=env.device, dtype=torch.float64)
    perturb_count = torch.zeros((), device=env.device, dtype=torch.float64)
    # Trajectory dumps live in preallocated device buffers; the host copy and
    # the JSON/npz export happen once after the rollout.
    dump_ids = sorted(int(i) for i in (dump_replicas or []))
    dump_index = (
        torch.tensor(dump_ids, device=env.device, dtype=torch.long) if dump_ids else None
    )
    cap = int(max_steps)
    if dump_ids:
        d_obs = torch.zeros(
            (cap, len(dump_ids), obs["policy"].shape[-1]), device=env.device, dtype=torch.float32
        )
        d_applied = torch.zeros(
            (cap, len(dump_ids), int(env.num_actions)), device=env.device, dtype=torch.float32
        )
        d_done = torch.zeros((cap, len(dump_ids)), device=env.device, dtype=torch.bool)
    d_nominal = d_residual = None
    torch.manual_seed(int(eval_seed) * 101 + (0 if deterministic else 1))
    start = time.perf_counter()
    t = 0
    with torch.no_grad():
        while t < int(max_steps):
            pre_seen = cohort.seen.clone()
            active = ~pre_seen
            mean = None
            if method == "traditional":
                action = controller.actions(obs)
            else:
                mean = actor(obs)
                action = mean if deterministic else actor(obs, stochastic_output=True)
            if not deterministic and mean is not None:
                if method in ("residual_ppo", "frozen_bc_residual_ppo"):
                    lim = env.residual_limit.to(action.dtype)
                    delta = lim * (torch.tanh(action) - torch.tanh(mean))
                else:
                    delta = action - mean
                perturb_sq += (delta**2).sum().to(torch.float64)
                perturb_count += float(n * delta.shape[-1])
            pre_obs = obs
            obs, _, term, trunc, extras = env.step(action)
            applied = getattr(
                env,
                "last_applied",
                action.clamp(float(env.clip_actions) * -1.0, float(env.clip_actions)),
            )
            nominal = getattr(env, "last_nominal", None)
            residual = getattr(env, "last_residual", None)
            done = (term | trunc) & active
            terminal_obs = (
                extras["terminal_observations"]["policy"]
                if "terminal_observations" in extras
                else obs["policy"]
            )
            accumulate(
                task,
                env,
                acc,
                obs=pre_obs["policy"],
                applied_action=applied,
                nominal_action=nominal,
                active=active,
                done=done,
                terminal_obs=terminal_obs,
                dt_control=dt_control,
            )
            # dump first-episode trajectories for the selected replicas
            if dump_ids:
                d_obs[t] = pre_obs["policy"][dump_index]
                d_applied[t] = applied[dump_index]
                if nominal is not None:
                    if d_nominal is None:
                        d_nominal = torch.zeros(
                            (cap, len(dump_ids), nominal.shape[-1]),
                            device=env.device,
                            dtype=torch.float32,
                        )
                    d_nominal[t] = nominal[dump_index]
                if residual is not None:
                    if d_residual is None:
                        d_residual = torch.zeros(
                            (cap, len(dump_ids), residual.shape[-1]),
                            device=env.device,
                            dtype=torch.float32,
                        )
                    d_residual[t] = residual[dump_index]
                d_done[t] = done[dump_index]
            cohort.update(env)
            new = cohort.seen & ~pre_seen
            first_steps[new] = t + 1
            first_terminal |= new
            t += 1
            # Coarse completion check: seen.all() is the only host sync.
            if t % 32 == 0 and cohort.finished():
                break
    wall = time.perf_counter() - start
    # Slice the dumps at each replica's first episode end (inclusive) -- one
    # host read for the whole cohort instead of per-step/per-replica copies.
    trajectories = {}
    if dump_ids and t:
        done_seen = d_done[:t]
        first_true = done_seen.to(torch.int32).argmax(dim=0)
        has = done_seen.any(dim=0)
        ends = torch.where(has, first_true + 1, torch.full_like(first_true, t)).tolist()
        for j, i in enumerate(dump_ids):
            k = int(ends[j])
            trajectories[i] = {
                "obs": d_obs[:k, j].contiguous().cpu(),
                "applied": d_applied[:k, j].contiguous().cpu(),
                "nominal": None if d_nominal is None else d_nominal[:k, j].contiguous().cpu(),
                "residual": None if d_residual is None else d_residual[:k, j].contiguous().cpu(),
                "done": d_done[:k, j].cpu(),
            }
    physical = finalize(task, env, acc, dt_control=dt_control)
    report = cohort.report()
    rows = []
    for i, row in enumerate(report["episodes"]):
        matched = report["successes"]  # noqa: F841 (documented below)
        pr = {
            k: (v[i] if isinstance(v, list) and len(v) == n else v)
            for k, v in physical["per_replica"].items()
        }
        rows.append(
            {
                **row,
                "first_episode_steps": int(first_steps[i]),
                "terminal_obs_captured": bool(first_terminal[i]),
                "physical": pr,
            }
        )
    # aggregate physical metrics over COMPLETED first episodes only; the count
    # is reported so success-only selection can never masquerade as full coverage
    completed = [r for r in rows if r["complete"]]
    n_completed = len(completed)
    agg = {}
    for k in physical["per_replica"]:
        vals = []
        for r in completed:
            v = r["physical"][k]
            if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                vals.append(float(v))
        agg[k] = sum(vals) / len(vals) if vals else None
    # success-conditioned cost is reported NEXT TO its success fraction
    succ_rows = [r for r in completed if r["success"]]
    succ_agg = {}
    for k in physical["per_replica"]:
        vals = []
        for r in succ_rows:
            v = r["physical"][k]
            if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                vals.append(float(v))
        succ_agg[k] = sum(vals) / len(vals) if vals else None
    expl_rms = (
        float((perturb_sq / perturb_count.clamp_min(1.0)).sqrt())
        if float(perturb_count) > 0
        else None
    )
    return dict(
        report=report,
        physical={
            "mean_completed": agg,
            "mean_successful": succ_agg,
            "success_fraction_for_cost": (len(succ_rows) / n_completed if n_completed else None),
            "completed_replicas": n_completed,
            "profile_note": PROFILE_NOTE.get(task),
        },
        rows=rows,
        init_digest=init_digest,
        eval_wall_seconds=wall,
        env_steps=int(t),
        exploration_env_action_rms=expl_rms,
        trajectories=trajectories,
        residual_stats=env.stats_snapshot() if hasattr(env, "stats_snapshot") else None,
        dt_control=dt_control,
    )


def _save_trajectories(path, trajectories, task):
    arrays = {}
    for i, tr in trajectories.items():
        if tr["obs"] is None or tr["obs"].numel() == 0:
            continue
        arrays[f"obs_{i}"] = tr["obs"].numpy()
        arrays[f"applied_{i}"] = tr["applied"].numpy()
        if tr["nominal"] is not None:
            arrays[f"nominal_{i}"] = tr["nominal"].numpy()
        if tr["residual"] is not None:
            arrays[f"residual_{i}"] = tr["residual"].numpy()
        arrays[f"done_{i}"] = tr["done"].numpy()
    if arrays:
        np.savez_compressed(path, **arrays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--method", required=True, choices=LEARNED + ("traditional",))
    p.add_argument("--condition", default="default")
    p.add_argument("--seed", type=int, default=7, help="training seed / run directory")
    p.add_argument(
        "--eval-seed",
        type=int,
        default=7,
        help="protocol seed shared by ALL arms for paired initial states",
    )
    p.add_argument(
        "--stage",
        default=None,
        help='evaluate stage_<n>.pt, or "best" (selection/best_checkpoint.pt)',
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None, help="evaluate an explicit checkpoint path"
    )
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--backend", default="gpu")
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--run-tag", default=None)
    p.add_argument("--no-stochastic", action="store_true")
    p.add_argument(
        "--overwrite-eval",
        action="store_true",
        help="overwrite existing eval JSON for this (arm, stage, mode)",
    )
    args = p.parse_args()

    cfg = task_config(args.task)
    cond = conditions.resolve(args.task, args.condition)
    out = run_dir(args.task, args.method, args.condition, args.seed, args.out_root, args.run_tag)
    out.mkdir(parents=True, exist_ok=True)
    stage_token = str(args.stage) if args.stage is not None else None
    if args.checkpoint is not None:
        stage_token = "selected"
    target = out / "eval" / (f"summary_stage_{stage_token}.json" if stage_token else "summary.json")
    if target.exists() and not args.overwrite_eval:
        raise FileExistsError(
            f"evaluation already exists: {target}; pass --overwrite-eval to replace"
        )
    ev = _load_eval_config(out, args.method, args.task, cond, cfg)
    num_envs = int(args.num_envs or cfg["eval"]["num_envs"])
    max_steps = int(args.max_steps or ev["max_steps"])
    ckpt = None
    if args.method != "traditional":
        if args.checkpoint is not None:
            ckpt = args.checkpoint
        elif args.stage == "best":
            ckpt = out / "selection" / "best_checkpoint.pt"
        elif args.stage is not None:
            ckpt = out / f"stage_{int(args.stage)}.pt"
        else:
            ckpt = out / "checkpoint.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
    modes = (
        [True]
        if args.method == "traditional"
        else ([True] if args.no_stochastic else [True, False])
    )
    results = {}
    for deterministic in modes:
        torch.manual_seed(args.eval_seed)
        env, controller = _make_eval_env(
            args.task,
            cfg,
            ev,
            num_envs=num_envs,
            backend=args.backend,
            eval_seed=args.eval_seed,
            method=args.method,
            cond=cond,
        )
        actor = None
        if ckpt is not None:
            actor, _ = _load_actor(env, ckpt)
        roll = _rollout(
            args.task,
            cfg,
            env,
            method=args.method,
            actor=actor,
            controller=controller,
            deterministic=deterministic,
            eval_seed=args.eval_seed,
            max_steps=max_steps,
            dump_replicas=ev["dump_replicas"],
        )
        tag = "det" if deterministic else "stoch"
        report = {
            "task": args.task,
            "method": args.method,
            "condition": args.condition,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "checkpoint": str(ckpt) if ckpt else None,
            "stage": stage_token,
            "policy_mode": "deterministic_mean" if deterministic else "stochastic_sample",
            "num_envs": num_envs,
            "max_steps": max_steps,
            "eval_env_kwargs": ev["eval_env_kwargs"],
            "controller_design": ev["controller_design"],
            "baseline_variant": ev["baseline_variant"],
            "controller": {
                "name": getattr(controller, "name", None),
                "config": controller.config() if hasattr(controller, "config") else None,
            },
            "residual": (
                {
                    "rho": ev["residual_rho"],
                    "squash": ev["residual_squash"],
                    "stats": roll["residual_stats"],
                }
                if args.method in ("residual_ppo", "frozen_bc_residual_ppo")
                else None
            ),
            "action_semantics": ev["action_semantics"],
            "dt_control": roll["dt_control"],
            "initial_condition_digest": roll["init_digest"],
            "exploration_env_action_rms": roll["exploration_env_action_rms"],
            "eval_wall_seconds": roll["eval_wall_seconds"],
            "env_steps": roll["env_steps"],
            "physical": roll["physical"],
            "per_replica": roll["rows"],
            "protocol": "first_episode_per_replica (scripts/first_episode.py)",
            "incomplete_cohort": not roll["report"]["complete"],
        }
        report.update({k: v for k, v in roll["report"].items() if k != "episodes"})
        report["cohort"] = {k: v for k, v in roll["report"].items() if k == "episodes"}
        if roll["report"]["complete"] is False:
            report["incomplete_reason"] = (
                "safety step cap reached before every replica finished one episode"
            )
        results[tag] = report
        write_json(
            out / "eval" / (f"{tag}_stage_{stage_token}.json" if stage_token else f"{tag}.json"),
            report,
        )
        if deterministic:
            _save_trajectories(
                out
                / "eval"
                / (f"trajectory_stage_{stage_token}.npz" if stage_token else "trajectory.npz"),
                roll["trajectories"],
                args.task,
            )
        print(
            json.dumps(
                {
                    "method": args.method,
                    "mode": tag,
                    "complete": roll["report"]["complete"],
                    "success_rate": roll["report"]["success_rate"],
                    "episodes_finished": roll["report"]["episodes_finished"],
                    "mean_episode_return": roll["report"]["mean_episode_return"],
                    "physical_mean": roll["physical"]["mean_completed"],
                    "init_digest": roll["init_digest"],
                    "exploration_env_action_rms": roll["exploration_env_action_rms"],
                    "eval_wall_seconds": round(roll["eval_wall_seconds"], 1),
                },
                indent=2,
            ),
            flush=True,
        )
        env.close()
    write_json(
        out / "eval" / (f"summary_stage_{stage_token}.json" if stage_token else "summary.json"),
        {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk
                in (
                    "success_rate",
                    "episodes_finished",
                    "episodes_expected",
                    "complete",
                    "mean_episode_return",
                    "eval_wall_seconds",
                    "exploration_env_action_rms",
                    "initial_condition_digest",
                )
            }
            for k, v in results.items()
        },
    )


if __name__ == "__main__":
    main()
