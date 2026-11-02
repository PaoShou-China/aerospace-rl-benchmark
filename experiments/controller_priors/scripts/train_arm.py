"""Train one controller-prior arm (PPO only; the experiment is PPO-scoped)."""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from algorithms.train_utils import run_training  # noqa: E402
from algorithms.residual.residual_env import (  # noqa: E402
    ResidualVecEnv,
    zero_init_policy_output,
    success_gated_authority_curriculum,
)
from algorithms.residual.action_space import (  # noqa: E402
    DeltaActionVecEnv,
    ActionRatePenaltyVecEnv,
)
from scripts.common import (  # noqa: E402
    run_dir,
    ensure_new_run_dir,
    env_class,
    make_controller,
    load_frozen_bc_policy,
    write_json,
    run_metadata,
    task_config,
)
from scripts import conditions  # noqa: E402

LEARNED_METHODS = (
    "bc_only",
    "ppo_scratch",
    "bc_ppo",
    "residual_ppo",
    "frozen_bc_residual_ppo",
    "bc_dagger",
    "bc_dagger_ppo",
    "bc_bootstrap",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--method", required=True, choices=LEARNED_METHODS)
    p.add_argument("--condition", default="default")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--backend", default="gpu")
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--run-tag", default=None)
    p.add_argument(
        "--stage-updates",
        default=None,
        help="comma list; default: tasks.json stage_updates (-1 disables)",
    )
    p.add_argument("--bc-dataset", type=Path, default=None)
    p.add_argument("--bc-checkpoint", type=Path, default=None)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--no-eval-config", action="store_true")
    # ---- improvement ablations (all opt-in; defaults preserve the pilot) ----
    p.add_argument("--bc-extra-dataset", type=Path, default=None)
    p.add_argument("--bc-anchor-coef", type=float, default=None)
    p.add_argument("--bc-anchor-decay", type=int, default=0)
    p.add_argument(
        "--terminal-weight", default=None, help="JSON sample weighting spec for BC (phase_weights)"
    )
    p.add_argument("--rho-channels", default=None, help="comma list of per-channel residual rho")
    p.add_argument(
        "--phase-gate", default=None, help='JSON {"index":[..],"near":..,"soft":..,"gain":..}'
    )
    p.add_argument(
        "--authority-curriculum",
        default=None,
        help='JSON {"start":..,"end":..,"success_threshold":..,"step":..}',
    )
    p.add_argument("--lr-warmup", type=int, default=None)
    p.add_argument("--sigma-end", type=float, default=None)
    p.add_argument("--freeze-actor-updates", type=int, default=None)
    p.add_argument("--segment-updates", type=int, default=None)
    p.add_argument("--curriculum", default=None, help='JSON list of {"fraction":..,"kwargs":{..}}')
    p.add_argument("--action-mode", choices=("absolute", "delta"), default="absolute")
    p.add_argument("--delta-scale", type=float, default=0.25)
    p.add_argument("--action-rate-penalty", type=float, default=None)
    p.add_argument("--select-checkpoint", action="store_true")
    args = p.parse_args()

    cfg = task_config(args.task)
    cond = conditions.resolve(args.task, args.condition)
    out = run_dir(args.task, args.method, args.condition, args.seed, args.out_root, args.run_tag)
    if args.reuse and (out / "run.json").exists():
        print(f"[arm] reusing {out}", flush=True)
        return
    ensure_new_run_dir(out)
    torch.manual_seed(args.seed)

    updates = int(args.updates if args.updates is not None else cfg["updates"])
    n_envs = int(args.num_envs or cfg["num_envs"])
    if args.method in ("bc_only", "bc_dagger", "bc_bootstrap"):
        updates = 0
    if args.stage_updates is not None:
        stages = (
            []
            if args.stage_updates.strip() == "-1"
            else [int(x) for x in args.stage_updates.split(",") if x.strip()]
        )
    else:
        stages = [s for s in cfg.get("stage_updates", []) if s <= updates]
    train_kw = conditions.train_env_kwargs(args.task, cond, cfg)
    eval_kw = conditions.eval_env_kwargs(args.task, cond, cfg)
    bc_cfg = conditions.bc_overrides(args.task, cond, cfg)
    cls = env_class(args.task, cfg)

    # ---- improvement ablation settings (condition first, explicit CLI wins) ----
    imp = dict(cond.get("improvements") or {})
    action_mode = (
        args.action_mode if args.action_mode != "absolute" else imp.get("action_mode", "absolute")
    )
    delta_scale = float(imp.get("delta_scale", args.delta_scale))
    rate_penalty = (
        args.action_rate_penalty
        if args.action_rate_penalty is not None
        else imp.get("action_rate_penalty")
    )
    rho_channels = args.rho_channels if args.rho_channels is not None else imp.get("rho_channels")
    phase_gate = json.loads(args.phase_gate) if args.phase_gate else imp.get("phase_gate")
    if rho_channels is not None:
        chans = (
            [float(x) for x in str(rho_channels).split(",")]
            if isinstance(rho_channels, str)
            else list(rho_channels)
        )
        cond["residual_rho"] = {"channels": chans, "gate": phase_gate} if phase_gate else chans
    elif phase_gate:
        cond["residual_rho"] = {"channels": cond["residual_rho"], "gate": phase_gate}
    bc_anchor_coef = (
        args.bc_anchor_coef if args.bc_anchor_coef is not None else imp.get("bc_anchor_coef")
    )
    bc_anchor_decay = (
        args.bc_anchor_decay if args.bc_anchor_decay else int(imp.get("bc_anchor_decay", 0))
    )
    terminal_weight = (
        json.loads(args.terminal_weight) if args.terminal_weight else imp.get("terminal_weight")
    )
    lr_warmup = args.lr_warmup if args.lr_warmup is not None else imp.get("lr_warmup")
    sigma_end = args.sigma_end if args.sigma_end is not None else imp.get("sigma_end")
    freeze_actor = (
        args.freeze_actor_updates
        if args.freeze_actor_updates is not None
        else imp.get("freeze_actor_updates")
    )
    segment_updates = (
        args.segment_updates if args.segment_updates is not None else imp.get("segment_updates")
    )
    curriculum = json.loads(args.curriculum) if args.curriculum else imp.get("curriculum")
    authority = (
        json.loads(args.authority_curriculum)
        if args.authority_curriculum
        else imp.get("authority_curriculum")
    )
    bc_extra = args.bc_extra_dataset or imp.get("bc_extra_dataset")
    if action_mode == "delta" and args.method in ("residual_ppo", "frozen_bc_residual_ppo"):
        raise ValueError(
            "delta actions compose with the raw env; residual arms "
            "already provide bounded increments"
        )
    if rate_penalty is not None and float(rate_penalty) < 0:
        raise ValueError("action rate penalty must be non-negative")
    anchor_applies = args.method in ("bc_ppo", "bc_dagger_ppo")
    if (
        bc_anchor_coef
        and not anchor_applies
        and args.method not in ("bc_only", "bc_dagger", "bc_bootstrap")
    ):
        raise ValueError(
            "the KL anchor regularizes a BC->PPO refinement; use method bc_ppo / bc_dagger_ppo"
        )
    # bc_only / bc_bootstrap are the shared initialization providers for a
    # condition; the anchor is inert there and is applied only by bc_ppo.

    frozen_bc_checkpoint = None

    def env_wrapper(base_env):
        wrapped = base_env
        if action_mode == "delta":
            wrapped = DeltaActionVecEnv(wrapped, delta_scale=delta_scale)
        if rate_penalty is not None:
            wrapped = ActionRatePenaltyVecEnv(wrapped, coef=float(rate_penalty))
        if args.method in ("residual_ppo", "frozen_bc_residual_ppo"):
            if args.method == "residual_ppo":
                controller = make_controller(args.task, wrapped, cond)
            else:
                controller = load_frozen_bc_policy(frozen_bc_checkpoint, wrapped)
            cond["_teacher_name"] = getattr(controller, "name", type(controller).__name__)
            wrapped = ResidualVecEnv(
                wrapped,
                controller,
                variant=cond.get("baseline_variant"),
                residual_rho=cond["residual_rho"],
                residual_squash=cond["residual_squash"],
                task=args.task,
                controller_design=cond.get("controller_design"),
                teacher_identity=cond["_teacher_name"],
                name=args.method,
            )
        return wrapped

    needs_wrapper = (
        args.method in ("residual_ppo", "frozen_bc_residual_ppo")
        or action_mode == "delta"
        or rate_penalty is not None
    )
    extra_config = {
        "task": args.task,
        "method": args.method,
        "condition": args.condition,
        "ppo_protocol": cfg["ppo"],
        "baseline_variant": cond.get("baseline_variant"),
        "controller_design": cond.get("controller_design"),
        "controller_kwargs": cond.get("controller_kwargs"),
        "residual_rho": cond["residual_rho"],
        "train_env_kwargs": train_kw,
        "eval_env_kwargs": eval_kw,
        "action_semantics": cfg.get("action_semantics"),
        "action_mode": action_mode,
        "delta_scale": delta_scale,
        "action_rate_penalty": rate_penalty,
        "physical_cost": cfg.get("physical_cost"),
        "ppo_overrides_applied": dict(cond.get("ppo_overrides") or {}),
        "improvements": imp,
        "disable_stages": bool(cond.get("disable_stages")),
    }
    runner_init_hook = None
    segment_hook = None
    bc_warmstart = None

    if args.method in ("residual_ppo", "frozen_bc_residual_ppo"):
        runner_init_hook = zero_init_policy_output
        extra_config["residual"] = {
            "rho": cond["residual_rho"],
            "squash": cond["residual_squash"],
            "phase_gate": phase_gate,
        }
        if args.method == "frozen_bc_residual_ppo":
            base_cond = cond.get("bc_base_condition") or args.condition
            ckpt = args.bc_checkpoint
            if ckpt is None:
                ckpt = (
                    run_dir(
                        args.task,
                        "bc_only",
                        base_cond,
                        args.seed,
                        args.out_root,
                        args.run_tag,
                    )
                    / "bc_checkpoint.pt"
                )
                if not ckpt.exists() and base_cond != "default":
                    # tuning conditions share the task's default BC base
                    ckpt = (
                        run_dir(
                            args.task,
                            "bc_only",
                            "default",
                            args.seed,
                            args.out_root,
                            args.run_tag,
                        )
                        / "bc_checkpoint.pt"
                    )
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"frozen BC base checkpoint missing: {ckpt}; run bc_only "
                    "(the same initialization must be shared across residual variants)"
                )
            frozen_bc_checkpoint = ckpt
            extra_config["frozen_bc"] = {
                "checkpoint": str(ckpt),
                "base_condition": base_cond,
            }
        if authority:
            segment_hook = success_gated_authority_curriculum(**authority)
    elif args.method in ("bc_only", "bc_ppo", "bc_dagger", "bc_dagger_ppo", "bc_bootstrap"):
        if args.method in ("bc_only", "bc_bootstrap", "bc_dagger"):
            # The reported BC arm (``bc_only``) is always trained on the
            # DAgger aggregation.  ``bc_bootstrap`` is the internal student
            # used to collect DAgger iterations and is never reported as an
            # arm; its dataset is the pure-expert iteration-0 split.
            if args.method == "bc_bootstrap":
                dataset = args.bc_dataset or (
                    run_dir(
                        args.task,
                        "bc_dataset",
                        args.condition,
                        args.seed,
                        args.out_root,
                        args.run_tag,
                    )
                    / "dataset.pt"
                )
                hint = "collect_bc.py"
            else:
                dataset = args.bc_dataset or (
                    run_dir(
                        args.task,
                        "bc_dagger_dataset",
                        args.condition,
                        args.seed,
                        args.out_root,
                        args.run_tag,
                    )
                    / "dagger_dataset.pt"
                )
                hint = "collect_dagger.py"
            if not dataset.exists():
                raise FileNotFoundError(f"BC dataset missing: {dataset}; run {hint} first")
            bc_warmstart = {
                **bc_cfg,
                "dataset": str(dataset),
                "dataset_split": "train",
                "save_checkpoint": str(out / "bc_checkpoint.pt"),
                "save_bc_checkpoint": True,
            }
            # Under the delta parameterization the actor emits an increment, so
            # the supervised target is the expert's per-step command delta.
            # The bootstrap student must instead emit ABSOLUTE actions: DAgger
            # mixes its output with the controller's absolute labels.
            if action_mode == "delta" and args.method != "bc_bootstrap":
                bc_warmstart["delta_action_scale"] = delta_scale
            if bc_extra:
                bc_warmstart["extra_datasets"] = [str(bc_extra)]
        else:
            dagger = args.method == "bc_dagger_ppo"
            ckpt = args.bc_checkpoint
            if ckpt is None:
                ckpt = (
                    run_dir(
                        args.task,
                        "bc_dagger" if dagger else "bc_only",
                        args.condition,
                        args.seed,
                        args.out_root,
                        args.run_tag,
                    )
                    / "bc_checkpoint.pt"
                )
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"BC checkpoint missing: {ckpt}; run "
                    + ("bc_dagger" if dagger else "bc_only")
                    + " first (the same initialization must be shared)"
                )
            bc_warmstart = {**bc_cfg, "checkpoint": str(ckpt), "fit_critic": False}
        if terminal_weight:
            bc_warmstart["sample_weighting"] = terminal_weight
    elif args.method == "ppo_scratch":
        pass

    base_cfg = dict(cfg["ppo"])
    # run_training expects lr_schedule / learning_rate inside algo_overrides
    ppo_overrides = dict(base_cfg)
    ppo_overrides["lr_schedule"] = ppo_overrides.pop("lr_schedule", "fixed")
    ppo_overrides.update(cond.get("ppo_overrides") or {})
    if bc_anchor_coef and anchor_applies:
        ppo_overrides["kl_anchor_coef"] = float(bc_anchor_coef)
        ppo_overrides["kl_anchor_decay_updates"] = int(bc_anchor_decay)
    if lr_warmup:
        ppo_overrides["lr_cosine"] = True
        ppo_overrides["lr_warmup_updates"] = int(lr_warmup)
    if sigma_end is not None:
        ppo_overrides["sigma_anneal_to"] = float(sigma_end)
    if freeze_actor:
        ppo_overrides["actor_freeze_updates"] = int(freeze_actor)
    if segment_updates:
        ppo_overrides["segment_updates"] = int(segment_updates)
    if curriculum:
        ppo_overrides["env_curriculum"] = list(curriculum)
    if segment_hook is not None:
        ppo_overrides["segment_hook"] = segment_hook
    bc_anchor = (
        {"coef": float(bc_anchor_coef), "decay_updates": int(bc_anchor_decay)}
        if (bc_anchor_coef and anchor_applies)
        else None
    )
    if cond.get("disable_stages"):
        stages = []

    meta = run_metadata(
        task=args.task,
        method=args.method,
        condition=args.condition,
        seed=args.seed,
        extra={
            "num_envs": n_envs,
            "updates": updates,
            "stage_updates": stages,
            "bc_overrides": bc_cfg,
            "train_env_kwargs": train_kw,
            "eval_env_kwargs": eval_kw,
            "controller_design": cond.get("controller_design"),
            "baseline_variant": cond.get("baseline_variant"),
            "residual_rho": cond["residual_rho"],
            "ppo_overrides_applied": dict(cond.get("ppo_overrides") or {}),
            "action_mode": action_mode,
            "delta_scale": delta_scale,
            "action_rate_penalty": rate_penalty,
            "improvements": imp,
            "transition_budget": updates * int(base_cfg["num_steps_per_env"]) * n_envs,
            "bc_dataset": (
                str(bc_warmstart.get("dataset"))
                if bc_warmstart and bc_warmstart.get("dataset")
                else None
            ),
            "bc_checkpoint": (
                str(bc_warmstart.get("checkpoint"))
                if bc_warmstart and bc_warmstart.get("checkpoint")
                else None
            ),
            "frozen_bc_checkpoint": (str(frozen_bc_checkpoint) if frozen_bc_checkpoint else None),
            "bc_base_condition": (
                cond.get("bc_base_condition") if args.method == "frozen_bc_residual_ppo" else None
            ),
        },
    )
    t0 = time.perf_counter()
    report = run_training(
        args.task,
        cls,
        num_envs=n_envs,
        updates=updates,
        seed=args.seed,
        backend=args.backend,
        clean=True,
        out_dir=str(out),
        algo="ppo",
        env_kwargs=train_kw,
        bc_warmstart=bc_warmstart,
        env_wrapper=env_wrapper if needs_wrapper else None,
        experiment_meta=meta,
        extra_config=extra_config,
        stage_updates=stages or None,
        runner_init_hook=runner_init_hook,
        bc_anchor=bc_anchor,
        save_initial_checkpoint=(args.method not in ("bc_only", "bc_dagger", "bc_bootstrap")),
        **ppo_overrides,
    )
    wall = time.perf_counter() - t0
    meta["training_wall_seconds"] = wall
    meta["run_json"] = str(out / "run.json")
    write_json(out / "arm.json", meta)
    # The success-gated authority curriculum may end below full authority; the
    # evaluation must compose the residual exactly at the TRAINED fraction.
    final_authority = None
    if isinstance(report.get("residual_stats"), dict):
        final_authority = report["residual_stats"].get("authority_fraction")
    if not args.no_eval_config:
        write_json(
            out / "eval_config.json",
            {
                "task": args.task,
                "method": args.method,
                "condition": args.condition,
                "seed": args.seed,
                "num_envs_training": n_envs,
                "updates": updates,
                "eval_env_kwargs": eval_kw,
                "controller_design": cond.get("controller_design"),
                "controller_kwargs": cond.get("controller_kwargs"),
                "baseline_variant": cond.get("baseline_variant"),
                "residual_rho": cond["residual_rho"],
                "residual_squash": cond["residual_squash"],
                "action_semantics": cfg.get("action_semantics"),
                "action_mode": action_mode,
                "delta_scale": delta_scale,
                "phase_gate": phase_gate,
                "authority_fraction": final_authority,
                "bc_checkpoint": (
                    str(frozen_bc_checkpoint)
                    if frozen_bc_checkpoint
                    else (
                        str(bc_warmstart.get("checkpoint"))
                        if bc_warmstart and bc_warmstart.get("checkpoint")
                        else None
                    )
                ),
                "frozen_bc_base": (
                    {
                        "checkpoint": str(frozen_bc_checkpoint),
                        "base_condition": cond.get("bc_base_condition") or args.condition,
                    }
                    if frozen_bc_checkpoint
                    else None
                ),
                "bc_dataset": (
                    str(bc_warmstart.get("dataset"))
                    if bc_warmstart and bc_warmstart.get("dataset")
                    else None
                ),
                "bc_extra_datasets": (bc_warmstart or {}).get("extra_datasets"),
                "eval": cfg["eval"],
            },
        )
    if args.select_checkpoint:
        import subprocess  # noqa: E402

        sel = [
            sys.executable,
            str(Path(__file__).resolve().parent / "select_checkpoint.py"),
            "--task",
            args.task,
            "--method",
            args.method,
            "--condition",
            args.condition,
            "--seed",
            str(args.seed),
            "--backend",
            args.backend,
        ]
        if args.out_root:
            sel += ["--out-root", str(args.out_root)]
        if args.run_tag:
            sel += ["--run-tag", args.run_tag]
        subprocess.run(sel, check=False)
    print(
        json.dumps(
            {
                "run_dir": str(out),
                "method": args.method,
                "train_wall_seconds": round(wall, 1),
                "updates": updates,
                "successes_train_rollout": sum(report.get("completed_successes", [])),
                "episodes_train_rollout": sum(report.get("completed_episodes", [])),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
