"""One-command orchestrator: smoke -> pilot -> factor scans -> plot.

Examples
--------
    python experiments/controller_priors/scripts/run_experiments.py --phase smoke
    python experiments/controller_priors/scripts/run_experiments.py --phase pilot
    python experiments/controller_priors/scripts/run_experiments.py --phase scans \
        --scan-task docking
    python experiments/controller_priors/scripts/run_experiments.py --phase plot

Runs are isolated by (task, method, condition, seed) plus an explicit run tag.
A run is only started when its directory is absent (or --force); results are
never silently overwritten. Logs live outside the run directories.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (  # noqa: E402
    PYTHON,
    RESULTS_ROOT,
    TUNING_ROOT,
    run_dir,
    write_json,
    run_metadata,
)
from scripts import conditions  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent

# pilot budgets (measured throughput is recorded in every run.json; the report
# marks these as PILOT-scale, not the final JGCD budget)
PILOT = {
    "docking": {"num_envs": 1024, "updates": 250, "seeds": [7, 13, 21]},
    "satellite": {"num_envs": 512, "updates": 100, "seeds": [7, 13, 21]},
    "rendezvous": {"num_envs": 512, "updates": 200, "seeds": [7, 13, 21]},
}
PILOT_METHODS = [
    "traditional",
    "bc_only",
    "ppo_scratch",
    "bc_ppo",
    "residual_ppo",
    "frozen_bc_residual_ppo",
]
STAGE_METHODS = [
    "bc_only",
    "ppo_scratch",
    "bc_ppo",
    "residual_ppo",
    "frozen_bc_residual_ppo",
]

SCAN_RUNS = {
    "quality_sluggish": ["traditional", "bc_only", "bc_ppo", "residual_ppo"],
    "quality_aggressive": ["traditional", "bc_only", "bc_ppo", "residual_ppo"],
    "quality_model_bias": ["traditional", "bc_only", "bc_ppo", "residual_ppo"],
    "coverage_narrow": ["bc_only", "bc_ppo"],
    "coverage_wide": ["bc_only", "bc_ppo"],
    "samples_1x": ["bc_only", "bc_ppo"],
    "samples_6x": ["bc_only", "bc_ppo"],
    "authority_0.01": ["residual_ppo"],
    "authority_0.025": ["residual_ppo"],
    "authority_0.05": ["residual_ppo"],
    "authority_0.1": ["residual_ppo"],
    "authority_0.2": ["residual_ppo"],
    "authority_0.4": ["residual_ppo"],
    "mismatch_plant": ["traditional", "ppo_scratch", "bc_only", "bc_ppo", "residual_ppo"],
    "mismatch_design": ["traditional", "ppo_scratch", "bc_only", "bc_ppo", "residual_ppo"],
    "mismatch_both": ["traditional", "ppo_scratch", "bc_only", "bc_ppo", "residual_ppo"],
    # optimizer ablation (NOT a mechanism comparison)
    "stability_pin_bc_ppo": ["bc_only", "bc_ppo"],
}


def _log_path(out_root, tag, task, method, condition, seed):
    p = Path(out_root) / "_logs" / (tag or "default") / task / method / condition
    p.mkdir(parents=True, exist_ok=True)
    return p / f"seed_{seed}.log"


def _run(cmd, log_path, dry=False):
    line = " ".join(str(x) for x in cmd)
    print(f"$ {line}\n  -> {log_path}", flush=True)
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n$ {line}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(_ROOT))
    return proc.returncode


def _eval_commands(
    task,
    method,
    cond,
    seed,
    out_root,
    tag,
    stages,
    eval_final,
    det_only_stages,
    num_envs,
    backend,
    force=False,
):
    cmds = []
    out = run_dir(task, method, cond, seed, out_root, tag)

    def _add(cmd, target):
        if target.exists() and not force:
            print(f"[skip] existing {target}", flush=True)
            return
        cmds.append(cmd)

    for st in stages:
        ck = out / f"stage_{st}.pt"
        if not ck.exists():
            continue
        _add(
            [
                PYTHON,
                str(SCRIPTS / "evaluate.py"),
                "--task",
                task,
                "--method",
                method,
                "--condition",
                cond,
                "--seed",
                str(seed),
                "--stage",
                str(st),
                "--num-envs",
                str(num_envs),
                "--backend",
                backend,
                "--out-root",
                str(out_root),
                "--run-tag",
                tag,
            ]
            + (["--no-stochastic"] if det_only_stages else [])
            + (["--overwrite-eval"] if force else []),
            out / "eval" / f"det_stage_{st}.json",
        )
    if eval_final and (method == "traditional" or (out / "checkpoint.pt").exists()):
        _add(
            [
                PYTHON,
                str(SCRIPTS / "evaluate.py"),
                "--task",
                task,
                "--method",
                method,
                "--condition",
                cond,
                "--seed",
                str(seed),
                "--num-envs",
                str(num_envs),
                "--backend",
                backend,
                "--out-root",
                str(out_root),
                "--run-tag",
                tag,
            ]
            + (["--overwrite-eval"] if force else []),
            out / "eval" / "summary.json",
        )
    return cmds


def _train_sequence(
    task,
    method,
    cond,
    seed,
    out_root,
    tag,
    num_envs,
    updates,
    stages,
    backend,
    force,
    episodes=None,
    bc_cond=None,
):
    cmds = []
    out = run_dir(task, method, cond, seed, out_root, tag)
    if method == "traditional":
        return []
    if (out / "run.json").exists():
        print(f"[skip] existing run {out}", flush=True)
        return cmds
    if method in ("bc_only", "bc_ppo", "bc_dagger", "bc_dagger_ppo", "frozen_bc_residual_ppo"):
        # tuning conditions reuse the task's shared BC base instead of fitting
        # a new one for every residual variant
        cmds.extend(
            _bc_pipeline_cmds(
                task, bc_cond or cond, seed, num_envs, backend, out_root, tag, episodes
            )
        )
        if method not in ("bc_only", "bc_dagger"):
            cmds.append(
                _improve_cmd(task, method, cond, seed, num_envs, updates, backend, out_root, tag)
            )
    else:
        cmds.append(
            _improve_cmd(task, method, cond, seed, num_envs, updates, backend, out_root, tag)
        )
    return cmds


def run_task_plan(
    task, methods, seeds, num_envs, updates, out_root, tag, backend, force, dry=False
):
    summary = []
    for seed in seeds:
        for method in methods:
            cond = "default"
            for cmd in _train_sequence(
                task, method, cond, seed, out_root, tag, num_envs, updates, [], backend, force
            ):
                rc = _run(cmd, _log_path(out_root, tag, task, method, cond, seed), dry)
                if rc != 0:
                    summary.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "stage": "train",
                            "returncode": rc,
                        }
                    )
                    break
            # stage / final evaluation
            stages = (
                []
                if method == "traditional"
                else [s for s in (0, 25, 50, 75, 100, 150, 200) if s < updates]
            )
            cmds = _eval_commands(
                task,
                method,
                cond,
                seed,
                out_root,
                tag,
                stages,
                eval_final=True,
                det_only_stages=True,
                num_envs=num_envs,
                backend=backend,
                force=force,
            )
            for cmd in cmds:
                rc = _run(cmd, _log_path(out_root, tag, task, method + "_eval", cond, seed), dry)
                if rc != 0:
                    summary.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "stage": "eval",
                            "cmd": str(cmd[-3:]),
                            "returncode": rc,
                        }
                    )
                    break
            summary.append({"task": task, "method": method, "seed": seed, "status": "done"})
    return summary


def run_scans(args):
    task = args.scan_task
    seeds = [int(x) for x in args.scan_seeds.split(",")]
    num_envs = args.scan_num_envs
    updates = args.scan_updates
    out_root = args.out_root or Path(str(RESULTS_ROOT) + "_scans")
    tag = args.tag or "scan_v1"
    summary = []
    for cond, methods in SCAN_RUNS.items():
        for seed in seeds:
            for method in methods:
                for cmd in _train_sequence(
                    task,
                    method,
                    cond,
                    seed,
                    out_root,
                    tag,
                    num_envs,
                    updates,
                    [],
                    args.backend,
                    args.force,
                ):
                    rc = _run(cmd, _log_path(out_root, tag, task, method, cond, seed), args.dry_run)
                    if rc != 0:
                        summary.append(
                            {
                                "task": task,
                                "condition": cond,
                                "method": method,
                                "seed": seed,
                                "stage": "train",
                                "returncode": rc,
                            }
                        )
                        break
                stages = [] if method in ("traditional", "bc_only") else [max(1, updates // 2)]
                cmds = _eval_commands(
                    task,
                    method,
                    cond,
                    seed,
                    out_root,
                    tag,
                    stages,
                    eval_final=True,
                    det_only_stages=True,
                    num_envs=num_envs,
                    backend=args.backend,
                    force=args.force,
                )
                for cmd in cmds:
                    rc = _run(
                        cmd,
                        _log_path(out_root, tag, task, method + "_eval", cond, seed),
                        args.dry_run,
                    )
                    if rc != 0:
                        summary.append(
                            {
                                "task": task,
                                "condition": cond,
                                "method": method,
                                "seed": seed,
                                "stage": "eval",
                                "returncode": rc,
                            }
                        )
                        break
                summary.append(
                    {
                        "task": task,
                        "condition": cond,
                        "method": method,
                        "seed": seed,
                        "status": "done",
                    }
                )
    return summary


# ---- improvement phase ----------------------------------------------------
# One condition = one mechanism (independent ablation).  Seeds are 5 by default
# so the analysis can report a paired bootstrap CI.  ``improve_selection`` adds
# deterministic validation-seed checkpoint selection on top of the arm.
IMPROVEMENT_RUNS = {
    "docking": {
        "improve_kl_anchor": ["bc_ppo"],
        "improve_terminal_weight": ["bc_only", "bc_ppo"],
        "improve_optimizer": ["ppo_scratch", "bc_ppo"],
        "improve_grad_clip": ["bc_ppo"],
        "improve_rate_penalty": ["bc_ppo"],
        "improve_phase_gate": ["residual_ppo"],
        "improve_authority_curriculum": ["residual_ppo"],
        "improve_scratch_curriculum": ["ppo_scratch"],
        "improve_selection": ["bc_ppo", "residual_ppo"],
    },
    "satellite": {
        "improve_kl_anchor": ["bc_ppo"],
        "improve_terminal_weight": ["bc_only", "bc_ppo"],
        "improve_optimizer": ["ppo_scratch", "bc_ppo"],
        "improve_grad_clip": ["bc_ppo"],
        "improve_rate_penalty": ["bc_ppo"],
        "improve_delta_action": ["ppo_scratch", "bc_ppo"],
        "improve_phase_gate": ["residual_ppo"],
        "improve_authority_curriculum": ["residual_ppo"],
        "improve_scratch_curriculum": ["ppo_scratch"],
        "improve_selection": ["bc_ppo", "residual_ppo"],
    },
    "rendezvous": {
        "improve_kl_anchor": ["bc_ppo"],
        "improve_terminal_weight": ["bc_only", "bc_ppo"],
        "improve_optimizer": ["ppo_scratch", "bc_ppo"],
        "improve_grad_clip": ["bc_ppo"],
        "improve_rate_penalty": ["bc_ppo"],
        "improve_phase_gate": ["residual_ppo"],
        "improve_authority_curriculum": ["residual_ppo"],
        "improve_scratch_curriculum": ["ppo_scratch"],
        "improve_selection": ["bc_ppo", "residual_ppo"],
    },
}
IMPROVEMENT_SEEDS = [7, 13, 21, 33, 45]
SELECT_CONDITIONS = ("improve_selection",)
DAGGER_ITERATIONS = 3


def _improve_cmd(
    task, method, cond, seed, num_envs, updates, backend, out_root, tag, select=False,
    bc_dataset=None,
):
    cmd = [
        PYTHON,
        str(SCRIPTS / "train_arm.py"),
        "--task",
        task,
        "--method",
        method,
        "--condition",
        cond,
        "--seed",
        str(seed),
        "--num-envs",
        str(num_envs),
        "--updates",
        str(updates),
        "--backend",
        backend,
        "--out-root",
        str(out_root),
    ]
    if tag:
        cmd += ["--run-tag", tag]
    if select:
        cmd += ["--select-checkpoint"]
    if bc_dataset is not None:
        cmd += ["--bc-dataset", str(bc_dataset)]
    return cmd


def _bc_pipeline_cmds(
    task, cond, seed, num_envs, backend, out_root, tag, episodes=None, data_cond=None
):
    """DAgger-first BC pipeline (shared by every BC/BC->PPO arm).

    Pure-expert rollouts are only the bootstrap student's material (DAgger
    iteration 0).  The reported ``bc_only`` checkpoint -- and therefore the
    ``bc_ppo`` initialization -- is always fitted on the merged DAgger dataset.

    ``data_cond`` names the condition whose demonstrations/bootstrapping are
    reused.  Improvement ablations only touch the PPO side, so they all share
    one representative dataset per (task, seed) under ``default`` instead of
    re-collecting near-identical data for every condition.  Each arm still
    fits its own ``bc_only`` checkpoint in its own run directory (the
    terminal-weight ablation changes the BC objective).
    """
    data_cond = data_cond or cond
    cmds = []
    raw_ds = run_dir(task, "bc_dataset", data_cond, seed, out_root, tag) / "dataset.pt"
    if not raw_ds.exists():
        cmd = [
            PYTHON,
            str(SCRIPTS / "collect_bc.py"),
            "--task",
            task,
            "--condition",
            data_cond,
            "--seed",
            str(seed),
            "--num-envs",
            str(num_envs),
            "--backend",
            backend,
            "--out-root",
            str(out_root),
        ]
        if tag:
            cmd += ["--run-tag", tag]
        if episodes is not None:
            cmd += ["--episodes-per-replica", str(episodes)]
        cmds.append(cmd)

    boot_out = run_dir(task, "bc_bootstrap", data_cond, seed, out_root, tag)
    if not (boot_out / "run.json").exists():
        cmds.append(
            _improve_cmd(
                task, "bc_bootstrap", data_cond, seed, num_envs, 0, backend, out_root, tag
            )
        )

    dg_ds = (
        run_dir(task, "bc_dagger_dataset", data_cond, seed, out_root, tag) / "dagger_dataset.pt"
    )
    if not dg_ds.exists():
        cmd = [
            PYTHON,
            str(SCRIPTS / "collect_dagger.py"),
            "--task",
            task,
            "--condition",
            data_cond,
            "--seed",
            str(seed),
            "--num-envs",
            str(num_envs),
            "--iterations",
            str(DAGGER_ITERATIONS),
            "--bc-checkpoint",
            str(boot_out / "bc_checkpoint.pt"),
            "--backend",
            backend,
            "--out-root",
            str(out_root),
        ]
        if tag:
            cmd += ["--run-tag", tag]
        if episodes is not None:
            cmd += ["--episodes-per-replica", str(episodes)]
        cmds.append(cmd)

    bc_out = run_dir(task, "bc_only", cond, seed, out_root, tag)
    if not (bc_out / "run.json").exists():
        # the dataset may not exist yet in a dry run, so only pass it when the
        # pipeline reused another condition's data
        cmds.append(
            _improve_cmd(
                task,
                "bc_only",
                cond,
                seed,
                num_envs,
                0,
                backend,
                out_root,
                tag,
                bc_dataset=(dg_ds if data_cond != cond else None),
            )
        )
    return cmds


def _ensure_bc_pipeline(
    task, cond, seed, num_envs, backend, out_root, tag, episodes=None, dry=False,
    data_cond=None,
):
    for cmd in _bc_pipeline_cmds(
        task, cond, seed, num_envs, backend, out_root, tag, episodes, data_cond=data_cond
    ):
        rc = _run(cmd, _log_path(out_root, tag, task, "bc_pipeline", cond, seed), dry)
        if rc != 0:
            return rc
    return 0


def _eval_best(task, method, cond, seed, out_root, tag, num_envs, backend, force=False):
    """Evaluate the validation-selected checkpoint on the REPORTING eval seed."""
    out = run_dir(task, method, cond, seed, out_root, tag)
    best = out / "selection" / "best_checkpoint.pt"
    if not best.exists():
        print(f"[skip] no selection for {out}", flush=True)
        return []
    # evaluate.py writes summary_stage_best.json LAST, so it is the completion
    # marker (det_stage_best.json alone can be a partially finished evaluation)
    target = out / "eval" / "summary_stage_best.json"
    if target.exists() and not force:
        print(f"[skip] existing {target}", flush=True)
        return []
    return [
        [
            PYTHON,
            str(SCRIPTS / "evaluate.py"),
            "--task",
            task,
            "--method",
            method,
            "--condition",
            cond,
            "--seed",
            str(seed),
            "--stage",
            "best",
            "--num-envs",
            str(num_envs),
            "--backend",
            backend,
            "--out-root",
            str(out_root),
        ]
        + (["--run-tag", tag] if tag else [])
    ]


def run_improvements(args):
    out_root = args.out_root or Path(str(RESULTS_ROOT) + "_improvements")
    tag = args.tag or "improve_v1"
    seeds = (
        [int(x) for x in args.improve_seeds.split(",")]
        if args.improve_seeds
        else list(IMPROVEMENT_SEEDS)
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary = []
    for task in tasks:
        plan = PILOT.get(task, {})
        num_envs = args.improve_num_envs or args.num_envs or plan.get("num_envs", 512)
        updates = args.improve_updates or args.updates or plan.get("updates", 100)
        for cond, methods in IMPROVEMENT_RUNS[task].items():
            select = cond in SELECT_CONDITIONS
            for seed in seeds:
                # every reported BC arm uses the DAgger aggregation; the
                # pipeline also materializes the shared bc_only initialization.
                # Improvement conditions only change the PPO side, so they reuse
                # one representative dataset per (task, seed) under "default"
                # instead of re-collecting near-identical demonstrations.
                if any(m in ("bc_only", "bc_ppo") for m in methods):
                    rc = _ensure_bc_pipeline(
                        task, cond, seed, num_envs, args.backend, out_root, tag,
                        dry=args.dry_run, data_cond="default",
                    )
                    if rc != 0:
                        summary.append(
                            {
                                "task": task,
                                "condition": cond,
                                "seed": seed,
                                "stage": "bc_pipeline",
                                "returncode": rc,
                            }
                        )
                        continue
                rc = 0
                for method in methods:
                    # the BC pipeline already produced the exact bc_only arm;
                    # never rerun into the same run directory.
                    if (
                        run_dir(task, method, cond, seed, out_root, tag) / "run.json"
                    ).exists() and not args.dry_run:
                        print(f"[skip] existing run {method}/{cond}/seed_{seed}", flush=True)
                        continue
                    cmd = _improve_cmd(
                        task,
                        method,
                        cond,
                        seed,
                        num_envs,
                        updates,
                        args.backend,
                        out_root,
                        tag,
                        select=select,
                    )
                    rc = _run(cmd, _log_path(out_root, tag, task, method, cond, seed), args.dry_run)
                    if rc != 0:
                        summary.append(
                            {
                                "task": task,
                                "condition": cond,
                                "method": method,
                                "seed": seed,
                                "stage": "train",
                                "returncode": rc,
                            }
                        )
                        break
                if rc != 0:
                    continue
                # evaluation: selected best for selection arms, final otherwise
                if select:
                    for method in methods:
                        for cmd in _eval_best(
                            task,
                            method,
                            cond,
                            seed,
                            out_root,
                            tag,
                            num_envs,
                            args.backend,
                            force=args.force,
                        ):
                            _run(
                                cmd,
                                _log_path(out_root, tag, task, "select_eval", cond, seed),
                                args.dry_run,
                            )
                else:
                    for method in methods:
                        if method == "bc_dagger":
                            # bc_dagger's checkpoint is evaluated under bc_dagger_ppo
                            continue
                        cmds = _eval_commands(
                            task,
                            method,
                            cond,
                            seed,
                            out_root,
                            tag,
                            [],
                            eval_final=True,
                            det_only_stages=True,
                            num_envs=num_envs,
                            backend=args.backend,
                            force=args.force,
                        )
                        for cmd in cmds:
                            _run(
                                cmd,
                                _log_path(out_root, tag, task, method + "_eval", cond, seed),
                                args.dry_run,
                            )
                summary.append(
                    {
                        "task": task,
                        "condition": cond,
                        "seeds": seeds,
                        "methods": methods,
                        "select": select,
                        "status": "done",
                    }
                )
    return summary


# ---- BC-residual PPO tuning phase ------------------------------------------
# The frozen-BC arm is the object under test; the other learned arms and the
# traditional controllers are the reference points, evaluated once per seed
# under the "default" condition.
TUNE_CONTROL_METHODS = [
    "traditional",
    "residual_ppo",
    "bc_ppo",
    "ppo_scratch",
    "frozen_bc_residual_ppo",
]
TUNE_SEEDS = [7, 13, 21, 33, 45]
TUNE_CONDITION_PREFIX = "tune_bcr_"


def run_tune(args):
    out_root = args.out_root or TUNING_ROOT
    tag = args.tag or "tune_bcr_v1"
    seeds = (
        [int(x) for x in args.tune_seeds.split(",")]
        if args.tune_seeds
        else list(TUNE_SEEDS)
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    wanted = (
        {c.strip() for c in args.tune_conditions.split(",") if c.strip()}
        if args.tune_conditions
        else None
    )
    summary = []
    for task in tasks:
        plan = PILOT.get(task, {})
        num_envs = args.tune_num_envs or args.num_envs or plan.get("num_envs", 512)
        updates = args.tune_updates or args.updates or plan.get("updates", 100)
        names = ["default"] + sorted(
            n for n in conditions.SCANS.get(task, {}) if n.startswith(TUNE_CONDITION_PREFIX)
        )
        if wanted:
            names = [n for n in names if n in wanted]
        for seed in seeds:
            for cond_name in names:
                cond_spec = conditions.resolve(task, cond_name)
                methods = (
                    TUNE_CONTROL_METHODS
                    if cond_name == "default"
                    else ["frozen_bc_residual_ppo"]
                )
                base_cond = cond_spec.get("bc_base_condition") or cond_name
                episodes = (cond_spec.get("bc") or {}).get("episodes_per_replica")
                if any(m in ("bc_only", "bc_ppo", "frozen_bc_residual_ppo") for m in methods):
                    rc = _ensure_bc_pipeline(
                        task,
                        base_cond,
                        seed,
                        num_envs,
                        args.backend,
                        out_root,
                        tag,
                        episodes=episodes,
                        dry=args.dry_run,
                    )
                    if rc != 0:
                        summary.append(
                            {
                                "task": task,
                                "condition": cond_name,
                                "seed": seed,
                                "stage": "bc_pipeline",
                                "returncode": rc,
                            }
                        )
                        continue
                for method in methods:
                    out = run_dir(task, method, cond_name, seed, out_root, tag)
                    if (out / "run.json").exists() and not args.force and not args.dry_run:
                        print(f"[skip] existing run {method}/{cond_name}/seed_{seed}", flush=True)
                        continue
                    rc = 0
                    for cmd in _train_sequence(
                        task,
                        method,
                        cond_name,
                        seed,
                        out_root,
                        tag,
                        num_envs,
                        updates,
                        [],
                        args.backend,
                        args.force,
                        episodes=episodes,
                        bc_cond=base_cond,
                    ):
                        rc = _run(
                            cmd,
                            _log_path(out_root, tag, task, method, cond_name, seed),
                            args.dry_run,
                        )
                        if rc != 0:
                            summary.append(
                                {
                                    "task": task,
                                    "condition": cond_name,
                                    "method": method,
                                    "seed": seed,
                                    "stage": "train",
                                    "returncode": rc,
                                }
                            )
                            break
                    if rc != 0:
                        continue
                    for cmd in _eval_commands(
                        task,
                        method,
                        cond_name,
                        seed,
                        out_root,
                        tag,
                        stages=[],
                        eval_final=True,
                        det_only_stages=False,
                        num_envs=num_envs,
                        backend=args.backend,
                        force=args.force,
                    ):
                        rc = _run(
                            cmd,
                            _log_path(out_root, tag, task, method + "_eval", cond_name, seed),
                            args.dry_run,
                        )
                        if rc != 0:
                            summary.append(
                                {
                                    "task": task,
                                    "condition": cond_name,
                                    "method": method,
                                    "seed": seed,
                                    "stage": "eval",
                                    "returncode": rc,
                                }
                            )
                            break
                    summary.append(
                        {
                            "task": task,
                            "condition": cond_name,
                            "method": method,
                            "seed": seed,
                            "status": "done",
                        }
                    )
    return summary


def run_smoke(args):
    out_root = args.out_root or Path(str(RESULTS_ROOT) + "_smoke")
    tag = args.tag or "smoke_v1"
    if args.force:
        import shutil  # noqa: E402

        shutil.rmtree(out_root, ignore_errors=True)
    summary = []
    for task in ("docking", "satellite"):
        num_envs = 64
        updates = 3
        for seed in (7,):
            for method in PILOT_METHODS:
                for cmd in _train_sequence(
                    task,
                    method,
                    "default",
                    seed,
                    out_root,
                    tag,
                    num_envs,
                    updates,
                    [],
                    args.backend,
                    args.force,
                    episodes=1,
                ):
                    rc = _run(
                        cmd, _log_path(out_root, tag, task, method, "default", seed), args.dry_run
                    )
                    if rc != 0:
                        summary.append(
                            {"task": task, "method": method, "stage": "train", "returncode": rc}
                        )
                        break
                cmds = _eval_commands(
                    task,
                    method,
                    "default",
                    seed,
                    out_root,
                    tag,
                    stages=[],
                    eval_final=True,
                    det_only_stages=False,
                    num_envs=num_envs,
                    backend=args.backend,
                    force=args.force,
                )
                for cmd in cmds:
                    if method == "traditional":
                        continue
                    rc = _run(
                        cmd,
                        _log_path(out_root, tag, task, method + "_eval", "default", seed),
                        args.dry_run,
                    )
                    if rc != 0:
                        summary.append(
                            {"task": task, "method": method, "stage": "eval", "returncode": rc}
                        )
                        break
                summary.append({"task": task, "method": method, "status": "done"})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=["smoke", "pilot", "scans", "improvements", "tune_bc_residual", "plot"],
        required=True,
    )
    p.add_argument("--tasks", default="docking,satellite,rendezvous")
    p.add_argument("--methods", default=",".join(PILOT_METHODS))
    p.add_argument("--seeds", default=None, help="override pilot seed list")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--tag-map", default=None, help="plot only: per-task tag selection (docking=pilot_v2,...)"
    )
    p.add_argument("--backend", default="gpu")
    p.add_argument(
        "--force",
        action="store_true",
        help="rerun even when a run/eval already exists (NEVER deletes other results)",
    )
    p.add_argument("--dry-run", action="store_true")
    # scan overrides
    p.add_argument("--scan-task", default="docking")
    p.add_argument("--scan-seeds", default="7")
    p.add_argument("--scan-num-envs", type=int, default=512)
    p.add_argument("--scan-updates", type=int, default=100)
    # improvement-phase overrides (5 seeds by default)
    p.add_argument(
        "--improve-seeds", default=None, help="comma list; default %s" % IMPROVEMENT_SEEDS
    )
    p.add_argument("--improve-num-envs", type=int, default=None)
    p.add_argument("--improve-updates", type=int, default=None)
    # BC-residual PPO tuning phase
    p.add_argument(
        "--tune-seeds", default=None, help="comma list; default %s" % TUNE_SEEDS
    )
    p.add_argument("--tune-num-envs", type=int, default=None)
    p.add_argument("--tune-updates", type=int, default=None)
    p.add_argument(
        "--tune-conditions",
        default=None,
        help="comma list to restrict the tuning grid (default: default + all tune_bcr_*)",
    )
    args = p.parse_args()

    if args.phase == "plot":
        from analysis.plot import main as plot_main  # noqa: E402

        sys.argv = ["plot.py"] + (["--out-root", str(args.out_root)] if args.out_root else [])
        if args.tag_map:
            sys.argv += ["--tag-map", args.tag_map]
        return plot_main()

    if args.phase == "smoke":
        summary = run_smoke(args)
    elif args.phase == "scans":
        summary = run_scans(args)
    elif args.phase == "improvements":
        summary = run_improvements(args)
    elif args.phase == "tune_bc_residual":
        summary = run_tune(args)
    else:
        out_root = args.out_root or RESULTS_ROOT
        tag = args.tag or "pilot_v1"
        summary = []
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for task in tasks:
            plan = PILOT[task]
            seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else plan["seeds"]
            n = args.num_envs or plan["num_envs"]
            u = args.updates or plan["updates"]
            summary += run_task_plan(
                task, methods, seeds, n, u, out_root, tag, args.backend, args.force, args.dry_run
            )
    write_json(
        Path(args.out_root or RESULTS_ROOT)
        / "_logs"
        / (args.tag or args.phase)
        / "orchestrator_last.json",
        {
            "phase": args.phase,
            "summary": summary,
            "meta": run_metadata(
                task="all", method=args.phase, condition=args.tag or args.phase, seed=0
            ),
        },
    )
    print(json.dumps(summary[-5:], indent=2), flush=True)


if __name__ == "__main__":
    main()
