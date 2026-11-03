"""5-seed benchmark protocol orchestrator (2026-11-03 protocol).

Runs the full method matrix on the 10 registered tasks:

    methods      ppo_scratch | bc_ppo | residual_ppo
                 (ppo_scratch = BC_WARMSTART None; bc_ppo shares one BC
                 initialization per task+seed; residual_ppo = zero-init
                 residual on the task's default classical controller,
                 rho = 0.05)
    train seeds  11 22 33 44 55
    eval seeds   1011 1022 1033 1044 1055
    layout       results/5seed/<task>/<method>/seed_<s>/
                 results/5seed/<task>/bc_shared/seed_<s>/bc_checkpoint.pt
                 results/5seed/<task>/baselines/<variant>/seed_<s>/

Phases (default --phase all runs them in order):
    train      3 methods x 5 seeds x 10 tasks, <=3 concurrent processes.
               The BC-only initialization (updates=0) is produced lazily into
               bc_shared/ and reused by every bc_ppo arm of the same task+seed.
               A run whose run.json already exists is skipped (resume-safe).
    eval       deterministic eval of every final checkpoint at each eval seed
               (scripts/eval.py --report-name eval_seed_<s>.json).
    baselines  the task's default classical controller + 2 structural variants
               x 5 eval seeds (scripts/baseline_eval.py), env_kwargs restored
               from the task's ppo_scratch training config.

Every run parameter (updates, hyperparameters, wind, reward variants) is
hard-coded in the per-task train entries; this script only overrides seed /
out_dir / bc_warmstart / residual wrapper per the protocol.

Usage: python scripts/run_5seed.py [--phase all] [--tasks ...] [--workers 3]
"""

import argparse
import importlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

TASKS = [
    "quadx",
    "quadx_track",
    "fixedwing",
    "quadplane",
    "rocket",
    "missile",
    "satellite",
    "transfer",
    "docking",
    "rendezvous",
]
TRAIN_SEEDS = [11, 22, 33, 44, 55]
EVAL_SEEDS = [1011, 1022, 1033, 1044, 1055]
METHODS = ["ppo_scratch", "bc_ppo", "residual_ppo"]
BC_RECIPE = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)
RESIDUAL_RHO = 0.05
BACKEND = "gpu"
OUT_ROOT = _ROOT / "results" / "5seed"
LOG = OUT_ROOT / "orchestrator.log"

TRAIN_MODULES = {
    "quadx": "environments.multirotor.train_quadx",
    "quadx_track": "environments.multirotor.train_quadx_track",
    "fixedwing": "environments.fixedwing.train_fixedwing",
    "quadplane": "environments.vtol.train_quadplane",
    "rocket": "environments.rocket.train_rocket",
    "missile": "environments.missile.train_missile",
    "satellite": "environments.satellite.attitude_control.train_satellite",
    "transfer": "environments.satellite.orbit_transfer.train_transfer",
    "docking": "environments.satellite.rendezvous_docking.train_docking",
    "rendezvous": "environments.satellite.rendezvous_docking.train_rendezvous",
}


def _log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# train jobs
# --------------------------------------------------------------------------- #
def _train_job(job):
    """One (task, method, seed) training run; runs in a spawned worker."""
    task, method, seed = job
    sys.path.insert(0, str(_ROOT))
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(TRAIN_MODULES[task])
        out = OUT_ROOT / task / method / f"seed_{seed}"
        if (out / "run.json").exists():
            _log(f"SKIP  {task}/{method}/seed_{seed} (run.json exists)")
            return
        overrides = dict(seed=seed, out_dir=str(out), backend=BACKEND, clean=True)
        if method == "ppo_scratch":
            overrides["bc_warmstart"] = None
        elif method == "bc_ppo":
            bc_dir = OUT_ROOT / task / "bc_shared" / f"seed_{seed}"
            ckpt = bc_dir / "bc_checkpoint.pt"
            if not ckpt.exists():
                _log(f"BC    {task}/seed_{seed}: fitting shared BC initialization")
                mod.main(
                    seed=seed,
                    out_dir=str(bc_dir),
                    updates=0,  # BC-only arm: no RL updates
                    clean=True,
                    backend=BACKEND,
                    bc_warmstart={**BC_RECIPE, "save_checkpoint": str(ckpt)},
                )
            overrides["bc_warmstart"] = {**BC_RECIPE, "checkpoint": str(ckpt)}
        elif method == "residual_ppo":
            from algorithms.baselines.baselines import build_baseline
            from algorithms.residual.residual_env import (
                ResidualVecEnv,
                zero_init_policy_output,
            )

            def wrap(env, _task=task):
                controller = build_baseline(_task, env, None)
                return ResidualVecEnv(
                    env, controller, residual_rho=RESIDUAL_RHO, task=_task
                )

            overrides.update(
                env_wrapper=wrap,
                runner_init_hook=zero_init_policy_output,
                bc_warmstart=None,
                extra_config={"method": method, "residual_rho": RESIDUAL_RHO},
            )
        else:
            raise ValueError(f"unknown method {method}")
        _log(f"START {task}/{method}/seed_{seed}")
        mod.main(**overrides)
        _log(f"DONE  {task}/{method}/seed_{seed} in {time.perf_counter() - t0:.0f}s")
    except Exception:
        _log(f"FAIL  {task}/{method}/seed_{seed}\n{traceback.format_exc()}")
        raise


def _train_jobs(tasks, seeds, methods):
    # interleave tasks so an early failure surfaces across the matrix
    jobs = []
    for task in tasks:
        for seed in seeds:
            for method in methods:
                jobs.append((task, method, seed))
    return jobs


# --------------------------------------------------------------------------- #
# eval jobs (trained checkpoints)
# --------------------------------------------------------------------------- #
def _eval_job(job):
    task, method, seed, eval_seed = job
    t0 = time.perf_counter()
    run_dir = OUT_ROOT / task / method / f"seed_{seed}"
    report = run_dir / f"eval_seed_{eval_seed}.json"
    if report.exists():
        return
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "eval.py"),
        task,
        "--out",
        str(run_dir),
        "--seed",
        str(eval_seed),
        "--report-name",
        report.name,
    ]
    _log(f"EVAL  {task}/{method}/seed_{seed} @ eval-seed {eval_seed}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _log(f"FAIL  eval {task}/{method}/seed_{seed}/{eval_seed}\n{proc.stderr[-4000:]}")
        raise RuntimeError(f"eval failed: {report}")
    _log(f"DONE  eval {task}/{method}/seed_{seed}/{eval_seed} in {time.perf_counter() - t0:.0f}s")


def _eval_jobs(tasks, seeds, eval_seeds, methods):
    return [
        (task, method, seed, es)
        for task in tasks
        for seed in seeds
        for method in methods
        for es in eval_seeds
    ]


# --------------------------------------------------------------------------- #
# baseline jobs (classical controllers)
# --------------------------------------------------------------------------- #
def _baseline_variants(task):
    """Default + 2 structural variants (the >=3 traditional baselines rule)."""
    from algorithms.baselines.baselines import BASELINE_VARIANTS, BUILDERS

    default_cls = BUILDERS[task]
    variants = [n for n, c in BASELINE_VARIANTS.get(task, {}).items() if c is not default_cls]
    return [None] + sorted(variants)[:2]


def _baseline_job(job):
    task, variant, eval_seed = job
    t0 = time.perf_counter()
    name = variant or "default"
    out = OUT_ROOT / task / "baselines" / name / f"seed_{eval_seed}"
    if (out / "baseline.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    # baseline_eval restores the training env_kwargs from out/config.json:
    # copy the task's ppo_scratch training config so baselines see the same
    # wind / reward-variant conditions as the learned policies.
    src = OUT_ROOT / task / "ppo_scratch" / "seed_11" / "config.json"
    if src.exists() and not (out / "config.json").exists():
        shutil.copy(src, out / "config.json")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "baseline_eval.py"),
        task,
        "--out",
        str(out),
        "--seed",
        str(eval_seed),
    ]
    if variant:
        cmd += ["--baseline", variant]
    _log(f"BASE  {task}/{name} @ eval-seed {eval_seed}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _log(f"FAIL  baseline {task}/{name}/{eval_seed}\n{proc.stderr[-4000:]}")
        raise RuntimeError(f"baseline failed: {out}")
    _log(f"DONE  baseline {task}/{name}/{eval_seed} in {time.perf_counter() - t0:.0f}s")


def _baseline_jobs(tasks, eval_seeds):
    return [
        (task, variant, es)
        for task in tasks
        for variant in _baseline_variants(task)
        for es in eval_seeds
    ]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _run_pool(jobs, worker, workers):
    if not jobs:
        return
    ctx = mp.get_context("spawn")
    # maxtasksperchild=1: one CUDA context per process, no cross-run state
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, jobs)):
            _log(f"progress {i + 1}/{len(jobs)} jobs finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("train", "eval", "baselines", "all"), default="all")
    parser.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", nargs="*", type=int, default=EVAL_SEEDS)
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.phase in ("train", "all"):
        jobs = _train_jobs(args.tasks, args.seeds, args.methods)
        _log(f"=== train phase: {len(jobs)} runs, {args.workers} workers ===")
        _run_pool(jobs, _train_job, args.workers)
    if args.phase in ("baselines", "all"):
        jobs = _baseline_jobs(args.tasks, args.eval_seeds)
        _log(f"=== baselines phase: {len(jobs)} evals, {args.workers} workers ===")
        _run_pool(jobs, _baseline_job, args.workers)
    if args.phase in ("eval", "all"):
        jobs = _eval_jobs(args.tasks, args.seeds, args.eval_seeds, args.methods)
        _log(f"=== eval phase: {len(jobs)} evals, {args.workers} workers ===")
        _run_pool(jobs, _eval_job, args.workers)
    _log(f"=== phase {args.phase} complete in {(time.perf_counter() - t0) / 3600:.2f} h ===")


if __name__ == "__main__":
    main()
