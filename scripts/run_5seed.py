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
RETRY_QUEUE = OUT_ROOT / "_retry_queue.json"
MAX_RETRIES = 3

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
def _load_retries():
    if RETRY_QUEUE.exists():
        try:
            return json.loads(RETRY_QUEUE.read_text())
        except (OSError, json.JSONDecodeError):
            return []
    return []


def _save_retries(retries):
    RETRY_QUEUE.write_text(json.dumps(retries, indent=2))


def _retry_job(job):
    """Re-run a failed training job. ``job`` is (task, method, seed) and we
    trust run_training(..., clean=True) to wipe the old directory."""
    return _train_job(job)


def _subprocess_train_job(job):
    """Run one (task, method, seed) training run as a fully-isolated Python
    subprocess. Genesis C++-level exceptions (the rigid solver's nan check)
    crash the spawned child but never touch the parent orchestrator, so a
    single bad run can't take down the whole schedule.

    We write a temp script and execute it instead of ``python -c`` because the
    latter's quoting/f-string nesting is a bottomless pit of SyntaxErrors;
    temp files give us plain, verifiable Python."""
    task, method, seed = job
    out = OUT_ROOT / task / method / f"seed_{seed}"
    if (out / "run.json").exists():
        _log(f"SKIP  {task}/{method}/seed_{seed} (run.json exists)")
        return
    out.mkdir(parents=True, exist_ok=True)

    bc_dir = OUT_ROOT / task / "bc_shared" / f"seed_{seed}"
    ckpt = bc_dir / "bc_checkpoint.pt"
    bc_src = "None"
    if method == "ppo_scratch":
        bc_src = "None"
    elif method == "bc_ppo":
        if not ckpt.exists():
            _log(f"BC    {task}/seed_{seed}: fitting shared BC initialization")
            bc_init_path = OUT_ROOT / "_tmp_bc_init.py"
            bc_init_path.write_text(
                bc_init_template(seed, str(bc_dir), str(ckpt), task)
            )
            proc = subprocess.run(
                [sys.executable, str(bc_init_path)], cwd=_ROOT,
                capture_output=True, text=True, timeout=900
            )
            bc_init_path.unlink(missing_ok=True)
            if proc.returncode != 0:
                _log(f"FAIL  bc_shared {task}/seed_{seed}\n{proc.stderr[-4000:]}")
                raise RuntimeError("BC init failed")
        bc_src = (
            "{'steps': %d, 'epochs': %d, 'batch_size': %d, 'lr': %g,\n"
            "               'fit_critic': %s, 'checkpoint': %r}" %
            (BC_RECIPE["steps"], BC_RECIPE["epochs"], BC_RECIPE["batch_size"],
             BC_RECIPE["lr"], BC_RECIPE["fit_critic"], str(ckpt))
        )

    _log(f"START {task}/{method}/seed_{seed}")
    t0 = time.perf_counter()
    runner_path = OUT_ROOT / "_tmp_runner.py"
    runner_path.write_text(
        runner_template(
            task=task, method=method, seed=seed, out=str(out),
            bc_src=bc_src, residual_rho=RESIDUAL_RHO, backend=BACKEND,
        )
    )
    proc = subprocess.run(
        [sys.executable, str(runner_path)], cwd=_ROOT,
        capture_output=True, text=True, timeout=None,
    )
    runner_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        _log(f"FAIL  {task}/{method}/seed_{seed}\n{proc.stderr[-4000:]}")
        retries = _load_retries()
        retries.append({
            "job": list(job), "attempt": 1,
            "last_error": (proc.stderr.splitlines()[-1] if proc.stderr else "unknown")[:300],
        })
        _save_retries(retries)
        return
    _log(f"DONE  {task}/{method}/seed_{seed} in {time.perf_counter() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# temp-script template helpers
# --------------------------------------------------------------------------- #
def bc_init_template(seed, out_dir, save_checkpoint, task):
    return (
        "import sys; sys.path.insert(0, %r)\n"
        "from %s import main\n"
        "main(\n"
        "    seed=%d,\n"
        "    out_dir=%r,\n"
        "    updates=0,\n"
        "    clean=True,\n"
        "    performance=True,\n"
        "    bc_warmstart={\n"
        "        'steps': %d,\n"
        "        'epochs': %d,\n"
        "        'batch_size': %d,\n"
        "        'lr': %g,\n"
        "        'fit_critic': %s,\n"
        "        'save_checkpoint': %r,\n"
        "    },\n"
        ")\n" % (
            str(_ROOT),
            TRAIN_MODULES[task],
            seed,
            out_dir,
            BC_RECIPE["steps"], BC_RECIPE["epochs"], BC_RECIPE["batch_size"],
            BC_RECIPE["lr"], BC_RECIPE["fit_critic"],
            save_checkpoint,
        )
    )


def runner_template(*, task, method, seed, out, bc_src, residual_rho, backend):
    imports = [
        "import sys; sys.path.insert(0, %r)" % str(_ROOT),
        "import traceback",
        "from %s import main" % TRAIN_MODULES[task],
    ]
    extras = []
    if method == "residual_ppo":
        imports.extend([
            "from algorithms.baselines.baselines import build_baseline",
            "from algorithms.residual.residual_env import ResidualVecEnv, zero_init_policy_output",
        ])
        extras.append("def _wrap(env, _task=%r):" % task)
        extras.append(
            "    return ResidualVecEnv(env, build_baseline(_task, env, None),"
            " residual_rho=%s, task=_task)" % residual_rho
        )
        extras.append("overrides['env_wrapper'] = _wrap")
        extras.append("overrides['runner_init_hook'] = zero_init_policy_output")

    header = "\n".join(imports)
    body = "\n".join(extras)
    return (
        "%s\n"
        "overrides = {\n"
        "    'seed': %d,\n"
        "    'out_dir': %r,\n"
        "    'backend': %r,\n"
        "    'clean': True,\n"
        "    'performance': True,\n"
        "    'bc_warmstart': %s,\n"
        "}\n"
        "%s\n"
        "try:\n"
        "    main(**overrides)\n"
        "except BaseException:\n"
        "    sys.stderr.write(traceback.format_exc())\n"
        "    sys.stderr.flush()\n"
        "    raise\n" % (header, seed, out, backend, bc_src, body)
    )


def _train_job_dispatcher(job):
    """Pool worker entry -- dispatches to the subprocess runner. Kept as a thin
    wrapper so _run_pool(..., _train_job_dispatcher, workers) works identically
    to before but the actual heavy lifting happens in isolated subprocesses."""
    try:
        _subprocess_train_job(job)
    except Exception:
        _log(f"FAIL  {job} dispatcher\n{traceback.format_exc()}")


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
        return  # soft failure -- don't take down the pool
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
        # matched-condition comparison: the learned policies train/evaluate
        # with the env's per-episode randomization ON, so the baselines must
        # use it too (the plain CLI defaults to the nominal plant)
        "--domain-randomize",
        "on",
    ]
    if variant:
        cmd += ["--baseline", variant]
    _log(f"BASE  {task}/{name} @ eval-seed {eval_seed}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _log(f"FAIL  baseline {task}/{name}/{eval_seed}\n{proc.stderr[-4000:]}")
        return  # soft failure
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
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # Measured 2026-11-04 on one RTX 3090: 6 workers doubled per-job wall time
    # vs 3 (quadx 4.9h -> 9.1h) with zero throughput gain -- the GPU is the
    # bottleneck and each worker also burns a full CPU core launching kernels.
    # 2026-11-10: default reduced to 2 to keep GPU util below 100%.
    if args.workers > 3:
        _log(
            f"WARNING: --workers {args.workers} > 3 on a single GPU historically "
            "reduced total throughput; recommend --workers 3 (or 2)"
        )

    t0 = time.perf_counter()
    if args.phase in ("train", "all"):
        jobs = _train_jobs(args.tasks, args.seeds, args.methods)
        _log(f"=== train phase: {len(jobs)} runs, {args.workers} workers ===")
        _run_pool(jobs, _train_job_dispatcher, args.workers)
        # Auto-retry any Genesis-failed training runs (up to MAX_RETRIES).
        # These show up as the rigid solver's nan check under GPU saturation
        # and succeed deterministically on the next attempt.
        while True:
            retries = _load_retries()
            if not retries:
                break
            # Retry in waves: every entry with attempt < MAX_RETRIES gets one
            # more go; any successes are removed, any new failures increment
            # attempt; exhausted entries are moved to _exhausted_retry.json.
            wave = [tuple(r["job"]) for r in retries if r["attempt"] < MAX_RETRIES]
            exhausted = [r for r in retries if r["attempt"] >= MAX_RETRIES]
            if not wave:
                if exhausted:
                    _log(f"=== {len(exhausted)} jobs hit MAX_RETRIES -- see {RETRY_QUEUE} ===")
                RETRY_QUEUE.unlink(missing_ok=True)
                break
            _log(f"=== retry wave ({len(wave)} jobs exhausted={len(exhausted)}) ===")
            _run_pool(wave, _retry_job, args.workers)
            # Filter: succeeded jobs have run.json now; failed ones are re-queued
            # by _train_job itself and we re-load.
            current = _load_retries()
            if exhausted:
                # merge exhausted back so the next outer loop iteration can
                # surface them; we only break when current is empty OR all are
                # exhausted.
                pass
            # _train_job already appended new failures with attempt=1; bump
            # attempt counter on pre-existing entries we just retried.
            seen = {tuple(r["job"]): r for r in current}
            for entry in retries:
                j = tuple(entry["job"])
                if j in wave and j in seen:
                    seen[j]["attempt"] = max(seen[j]["attempt"], entry["attempt"] + 1)
            _save_retries(list(seen.values()))
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
