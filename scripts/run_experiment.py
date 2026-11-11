"""实验编排：训练 → 基线 → 评估。

用法:
    python scripts/run_experiment.py                     # 全部阶段
    python scripts/run_experiment.py --phase train       # 只训练
    python scripts/run_experiment.py --phase eval        # 只评估
    python scripts/run_experiment.py --phase baselines   # 只跑基线
    python scripts/run_experiment.py --tasks quadx missile --seeds 11 22

实验矩阵: 3 方法 × 5 种子 × 10 任务 = 150 次训练
    methods:  ppo_scratch | bc_ppo | residual_ppo
    seeds:    11 22 33 44 55
    eval种子: 1011 1022 1033 1044 1055

输出目录: results/5seed/<task>/<method>/seed_<s>/
已存在 run.json 的 run 会自动跳过（可断点续跑）。
日志只记录关键事件（开始/完成/失败），不保存子进程的训练 stdout。
"""

import argparse
import importlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 实验参数（与 run_5seed.py 一致，保证对比公平）
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

TASKS = [
    "quadx", "quadx_track", "fixedwing", "quadplane", "rocket",
    "missile", "satellite", "transfer", "docking", "rendezvous",
]
TRAIN_SEEDS = [11, 22, 33, 44, 55]
EVAL_SEEDS = [1011, 1022, 1033, 1044, 1055]
METHODS = ["ppo_scratch", "bc_ppo", "residual_ppo"]

BC_RECIPE = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)
RESIDUAL_RHO = 0.05
BACKEND = "gpu"

OUT_ROOT = _ROOT / "results" / "5seed"
LOG_FILE = OUT_ROOT / "experiment.log"

# 每个任务对应的训练模块（提供 main(**overrides) 入口）
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


def log(msg):
    """写一行日志到控制台和 experiment.log。"""
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# 训练阶段
# --------------------------------------------------------------------------- #
def _train_one(job):
    """跑一次 (task, method, seed) 训练，子进程隔离防止 Genesis 崩溃传染。"""
    task, method, seed = job
    out = OUT_ROOT / task / method / f"seed_{seed}"

    # 已完成则跳过
    if (out / "run.json").exists():
        return

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # BC 初始化：bc_ppo 复用同 task+seed 的共享 BC checkpoint
    bc_dir = OUT_ROOT / task / "bc_shared" / f"seed_{seed}"
    ckpt = bc_dir / "bc_checkpoint.pt"
    bc_src = "None"
    if method == "bc_ppo":
        if not ckpt.exists():
            log(f"BC    {task}/seed_{seed}")
            _run_bc_init(task, seed, bc_dir, ckpt)
        bc_src = (
            "{'steps': %d, 'epochs': %d, 'batch_size': %d, 'lr': %g,"
            " 'fit_critic': %s, 'checkpoint': %r}"
            % (BC_RECIPE["steps"], BC_RECIPE["epochs"], BC_RECIPE["batch_size"],
               BC_RECIPE["lr"], BC_RECIPE["fit_critic"], str(ckpt))
        )

    # 写临时 runner 脚本，避免 python -c 的引号嵌套问题
    runner = OUT_ROOT / "_tmp_runner.py"
    runner.write_text(_runner_template(task, method, seed, str(out), bc_src))
    proc = subprocess.run(
        [sys.executable, str(runner)], cwd=_ROOT,
        capture_output=True, text=True, timeout=None,
    )
    runner.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        err = proc.stderr.splitlines()[-1] if proc.stderr else "unknown"
        log(f"FAIL  {task}/{method}/seed_{seed} ({elapsed:.0f}s) {err[:120]}")
    else:
        log(f"DONE  {task}/{method}/seed_{seed} ({elapsed:.0f}s)")


def _run_bc_init(task, seed, bc_dir, ckpt):
    """生成共享 BC 初始化 checkpoint（同 task+seed 的 bc_ppo 复用）。"""
    bc_dir.mkdir(parents=True, exist_ok=True)
    script = OUT_ROOT / "_tmp_bc_init.py"
    script.write_text(_bc_init_template(task, seed, str(bc_dir), str(ckpt)))
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=_ROOT,
        capture_output=True, text=True, timeout=900,
    )
    script.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"BC init failed: {proc.stderr[-500:]}")


def _bc_init_template(task, seed, bc_dir, ckpt):
    """BC 初始化：调用训练模块的 main(updates=0)，只跑 BC 预热并保存 checkpoint。"""
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
        "        'steps': %d, 'epochs': %d, 'batch_size': %d, 'lr': %g,\n"
        "        'fit_critic': %s, 'save_checkpoint': %r,\n"
        "    },\n"
        ")\n"
        % (
            str(_ROOT), TRAIN_MODULES[task], seed, bc_dir,
            BC_RECIPE["steps"], BC_RECIPE["epochs"], BC_RECIPE["batch_size"],
            BC_RECIPE["lr"], BC_RECIPE["fit_critic"], ckpt,
        )
    )


def _runner_template(task, method, seed, out, bc_src):
    """生成训练 runner 脚本。"""
    # residual_ppo 需要包一层残差环境
    env_wrapper = (
        "('residual', {'rho': %g})" % RESIDUAL_RHO if method == "residual_ppo" else "None"
    )
    return (
        "import sys, importlib\n"
        "sys.path.insert(0, %r)\n"
        "mod = importlib.import_module(%r)\n"
        "mod.main(\n"
        "    seed=%d,\n"
        "    out_dir=%r,\n"
        "    bc_warmstart=%s,\n"
        "    env_wrapper=%s,\n"
        "    performance=True,\n"
        ")\n"
        % (str(_ROOT), TRAIN_MODULES[task], seed, out, bc_src, env_wrapper)
    )


# --------------------------------------------------------------------------- #
# 评估阶段
# --------------------------------------------------------------------------- #
def _eval_one(job):
    """评估一个训练好的 checkpoint 在指定 eval seed 下的表现。"""
    task, method, seed, eval_seed = job
    run_dir = OUT_ROOT / task / method / f"seed_{seed}"
    report = run_dir / f"eval_seed_{eval_seed}.json"
    if report.exists():
        return
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "eval.py"),
        task, "--out", str(run_dir), "--seed", str(eval_seed),
        "--report-name", report.name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"FAIL  eval {task}/{method}/seed_{seed}@{eval_seed}")


# --------------------------------------------------------------------------- #
# 基线阶段
# --------------------------------------------------------------------------- #
def _baseline_one(job):
    """跑经典控制器基线（default + 2 变体）。"""
    task, variant, eval_seed = job
    name = variant or "default"
    out = OUT_ROOT / task / "baselines" / name / f"seed_{eval_seed}"
    if (out / "baseline.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    # 基线使用与训练相同的环境配置（从 ppo_scratch/seed_11 复制 config.json）
    src = OUT_ROOT / task / "ppo_scratch" / "seed_11" / "config.json"
    if src.exists() and not (out / "config.json").exists():
        shutil.copy(src, out / "config.json")
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "baseline_eval.py"),
        task, "--out", str(out), "--seed", str(eval_seed),
    ]
    if variant:
        cmd += ["--baseline", variant]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"FAIL  baseline {task}/{name}@{eval_seed}")


def _baseline_variants(task):
    from algorithms.baselines.baselines import BASELINE_VARIANTS, BUILDERS
    default_cls = BUILDERS[task]
    others = [n for n, c in BASELINE_VARIANTS.get(task, {}).items() if c is not default_cls]
    return [None] + sorted(others)[:2]


# --------------------------------------------------------------------------- #
# 进程池
# --------------------------------------------------------------------------- #
def _run_pool(jobs, worker, workers):
    if not jobs:
        return
    # spawn 上下文：每个子进程独立 CUDA 上下文，maxtasksperchild=1 避免状态泄漏
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, jobs)):
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                log(f"进度 {i + 1}/{len(jobs)}")


def main():
    parser = argparse.ArgumentParser(description="实验编排：训练+基线+评估")
    parser.add_argument("--phase", choices=("train", "eval", "baselines", "all"), default="all")
    parser.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", nargs="*", type=int, default=EVAL_SEEDS)
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # 2 workers 保持 GPU 利用率在 70-80%，避免 100% 满载
    workers = min(args.workers, 3)

    # --- 训练 ---
    if args.phase in ("train", "all"):
        jobs = [(t, m, s) for t in args.tasks for s in args.seeds for m in args.methods]
        log(f"=== 训练阶段: {len(jobs)} runs, {workers} workers ===")
        _run_pool(jobs, _train_one, workers)

    # --- 基线 ---
    if args.phase in ("baselines", "all"):
        jobs = [(t, v, es) for t in args.tasks for v in _baseline_variants(t) for es in args.eval_seeds]
        log(f"=== 基线阶段: {len(jobs)} runs ===")
        _run_pool(jobs, _baseline_one, workers)

    # --- 评估 ---
    if args.phase in ("eval", "all"):
        jobs = [
            (t, m, s, es)
            for t in args.tasks for s in args.seeds for m in args.methods for es in args.eval_seeds
        ]
        log(f"=== 评估阶段: {len(jobs)} runs ===")
        _run_pool(jobs, _eval_one, workers)

    log("=== 全部完成 ===")


if __name__ == "__main__":
    main()
