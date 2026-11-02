"""Training smoke test: all 10 tasks with PPO.

Runs a tiny budget (5 updates, 32 envs) for every task to verify the
full pipeline boots, trains without crashing, and writes checkpoint.pt +
run.json + config.json.  Three runs execute concurrently (matching the
project's parallel-training preference).

Usage:
    python scripts/smoke_test.py [--updates 5] [--num-envs 32] [--workers 3]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from algorithms.train_utils import run_training  # noqa: E402
from environments import ENV_REGISTRY  # noqa: E402

ALGOS = ["ppo"]


def _run_one(task, algo, updates, num_envs, out_root):
    out_dir = out_root / f"{task}_{algo}"
    t0 = time.time()
    try:
        run_training(
            task,
            ENV_REGISTRY[task][0],
            num_envs=num_envs,
            updates=updates,
            seed=7,
            backend="gpu",
            out_dir=str(out_dir),
            clean=True,
            algo=algo,
        )
        wall = time.time() - t0
        # verify artifacts
        ckpt = out_dir / "checkpoint.pt"
        rj = out_dir / "run.json"
        cj = out_dir / "config.json"
        ok = ckpt.exists() and rj.exists() and cj.exists()
        return task, algo, "PASS" if ok else "FAIL(missing artifacts)", wall
    except Exception as exc:  # noqa: BLE001
        return task, algo, f"FAIL({type(exc).__name__}: {exc})", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=_ROOT / "results" / "smoke")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(t, a) for t in sorted(ENV_REGISTRY) for a in ALGOS]
    print(f"[smoke] {len(jobs)} runs, {args.workers} workers, updates={args.updates} envs={args.num_envs}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_run_one, t, a, args.updates, args.num_envs, args.out_root): (t, a)
            for t, a in jobs
        }
        for fut in as_completed(futs):
            task, algo, status, wall = fut.result()
            results.append((task, algo, status, wall))
            print(f"  {task:14s} {algo:10s} {status:30s} {wall:6.1f}s", flush=True)

    results.sort()
    n_pass = sum(1 for _, _, s, _ in results if s.startswith("PASS"))
    print(f"\n[smoke] {n_pass}/{len(results)} passed")
    for task, algo, status, wall in results:
        print(f"  {task:14s} {algo:10s} {status}")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
