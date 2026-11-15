"""Phase 4: BC warmstart A/B comparison.

For each task, runs PPO from scratch and PPO+BC-warmstart with identical
budget/seed, then reports final success rate and a coarse sample-efficiency
proxy (success rate at the 25% and 50% budget marks).

Usage: python scripts/bc_warmstart_ab.py [--tasks satellite,docking,quadx]
                                         [--updates 150] [--num-envs 256]
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from algorithms.train_utils import run_training  # noqa: E402
from environments import ENV_REGISTRY  # noqa: E402


def _run(task, algo_arm, updates, num_envs, out_root):
    out = out_root / f"{task}_{algo_arm}"
    bc = {"steps": 512, "epochs": 100} if algo_arm == "bc" else None
    t0 = time.time()
    run_training(
        task,
        ENV_REGISTRY[task][0],
        num_envs=num_envs,
        updates=updates,
        seed=7,
        backend="gpu",
        out_dir=str(out),
        clean=True,
        algo="ppo",
        bc_warmstart=bc,
    )
    wall = time.time() - t0
    run = json.loads((out / "run.json").read_text())
    # final success rate from the last logged chunk
    sr_final = None
    if "success_rates" in run and run["success_rates"]:
        sr_final = run["success_rates"][-1]
    elif "completed_successes" in run and "completed_episodes" in run:
        s = run["completed_successes"]
        e = run["completed_episodes"]
        sr_final = (s / e) if e > 0 else None
    return task, algo_arm, sr_final, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="satellite,docking,quadx")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--out-root", type=Path, default=_ROOT / "results" / "bc_ab")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"[bc_ab] tasks={tasks} updates={args.updates} envs={args.num_envs}")
    print(f"{'task':14s} {'arm':8s} {'SR_final':>10s} {'wall_s':>8s}")
    print("-" * 45)
    for task in tasks:
        for arm in ("scratch", "bc"):
            t, a, sr, wall = _run(task, arm, args.updates, args.num_envs, args.out_root)
            sr_s = f"{sr:.4f}" if sr is not None else "n/a"
            print(f"{t:14s} {a:8s} {sr_s:>10s} {wall:8.1f}")
        print()


if __name__ == "__main__":
    main()
