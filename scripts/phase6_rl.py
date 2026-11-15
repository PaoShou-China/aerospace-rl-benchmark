"""Phase 6 RL benchmark: PPO on fast tasks with multiple seeds."""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from algorithms.train_utils import run_training  # noqa: E402
from environments import ENV_REGISTRY  # noqa: E402


def _run(task, algo, seed, updates, num_envs, out_root):
    out = out_root / task / algo / f"seed_{seed}"
    t0 = time.time()
    run_training(
        task,
        ENV_REGISTRY[task][0],
        num_envs=num_envs,
        updates=updates,
        seed=seed,
        backend="gpu",
        out_dir=str(out),
        clean=True,
        algo=algo,
    )
    wall = time.time() - t0
    r = json.loads((out / "run.json").read_text())
    e = r["completed_episodes"]
    s = r["completed_successes"]
    sr = s / e if e > 0 else 0.0
    return sr, wall, r.get("train_throughput_steps_per_s", 0)


def main():
    tasks = ["satellite", "docking"]
    algos = ["ppo"]
    seeds = [7, 13]
    updates, num_envs = 200, 256
    out_root = _ROOT / "results" / "phase6_rl"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{'task':12s} {'algo':10s} {'seed':>5s} {'SR':>7s} {'wall_s':>8s} {'sps':>8s}")
    print("-" * 55)
    for task in tasks:
        for algo in algos:
            for seed in seeds:
                sr, wall, sps = _run(task, algo, seed, updates, num_envs, out_root)
                print(f"{task:12s} {algo:10s} {seed:5d} {sr:7.3f} {wall:8.1f} {sps:8.0f}")
        print()


if __name__ == "__main__":
    main()
