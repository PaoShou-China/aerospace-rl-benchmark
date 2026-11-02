"""Phase 5: Residual RL effect verification.

Runs PPO with residual=True (policy learns a correction on top of the task's
conventional controller) and compares against:
  * pure RL (PPO from scratch, residual=False)  -- from Phase 4 scratch runs
  * pure conventional controller                 -- from Phase 3 baseline suite

Reports cumulative success rate at the same budget.
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from algorithms.train_utils import run_training  # noqa: E402
from environments import ENV_REGISTRY  # noqa: E402


def _run_residual(task, updates, num_envs, out_root):
    out = out_root / f"{task}_residual"
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
        env_kwargs={"residual": True},
    )
    r = json.loads((out / "run.json").read_text())
    e = r["completed_episodes"]
    s = r["completed_successes"]
    return s / e if e > 0 else 0.0


def _scratch_sr(task):
    p = _ROOT / "results" / "bc_ab" / f"{task}_scratch" / "run.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    e = r["completed_episodes"]
    s = r["completed_successes"]
    return s / e if e > 0 else 0.0


def main():
    tasks = ["satellite", "docking"]
    updates, num_envs = 150, 256
    out_root = _ROOT / "results" / "residual_ab"
    out_root.mkdir(parents=True, exist_ok=True)
    # pure conventional controller SR from Phase 3 (seed 7, 256 envs)
    conv_sr = {"satellite": 1.0, "docking": 1.0}
    print(f"{'task':12s} {'conv':>7s} {'RL_scratch':>11s} {'RL_residual':>12s}")
    print("-" * 45)
    for task in tasks:
        sr_res = _run_residual(task, updates, num_envs, out_root)
        sr_scr = _scratch_sr(task)
        print(
            f"{task:12s} {conv_sr[task]:7.3f} "
            f"{sr_scr if sr_scr is not None else 'n/a':>11} "
            f"{sr_res:12.3f}"
        )


if __name__ == "__main__":
    main()
