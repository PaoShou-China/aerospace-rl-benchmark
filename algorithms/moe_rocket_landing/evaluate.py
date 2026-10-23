"""Evaluate the frozen policy on the historical CPU first-episode protocol."""
import argparse
from pathlib import Path
import shutil
import json

import runtime
from train import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=runtime.CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", type=int, choices=(1, 2, 3, 4), default=4,
                        help="four is the full development protocol; fewer is a diagnostic")
    parser.add_argument("--physics-threads", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    args.parent, args.backend, args.max_envs = args.checkpoint, "cpu", 256
    run = Experiment(args)
    shutil.copy2(args.checkpoint, run.out / "student-final.pt")
    payload = run.torch.load(run.out / "student-final.pt", map_location="cpu", weights_only=False)
    result = run.evaluate(payload, seeds=runtime.DEV_SEEDS[:args.cohorts])
    runtime.verify_snapshot(run.out)
    run.save(status="completed", summary=result["summary"], training_started=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
