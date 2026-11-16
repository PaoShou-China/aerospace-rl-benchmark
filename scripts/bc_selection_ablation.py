"""Compare BC expert-data curation variants before spending PPO budget.

Runs BC-only fits (``updates=0``) for the same task/seed and reports the
held-out expert-action MSE (the last 10% of replicas are withheld; no model
selection is performed).  This is the empirical check the curation literature
asks for: a ranking metric that looks good can still produce a worse policy, so
the subset should be validated on held-out actions (and ideally closed loop)
before it is used for the shipped PPO runs.

Variants:
    full      train on every collected expert step (no curation)
    tss       TSS/keyframe + per-episode stratified coverage (default recipe)
    uniform   stratified-uniform coverage with the same budget

Usage:
    python scripts/bc_selection_ablation.py --task rocket
    python scripts/bc_selection_ablation.py --task missile --num-envs 512 \
        --variants full,tss,uniform --epochs 100
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from algorithms.warmstart.bc_warmstart import BC_SELECT_DEFAULTS  # noqa: E402

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


def _bc_cfg(variant, args):
    select = None
    if variant != "full":
        select = {**BC_SELECT_DEFAULTS, "method": variant}
    return {
        "steps": None,  # full episode; curation reduces it below
        "epochs": args.epochs,
        "batch_size": 8192,
        "lr": 1e-3,
        "fit_critic": True,
        "select": select,
        "bc_val_fraction": args.val_fraction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rocket", choices=sorted(TRAIN_MODULES))
    ap.add_argument("--variants", default="full,tss,uniform")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--out-root", type=Path,
                    default=_ROOT / "results" / "bc_selection_ablation")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    module = importlib.import_module(TRAIN_MODULES[args.task])
    rows = []
    for seed in seeds:
        for variant in variants:
            out = args.out_root / args.task / variant / f"seed_{seed}"
            bc = _bc_cfg(variant, args)
            print(f"[ablation] {args.task}/{variant}/seed_{seed} -> {out}", flush=True)
            t0 = time.time()
            module.main(
                seed=seed,
                out_dir=str(out),
                updates=0,
                num_envs=args.num_envs,
                clean=True,
                performance=True,
                bc_warmstart=bc,
            )
            report = json.loads((out / "run.json").read_text())
            bc_report = report.get("bc_warmstart") or {}
            rows.append({
                "task": args.task,
                "seed": seed,
                "variant": variant,
                "pairs_collected": bc_report.get("pairs_collected"),
                "pairs_selected": bc_report.get("pairs_selected"),
                "pairs_val": bc_report.get("pairs_val"),
                "final_train_actor_mse": bc_report.get("final_train_actor_mse"),
                "final_val_actor_mse": bc_report.get("final_val_actor_mse"),
                "collect_wall_seconds": bc_report.get("collect_wall_seconds"),
                "fit_wall_seconds": bc_report.get("fit_wall_seconds"),
                "wall_seconds": round(time.time() - t0, 1),
            })
            print("[ablation] " + json.dumps(rows[-1], sort_keys=True), flush=True)

    summary = args.out_root / args.task / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, indent=2))
    print(f"[ablation] wrote {summary}", flush=True)
    print("\n| variant | pairs | val actor MSE | train actor MSE | wall s |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['variant']} | {r['pairs_selected']} | "
              f"{r['final_val_actor_mse']} | {r['final_train_actor_mse']} | "
              f"{r['wall_seconds']} |")


if __name__ == "__main__":
    main()
