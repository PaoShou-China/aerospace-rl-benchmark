"""Aggregate the 5-seed benchmark protocol results into a report.

Reads results/5seed/:
    <task>/<method>/seed_<s>/eval_seed_<es>.json   learned-policy evals
    <task>/<method>/seed_<s>/run.json              training cost records
    <task>/baselines/<variant>/seed_<es>/baseline.json  classical baselines

Statistics (protocol: fixed budget, no checkpoint selection):
    - per (task, method): success rate mean / std / min across the 5 training
      seeds, each seed's value = mean over the 5 held-out eval seeds.
    - cost, dual caliber: mean interactions (transitions from run.json) and
      mean train wall-clock seconds.
    - optional sample-efficiency curves from the tensorboard event files
      (first update reaching 90% of the run's own final rolling success).

Writes results/5seed/summary.json and results/5seed/report.md.
Usage: python scripts/aggregate_5seed.py
"""

import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = _ROOT / "results" / "5seed"

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
METHODS = ["ppo_scratch", "bc_ppo", "residual_ppo"]
TRAIN_SEEDS = [11, 22, 33, 44, 55]
EVAL_SEEDS = [1011, 1022, 1033, 1044, 1055]

SUCCESS_KEYS = ("success_rate", "success")


def _success_rate(report):
    for k in SUCCESS_KEYS:
        if k in report:
            return float(report[k])
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _success_tag(events_dir):
    """Find the rolling-success scalar tag in a run's tensorboard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None, None
    acc = EventAccumulator(str(events_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = [t for t in acc.Tags().get("scalars", []) if "success" in t.lower()]
    if not tags:
        return None, None
    tag = sorted(tags)[0]
    steps = [e.step for e in acc.Scalars(tag)]
    vals = [e.value for e in acc.Scalars(tag)]
    return steps, vals


def _steps_to_threshold(steps, vals, frac=0.9):
    """First step index reaching frac * final rolling value (sample efficiency)."""
    if not vals:
        return None
    window = max(1, len(vals) // 20)
    final = _mean(vals[-window:])
    thr = frac * final
    for s, v in zip(steps, vals):
        if v >= thr:
            return s
    return None


def aggregate():
    summary = {"tasks": {}}
    for task in TASKS:
        troot = OUT_ROOT / task
        if not troot.exists():
            continue
        tsum = {"methods": {}, "baselines": {}}

        for method in METHODS:
            per_seed = []
            costs = []
            sample_eff = []
            for seed in TRAIN_SEEDS:
                run_dir = troot / method / f"seed_{seed}"
                srs = []
                for es in EVAL_SEEDS:
                    rep = _load_json(run_dir / f"eval_seed_{es}.json")
                    if rep:
                        srs.append(_success_rate(rep))
                run_json = _load_json(run_dir / "run.json")
                if run_json:
                    costs.append(
                        {
                            "transitions": run_json.get("transitions"),
                            "train_wall_seconds": run_json.get("train_wall_seconds"),
                        }
                    )
                if srs:
                    per_seed.append(_mean(srs))
                steps, vals = _success_tag(run_dir)
                if vals:
                    sample_eff.append(_steps_to_threshold(steps, vals))
            tsum["methods"][method] = {
                "success_per_seed": per_seed,
                "success_mean": _mean(per_seed),
                "success_std": _std(per_seed),
                "success_min": min(per_seed) if per_seed else None,
                "n_seeds": len(per_seed),
                "mean_transitions": _mean([c["transitions"] for c in costs]),
                "mean_train_wall_seconds": _mean([c["train_wall_seconds"] for c in costs]),
                "steps_to_90pct_final": _mean(sample_eff),
            }

        broot = troot / "baselines"
        if broot.exists():
            for variant_dir in sorted(broot.iterdir()):
                srs = []
                for es in EVAL_SEEDS:
                    rep = _load_json(variant_dir / f"seed_{es}" / "baseline.json")
                    if rep:
                        srs.append(_success_rate(rep))
                if srs:
                    tsum["baselines"][variant_dir.name] = {
                        "success_mean": _mean(srs),
                        "success_std": _std(srs),
                        "n_eval_seeds": len(srs),
                    }
        summary["tasks"][task] = tsum

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(summary)
    print(f"wrote {OUT_ROOT / 'summary.json'} and {OUT_ROOT / 'report.md'}")
    return summary


def _write_report(summary):
    lines = [
        "# 5-Seed Benchmark Protocol -- Results",
        "",
        "Protocol: 5 training seeds (11/22/33/44/55), 5 held-out eval seeds",
        "(1011/1022/1033/1044/1055), fixed budget per task, no checkpoint",
        "selection. Each seed's success rate is the mean over the 5 eval seeds;",
        "mean +/- std is across training seeds. Baselines share the task's",
        "env configuration and read only the policy observation group.",
        "",
    ]
    for task, tsum in summary["tasks"].items():
        lines += [f"## {task}", "", "| method | success rate (mean±std) | min seed | transitions | train wall |", "|---|---|---|---|---|"]
        rows = [(f"baseline: {n}", tsum["baselines"][n]) for n in tsum["baselines"]]
        rows += [(m, tsum["methods"].get(m) or {}) for m in METHODS]
        for name, d in rows:
            if not d or d.get("success_mean") is None:
                continue
            sr = d["success_mean"]
            std = d.get("success_std")
            mn = d.get("success_min")
            tr = d.get("mean_transitions")
            wall = d.get("mean_train_wall_seconds")
            lines.append(
                "| {} | {:.3f} ± {:.3f} | {:.3f} | {} | {} |".format(
                    name,
                    sr,
                    std if std is not None else float("nan"),
                    mn if mn is not None else float("nan"),
                    f"{tr / 1e6:.1f}M" if tr else "--",
                    f"{wall / 3600:.2f} h" if wall else "--",
                )
            )
        se = {m: tsum["methods"][m].get("steps_to_90pct_final") for m in METHODS if tsum["methods"].get(m)}
        if any(v is not None for v in se.values()):
            lines += ["", "Sample efficiency (updates to 90% of the run's own final rolling success):"]
            lines += [f"- {m}: {v:.0f}" for m, v in se.items() if v is not None]
        lines.append("")
    (OUT_ROOT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    aggregate()
