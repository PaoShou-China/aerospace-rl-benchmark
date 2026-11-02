"""Aggregate controller-prior results into tables (only files that exist)."""

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "results" / "controller_priors"

PHYS_KEYS = (
    "propellant_kg",
    "maneuver_time_s",
    "terminal_range_m",
    "terminal_speed_m_s",
    "action_saturation_fraction",
    "gimbal_rate_squared_integral_proxy",
    "terminal_attitude_error_rad",
    "terminal_rate_rad_s",
    "min_manipulability",
    "min_range_m",
    "keepout_fraction",
    "mean_abs_action",
    "rms_action",
)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def discover_runs(out_root, tag=None):
    out_root = Path(out_root)
    pat = re.compile(r"seed_(\d+)(?:__(.*))?$")
    for p in sorted(out_root.glob("*/*/*/seed_*")):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        seed, run_tag = int(m.group(1)), m.group(2)
        if tag and run_tag != tag:
            continue
        parts = p.relative_to(out_root).parts
        if len(parts) != 4:
            continue
        task, method, condition, _ = parts
        yield dict(path=p, task=task, method=method, condition=condition, seed=seed, tag=run_tag)


def _mean_sd(values):
    vals = [v for v in values if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean (pure function, unit-tested).

    Returns ``(lo, hi)`` or ``(None, None)`` when no finite values exist.  The
    RNG is local and seeded so repeated analyses are reproducible.
    """
    vals = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
    ]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    if len(arr) == 1:
        # one seed cannot justify a CI; report the point estimate explicitly
        return float(arr[0]), float(arr[0])
    return lo, hi


def _parse_tag_map(spec):
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def collect(out_root=DEFAULT_ROOT, tag=None, tag_map=None):
    tag_map = _parse_tag_map(tag_map) if isinstance(tag_map, str) else (tag_map or {})
    rows, stage_rows = [], []
    datasets = []
    for r in discover_runs(out_root, tag):
        if r["task"] in tag_map and r["tag"] != tag_map[r["task"]]:
            continue
        p = r["path"]
        run = _read(p / "run.json") or {}
        train_cfg = _read(p / "config.json") or {}
        protocol = {
            k: train_cfg.get(k) for k in ("num_steps_per_env", "actor", "critic", "algorithm")
        }
        import hashlib

        protocol_digest = (
            hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()[
                :16
            ]
            if train_cfg.get("algorithm")
            else None
        )
        det = _read(p / "eval" / "det.json")
        stoch = _read(p / "eval" / "stoch.json")
        # selection arms evaluate the chosen checkpoint and write
        # ``*_stage_best.json`` instead of the final ``det/stoch`` files; fall
        # back to them so those arms are not reported as ``not_evaluated``.
        best_det = None if det else _read(p / "eval" / "det_stage_best.json")
        best_stoch = None if stoch else _read(p / "eval" / "stoch_stage_best.json")
        stage_files = sorted((p / "eval").glob("det_stage_*.json"))
        for f in stage_files:
            d = _read(f)
            if not d:
                continue
            st_raw = f.stem.split("_")[-1]
            st = int(st_raw) if st_raw.isdigit() else st_raw
            phys = (d.get("physical") or {}).get("mean_completed") or {}
            row = dict(
                task=r["task"],
                method=r["method"],
                condition=r["condition"],
                seed=r["seed"],
                run_tag=r["tag"],
                stage=st,
                mode="det",
                success_rate=d.get("success_rate"),
                complete=d.get("complete"),
                episodes_finished=d.get("episodes_finished"),
                mean_episode_return=d.get("mean_episode_return"),
                exploration_env_action_rms=d.get("exploration_env_action_rms"),
                init_digest=d.get("initial_condition_digest"),
                physical=phys,
                eval_wall_seconds=d.get("eval_wall_seconds"),
                residual_stats=(d.get("residual") or {}).get("stats"),
            )
            stage_rows.append(row)
        selection = _read(p / "selection" / "selection.json")
        meta_run = run.get("experiment_meta") or {}
        base = dict(
            task=r["task"],
            method=r["method"],
            condition=r["condition"],
            seed=r["seed"],
            run_tag=r["tag"],
            updates=run.get("updates"),
            transitions=run.get("transitions"),
            action_mode=meta_run.get("action_mode"),
            improvements=meta_run.get("improvements"),
            selected_best=(selection or {}).get("best_name"),
            selected_val_success=(
                (selection or {}).get("ranking", [{}])[0].get("success_rate") if selection else None
            ),
            protocol_digest=protocol_digest,
            train_wall_seconds=run.get("train_wall_seconds"),
            train_throughput=run.get("train_throughput_steps_per_s"),
            peak_vram_gib=run.get("peak_vram_gib"),
            bc=run.get("bc_warmstart"),
            residual_stats=run.get("residual_stats"),
            num_envs=(run.get("experiment_meta") or {}).get("num_envs"),
            residual_rho=(run.get("experiment_meta") or {}).get("residual_rho"),
            controller_design=(run.get("experiment_meta") or {}).get("controller_design"),
            baseline_variant=(run.get("experiment_meta") or {}).get("baseline_variant"),
        )
        for mode, d, is_best in (
            ("det", det, False),
            ("stoch", stoch, False),
            ("det", best_det, True),
            ("stoch", best_stoch, True),
        ):
            if not d:
                continue
            phys = (d.get("physical") or {}).get("mean_completed") or {}
            succ_phys = (d.get("physical") or {}).get("mean_successful") or {}
            row = dict(
                base,
                stage=(
                    "best"
                    if is_best
                    else (d.get("updates") if d.get("updates") not in (None, 0) else None)
                ),
                mode=mode,
                success_rate=d.get("success_rate"),
                complete=d.get("complete"),
                episodes_finished=d.get("episodes_finished"),
                episodes_expected=d.get("episodes_expected"),
                mean_episode_return=d.get("mean_episode_return"),
                exploration_env_action_rms=d.get("exploration_env_action_rms"),
                init_digest=d.get("initial_condition_digest"),
                physical=phys,
                physical_successful=succ_phys,
                success_fraction_for_cost=(d.get("physical") or {}).get(
                    "success_fraction_for_cost"
                ),
                eval_wall_seconds=d.get("eval_wall_seconds"),
                residual_stats=(d.get("residual") or {}).get("stats"),
                per_replica=d.get("per_replica"),
            )
            rows.append(row)
        # a run with training but no evaluation still appears, explicitly incomplete
        if not det and not stoch and not best_det and not best_stoch and run:
            rows.append(
                dict(
                    base,
                    stage=None,
                    mode="not_evaluated",
                    success_rate=None,
                    complete=False,
                    physical={},
                    residual_stats=run.get("residual_stats"),
                )
            )
        ds = _read(p / "dataset.json")
        if ds:
            datasets.append(ds)
    return rows, stage_rows, datasets


def summarise(rows, stage_rows):
    def key(r):
        return (r["task"], r["method"], r["condition"], r["mode"], r.get("stage"))

    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        succ = [r["success_rate"] for r in rs]
        ret = [r.get("mean_episode_return") for r in rs]
        s_mean, s_sd, n = _mean_sd(succ)
        r_mean, r_sd, _ = _mean_sd(ret)
        import hashlib

        ci_seed = int(hashlib.sha256("|".join(str(x) for x in k).encode()).hexdigest()[:8], 16)
        s_lo, s_hi = bootstrap_ci(succ, seed=ci_seed)
        r_lo, r_hi = bootstrap_ci(ret, seed=ci_seed + 1)
        rec = dict(
            task=k[0],
            method=k[1],
            condition=k[2],
            mode=k[3],
            stage=k[4],
            seeds=len(rs),
            success_mean=s_mean,
            success_sd=s_sd,
            success_ci_lo=s_lo,
            success_ci_hi=s_hi,
            return_mean=r_mean,
            return_sd=r_sd,
            return_ci_lo=r_lo,
            return_ci_hi=r_hi,
            action_mode=rs[0].get("action_mode"),
            improvements=rs[0].get("improvements"),
            selected_best=rs[0].get("selected_best"),
            selected_val_success=rs[0].get("selected_val_success"),
            incomplete=sum(1 for r in rs if not r.get("complete")),
            init_digests=sorted({r.get("init_digest") for r in rs if r.get("init_digest")}),
            eval_wall_mean=_mean_sd([r.get("eval_wall_seconds") for r in rs])[0],
            exploration_env_action_rms=_mean_sd([r.get("exploration_env_action_rms") for r in rs])[
                0
            ],
            updates=rs[0].get("updates"),
            num_envs=rs[0].get("num_envs"),
            transitions_mean=_mean_sd([r.get("transitions") for r in rs])[0],
            train_wall_mean=_mean_sd([r.get("train_wall_seconds") for r in rs])[0],
            peak_vram_gib=_mean_sd([r.get("peak_vram_gib") for r in rs])[0],
            baseline_variant=rs[0].get("baseline_variant"),
            controller_design=rs[0].get("controller_design"),
            residual_rho=rs[0].get("residual_rho"),
            bc_transitions=((rs[0].get("bc") or {}).get("transitions")),
            bc_wall_seconds=((rs[0].get("bc") or {}).get("wall_seconds")),
            success_fraction_for_cost=_mean_sd([r.get("success_fraction_for_cost") for r in rs])[0],
            physical={},
        )
        for pk in PHYS_KEYS:
            vals = [(r.get("physical") or {}).get(pk) for r in rs]
            rec["physical"][pk] = _mean_sd(vals)[0]
            rec["physical_sd_" + pk] = _mean_sd(vals)[1]
        rec["physical_successful"] = {
            pk: _mean_sd([(r.get("physical_successful") or {}).get(pk) for r in rs])[0]
            for pk in PHYS_KEYS
        }
        stat_dicts = [
            r.get("residual_stats") for r in rs if isinstance(r.get("residual_stats"), dict)
        ]
        if stat_dicts:
            keys = sorted(
                {k for d in stat_dicts for k, v in d.items() if isinstance(v, (int, float))}
            )
            rec["residual_stats"] = {k: _mean_sd([d.get(k) for d in stat_dicts])[0] for k in keys}
            rec["residual_stats_first"] = stat_dicts[0]
        else:
            rec["residual_stats"] = None
        out.append(rec)
    # stage table: per (task,method,seed,stage) already in stage_rows; add group summary
    sg = {}
    for r in stage_rows:
        sg.setdefault((r["task"], r["method"], r["condition"], r["stage"]), []).append(r)
    stage_summary = []
    for k, rs in sorted(sg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        stage_summary.append(
            dict(
                task=k[0],
                method=k[1],
                condition=k[2],
                stage=k[3],
                seeds=len(rs),
                success_mean=_mean_sd([r["success_rate"] for r in rs])[0],
                success_sd=_mean_sd([r["success_rate"] for r in rs])[1],
                return_mean=_mean_sd([r["mean_episode_return"] for r in rs])[0],
                complete=all(r.get("complete") for r in rs),
            )
        )
    return out, stage_summary


def paired_check(rows):
    """Verify that every arm evaluated the SAME first-episode initial states."""
    groups = {}
    for r in rows:
        if r.get("mode") != "det" or r.get("stage") not in (None, "best"):
            continue
        groups.setdefault((r["task"], r["condition"]), set()).add(r.get("init_digest"))
    return {f"{k[0]}/{k[1]}": sorted(v) for k, v in groups.items() if len(v) > 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--tag-map",
        default=None,
        help="per-task tag selection, e.g. docking=pilot_v2,satellite=pilot_v1",
    )
    p.add_argument(
        "--analysis-subdir",
        default="_analysis",
        help="output subdirectory (use to keep holdout/scans separate)",
    )
    args = p.parse_args()
    rows, stages, datasets = collect(args.out_root, args.tag, args.tag_map)
    summary, stage_summary = summarise(rows, stages)
    analysis_dir = Path(args.out_root) / args.analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "runs.json").write_text(json.dumps(rows, indent=2, default=str))
    (analysis_dir / "stages.json").write_text(json.dumps(stages, indent=2, default=str))
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (analysis_dir / "stage_summary.json").write_text(
        json.dumps(stage_summary, indent=2, default=str)
    )
    (analysis_dir / "datasets.json").write_text(json.dumps(datasets, indent=2, default=str))
    (analysis_dir / "paired_condition_check.json").write_text(
        json.dumps(paired_check(rows), indent=2, default=str)
    )
    # same-task arms must share the PPO protocol (mechanism, not optimizer, is
    # the variable under test)
    proto = {}
    for r in rows:
        if r.get("protocol_digest"):
            proto.setdefault((r["task"], r["condition"]), set()).add(r["protocol_digest"])
    (analysis_dir / "protocol_consistency.json").write_text(
        json.dumps({f"{k[0]}/{k[1]}": sorted(v) for k, v in proto.items()}, indent=2)
    )
    # CSV for the paper tables (only existing data)
    import csv

    flat_keys = [
        "task",
        "method",
        "condition",
        "mode",
        "stage",
        "seeds",
        "success_mean",
        "success_sd",
        "success_ci_lo",
        "success_ci_hi",
        "return_mean",
        "return_sd",
        "return_ci_lo",
        "return_ci_hi",
        "incomplete",
        "updates",
        "action_mode",
        "selected_best",
        "num_envs",
        "transitions_mean",
        "train_wall_mean",
        "peak_vram_gib",
        "baseline_variant",
        "residual_rho",
        "bc_transitions",
        "bc_wall_seconds",
        "success_fraction_for_cost",
    ] + ["physical_" + k for k in PHYS_KEYS]
    with open(analysis_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        for r in summary:
            row = {k: r.get(k) for k in flat_keys if k not in ("physical_" + x for x in PHYS_KEYS)}
            for pk in PHYS_KEYS:
                row["physical_" + pk] = r["physical"].get(pk)
            w.writerow(row)
    with open(analysis_dir / "stages.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "method",
                "condition",
                "stage",
                "seeds",
                "success_mean",
                "success_sd",
                "return_mean",
                "complete",
            ],
        )
        w.writeheader()
        w.writerows(stage_summary)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "stage_points": len(stages),
                "summary_groups": len(summary),
                "datasets": len(datasets),
                "paired_condition_mismatches": paired_check(rows),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
