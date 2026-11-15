"""Gain-sweep harness for the classical (non-learned) baselines.

Tuning tool only: it evaluates candidate gain sets on the frozen benchmark
protocol (calm air, benchmark reset distribution, first episode per replica)
and prints a ranked table.  It never writes the official
``results/<task>/baseline*.json`` artifacts.

Usage
-----
    python scripts/baseline_tune.py <task> [--baseline KEY]
        --grid '{"KX": [3, 4, 5], "KV": [2, 3]}'
        [--seeds 7] [--num-envs 128] [--max-steps 5000] [--out DIR]

Every candidate gets a FRESH environment built with the same seed, so all
candidates see identical initial conditions; ``--seeds 7,13`` repeats that for
several seeds and reports the mean.  Candidate keys are controller class
attributes (declared ``TUNABLE_PARAMS`` entries and plain class constants
alike).  Results go to ``results/tuning/<task>/<controller>/<candidate>.json``.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from algorithms.baselines.baselines import (  # noqa: E402
    BASELINE_VARIANTS,
    BUILDERS,
    build_baseline,
)
from scripts.baseline_eval import ENVS  # noqa: E402
from scripts.first_episode import FirstEpisodes  # noqa: E402


def _controller_class(task, key):
    if key in (None, "default"):
        return BUILDERS[task]
    return BASELINE_VARIANTS[task][key]


class patched_gains:
    """Temporarily override controller class constants (and TUNABLE defaults)."""

    def __init__(self, cls, overrides):
        self.cls = cls
        self.overrides = overrides
        self.saved_attrs = {}
        self.created = []
        self.saved_tunable = None

    def __enter__(self):
        cls = self.cls
        if any(k in cls.TUNABLE_PARAMS for k in self.overrides):
            self.saved_tunable = cls.TUNABLE_PARAMS
            table = dict(cls.TUNABLE_PARAMS)
            for k, v in self.overrides.items():
                if k in table:
                    table[k] = dict(table[k], default=v)
            cls.TUNABLE_PARAMS = table
        for k, v in self.overrides.items():
            if hasattr(cls, k):
                self.saved_attrs[k] = getattr(cls, k)
            else:
                self.created.append(k)
            setattr(cls, k, v)
        return self

    def __exit__(self, *exc):
        cls = self.cls
        for k, v in self.saved_attrs.items():
            setattr(cls, k, v)
        for k in self.created:
            try:
                delattr(cls, k)
            except AttributeError:
                pass
        if self.saved_tunable is not None:
            cls.TUNABLE_PARAMS = self.saved_tunable
        return False


def _build_env(env_cls, num_envs, seed, backend):
    try:
        return env_cls(
            num_envs=num_envs, backend=backend, seed=seed, domain_randomize=False
        )
    except TypeError:
        return env_cls(num_envs=num_envs, backend=backend, seed=seed)


class patched_config:
    """Point a controller's CONFIG_PATH at a temp JSON with dotted-key updates."""

    def __init__(self, cls, overrides):
        self.cls = cls
        self.overrides = overrides
        self.base = None
        self.tmp = None

    def __enter__(self):
        base = self.cls.CONFIG_PATH
        cfg = json.loads(base.read_text())
        for dotted, value in self.overrides.items():
            node = cfg
            keys = dotted.split(".")
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        self.base = base
        self.tmp = base.with_name(base.stem + "_tune_tmp.json")
        self.tmp.write_text(json.dumps(cfg, indent=2))
        self.cls.CONFIG_PATH = self.tmp
        return self

    def __exit__(self, *exc):
        self.cls.CONFIG_PATH = self.base
        if self.tmp is not None:
            self.tmp.unlink(missing_ok=True)
        return False


def run_candidate(
    env_cls,
    controller_cls,
    override,
    seed,
    num_envs,
    backend,
    max_steps,
    config_mode=False,
    design_mode=False,
):
    torch.manual_seed(seed)
    env = _build_env(env_cls, num_envs, seed, backend)
    try:
        if config_mode:
            with patched_config(controller_cls, override):
                controller = controller_cls(env)
        elif design_mode:
            design = dict(getattr(env, "controller_design", None) or {})
            design.update(override)
            controller = controller_cls(env, design=design)
        else:
            with patched_gains(controller_cls, override):
                controller = controller_cls(env)
        cohort = FirstEpisodes(env)
        obs = env.get_observations()
        steps = 0
        with torch.no_grad():
            while steps < max_steps:
                actions = controller.actions(obs)
                obs, _, term, trunc, _ = env.step(actions)
                controller.reset_idx(term | trunc)
                steps += 1
                cohort.update(env)
                if steps % 32 == 0 and cohort.finished():
                    break
        report = cohort.report()
        n = max(report["episodes_finished"], 1)
        metrics = {k: v / n for k, v in report["metric_counts"].items()}
        m = max(report["episodes_completed_total"], 1)
        metrics.update({k: v / m for k, v in report["global_metric_counts"].items()})
        return dict(
            seed=seed,
            steps=steps,
            success_rate=report["success_rate"],
            episodes=report["episodes_finished"],
            complete=report["complete"],
            mean_return=report["mean_episode_return"],
            task_metrics=metrics,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vehicle", choices=list(ENVS))
    parser.add_argument("--baseline", default="default")
    parser.add_argument("--grid", required=True, help='JSON {"name": [values]}')
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--config-json",
        action="store_true",
        help="treat grid keys as dotted paths into the controller CONFIG_PATH json",
    )
    parser.add_argument(
        "--design",
        action="store_true",
        help="pass grid keys through the controller design dict (gains read via _param)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    grid = json.loads(args.grid)
    names = list(grid)
    candidates = [dict(zip(names, vals)) for vals in itertools.product(*(grid[n] for n in names))]
    cls = _controller_class(args.vehicle, args.baseline)
    out = args.out or (
        _ROOT
        / "results"
        / "tuning"
        / args.vehicle
        / (args.baseline if args.baseline != "default" else cls.name)
    )
    out.mkdir(parents=True, exist_ok=True)
    env_cls = ENVS[args.vehicle]

    rows = []
    for i, cand in enumerate(candidates, 1):
        t0 = time.perf_counter()
        runs = [
            run_candidate(
                env_cls,
                cls,
                cand,
                seed,
                args.num_envs,
                args.backend,
                args.max_steps,
                args.config_json,
                args.design,
            )
            for seed in seeds
        ]
        srs = [r["success_rate"] for r in runs]
        mean = sum(srs) / len(srs)
        row = dict(
            candidate=cand,
            mean_success=mean,
            per_seed=srs,
            seeds=seeds,
            runs=runs,
            wall_s=round(time.perf_counter() - t0, 1),
        )
        rows.append(row)
        tag = "_".join(f"{k}{v}" for k, v in cand.items()) or "base"
        (out / f"{tag}.json").write_text(json.dumps(row, indent=2))
        print(
            f"[{i}/{len(candidates)}] {tag:40s} SR mean {mean * 100:5.1f}%  "
            f"per-seed {[round(s * 100, 1) for s in srs]}  {row['wall_s']:.0f}s",
            flush=True,
        )

    rows.sort(key=lambda r: -r["mean_success"])
    print("\n=== ranked ===")
    for r in rows:
        print(
            f"{r['mean_success'] * 100:5.1f}%  {json.dumps(r['candidate'])}  "
            f"per-seed {[round(s * 100, 1) for s in r['per_seed']]}"
        )
    (out / "sweep_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
