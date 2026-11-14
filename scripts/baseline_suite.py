"""Full classical-baseline test suite: every task × every registered
controller × the frozen-protocol seeds, one first-episode cohort per replica.

Usage
-----
    python scripts/baseline_suite.py run       [--tasks ...] [--seeds 7,13,21,101]
                                              [--difficulty 1.0,0.5,0.0]
                                              [--shard 0/3] [--num-envs 1024]
    python scripts/baseline_suite.py report    [--out docs/baseline_test_report.md]

`run` evaluates every (task, controller, seed[, difficulty]) combination with
the same unbiased accounting as `scripts/baseline_eval.py` (loop until EVERY
replica has finished one full episode) and writes one JSON per cell to
    results/<task>/baselines/<controller>_seed<seed>[_d<difficulty>].json
Existing official `results/<task>/baseline*.json` files are not touched.

`report` aggregates those JSONs into the markdown test report (mean/min/max
over seeds, outcome breakdown, action/robustness audits).

Sharding (`--shard k/n`) splits the deterministic job list so several GPU
processes can run in parallel without duplicating work.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from scripts.baseline_eval import ENVS  # noqa: E402
from scripts.first_episode import FirstEpisodes  # noqa: E402
from algorithms.baselines.baselines import (  # noqa: E402
    BASELINE_VARIANTS,
    BUILDERS,
    build_baseline,
)

# the legacy extra registry key is the same class as the 'cwh_pd' variant
_DUPLICATE_KEYS = {"rendezvous_cwhpd"}

MAX_STEPS = {"fixedwing": 8000, "quadplane": 4000}  # safety caps per task
DEFAULT_MAX_STEPS = 4000


def controller_matrix():
    """[(task, label, variant-or-None)] -- default first, then variants."""
    jobs = []
    for task in ENVS:
        jobs.append((task, BUILDERS[task].name, None))
        for variant, cls in sorted(BASELINE_VARIANTS.get(task, {}).items()):
            if variant in _DUPLICATE_KEYS:
                continue
            jobs.append((task, cls.name, variant))
    # de-duplicate identical (task, controller) cells (a variant may share the
    # default's class, e.g. satellite's damped-CMG arm)
    seen, unique = set(), []
    for job in jobs:
        key = (job[0], job[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def _env_kwargs(task, out_root, difficulty):
    """Evaluation always uses the frozen benchmark configuration: calm air,
    env defaults, benchmark reset distribution.  Training-side env_kwargs
    (e.g. fixedwing dryden wind) are deliberately NOT inherited -- the frozen
    protocol requires a wind-free baseline evaluation.

    Domain randomization is explicitly disabled here so the published
    classical-controller SR reflects the nominal plant (benchmark contract).
    Robustness experiments enable it by omitting this override.
    """
    kw = {"domain_randomize": False}
    if difficulty is not None:
        kw["difficulty"] = difficulty
    return kw


def run_cell(
    env_cls,
    controller_name,
    variant,
    task,
    seed,
    difficulty,
    num_envs,
    backend,
    out_root,
    max_steps=None,
):
    env_kwargs = _env_kwargs(task, out_root, difficulty)
    torch.manual_seed(seed)
    try:
        env = env_cls(num_envs=num_envs, backend=backend, seed=seed, **env_kwargs)
    except TypeError:
        if difficulty is not None:
            raise
        env = env_cls(num_envs=num_envs, backend=backend, seed=seed, domain_randomize=False)
        env_kwargs = {"domain_randomize": False}
    controller = build_baseline(task, env, variant)
    cap = max_steps or MAX_STEPS.get(task, DEFAULT_MAX_STEPS)
    cohort = FirstEpisodes(env)
    obs = env.get_observations()
    start = time.perf_counter()
    steps = 0
    actions_finite = True
    with torch.no_grad():
        while steps < cap:
            actions = controller.actions(obs)
            if actions_finite and not torch.isfinite(actions).all():
                actions_finite = False
            obs, _, term, trunc, _ = env.step(actions)
            controller.reset_idx(term | trunc)
            steps += 1
            cohort.update(env)
            if steps % 32 == 0 and cohort.finished():
                break
    wall = time.perf_counter() - start
    report = {
        "task": task,
        "controller": controller_name,
        "variant": variant,
        "seed": seed,
        "difficulty": difficulty,
        "num_envs": num_envs,
        "backend": backend,
        "env_kwargs": env_kwargs,
        "env_steps": steps,
        "eval_wall_seconds": wall,
        "actions_finite": actions_finite,
    }
    report.update(cohort.report())
    n = max(report["episodes_finished"], 1)
    metrics = {k: v / n for k, v in report["metric_counts"].items()}
    m = max(report["episodes_completed_total"], 1)
    metrics.update({k: v / m for k, v in report["global_metric_counts"].items()})
    report["task_metrics"] = metrics
    report["benchmark"] = env.cfg.get("benchmark")
    if hasattr(controller, "config"):
        report["controller_config"] = controller.config()
    if hasattr(controller, "metrics"):
        report["controller_metrics"] = {
            k: float(v.detach().cpu()) if torch.is_tensor(v) else v
            for k, v in controller.metrics.items()
        }
    env.close()
    return report


def cmd_run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    diffs = (
        [None]
        if args.difficulty in ("", "default")
        else [float(d) for d in args.difficulty.split(",")]
    )
    jobs = []
    tasks = list(ENVS) if args.tasks in ("", "all") else [t.strip() for t in args.tasks.split(",")]
    for task, label, variant in controller_matrix():
        if task not in tasks:
            continue
        for seed in seeds:
            for d in diffs:
                jobs.append((task, label, variant, seed, d))
    shard_id, shard_n = (0, 1)
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))
    mine = jobs[shard_id::shard_n]
    print(f"[suite] {len(mine)}/{len(jobs)} cells (shard {shard_id}/{shard_n})", flush=True)
    for i, (task, label, variant, seed, d) in enumerate(mine, 1):
        tag = f"seed{seed}" + (f"_d{d:g}" if d is not None else "")
        dest = args.out_root / task / "baselines" / f"{label}_{tag}.json"
        if dest.exists() and not args.force:
            print(f"[{i}/{len(mine)}] skip (exists) {task}/{label}/{tag}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            report = run_cell(
                ENVS[task],
                label,
                variant,
                task,
                seed,
                d,
                args.num_envs,
                args.backend,
                args.out_root,
                args.max_steps,
            )
        except Exception as exc:  # keep the sweep alive
            print(
                f"[{i}/{len(mine)}] FAIL {task}/{label}/{tag}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2))
        sr = report.get("success_rate")
        print(
            f"[{i}/{len(mine)}] {task:11s} {label:22s} {tag:14s} "
            f"SR={sr if sr is not None else float('nan'):.4f} "
            f"n={report['episodes_finished']} "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _cells(out_root):
    cells = {}
    for path in sorted(out_root.glob("*/baselines/*.json")):
        data = json.loads(path.read_text())
        cells.setdefault((data["task"], data["controller"]), []).append(data)
    return cells


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


PROTOCOL_SEEDS = (7, 13, 21)
HELDOUT_SEED = 101

# one-line method family for the report's method table
METHOD_NOTES = {
    "se3_geometric": "Lee SE(3) 几何控制（默认）",
    "cascaded_pid": "位置→姿态级联 PID（Euler）",
    "smc": "滑模位置外环（指数趋近律+边界层）+ Euler 姿态级联",
    "cv_kf_pid": "CV 卡尔曼滤波估计目标 + SE(3) 跟踪（默认）",
    "pursuit": "无滤波纯追踪（原始含噪相对测量）",
    "pid_ema": "一阶 EMA 目标滤波 + SE(3) 跟踪",
    "l1_tecs": "L1 横向制导 + TECS 纵向（默认）",
    "pid_cascade": "常规级联 PID 姿态/高度/速度",
    "nlgl_tecs": "Park 非线性制导律（NLGL）+ TECS",
    "zem_zev_pdg": "ZEM/ZEV 中段 + 虚拟平台 PDG + 速度带脉冲伺服（默认）",
    "pd_landing": "纯 PD 着陆（无制导层）",
    "eguidance": "Apollo E-guidance 二次多项式加速制导 + 终端脉冲伺服",
    "apn_rate": "增广比例导引 APN + half-N 重力补偿（默认）",
    "ppn": "纯追踪导引（无前置量）",
    "tpn": "真比例导引 TPN（无 (N/2)g 增广）",
    "se3_pusher": "四旋翼 SE(3) 全程 + 前向推进器分配（默认）",
    "hybrid_cascade": "模式切换混合自驾（低速四旋翼/高速机翼），含机翼升力卸荷项",
    "l1_wing": "机翼模式 L1 横向 + 四旋翼低速助推",
    "quat_pd_damped_cmg": "四元数 PD + Schaub-Junkins 阻尼伪逆 CMG（默认）",
    "quat_pd_mp_pinv_cmg": "四元数 PD + 普通 Moore-Penrose 伪逆 CMG",
    "quat_pd_wpinv_cmg": "四元数 PD + Oh-Fleming 加权阻尼最小范数 + Vadali 零空间奇异规避",
    "gauss_tangential": "Gauss 行星方程切向推力单步贪心（默认）",
    "q_law_petropoulos": "Petropoulos Q-law Lyapunov 制导",
    "edelbaum": "Edelbaum 解析 Δv 预算开环切向推力",
    "cwh_lqr": "CWH 全状态 LQR + 走廊/捕获网/燃料护栏（默认）",
    "cwh_pole_placement": "CWH 极点配置（无附加约束层）",
    "zem_zev": "CWH ZEM/ZEV 单脉冲滚动时域制导 + 燃料/距离护栏",
    "cwh_two_impulse": "CWH 滚动时域两脉冲终端制导（默认）",
    "cwh_compensated_pd": "CWH 补偿 PD 直趋原点",
    "glideslope": "距离-速率刹车走廊（glideslope）V-bar 逼近",
}


def _ci95(p, n):
    """Wilson score 95% interval for a binomial success rate."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def cmd_report(args):
    task_order = list(ENVS)
    cells = _cells(args.out_root)
    rows = []
    for (task, label), runs in cells.items():
        bench = {(r["seed"]): r for r in runs if r.get("difficulty") is None}
        proto = [bench[s] for s in PROTOCOL_SEEDS if s in bench]
        if not proto:
            continue
        srs = [r["success_rate"] for r in proto]
        mean = sum(srs) / len(srs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in srs) / (len(srs) - 1)) if len(srs) > 1 else 0.0
        n_eps = sum(r["episodes_finished"] for r in proto)
        p_pool = sum(r["successes"] for r in proto) / max(n_eps, 1)
        lo_ci, hi_ci = _ci95(p_pool, n_eps)
        worst = min(proto, key=lambda r: r["success_rate"])
        tm = worst.get("task_metrics", {})
        held = bench.get(HELDOUT_SEED)
        rows.append(
            dict(
                task=task,
                controller=label,
                variant=proto[0].get("variant"),
                seeds=len(srs),
                mean=mean,
                sd=sd,
                lo=min(srs),
                hi=max(srs),
                pool=p_pool,
                ci_lo=lo_ci,
                ci_hi=hi_ci,
                held=(held["success_rate"] if held else None),
                ret=sum(r.get("mean_episode_return") or 0.0 for r in proto) / len(proto),
                episodes=n_eps,
                steps=max(r["env_steps"] for r in proto),
                wall=sum(r["eval_wall_seconds"] for r in proto) / len(proto),
                finite=all(r.get("actions_finite", True) for r in proto),
                fails=dict(
                    crash=tm.get("crash_rate", tm.get("crashes")),
                    oob=tm.get("out_of_bounds_rate", tm.get("out_of_bounds")),
                    timeout=tm.get("timeout_rate", tm.get("timeouts")),
                    fatal=tm.get("fatal_impact_rate", tm.get("fatal_impacts")),
                ),
            )
        )
    rows.sort(
        key=lambda r: (task_order.index(r["task"]) if r["task"] in task_order else 99, -r["mean"])
    )

    n_envs = max((r.get("num_envs") or 0) for runs in cells.values() for r in runs) if cells else 0
    L = [
        "# 传统（非学习）基线完整测试报告",
        "",
        "## 测试口径",
        "",
        "- 冻结评测协议：官方 benchmark 初始分布、静风、**标称 plant（域随机化关闭）**、"
        f"{n_envs or 'N'} 并行环境、每个并行副本只采信**首回合**"
        "（`FirstEpisodes`，与 `scripts/baseline_eval.py` 同一记账）。",
        "- 每个任务 × 每个控制器在协议种子 {7, 13, 21} 各跑一次，另跑留出种子 101 一次；"
        "成功率 = `completed_successes / completed_episodes`。",
        "- 95% 区间为 pooling 三个协议种子后的 Wilson 区间（样本量 ≈ 3×副本数 回合）。",
        "- 训练侧 `env_kwargs`（如固定翼 Dryden 风）**不继承**：基线一律静风评测。",
        "",
        "## 总表（协议种子 7/13/21 均值，附留出种子 101）",
        "",
        "| 任务 | 控制器 | 成功率 mean±sd | [min, max] | 95% CI | 留出 101 | "
        "mean return | 回合 | 步数窗口 | 平均墙钟 s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        held = "—" if r["held"] is None else f"{r['held'] * 100:.1f}%"
        L.append(
            f"| {r['task']} | {r['controller']} | "
            f"**{r['mean'] * 100:.1f}%** ± {r['sd'] * 100:.1f} | "
            f"[{r['lo'] * 100:.1f}, {r['hi'] * 100:.1f}] | "
            f"[{r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}] | {held} | "
            f"{_fmt(r['ret'])} | {r['episodes']} | {r['steps']} | "
            f"{_fmt(r['wall'], 0)} |"
        )
    L.append("")
    L.append("## 失败模式（取协议种子中最差那次的回合平均率）")
    L.append("")
    L.append("| 任务 | 控制器 | crash | 出界 | 超时 | 致命撞击 | 动作有限值 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["fails"]

        def fmt(v):
            return "—" if v is None else f"{v * 100:.1f}%"

        L.append(
            f"| {r['task']} | {r['controller']} | {fmt(f['crash'])} | "
            f"{fmt(f['oob'])} | {fmt(f['timeout'])} | {fmt(f['fatal'])} | "
            f"{'✓' if r['finite'] else '✗'} |"
        )
    L.append("")
    L.append("## 方法族谱")
    L.append("")
    L.append("| 任务 | 控制器 | 方法 |")
    L.append("|---|---|---|")
    for r in rows:
        L.append(f"| {r['task']} | {r['controller']} | {METHOD_NOTES.get(r['controller'], '—')} |")
    L.append("")
    L.append("## 每任务最优经典方法")
    L.append("")
    L.append("| 任务 | 最优（协议种子均值） | 次优 | 最弱 | 方法间极差 |")
    L.append("|---|---|---|---|---|")
    for task in task_order:
        group = [r for r in rows if r["task"] == task]
        if not group:
            continue
        ranked = sorted(group, key=lambda r: -r["mean"])
        best, second, worst = ranked[0], (ranked[1] if len(ranked) > 1 else None), ranked[-1]
        spread = best["mean"] - worst["mean"]
        if second:
            second_txt = second["controller"] + f" ({second['mean'] * 100:.1f}%)"
        else:
            second_txt = "—"
        L.append(
            f"| {task} | {best['controller']} ({best['mean'] * 100:.1f}%) | "
            f"{second_txt} | "
            f"{worst['controller']} ({worst['mean'] * 100:.1f}%) | "
            f"{spread * 100:.1f} pp |"
        )
    L.append("")
    args.out.write_text("\n".join(L) + "\n")
    print(
        f"wrote {args.out} ({len(rows)} controllers, {sum(r['episodes'] for r in rows)} episodes)"
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tasks", default="all")
    p.add_argument("--seeds", default="7")
    p.add_argument(
        "--difficulty",
        default="default",
        help="'default' (benchmark), or comma list e.g. 1.0,0.5,0.0",
    )
    p.add_argument("--shard", default=None, help="k/n")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=_ROOT / "results")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--out-root", type=Path, default=_ROOT / "results")
    q.add_argument("--out", type=Path, default=_ROOT / "docs" / "baseline_test_report.md")
    q.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
