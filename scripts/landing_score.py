"""Read full-scale landing scores from current and historical first-episode reports."""

import json
import math

TERMINAL = "terminal_stable_2s"


def benchmark_method_dir(task_dir, method, *, read_existing=False):
    """Keep the revised rocket BC weights and evals separate from archived runs."""
    method_dir = task_dir / method
    if task_dir.name == "rocket" and method in ("bc_shared", "bc_ppo"):
        revised = method_dir / "bc_recipe_2"
        if not read_existing or revised.is_dir():
            return revised
    return method_dir


def rocket_version(report):
    benchmark = report.get("benchmark") or {}
    return (benchmark.get("version") or report.get("benchmark_version")
            or (report.get("env_kwargs") or {}).get("benchmark_version"))


def terminal_count(report):
    """Historical v1 reports contain the terminal count even when success meant ever stable."""
    n = report.get("episodes_finished")
    if (report.get("complete") is not True or type(n) is not int or n <= 0
            or report.get("episodes_expected") != n):
        return None
    count = (report.get("metric_counts") or {}).get(TERMINAL)
    if (type(count) not in (int, float) or not math.isfinite(count)
            or not 0 <= count <= n or int(count) != count):
        return None
    return int(count)


def evaluation_rate(task, report):
    if not report or report.get("complete") is False:
        return None
    if task == "rocket" and rocket_version(report) == "v1":
        count = terminal_count(report)
        return count / report["episodes_finished"] if count is not None else None
    if "episodes_expected" in report and report.get("episodes_finished") != report["episodes_expected"]:
        return None
    for key in ("success_rate", "success"):
        if report.get(key) is not None:
            return float(report[key])
    return None


def reusable_evaluation(task, path, config_path=None, *, expected_version=None):
    """A saved v1 artifact must have terminal per-episode labels before it can skip replay."""
    if not path.is_file():
        return False
    if task != "rocket":
        return True
    try:
        report = json.loads(path.read_text())
        version = rocket_version(report)
        if config_path is not None and config_path.is_file():
            expected_version = rocket_version(json.loads(config_path.read_text()))
        if expected_version is not None and expected_version != version:
            return False
        if report.get("complete") is not True:
            return False
        if version != "v1":
            return version == "v2"
        count = terminal_count(report)
        rows = report.get("episodes") or []
        if count is None or len(rows) != report["episodes_expected"]:
            return False
        return (
            report.get("benchmark", {}).get("primary_metric") == TERMINAL
            and report.get("successes") == count
            and report.get("success_rate") == count / len(rows)
            and all(row.get("complete") is True and type(row.get("success")) is bool for row in rows)
            and sum(row["success"] for row in rows) == count
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False
