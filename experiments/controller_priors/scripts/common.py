"""Shared plumbing for the controller-prior experiments (no new dependencies)."""

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
EXP_ROOT = Path(__file__).resolve().parents[1]  # experiments/controller_priors
CONFIG_DIR = EXP_ROOT / "configs"
RESULTS_ROOT = ROOT / "results" / "controller_priors"
TUNING_ROOT = ROOT / "results" / "controller_priors_tuning"
PYTHON = sys.executable


def load_json(path):
    return json.loads(Path(path).read_text())


def task_table():
    return load_json(CONFIG_DIR / "tasks.json")


def task_config(task, config=None):
    table = task_table() if config is None else config
    if task not in table:
        raise KeyError(f"unknown task {task!r}; known: {sorted(table)}")
    return copy.deepcopy(table[task])


def env_class(task, config):
    import importlib

    module = importlib.import_module(config["env_module"])
    return getattr(module, config["env_class"])


def make_env(task, config, *, num_envs=None, seed=None, backend="gpu", env_kwargs=None):
    cls = env_class(task, config)
    kw = dict(config.get("env_kwargs") or {})
    kw.update(env_kwargs or {})
    n = int(num_envs if num_envs is not None else config.get("num_envs", 256))
    s = int(seed if seed is not None else config.get("seed", 7))
    env = cls(num_envs=n, backend=backend, seed=s, **kw)
    return env


def make_controller(task, env, condition):
    from algorithms.baselines.baselines import build_baseline

    kwargs = dict(condition.get("controller_kwargs") or {})
    design = condition.get("controller_design")
    if design is not None:
        kwargs["design"] = design
    return build_baseline(task, env, condition.get("baseline_variant"), **kwargs)


def load_frozen_bc_policy(checkpoint, env, *, device=None, name=None):
    """Load a BC actor checkpoint as a frozen deterministic base controller.

    The checkpoint's adjacent ``config.json`` describes the actor's network and
    observation normalizer, so the same artifact used by ``bc_ppo`` is reused
    verbatim -- no retraining and no environment state is read.  The returned
    object implements the ``Baseline`` controller protocol consumed by
    ``ResidualVecEnv``.
    """
    import copy

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from algorithms.residual.residual_env import PolicyBaseController
    from algorithms.train_utils import RslVecEnv

    ckpt = Path(checkpoint)
    cfg_path = ckpt.parent / "config.json"
    if not cfg_path.exists():
        # selection/best_checkpoint.pt lives one level below the seed dir
        cfg_path = ckpt.parent.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json next to {ckpt}")
    cfg = json.loads(cfg_path.read_text())
    init_std = cfg.get("actor", {}).get("distribution_cfg", {}).get("init_std")
    if isinstance(init_std, list):
        cfg["actor"]["distribution_cfg"]["init_std"] = torch.tensor(init_std)
    target_device = device if device is not None else getattr(env, "device", "cpu")
    runner = OnPolicyRunner(
        RslVecEnv(env),
        copy.deepcopy(cfg),
        log_dir="/tmp/cp_frozen_bc_tb",
        device=str(target_device),
    )
    runner.load(str(ckpt))
    actor = runner.alg.actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    bound = float(getattr(env, "clip_actions", 1.0))
    return PolicyBaseController(
        actor,
        device=target_device,
        action_bound=bound,
        name=name or f"frozen_bc:{ckpt.parent.name}",
    )


def git_state():
    def run(args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    commit = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain=v1"])
    diff = run(["diff", "--stat"])
    names = [line[3:].strip() for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_entries": len(names),
        "dirty_names": names[:200],
        "dirty_digest": hashlib.sha256(status.encode()).hexdigest()[:16],
        "diff_stat_digest": hashlib.sha256(diff.encode()).hexdigest()[:16],
        "diff_stat": diff[-4000:],
    }


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    try:
        import torch

        info.update(
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda=torch.version.cuda,
        )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            )
    except Exception:
        pass
    for mod in ("numpy", "scipy", "tensordict", "rsl_rl"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    return info


def run_dir(task, method, condition, seed, out_root=None, run_tag=None):
    root = Path(out_root) if out_root else RESULTS_ROOT
    name = f"seed_{int(seed)}"
    if run_tag:
        name += f"__{run_tag}"
    return root / task / method / condition / name


def ensure_new_run_dir(path, *, allow_existing=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in path.iterdir() if p.name != ".gitkeep"]
    if existing and not allow_existing:
        raise FileExistsError(
            f"run directory already contains results: {path} "
            f"({len(existing)} entries). Use a unique --run-tag; "
            "controller-prior results are never silently overwritten."
        )
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json_safe(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def run_metadata(*, task, method, condition, seed, extra=None):
    meta = {
        "task": task,
        "method": method,
        "condition": condition,
        "seed": int(seed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_state(),
        "environment": environment_info(),
    }
    if extra:
        meta.update(extra)
    return meta


def digest_tensor(tensor):
    import torch

    flat = tensor.detach().to("cpu", torch.float32).contiguous().flatten()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
