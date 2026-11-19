"""Paired, first-episode evaluation of frozen civil task policies in Genesis or Torch."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environments import make
from environments.civil_scenarios import CIVIL_TASKS, scenario_kwargs
from scripts.first_episode import FirstEpisodes
from scripts.civil_accuracy import CivilAccuracy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cfg = json.loads(args.config.read_text())
    ref = cfg.get("env_kwargs", {})
    options = {**ref, **scenario_kwargs(args.task, args.profile, args.seed)}
    env = make(args.task, args.num_envs, args.backend, args.seed, civil_profile=args.profile, **ref)
    try:
        obs = env.get_observations()
        initial_hash = hashlib.sha256(obs["policy"].cpu().numpy().tobytes()).hexdigest()
        ckpt = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        after_step = None
        kw = copy.deepcopy(cfg["actor"])
        class_name = kw.pop("class_name")
        if class_name != "MLPModel":
            raise ValueError(
                f"Unsupported actor {class_name!r}; archived custom actors require "
                "a separately verified source package"
            )
        std = kw.get("distribution_cfg", {}).get("init_std")
        if isinstance(std, list):
            kw["distribution_cfg"]["init_std"] = torch.tensor(std, device=env.device)
        actor = MLPModel(obs, cfg["obs_groups"], "actor", env.num_actions, **kw).to(env.device)
        actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        actor.eval()
        if getattr(actor, "is_recurrent", False):
            after_step = actor.reset
        cohort = FirstEpisodes(env)
        accuracy = CivilAccuracy(args.task, env)
        first_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_xy = torch.zeros_like(first_touch, dtype=torch.float32)
        first_vz = first_xy.clone()
        end_xy = first_xy.clone()
        with torch.no_grad():
            for steps in range(1, env.max_steps + 1):
                action = actor(obs)
                if not torch.isfinite(action).all():
                    raise ValueError(f"nonfinite action at step {steps}")
                obs, rew, term, trunc, extras = env.step(action)
                if after_step is not None:
                    after_step(term | trunc)
                terminal = extras.get("civil_truth_terminal", extras["terminal_observations"]["policy"])
                if not (torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
                        and torch.isfinite(terminal[term | trunc]).all()):
                    raise ValueError(f"nonfinite simulation at step {steps}")
                if args.task == "rocket":
                    # _contact_any and _contact_vz are captured across inner physics
                    # steps. The cached observation is the end of that policy step,
                    # before auto-reset (including fatal first contacts).
                    new_touch = ~cohort.seen & ~first_touch & env._contact_any
                    radius = terminal[:, 10:12].norm(dim=-1)
                    first_xy[new_touch] = radius[new_touch]
                    first_vz[new_touch] = env._contact_vz[new_touch].float()
                    first_touch |= new_touch
                    done = ~cohort.seen & (term | trunc)
                    end_xy[done] = radius[done]
                accuracy.update(terminal, ~cohort.seen, term | trunc)
                cohort.update(env)
                if cohort.finished():
                    break
        report = cohort.report()
        if not report["complete"]:
            raise RuntimeError("incomplete first-episode cohort")
        accuracy.attach(cohort, report["episodes"])
        if args.task == "rocket":
            benchmark = env.cfg["benchmark"]
            # Optional metadata differs across preserved rocket benchmark versions.
            for key in ("contact_geometry", "landing_terrain", "ground_shutdown",
                        "pad_load_feedback", "collision_asset_sha256"):
                if key in benchmark:
                    report[key] = benchmark[key]
            for i, row in enumerate(report["episodes"]):
                row["first_pad_contact_step_xy_m"] = float(first_xy[i]) if first_touch[i] else None
                row["first_pad_contact_vz_m_s"] = float(first_vz[i]) if first_touch[i] else None
                row["terminal_xy_m"] = float(end_xy[i])
        else:
            report.pop("global_metric_counts", None)  # mixes later episodes in some tasks
        report.update(task=args.task, profile=args.profile, seed=args.seed, backend=args.backend,
                      checkpoint_sha256=digest(args.checkpoint), config_sha256=digest(args.config),
                      original_checkpoint_sha256=ckpt.get("original_checkpoint_sha256"),
                      env_kwargs=options, actor_class=class_name, initial_obs_sha256=initial_hash,
                      steps=steps, torch=torch.__version__, genesis=package_version("genesis-world"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.out)
        print(f"{args.task} {args.profile}: {report['successes']}/{report['episodes_finished']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=CIVIL_TASKS)
    p.add_argument("--profile", choices=("reference", "development", "hard", "navigation_outage"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    run(p.parse_args())
