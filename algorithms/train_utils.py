"""Shared machinery for the per-task training entries (train_<task>.py).

Importing requires only the repository root on sys.path (so `algorithms.*` and
`environments.common.*`, including the vendored
`environments.common.original_pyflyt.*` sources, resolve).

`run_training()` drives the rsl-rl PPO loop with the checkpoint handling and
run reporting of the original train.py. All run parameters are passed in as
plain constants from the per-task entry files -- nothing is read from the
command line -- and the PPO/model configuration defaults are shared across
tasks so results stay directly comparable; an entry pins only the variables
under test.
"""

import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


def _warmup_cosine_decay(step, init, peak, end, warmup_steps, decay_steps):
    """Linear warmup from ``init`` to ``peak`` over ``warmup_steps``, then a
    cosine decay from ``peak`` to ``end`` over the remaining ``decay_steps``.

    Used by PPO's opt-in ``lr_cosine`` schedule.
    """
    if step < warmup_steps:
        return init + (peak - init) * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.0)
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * progress))


class RslVecEnv(VecEnv):
    """Adapts the project envs' 5-tuple step (obs, rew, terminated, truncated,
    extras) to the 4-tuple (obs, rew, dones, extras) rsl_rl expects.
    dones = terminated | truncated, identical to the env's internal reset_buf;
    timeout bootstrapping still flows through extras['time_outs'] untouched."""

    def __init__(self, env):
        self.__dict__["env"] = env  # bypass __getattr__ during construction

    def get_observations(self):
        return self.env.get_observations()

    def step(self, actions):
        obs, rew, term, trunc, extras = self.env.step(actions)
        return obs, rew, term | trunc, extras

    def __getattr__(self, name):
        return getattr(self.env, name)


def _actor_std(runner):
    dist = runner.alg.actor.distribution
    with torch.no_grad():
        if hasattr(dist, "log_std_param"):
            return float(dist.log_std_param.exp().mean())
        if hasattr(dist, "std_param"):
            return float(dist.std_param.mean())
        # BetaDistribution: std = sqrt(alpha*beta / ((alpha+beta)^2*(alpha+beta+1)))
        # scaled by range; report the mean unscaled std as a proxy.
        if hasattr(dist, "_alpha") and getattr(dist, "_alpha", None) is not None:
            a = dist._alpha.mean()
            b = dist._beta.mean()
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            return float(var.sqrt() * getattr(dist, "_range_scale", 2.0))
    return float("nan")


def build_cfg(
    *,
    lr_schedule="fixed",
    learning_rate=0.001,
    desired_kl=0.01,
    entropy_coef=0.001,
    init_std=0.5,
    gamma=0.99,
    lam=0.95,
    num_steps_per_env=32,
    num_learning_epochs=4,
    num_mini_batches=4,
    clip_param=0.2,
    value_loss_coef=1.0,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    normalize_advantage_per_mini_batch=False,
    kl_anchor_coef=0.0,
    kl_anchor_decay_updates=0,
    distribution="beta",
):
    """PPO/model config, identical for every task (fair-comparison baseline).

    Keyword overrides exist so a task entry can vary exactly the variables
    under test while everything else stays at the shared defaults.

    Actor and critic are the 2x128 tanh MLP (rsl-rl MLPModel) with running-
    stats EmpiricalNormalization.

    Shared lr defaults follow the 2026-10-20 tuning history: the KL-adaptive
    schedule is the project-wide pathology (rides its 1e-2 ceiling on
    rendezvous/satellite, slams to its 1e-5 floor on fixedwing, thrashes
    1e-5..8.65e-4 elsewhere) and fixed 3e-4 stalls from scratch (quadx stuck
    at -377 with 0% success, results/quadx_sweep). Fixed 1e-3 is the A/B
    winner on every task where the two were compared (quadx 300-update sweep,
    fixedwing 150-update A/B, docking, rendezvous), so it is the shared
    default; the shipped entries pin their per-task proven optimum
    (quadx/fixedwing/rocket/satellite/rendezvous/docking/missile fixed 3e-4,
    transfer fixed 2.5e-4)."""
    distribution = distribution.lower()
    if distribution == "beta":
        # Bounded action space: Beta(alpha, beta) rescaled to [-1, 1].
        # init_std is accepted for back-compat but has no effect on Beta.
        distribution_cfg = {
            "class_name": "BetaDistribution",
            "action_range": (-1.0, 1.0),
        }
    elif distribution == "gaussian":
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": init_std,
            "std_type": "log",
        }
    else:
        raise ValueError(f"unknown distribution {distribution!r}; use 'beta' or 'gaussian'")
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128],
        "activation": "tanh",
        "obs_normalization": True,
        "distribution_cfg": distribution_cfg,
    }
    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [128, 128],
        "activation": "tanh",
        "obs_normalization": True,
    }
    algorithm = {
        "class_name": "PPO",
        "learning_rate": learning_rate,
        "num_learning_epochs": num_learning_epochs,
        "num_mini_batches": num_mini_batches,
        "schedule": lr_schedule,
        "desired_kl": desired_kl,
        "entropy_coef": entropy_coef,
        "gamma": gamma,
        "lam": lam,
        "clip_param": clip_param,
        "value_loss_coef": value_loss_coef,
        "max_grad_norm": max_grad_norm,
        "use_clipped_value_loss": bool(use_clipped_value_loss),
        "normalize_advantage_per_mini_batch": bool(normalize_advantage_per_mini_batch),
    }
    if kl_anchor_coef > 0:
        # Optimization ablation: freeze the BC actor as a KL target.  The
        # class lives in this repo, resolved by rsl-rl's qualified-name path.
        algorithm["class_name"] = "algorithms.ppo.kl_anchor:KLAnchorPPO"
        algorithm["kl_anchor_coef"] = float(kl_anchor_coef)
        algorithm["kl_anchor_decay_updates"] = int(kl_anchor_decay_updates)
    return {
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 10_000_000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": actor,
        "critic": critic,
        "algorithm": algorithm,
        "logger": "tensorboard",
    }


def _save_runner(runner, path):
    """Save a runner even before the logger has been initialised (BC-only)."""
    logger = getattr(runner, "logger", None)
    if hasattr(logger, "writer") or not hasattr(runner, "alg"):
        # OnPolicyRunner with an initialised logger, or a runner that owns its
        # own save format (no ``alg`` attribute).
        runner.save(str(path))
    else:
        saved = runner.alg.save()
        saved["iter"] = runner.current_learning_iteration
        saved["infos"] = None
        torch.save(saved, str(path))


def run_training(
    task,
    env_cls,
    num_envs=64,
    updates=1000,
    seed=7,
    backend="gpu",
    resume=False,
    clean=True,
    out_dir=None,
    lr_decay_updates=None,
    lr_cosine=False,
    learning_rate_end=1.5e-4,
    freeze_obs_norm=False,
    lr_sigma_coupling=None,
    sigma_pin=None,
    algo="ppo",
    env_kwargs=None,
    bc_warmstart=None,
    env_wrapper=None,
    experiment_meta=None,
    extra_config=None,
    stage_updates=None,
    runner_init_hook=None,
    save_initial_checkpoint=False,
    bc_anchor=None,
    lr_warmup_updates=None,
    sigma_anneal_to=None,
    actor_freeze_updates=None,
    segment_updates=None,
    segment_hook=None,
    env_curriculum=None,
    **algo_overrides,
):
    """Train one task end-to-end and write checkpoint.pt / run.json / config.json.

    env_kwargs: extra keyword arguments forwarded to the env constructor
    (e.g. the atmospheric tasks' wind disturbance config); None leaves
    every other task entry's call signature unchanged.

    lr_decay_updates: optional exponential lr half-life in updates (opt-in;
    None keeps the constant lr the other task entries use).

    lr_cosine (PPO only): warmup-cosine lr schedule. ``learning_rate_end``
    is the cosine floor. Applied at the START of each ``updates//100``
    learn-iteration chunk (the value at the chunk's START; the first chunk
    gets the schedule's step-0 value instead of the raw initial lr).

    freeze_obs_norm (PPO only): pin both empirical obs normalizers in eval
    mode after the checkpoint load -- EmpiricalNormalization.update() no-ops
    there, so a warm start's fitted input mapping cannot drift while PPO
    rollouts stream through. The pin is re-applied inside a wrapped
    alg.train_mode() because every learn() call recursively re-enables it.

    lr_sigma_coupling (PPO only): opt-in lr = k * sigma schedule, mutually
    exclusive with lr_decay_updates. The per-update mean policy drift scales
    as lr/sigma, so a fixed lr pushes the policy progressively harder as the
    exploration std anneals; after every 25-update chunk the lr is rewritten
    as k * mean action std to keep the drift rate constant. The coupling is
    applied once BEFORE the first chunk too -- at a pinned small sigma the
    uncoupled initial lr can sit in the collapse band (lr/sigma > ~3.3e-4).

    sigma_pin (PPO only): re-pin the exploration std to this value at every
    chunk (the entropy bonus and policy gradient otherwise drift it). With
    lr_sigma_coupling the lr follows the pinned sigma. Pinned sigma keeps the
    noisy-rollout objective aligned with greedy behavior: at sigma=0.5 the
    in-rollout success collapses and PPO's gradient chases noise (measured
    dose-response, 100-update gates: sigma 0.5 -> 12% greedy, 0.35 -> 46%,
    0.2 -> holds ~60-64%).

    algo: 'ppo' (default, rsl-rl OnPolicyRunner). Consumes the identical
    batched env protocol, checkpoint naming and reporting pipeline so results
    stay directly comparable across runs.

    bc_warmstart (PPO only): None disables; a dict enables the behavior-
    cloning warm start (algorithms/warmstart/bc_warmstart.py). Keys with defaults:
    'steps' (512 vectorized env steps of expert data, collected ON the
    training env by driving the task's classical baseline controller (its
    ``conventional_controller``) directly through collect_expert), 'epochs'
    (100 supervised epochs), 'batch_size' (8192),
    'lr' (1e-3, a separate Adam over the actor's parameters), 'gamma'
    (None -> the PPO config's gamma, used only for the critic's
    returns-to-go targets) and 'fit_critic' (True). The env's episode
    counters are zeroed and the env reset after collection so the PPO run's
    own reporting stays clean. Skipped when `resume` actually loads a
    checkpoint (the checkpoint would overwrite the warm start anyway).

    Extra keyword arguments are forwarded to ``build_cfg`` so a task entry can
    vary exactly the variables under test while everything else stays at the
    shared fair-comparison defaults.
    """
    torch.manual_seed(seed)
    root = Path(__file__).resolve().parents[1]  # repo root (this file: algorithms/train_utils.py)
    out = Path(out_dir) if out_dir is not None else root / "results" / task
    out.mkdir(parents=True, exist_ok=True)
    if not resume and clean:
        # fresh run: drop stale models, tensorboard events and reports
        for p in list(out.glob("model_*.pt")) + [
            out / "checkpoint.pt",
            out / "run.json",
            out / "config.json",
            out / "params",
        ]:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        for p in out.glob("events.out.tfevents.*"):
            p.unlink()

    env_kwargs = dict(env_kwargs or {})
    # A saved rocket contract pins the physical variation and policy rate.
    # Resuming optimization must not silently change its physics/score.
    if resume and env_cls.__name__ == "RocketLandingEnv" and (out / "config.json").exists():
        saved = json.loads((out / "config.json").read_text()).get("env_kwargs") or {}
        for key, default in (("variation", "nominal"), ("agent_hz", 40)):
            value = saved.get(key, default)
            if key in env_kwargs and env_kwargs[key] != value:
                raise ValueError(f"Cannot change {key} while resuming; start a separate run")
            env_kwargs[key] = value
    env = env_cls(num_envs=num_envs, backend=backend, seed=seed, **env_kwargs)
    base_env = env
    if env_wrapper is not None:
        # Residual / controller-prior experiments own a controller OUTSIDE the
        # env; the wrapper keeps the five-tuple step protocol intact.
        env = env_wrapper(env)
    if algo == "ppo":
        cfg = build_cfg(**algo_overrides)
        # OnPolicyRunner mutates cfg in place (resolving/popping class_name); pass a
        # deep copy so the dumped config.json stays a faithful training config.
        runner = OnPolicyRunner(
            RslVecEnv(env), copy.deepcopy(cfg), log_dir=str(out), device=str(env.device)
        )
        steps_per_iter = cfg["num_steps_per_env"]
    else:
        raise ValueError(f"unknown algo {algo!r} (expected 'ppo')")
    if runner_init_hook is not None:
        # residual arms zero the actor's output layer so the untrained
        # deterministic residual is exactly zero; the hook never runs on resume
        # (a loaded checkpoint must be preserved).
        if not resume:
            hook_info = runner_init_hook(runner)
            if hook_info:
                experiment_meta = dict(experiment_meta or {})
                experiment_meta["runner_init_hook"] = hook_info
    cfg["env_kwargs"] = dict(env_kwargs)
    cfg["env_kwargs"].update(getattr(env, "benchmark_env_kwargs", {}))
    if "benchmark" in env.cfg:
        cfg["benchmark"] = env.cfg["benchmark"]
    if extra_config:
        cfg.update(copy.deepcopy(extra_config))
    if experiment_meta:
        cfg["experiment_meta"] = copy.deepcopy(experiment_meta)
    if hasattr(env, "physics_limits"):
        cfg["residual"] = {
            "controller": getattr(env, "teacher_identity", None),
            "controller_design": getattr(env, "controller_design", {}),
            "variant": getattr(env, "variant", None),
            "authority": env.physics_limits(),
        }
    # Persist the contract before training, including interrupted runs.
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if resume:
        cands = sorted(out.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        ckpt = cands[-1] if cands else out / "checkpoint.pt"
        if ckpt.exists():
            runner.load(str(ckpt))
            print(f"resumed from {ckpt}", flush=True)
            resumed = True
        else:
            print("no checkpoint found, training from scratch", flush=True)
            resumed = False
    else:
        resumed = False
    if bc_warmstart:
        if algo != "ppo":
            raise ValueError("bc_warmstart is PPO-only")
        if resumed:
            print("bc_warmstart skipped: a checkpoint was resumed", flush=True)
        else:
            from algorithms.warmstart.bc_warmstart import (
                BC_DEFAULTS,
                collect_expert,
                fit_behavior_cloning,
            )

            bc_cfg = {**BC_DEFAULTS, **bc_warmstart}
            if bc_cfg["gamma"] is None:
                bc_cfg["gamma"] = cfg["algorithm"]["gamma"]
            if bc_cfg.get("checkpoint"):
                # Reuse a BC-only checkpoint (same initialization for BC-only
                # and BC->PPO) instead of refitting the same data.
                runner.load(str(bc_cfg["checkpoint"]))
                bc_report = {
                    "initialization": "bc_checkpoint_loaded",
                    "checkpoint": str(bc_cfg["checkpoint"]),
                    "fit_critic": bool(bc_cfg["fit_critic"]),
                    "initial_output_std": _actor_std(runner),
                }
                print(f"[bc] loaded initialization from {bc_cfg['checkpoint']}", flush=True)
            elif bc_cfg.get("dataset"):
                from algorithms.warmstart.expert_dataset import (
                    load_expert_dataset,
                    split_dataset,
                    flatten_split,
                    merge_datasets,
                )

                paths = [bc_cfg["dataset"]] + list(bc_cfg.get("extra_datasets") or [])
                ds = load_expert_dataset(paths[0])
                if len(paths) > 1:
                    extra = [load_expert_dataset(p) for p in paths[1:]]
                    ds = merge_datasets([ds] + extra)
                    print(f"[bc] merged {len(paths)} datasets: {paths}", flush=True)
                split_name = bc_cfg.get("dataset_split", "train")
                splits = split_dataset(ds, seed=seed)
                pairs = flatten_split(ds, splits[split_name])
                val_pairs = flatten_split(ds, splits["val"]) if splits.get("val") else None
                sample_weights = None
                if bc_cfg.get("sample_weighting"):
                    from algorithms.warmstart.bc_warmstart import phase_weights

                    sample_weights = phase_weights(pairs["obs"], bc_cfg["sample_weighting"])
                obs = pairs["obs"].reshape(-1, 1, pairs["obs"].shape[-1])
                # delta parameterization: the supervised target must be the
                # increment, not the absolute command (train_arm sets the scale
                # only for the dataset-fit arms under action_mode='delta').
                act_flat = pairs["act"]
                val_act_flat = None if val_pairs is None else val_pairs["act"]
                if bc_cfg.get("delta_action_scale") is not None:
                    from algorithms.warmstart.bc_warmstart import delta_action_targets

                    scale = float(bc_cfg["delta_action_scale"])
                    act_flat = delta_action_targets(act_flat, pairs["t_in_episode"], scale)
                    if val_act_flat is not None:
                        val_act_flat = delta_action_targets(
                            val_act_flat, val_pairs["t_in_episode"], scale
                        )
                acts = act_flat.reshape(-1, 1, act_flat.shape[-1])
                rews = pairs["rew"].reshape(-1, 1)
                dones = pairs["done"].reshape(-1, 1)
                # success ends the episode as a TRUE terminal event (the task
                # env marks it truncated); keep timeout tails out of critic
                # targets unless a bootstrap is supplied.
                term_eff = (pairs["terminated"] | pairs["success"]).reshape(-1, 1)
                trunc_eff = (pairs["truncated"] & ~pairs["success"]).reshape(-1, 1)
                cut = pairs["cut"].reshape(-1, 1)
                val = None
                if val_pairs is not None and val_pairs["obs"].shape[0]:
                    val = {
                        "obs": val_pairs["obs"].reshape(-1, 1, val_pairs["obs"].shape[-1]),
                        "acts": val_act_flat.reshape(-1, 1, val_act_flat.shape[-1]),
                    }
                print(
                    f"[bc] dataset {bc_cfg['dataset']} split={split_name} "
                    f"pairs={obs.shape[0]} val_pairs={0 if val is None else val['obs'].shape[0]}",
                    flush=True,
                )
                bc_report = fit_behavior_cloning(
                    runner,
                    obs,
                    acts,
                    rews,
                    dones,
                    epochs=bc_cfg["epochs"],
                    batch_size=bc_cfg["batch_size"],
                    lr=bc_cfg["lr"],
                    gamma=bc_cfg["gamma"],
                    fit_critic=bc_cfg["fit_critic"],
                    val=val,
                    normalizer_mode=bc_cfg.get("normalizer_mode", "per_batch"),
                    freeze_normalizer=bc_cfg.get("freeze_normalizer", False),
                    terminated=term_eff,
                    truncated=trunc_eff,
                    cut=cut,
                    sample_weights=sample_weights,
                )
                bc_report["dataset_path"] = str(bc_cfg["dataset"])
                bc_report["extra_datasets"] = list(bc_cfg.get("extra_datasets") or [])
                bc_report["sample_weighting"] = bc_cfg.get("sample_weighting")
                bc_report["delta_action_scale"] = bc_cfg.get("delta_action_scale")
                bc_report["dataset_split"] = split_name
                bc_report["split_episodes"] = {k: len(v) for k, v in splits.items()}
                bc_report["initial_output_std"] = _actor_std(runner)
            else:
                print(
                    f"[bc] collecting expert data: {bc_cfg['steps']} steps x {num_envs} envs",
                    flush=True,
                )
                t0 = time.perf_counter()
                obs, acts, rews, dones = collect_expert(env, bc_cfg["steps"])
                collect_wall = time.perf_counter() - t0
                print(
                    f"[bc] collected {obs.shape[0] * num_envs} pairs in "
                    f"{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                bc_report = fit_behavior_cloning(
                    runner,
                    obs,
                    acts,
                    rews,
                    dones,
                    epochs=bc_cfg["epochs"],
                    batch_size=bc_cfg["batch_size"],
                    lr=bc_cfg["lr"],
                    gamma=bc_cfg["gamma"],
                    fit_critic=bc_cfg["fit_critic"],
                )
                bc_report["collect_wall_seconds"] = round(collect_wall, 1)
            bc_warmstart = {**bc_cfg, **bc_report}
            if bc_cfg.get("save_checkpoint"):
                _save_runner(runner, bc_cfg["save_checkpoint"])
                print(f"[bc] saved initialization to {bc_cfg['save_checkpoint']}", flush=True)
            # fresh MDP state for the PPO run; zero the counters the expert
            # rollout inflated so run.json only reports PPO's own episodes
            env.reset()
            env.completed_episodes.zero_()
            env.completed_successes.zero_()
            env.completed_returns.zero_()
            for v in (getattr(env, "completed_metrics", None) or {}).values():
                v.zero_()
    if freeze_obs_norm:
        if algo != "ppo":
            raise ValueError("freeze_obs_norm is PPO-only")

        def _pin_norms():
            for m in (runner.alg.actor, runner.alg.critic):
                if hasattr(m, "obs_normalizer"):
                    m.obs_normalizer.eval()

        _orig_train_mode = runner.alg.train_mode

        def _train_mode():
            _orig_train_mode()
            _pin_norms()

        runner.alg.train_mode = _train_mode
        _pin_norms()
        print("obs normalizers frozen (eval mode)", flush=True)

    if bc_anchor:
        if algo != "ppo" or not hasattr(runner.alg, "set_bc_anchor"):
            raise ValueError("bc_anchor requires the KLAnchorPPO algorithm (kl_anchor_coef > 0)")
        runner.alg.set_bc_anchor(
            coef=bc_anchor.get("coef"), decay_updates=bc_anchor.get("decay_updates")
        )
        experiment_meta = dict(experiment_meta or {})
        experiment_meta["bc_anchor"] = dict(bc_anchor)
        print(f"[anchor] frozen BC KL target enabled: {bc_anchor}", flush=True)

    improvement_schedule = any(
        (
            sigma_anneal_to is not None,
            actor_freeze_updates,
            env_curriculum,
            segment_hook,
            segment_updates,
        )
    )
    if improvement_schedule and (lr_decay_updates or lr_sigma_coupling or sigma_pin):
        raise ValueError(
            "improvement schedules (sigma/freeze/curriculum/hook) are "
            "mutually exclusive with lr_decay/lr_sigma/sigma_pin; "
            "warmup-cosine lr (lr_cosine) is applied per chunk instead"
        )
    if lr_decay_updates and (lr_sigma_coupling or sigma_pin):
        raise ValueError("lr_decay_updates and lr_sigma_coupling/sigma_pin are mutually exclusive")
    if lr_cosine and (lr_decay_updates or lr_sigma_coupling or sigma_pin):
        raise ValueError(
            "lr_cosine is mutually exclusive with lr_decay_updates/lr_sigma_coupling/sigma_pin"
        )
    if (
        stage_updates
        and not improvement_schedule
        and (lr_cosine or lr_decay_updates or lr_sigma_coupling or sigma_pin)
    ):
        # the improvement-schedule branch applies lr_cosine per chunk itself and
        # still honours stage milestones, so only the plain branches conflict
        raise ValueError("stage_updates cannot be combined with a custom lr schedule")
    if save_initial_checkpoint:
        # stage 0 = the initialization under test (random / BC / zero residual),
        # evaluated with the same protocol as every learned stage.
        _save_runner(runner, out / "stage_0.pt")
    start = time.perf_counter()
    stage_saved = []
    hook_history = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if updates <= 0:
        # BC-only arm: evaluate the warm start as-is; the final checkpoint is
        # the initialization (no RL updates, no rollout statistics).
        stage_saved = []
    elif improvement_schedule:
        # Chunked training for the improvement ablations: exploration-std
        # annealing, actor freeze/warmup, initial-condition curriculum and a
        # segment hook (used by the success-gated residual-authority
        # curriculum).  Stage checkpoints still work here.
        milestones = sorted({int(x) for x in (stage_updates or []) if int(x) > 0} | {int(updates)})
        chunk = max(1, int(segment_updates or 10))
        sigma0 = _actor_std(runner)
        raw = env
        seen = set()
        while hasattr(raw, "env") and id(raw) not in seen:
            seen.add(id(raw))
            raw = raw.env
        applied_curriculum = -1
        frozen = False
        hook_history = []
        done = 0
        applied_curriculum_items = []
        if lr_cosine:
            lr0 = cfg["algorithm"]["learning_rate"]
            warmup = (
                max(1, int(lr_warmup_updates)) if lr_warmup_updates else max(1, int(1e-6 * updates))
            )

            def _apply_lr(at):
                lr = _warmup_cosine_decay(at, lr0, lr0, learning_rate_end, warmup, updates)
                runner.alg.learning_rate = lr
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = lr

            _apply_lr(0)
        while done < updates:
            if actor_freeze_updates:
                should_freeze = done < int(actor_freeze_updates)
                if should_freeze != frozen:
                    frozen = should_freeze
                    for p in runner.alg.actor.parameters():
                        p.requires_grad_(not frozen)
                    print(
                        f"[freeze] actor requires_grad={not frozen} at {done}/{updates}", flush=True
                    )
            n = min(chunk, updates - done)
            runner.learn(n)
            done += n
            if lr_cosine:
                _apply_lr(done)
            if sigma_anneal_to is not None:
                frac = min(1.0, done / float(updates))
                sigma = sigma0 + (float(sigma_anneal_to) - sigma0) * frac
                with torch.no_grad():
                    dist = runner.alg.actor.distribution
                    if hasattr(dist, "log_std_param"):
                        dist.log_std_param.fill_(math.log(sigma))
                    elif hasattr(dist, "std_param"):
                        dist.std_param.fill_(sigma)
                print(f"[sigma] update {done}/{updates}: sigma {sigma:.4f}", flush=True)
            if env_curriculum:
                progress = done / float(updates)
                for idx, item in enumerate(env_curriculum):
                    if idx <= applied_curriculum:
                        continue
                    if progress >= float(item.get("fraction", 0.0)):
                        for key, value in (item.get("kwargs") or {}).items():
                            setattr(raw, key, value)
                        applied_curriculum = idx
                        applied_curriculum_items.append(item)
                        print(f"[curriculum] update {done}/{updates}: {item}", flush=True)
            if segment_hook is not None:
                info = segment_hook(runner, env, done, updates)
                if info:
                    hook_history.append({"update": done, **dict(info)})
            if done in milestones and done != updates:
                path = out / f"stage_{done}.pt"
                _save_runner(runner, path)
                stage_saved.append(str(path))
                print(f"[stage] saved {path}", flush=True)
        for p in runner.alg.actor.parameters():
            p.requires_grad_(True)
    elif stage_updates:
        milestones = sorted({int(x) for x in stage_updates if int(x) > 0} | {int(updates)})
        done = 0
        stage_saved = []
        while done < updates:
            nxt = min([m for m in milestones if m > done] or [updates])
            n = min(nxt, updates) - done
            runner.learn(n)
            done += n
            if done in milestones and done != updates:
                path = out / f"stage_{done}.pt"
                _save_runner(runner, path)
                stage_saved.append(str(path))
                print(f"[stage] saved {path}", flush=True)
    elif algo == "ppo" and lr_cosine:
        # warmup-cosine lr. Applied at the START of each chunk -- including
        # BEFORE the first chunk, so no update ever runs at a lr the schedule
        # never prescribes.
        lr0 = cfg["algorithm"]["learning_rate"]
        warmup = (
            max(1, int(lr_warmup_updates)) if lr_warmup_updates else max(1, int(1e-6 * updates))
        )
        step = max(1, updates // 100)
        done = 0
        lr = _warmup_cosine_decay(0, lr0, lr0, learning_rate_end, warmup, updates)
        runner.alg.learning_rate = lr
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = lr
        while done < updates:
            n = min(step, updates - done)
            runner.learn(n)
            done += n
            lr = _warmup_cosine_decay(done, lr0, lr0, learning_rate_end, warmup, updates)
            runner.alg.learning_rate = lr
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = lr
    elif algo == "ppo" and lr_decay_updates:
        # exponential lr half-life; with schedule='fixed' rsl-rl never reapplies
        # alg.learning_rate to the optimizer (that path is inside the adaptive
        # branch), so write the optimizer's param_groups directly
        lr0 = cfg["algorithm"]["learning_rate"]
        step = max(1, lr_decay_updates // 4)
        done = 0
        while done < updates:
            n = min(step, updates - done)
            runner.learn(n)
            done += n
            lr = lr0 * 0.5 ** (done / lr_decay_updates)
            runner.alg.learning_rate = lr
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = lr
    elif algo == "ppo" and (lr_sigma_coupling or sigma_pin):

        def _couple():
            if sigma_pin:
                with torch.no_grad():
                    runner.alg.actor.distribution.log_std_param.fill_(math.log(sigma_pin))
                sigma = sigma_pin
            else:
                with torch.no_grad():
                    sigma = float(runner.alg.actor.distribution.log_std_param.exp().mean())
            lr = lr_sigma_coupling * sigma if lr_sigma_coupling else None
            if lr is not None:
                runner.alg.learning_rate = lr
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = lr
            return sigma, lr

        sigma, lr = _couple()
        print(f"[lr-sigma] init: sigma {sigma:.3f} -> lr {lr:.2e}", flush=True)
        done = 0
        while done < updates:
            n = min(25, updates - done)
            runner.learn(n)
            done += n
            sigma, lr = _couple()
            print(
                f"[lr-sigma] update {done}/{updates}: sigma {sigma:.3f} -> lr {lr:.2e}", flush=True
            )
    else:
        runner.learn(updates)
    elapsed = time.perf_counter() - start
    _save_runner(runner, out / "checkpoint.pt")
    # keep results concise: drop periodic mid-training checkpoints, only the
    # final checkpoint.pt stays (PPO/rsl_rl saves model_{it}.pt along the way)
    for p in out.glob("model_*.pt"):
        p.unlink()

    report = {
        "task": env.cfg,
        "device": str(env.device),
        "updates": updates,
        "algo": algo,
        "transitions": updates * steps_per_iter * num_envs,
        "train_wall_seconds": elapsed,
        "train_throughput_steps_per_s": round(
            updates * steps_per_iter * num_envs / max(elapsed, 1e-9), 1
        ),
        "peak_vram_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if "cuda" in str(env.device)
            else None
        ),
        "completed_episodes": env.completed_episodes.detach().cpu().tolist(),
        "completed_successes": env.completed_successes.detach().cpu().tolist(),
        "physics_step_calls": getattr(getattr(env, "plant", None), "physics_steps", None),
        "note": "Original default tasks on device-resident tensors; NumPy seeded RNG bitstreams and physics engine numerics are not identical to PyFlyt.",  # noqa: E501
    }
    if getattr(env, "completed_metrics", None):
        eps = max(int(env.completed_episodes.detach().sum()), 1)
        report["completed_metrics"] = {
            k: float(v.detach().sum()) / eps for k, v in env.completed_metrics.items()
        }
    if hasattr(env, "stats_snapshot"):
        report["residual_stats"] = env.stats_snapshot()
    if stage_saved:
        report["stage_checkpoints"] = stage_saved
    if improvement_schedule:
        report["improvement_schedule"] = {
            "sigma_anneal_to": sigma_anneal_to,
            "sigma_initial": locals().get("sigma0"),
            "sigma_final": _actor_std(runner),
            "actor_freeze_updates": actor_freeze_updates,
            "segment_updates": segment_updates,
            "env_curriculum": env_curriculum,
            "curriculum_applied": locals().get("applied_curriculum_items"),
            "hook_history": hook_history,
        }
    if experiment_meta:
        report["experiment_meta"] = copy.deepcopy(experiment_meta)
    if isinstance(bc_warmstart, dict):
        report["bc_warmstart"] = bc_warmstart
    # Persist a slim run.json: per-env completed_* lists (num_envs entries)
    # bloat the file for no downstream benefit; summarize.py only needs the
    # totals. The returned report keeps the full lists for evolve_reward.py's
    # tail-quarter success-rate calculation.
    run_json = dict(report)
    run_json["completed_episodes"] = int(sum(report["completed_episodes"]))
    run_json["completed_successes"] = int(sum(report["completed_successes"]))
    (out / "run.json").write_text(json.dumps(run_json, indent=2))
    cfg["env_kwargs"] = dict(env_kwargs) if env_kwargs else {}
    cfg["env_kwargs"].update(getattr(env, "benchmark_env_kwargs", {}))
    if "benchmark" in env.cfg:
        cfg["benchmark"] = env.cfg["benchmark"]
    if isinstance(bc_warmstart, dict):
        cfg["bc_warmstart"] = bc_warmstart
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    return report
