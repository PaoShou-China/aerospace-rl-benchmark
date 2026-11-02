"""Condition definitions for the controller-prior study.

A condition is a dict of experimental factors applied identically to every
mechanism arm:

- ``train_env_kwargs``: true-plant / spawn overrides while TRAINING and
  collecting demonstrations;
- ``eval_env_kwargs``: true-plant overrides at evaluation (default: the
  official benchmark contract, empty);
- ``controller_design``: explicit nominal design model handed to the
  controller.  This is mandatory for a model-mismatch condition: without it
  the controller would read the shifted env constants at construction and
  would silently know the "unknown" test-time parameters;
- ``controller_kwargs``: design gains / variants (controller-quality scan);
- ``baseline_variant``: named controller A/B arm;
- ``residual_rho``: residual authority fraction of the full action range;
- ``bc``: demonstration-collection overrides (coverage / sample count).

``ppo_overrides`` are forwarded to `run_training` unchanged. ``init_std``,
``sigma_pin``, ``lr_sigma_coupling`` and the ``sigma_end`` improvement are
Gaussian-policy knobs: the shipped default policy is the bounded
``BetaDistribution``, which ignores ``init_std`` and has no ``log_std_param``
for the sigma controls, so those conditions only differ under
``distribution='gaussian'``.
"""

import copy

from scripts.common import task_config

# Nominal design knowledge pinned for mismatch conditions (the controller must
# NOT inherit the shifted plant constants from the env).
NOMINAL_DESIGN = {
    "docking": {
        "mass": 500.0,
        "f_max": 10.0,
        "isp": 300.0,
        "g0": 9.80665,
        "pos_goal": 1.0,
        "vel_goal": 0.1,
        "range_fail": 200.0,
        "fuel0": 2.0,
        "max_steps": 500,
        "dt": 1.0,
    },
    "satellite": {
        "inertia": [75.0, 60.0, 113.0],
        "h_dyn": 20.0,
        "rotor_speed": 100.0,
        "action_scale": 3.0,
        "kp": 3.0,
        "kd": 4.0,
    },
    "rendezvous": {
        "mass": 500.0,
        "f_max": 10.0,
        "isp": 300.0,
        "dt": 1.0,
        "max_steps": 1000,
        "keep_out": 5.0,
        "pos_goal": 10.0,
    },
}

_DOCKING_PLANT = {"mass": 550.0, "f_max": 9.0, "isp": 250.0}
_DOCKING_DESIGN = {"mass": 550.0, "f_max": 9.0, "isp": 250.0}
_SAT_PLANT = {"inertia": [90.0, 72.0, 135.6], "rotor_h_dyn": 16.0}
_SAT_DESIGN = {"inertia": [90.0, 72.0, 135.6], "h_dyn": 16.0}

SCANS = {
    "docking": {
        # controller quality, TRUE PLANT UNCHANGED
        "quality_sluggish": {"controller_kwargs": {"kappa": 0.12, "w_vel": 1.0}},
        "quality_aggressive": {"controller_kwargs": {"kappa": 1.2, "w_vel": 6.0}},
        "quality_model_bias": {"controller_design": {"mass": 550.0, "f_max": 9.0, "isp": 250.0}},
        # demonstration coverage, SAME sample count (episodes_per_replica fixed)
        "coverage_narrow": {"train_env_kwargs": {"init_range": [80.0, 90.0]}},
        "coverage_wide": {"train_env_kwargs": {"init_range": [40.0, 160.0]}},
        # demonstration sample count, SAME coverage
        "samples_1x": {"bc": {"episodes_per_replica": 1}},
        "samples_6x": {"bc": {"episodes_per_replica": 6}},
        # residual authority (fraction of the full normalized action range)
        "authority_0.01": {"residual_rho": 0.01},
        "authority_0.025": {"residual_rho": 0.025},
        "authority_0.05": {},
        "authority_0.1": {"residual_rho": 0.1},
        "authority_0.2": {"residual_rho": 0.2},
        "authority_0.4": {"residual_rho": 0.4},
        # model mismatch: true plant shifted, controller design pinned nominal
        "mismatch_plant": {
            "train_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "controller_design": NOMINAL_DESIGN["docking"],
        },
        # controller design model shifted, true plant nominal
        "mismatch_design": {"controller_design": _DOCKING_DESIGN},
        # both shifted by the same factor (controller has a wrong, not merely stale, model)
        "mismatch_both": {
            "train_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "controller_design": _DOCKING_DESIGN,
        },
        # optimizer ablation, NOT part of the mechanism comparison: the BC-PPO
        # collapse is tested under a pinned exploration std / lr-sigma coupling
        "stability_pin_bc_ppo": {
            "ppo_overrides": {"sigma_pin": 0.05, "lr_sigma_coupling": 1.5e-3},
            "disable_stages": True,
        },
        # ---- improvement ablations (one mechanism per condition) ----
        "improve_kl_anchor": {
            "improvements": {"bc_anchor_coef": 0.05, "bc_anchor_decay": 100},
            "disable_stages": True,
        },
        "improve_terminal_weight": {
            "improvements": {"terminal_weight": {"index": [0, 1, 2], "near": 10.0, "weight": 3.0}}
        },
        "improve_optimizer": {
            "improvements": {
                "lr_warmup": 25,
                "sigma_end": 0.2,
                "freeze_actor_updates": 10,
                "segment_updates": 10,
            },
            "disable_stages": True,
        },
        "improve_grad_clip": {
            "ppo_overrides": {"max_grad_norm": 0.5, "normalize_advantage_per_mini_batch": True}
        },
        "improve_rate_penalty": {"improvements": {"action_rate_penalty": 0.01}},
        "improve_phase_gate": {
            "improvements": {
                "rho_channels": [0.05, 0.05, 0.1],
                "phase_gate": {"index": [0, 1, 2], "near": 20.0, "soft": 5.0, "gain": 1.0},
            }
        },
        "improve_authority_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "authority_curriculum": {
                    "start": 0.01,
                    "end": 1.0,
                    "success_threshold": 0.9,
                    "step": 0.1,
                    "min_episodes": 8,
                },
            }
        },
        "improve_scratch_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "curriculum": [
                    {"fraction": 0.0, "kwargs": {"init_range": [40.0, 60.0]}},
                    {"fraction": 0.4, "kwargs": {"init_range": [50.0, 100.0]}},
                ],
            }
        },
        # checkpoint-selection ablation: training recipe identical to the baseline
        "improve_selection": {},
    },
    "satellite": {
        "quality_sluggish": {"controller_design": {"kp": 1.2, "kd": 2.0}},
        "quality_aggressive": {"controller_design": {"kp": 6.0, "kd": 5.0}},
        "quality_model_bias": {
            "controller_design": {"inertia": [90.0, 72.0, 135.6], "h_dyn": 16.0}
        },
        "coverage_narrow": {"train_env_kwargs": {"reset_noise_scale": 0.002}},
        "coverage_wide": {"train_env_kwargs": {"reset_noise_scale": 0.02}},
        "samples_1x": {"bc": {"episodes_per_replica": 1}},
        "samples_2x": {"bc": {"episodes_per_replica": 2}},
        "samples_4x": {"bc": {"episodes_per_replica": 4}},
        "authority_0.01": {"residual_rho": 0.01},
        "authority_0.025": {"residual_rho": 0.025},
        "authority_0.05": {},
        "authority_0.1": {"residual_rho": 0.1},
        "authority_0.2": {"residual_rho": 0.2},
        "authority_0.4": {"residual_rho": 0.4},
        "mismatch_plant": {
            "train_env_kwargs": {"plant_kwargs": _SAT_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _SAT_PLANT},
            "controller_design": NOMINAL_DESIGN["satellite"],
        },
        "mismatch_design": {"controller_design": _SAT_DESIGN},
        "mismatch_both": {
            "train_env_kwargs": {"plant_kwargs": _SAT_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _SAT_PLANT},
            "controller_design": _SAT_DESIGN,
        },
        "stability_pin_bc_ppo": {
            "ppo_overrides": {"sigma_pin": 0.05, "lr_sigma_coupling": 1.5e-3},
            "disable_stages": True,
        },
        # ---- improvement ablations (one mechanism per condition) ----
        "improve_kl_anchor": {
            "improvements": {"bc_anchor_coef": 0.05, "bc_anchor_decay": 100},
            "disable_stages": True,
        },
        "improve_terminal_weight": {
            "improvements": {"terminal_weight": {"index": [4, 5, 6], "near": 0.5, "weight": 3.0}}
        },
        "improve_optimizer": {
            "improvements": {
                "lr_warmup": 25,
                "sigma_end": 0.2,
                "freeze_actor_updates": 10,
                "segment_updates": 10,
            },
            "disable_stages": True,
        },
        "improve_grad_clip": {
            "ppo_overrides": {"max_grad_norm": 0.5, "normalize_advantage_per_mini_batch": True}
        },
        "improve_rate_penalty": {"improvements": {"action_rate_penalty": 0.01}},
        "improve_delta_action": {"improvements": {"action_mode": "delta", "delta_scale": 0.25}},
        "improve_phase_gate": {
            "improvements": {
                "rho_channels": [0.05, 0.05, 0.05, 0.1],
                "phase_gate": {"index": [4, 5, 6], "near": 0.3, "soft": 0.1, "gain": 1.0},
            }
        },
        "improve_authority_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "authority_curriculum": {
                    "start": 0.01,
                    "end": 1.0,
                    "success_threshold": 0.9,
                    "step": 0.1,
                    "min_episodes": 8,
                },
            }
        },
        "improve_scratch_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "curriculum": [
                    {"fraction": 0.0, "kwargs": {"reset_noise_scale": 0.002}},
                    {"fraction": 0.4, "kwargs": {"reset_noise_scale": 0.01}},
                ],
            }
        },
        "improve_selection": {},
    },
    "rendezvous": {
        "coverage_narrow": {"train_env_kwargs": {"init_range": [1100.0, 1300.0]}},
        "coverage_wide": {"train_env_kwargs": {"init_range": [700.0, 1700.0]}},
        "samples_1x": {"bc": {"episodes_per_replica": 1}},
        "samples_4x": {"bc": {"episodes_per_replica": 4}},
        "authority_0.01": {"residual_rho": 0.01},
        "authority_0.025": {"residual_rho": 0.025},
        "authority_0.05": {},
        "authority_0.1": {"residual_rho": 0.1},
        "authority_0.2": {"residual_rho": 0.2},
        "mismatch_plant": {
            "train_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "controller_design": NOMINAL_DESIGN["rendezvous"],
        },
        "mismatch_design": {"controller_design": _DOCKING_DESIGN},
        "mismatch_both": {
            "train_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "eval_env_kwargs": {"plant_kwargs": _DOCKING_PLANT},
            "controller_design": _DOCKING_DESIGN,
        },
        "authority_0.4": {"residual_rho": 0.4},
        "stability_pin_bc_ppo": {
            "ppo_overrides": {"sigma_pin": 0.05, "lr_sigma_coupling": 1.5e-3},
            "disable_stages": True,
        },
        "stability_pin02_bc_ppo": {
            "ppo_overrides": {"sigma_pin": 0.2, "lr_sigma_coupling": 1.5e-3},
            "disable_stages": True,
        },
        # ---- improvement ablations (one mechanism per condition) ----
        "improve_kl_anchor": {
            "improvements": {"bc_anchor_coef": 0.05, "bc_anchor_decay": 100},
            "disable_stages": True,
        },
        "improve_terminal_weight": {
            "improvements": {"terminal_weight": {"index": [0, 1, 2], "near": 50.0, "weight": 3.0}}
        },
        "improve_optimizer": {
            "improvements": {
                "lr_warmup": 25,
                "sigma_end": 0.2,
                "freeze_actor_updates": 10,
                "segment_updates": 10,
            },
            "disable_stages": True,
        },
        "improve_grad_clip": {
            "ppo_overrides": {"max_grad_norm": 0.5, "normalize_advantage_per_mini_batch": True}
        },
        "improve_rate_penalty": {"improvements": {"action_rate_penalty": 0.01}},
        "improve_phase_gate": {
            "improvements": {
                "rho_channels": [0.05, 0.05, 0.1],
                "phase_gate": {"index": [0, 1, 2], "near": 100.0, "soft": 20.0, "gain": 1.0},
            }
        },
        "improve_authority_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "authority_curriculum": {
                    "start": 0.01,
                    "end": 1.0,
                    "success_threshold": 0.9,
                    "step": 0.1,
                    "min_episodes": 8,
                },
            }
        },
        "improve_scratch_curriculum": {
            "improvements": {
                "segment_updates": 10,
                "curriculum": [
                    {"fraction": 0.0, "kwargs": {"init_range": [1300.0, 1500.0]}},
                    {"fraction": 0.4, "kwargs": {"init_range": [1000.0, 1500.0]}},
                ],
            }
        },
        "improve_selection": {},
    },
}


# ---- BC-residual PPO tuning grid (frozen BC base + residual PPO) ------------
# Every entry shares one BC base per (task, seed): ``bc_base_condition`` points
# at the ``bc_only`` run that provides the frozen policy.  Only the listed
# residual/optimizer factor changes; everything else stays at the task recipe.
_BCR_SHARED = {"bc_base_condition": "default"}

TUNE_BCR = {
    # reference: task default recipe
    "tune_bcr_default": dict(_BCR_SHARED),
    # residual authority (fraction of the full normalized action range)
    "tune_bcr_rho_001": {**_BCR_SHARED, "residual_rho": 0.01},
    "tune_bcr_rho_0025": {**_BCR_SHARED, "residual_rho": 0.025},
    "tune_bcr_rho_005": {**_BCR_SHARED, "residual_rho": 0.05},
    "tune_bcr_rho_01": {**_BCR_SHARED, "residual_rho": 0.1},
    "tune_bcr_rho_02": {**_BCR_SHARED, "residual_rho": 0.2},
    # exploration std (task default is 0.5 docking/rendezvous, 1.0 satellite).
    # NOTE: `init_std` is a Gaussian-policy knob; the default BetaDistribution
    # ignores it, so these three arms only differ when the policy is switched
    # to `distribution='gaussian'`.
    "tune_bcr_sigma_02": {**_BCR_SHARED, "ppo_overrides": {"init_std": 0.2}},
    "tune_bcr_sigma_05": {**_BCR_SHARED, "ppo_overrides": {"init_std": 0.5}},
    "tune_bcr_sigma_10": {**_BCR_SHARED, "ppo_overrides": {"init_std": 1.0}},
    # pinned sigma + lr-sigma coupling (same drift rate at every update).
    # NOTE: these read `distribution.log_std_param` and therefore also require
    # `distribution='gaussian'`.
    "tune_bcr_pin_005": {
        **_BCR_SHARED,
        "ppo_overrides": {"sigma_pin": 0.05, "lr_sigma_coupling": 1.5e-3},
        "disable_stages": True,
    },
    "tune_bcr_pin_02": {
        **_BCR_SHARED,
        "ppo_overrides": {"sigma_pin": 0.2, "lr_sigma_coupling": 1.5e-3},
        "disable_stages": True,
    },
    # learning rate
    "tune_bcr_lr_1e4": {**_BCR_SHARED, "ppo_overrides": {"learning_rate": 1e-4}},
    "tune_bcr_lr_3e4": {**_BCR_SHARED, "ppo_overrides": {"learning_rate": 3e-4}},
    "tune_bcr_lr_1e3": {**_BCR_SHARED, "ppo_overrides": {"learning_rate": 1e-3}},
    # BC base quality: different demonstration sample budget, own bc_only run
    "tune_bcr_samples_1x": {
        "bc_base_condition": "tune_bcr_samples_1x",
        "bc": {"episodes_per_replica": 1},
    },
    "tune_bcr_samples_3x": {
        "bc_base_condition": "tune_bcr_samples_3x",
        "bc": {"episodes_per_replica": 3},
    },
}

# per-task phase-gate values (observation-only terminal phase signal)
_BCR_PHASE_GATE = {
    "docking": {
        "improvements": {
            "rho_channels": [0.05, 0.05, 0.1],
            "phase_gate": {"index": [0, 1, 2], "near": 20.0, "soft": 5.0, "gain": 1.0},
        },
        "bc_base_condition": "default",
    },
    "satellite": {
        "improvements": {
            "rho_channels": [0.05, 0.05, 0.05, 0.1],
            "phase_gate": {"index": [4, 5, 6], "near": 0.3, "soft": 0.1, "gain": 1.0},
        },
        "bc_base_condition": "default",
    },
    "rendezvous": {
        "improvements": {
            "rho_channels": [0.05, 0.05, 0.1],
            "phase_gate": {"index": [0, 1, 2], "near": 100.0, "soft": 20.0, "gain": 1.0},
        },
        "bc_base_condition": "default",
    },
}

for _task in ("docking", "satellite", "rendezvous"):
    _specs = {name: copy.deepcopy(spec) for name, spec in TUNE_BCR.items()}
    _specs["tune_bcr_phase_gate"] = copy.deepcopy(_BCR_PHASE_GATE[_task])
    SCANS[_task].update(_specs)


def resolve(task, condition, config=None):
    cfg = task_config(task, config)
    cond = {
        "name": condition,
        "task": task,
        "train_env_kwargs": {},
        "eval_env_kwargs": {},
        "controller_design": None,
        "controller_kwargs": {},
        "baseline_variant": None,
        "residual_rho": cfg["residual"]["rho"],
        "residual_squash": cfg["residual"].get("squash", "identity"),
        "bc": {},
        "student_base": "conventional",
        "bc_base_condition": None,
        "ppo_overrides": {},
        "disable_stages": False,
        "improvements": {},
    }
    if condition not in ("default", None):
        if condition not in SCANS.get(task, {}):
            raise KeyError(
                f"unknown condition {condition!r} for task {task!r}; "
                f"known: default + {sorted(SCANS.get(task, {}))}"
            )
        over = copy.deepcopy(SCANS[task][condition])
        cond.update(over)
        if "bc" in over:
            cond["bc"] = dict(over["bc"])
    # evaluation env defaults to the official benchmark contract unless the
    # condition explicitly selects the shifted plant it was trained/evaluated on
    if "eval_env_kwargs" not in cond:
        cond["eval_env_kwargs"] = {}
    if condition and condition.startswith("coverage"):
        # evaluation stays on the official task distribution (distribution shift
        # is the scientific question, not a changed benchmark)
        cond["eval_env_kwargs"] = {}
    return cond


def train_env_kwargs(task, cond, cfg):
    kw = dict(cfg.get("env_kwargs") or {})
    kw.update(cond.get("train_env_kwargs") or {})
    return kw


def eval_env_kwargs(task, cond, cfg):
    kw = dict(cfg.get("env_kwargs") or {})
    kw.update(cond.get("eval_env_kwargs") or {})
    return kw


def bc_overrides(task, cond, cfg):
    bc = dict(cfg.get("bc") or {})
    bc.update(cond.get("bc") or {})
    return bc
