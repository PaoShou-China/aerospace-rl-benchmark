"""Shared entry point for online gain-scheduling RL across all tasks.

Each task uses its best classical baseline as the base controller.  The RL
policy schedules the controller's tunable gains (declared in
``TUNABLE_PARAMS``) each step instead of outputting environment actions.

Per-task training recipe
------------------------
For a same-task A/B, everything except the action space (gain settings vs env
actions) must match the task's own ``train_<task>.py`` entry.  ``RECIPES``
below mirrors each entry one-by-one: algorithm (all tasks use PPO), the
model/optimiser overrides, the update budget, ``env_kwargs`` and, where the
entry uses one, the training-only env subclass (rendezvous' eased spawn) and
the zero-init head patch (missile).  Values not listed fall back to the
shared ``build_cfg`` defaults, exactly as they do in the task entry.

NOT mirrored: BC warm start.  Its collector supervises a policy whose action
is the ENV action, while this arm's action is the gain vector, so the targets
are not interchangeable; the gain-scheduling arm therefore always starts
from scratch.  This asymmetry is a property of the method under test, not a
hyperparameter choice -- report it when comparing against warm-started arms.

Usage
-----
Edit ``TASK`` below and run ``python train_gainsched.py``.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))

from algorithms.baselines.baselines import build_baseline  # noqa: E402
from algorithms.gainsched import GainSchedulingVecEnv  # noqa: E402
from algorithms.train_utils import run_training  # noqa: E402

# ---------------------------------------------------------------- task table
# task -> (env module path, env class name, best baseline variant or None)
# The 'variant' is passed to build_baseline so the BEST classical method is
# used as the scheduling base (see docs/baseline_test_report.md).
TASK_TABLE = {
    "quadx": ("environments.multirotor", "QuadXWaypointsEnv", None),
    "quadx_track": ("environments.multirotor", "QuadXTrackEnv", None),
    "rocket": ("environments.rocket", "RocketLandingEnv", None),
    "satellite": ("environments.satellite.attitude_control", "SatelliteAttitudeEnv", None),
    "transfer": ("environments.satellite.orbit_transfer", "LowThrustTransferEnv", None),
    "docking": ("environments.satellite.rendezvous_docking", "SpaceDockingEnv", None),
    "rendezvous": ("environments.satellite.rendezvous_docking", "SpaceRendezvousEnv", "cwh_pd"),
    "fixedwing": ("environments.fixedwing", "FixedwingWaypointsEnv", "nlgl_tecs"),
    "missile": ("environments.missile", "MissileInterceptEnv", "tpn"),
    "quadplane": ("environments.vtol.env", "QuadplaneWaypointsEnv", None),
}

# ------------------------------------------------------- per-task recipes
# Mirrored from each task's own train entry (2026-10-31).  Keys:
#   algo            : 'ppo'                     (entry's ALGORITHM)
#   updates         : update budget             (entry's UPDATES)
#   env_kwargs      : env-ctor kwargs           (entry's WIND / ENV_KWARGS)
#   ppo             : model/optimiser overrides consumed by build_cfg
#   train_env_entry : (module, class) training-only env subclass, if any
#   zero_init_entry : task entry imported for its zero-init head patch, if any
_WIND = dict(steady_wind=None, dryden=0.3, severity="moderate")

RECIPES = {
    # train_quadx.py: fixed 3e-4, 5 epochs, entropy .01, init_std 1.0, nsp 100
    "quadx": dict(
        algo="ppo",
        updates=1000,
        env_kwargs=dict(_WIND),
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            entropy_coef=0.01,
            init_std=1.0,
            num_steps_per_env=100,
            num_learning_epochs=5,
        ),
    ),
    # train_quadx_track.py: same recipe, 20_000 env steps / 100 = 200 updates
    "quadx_track": dict(
        algo="ppo",
        updates=200,
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            entropy_coef=0.01,
            init_std=1.0,
            num_steps_per_env=100,
            num_learning_epochs=5,
        ),
    ),
    # train_fixedwing.py: fixed 3e-4, nsp 64, 10 epochs, entropy 0, vf 0.5
    "fixedwing": dict(
        algo="ppo",
        updates=1000,
        env_kwargs=dict(_WIND),
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            num_steps_per_env=64,
            num_learning_epochs=10,
            entropy_coef=0.0,
            value_loss_coef=0.5,
        ),
    ),
    # train_rocket.py: PPO, 2000 updates, gamma 0.995, lr 3e-4, nsp 64
    "rocket": dict(
        algo="ppo",
        updates=2000,
        env_kwargs=dict(
            steady_wind=None, dryden=0.0, severity="moderate", benchmark_version="v2"
        ),
        ppo=dict(
            num_steps_per_env=64,
            lr_schedule="fixed",
            learning_rate=3e-4,
            gamma=0.995,
            lam=0.97,
            entropy_coef=0.007,
            num_learning_epochs=10,
            clip_param=0.2,
            value_loss_coef=1.0,
            desired_kl=0.01,
        ),
    ),
    # train_missile.py: nsp 64, gamma .995, lam .97, entropy .007, init_std .5,
    # 10 epochs; zero-init head; terminal_gauss reward variant
    "missile": dict(
        algo="ppo",
        updates=1000,
        env_kwargs={"reward_variant": "terminal_gauss"},
        zero_init_entry="environments.missile.train_missile",
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            num_steps_per_env=64,
            gamma=0.995,
            lam=0.97,
            entropy_coef=0.007,
            init_std=0.5,
            num_learning_epochs=10,
            clip_param=0.2,
            value_loss_coef=1.0,
            desired_kl=0.01,
        ),
    ),
    # train_quadplane.py: same recipe as train_fixedwing.py
    "quadplane": dict(
        algo="ppo",
        updates=1000,
        env_kwargs=dict(_WIND),
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            num_steps_per_env=64,
            num_learning_epochs=10,
            entropy_coef=0.0,
            value_loss_coef=0.5,
        ),
    ),
    # train_satellite.py: gamma .999, nsp 150, mb 5, 40 epochs, entropy 0,
    # init_std 1.0, fixed lr 3e-4
    "satellite": dict(
        algo="ppo",
        updates=1000,
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            gamma=0.999,
            num_steps_per_env=150,
            num_mini_batches=5,
            num_learning_epochs=40,
            entropy_coef=0.0,
            init_std=1.0,
            clip_param=0.2,
        ),
    ),
    # train_transfer.py: gamma .9999, lr 2.5e-4, lam .99, entropy 4.75e-8,
    # 30 epochs, clip .3, vf 0.5
    "transfer": dict(
        algo="ppo",
        updates=1000,
        ppo=dict(
            gamma=0.9999,
            lr_schedule="fixed",
            learning_rate=2.5e-4,
            lam=0.99,
            entropy_coef=4.75e-8,
            num_learning_epochs=30,
            clip_param=0.3,
            value_loss_coef=0.5,
        ),
    ),
    # train_docking.py: fixed lr 3e-4, entropy .005, gamma .999 (other knobs at
    # the build_cfg defaults: nsp 32, 4 epochs)
    "docking": dict(
        algo="ppo",
        updates=1000,
        ppo=dict(
            lr_schedule="fixed",
            learning_rate=3e-4,
            entropy_coef=0.005,
            gamma=0.999,
        ),
    ),
    # train_rendezvous.py: nsp 250, gamma .999, mb 1, entropy 0, sigma pinned
    # 0.2 with lr = 1.5e-3 * sigma (= 3e-4), 1500 updates, eased training spawn
    "rendezvous": dict(
        algo="ppo",
        updates=1500,
        train_env_entry=(
            "environments.satellite.rendezvous_docking.train_rendezvous",
            "_EasedSpawnEnv",
        ),
        ppo=dict(
            num_steps_per_env=250,
            gamma=0.999,
            num_mini_batches=1,
            entropy_coef=0.0,
            lr_sigma_coupling=1.5e-3,
            sigma_pin=0.2,
        ),
    ),
}

# ---------------------------------------------------------------- config
TASK = "rocket"  # <-- edit this to switch tasks
NUM_ENVS = 1024
SEED = 7
BACKEND = "gpu"
RESUME = False
CLEAN = True

# Gain-scheduling wrapper settings
GAIN_SCALE_FRAC = 1.0  # 1.0 = full authority over [low,high]; <1 keeps schedule near nominal
SMOOTHING = 0.0  # 0 = raw request, >0 = EMA filter


def _import_env(module_path, class_name):
    import importlib  # noqa: E402

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def make_gain_sched_wrapper(task, gain_scale_frac, smoothing):
    """Return an env_wrapper callable that builds the baseline + scheduler."""
    variant = TASK_TABLE[task][2]

    def wrapper(env):
        controller = build_baseline(task, env, variant=variant)
        if not controller.TUNABLE_PARAMS:
            raise RuntimeError(
                f"{task} baseline {controller.name} has no TUNABLE_PARAMS; cannot schedule gains"
            )
        return GainSchedulingVecEnv(
            env,
            controller,
            gain_scale_frac=gain_scale_frac,
            smoothing=smoothing,
            task=task,
            name=f"gainsched_{controller.name}",
        )

    return wrapper


if __name__ == "__main__":
    if TASK not in TASK_TABLE:
        raise ValueError(f"unknown task {TASK!r}; choose from {sorted(TASK_TABLE)}")
    recipe = RECIPES[TASK]

    if recipe.get("zero_init_entry"):
        # Importing the task entry applies its zero-init head patch, mirroring
        # the shipped arm's initialization (the untrained mean gain is 0 ->
        # exactly the hand-tuned controller).
        import importlib

        importlib.import_module(recipe["zero_init_entry"])

    if recipe.get("train_env_entry"):
        EnvCls = _import_env(*recipe["train_env_entry"])
    else:
        module_path, env_cls_name, _ = TASK_TABLE[TASK]
        EnvCls = _import_env(module_path, env_cls_name)

    wrapper = make_gain_sched_wrapper(TASK, GAIN_SCALE_FRAC, SMOOTHING)
    out_dir = _ROOT / "results" / f"{TASK}_gainsched"
    extra = dict(recipe.get("ppo") or {})
    run_training(
        TASK,
        EnvCls,
        num_envs=NUM_ENVS,
        updates=recipe["updates"],
        seed=SEED,
        backend=BACKEND,
        resume=RESUME,
        clean=CLEAN,
        algo=recipe["algo"],
        env_kwargs=recipe.get("env_kwargs"),
        env_wrapper=wrapper,
        out_dir=str(out_dir),
        experiment_meta={
            "method": "gain_scheduling",
            "gain_scale_frac": GAIN_SCALE_FRAC,
            "smoothing": SMOOTHING,
            "task_recipe": recipe,
        },
        **extra,
    )
