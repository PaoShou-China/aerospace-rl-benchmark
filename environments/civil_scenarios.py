"""Opt-in simulation challenges for civilian flight and spacecraft operations.

The numeric ranges are engineering stress hypotheses. They are not fitted to
flight data. The reference environment retains the task's native settings.
"""
from copy import deepcopy
from math import cos, pi, sin
from random import Random

# For each task, two fixed levels support development and out-of-range tests.
# Relative bounds multiply the nominal hardware value, distances are metres.
_LEVELS = {
    "rocket": (
        dict(variation="shifted", benchmark_version="v2", dryden=0.3),
        dict(variation="stress", benchmark_version="v2", dryden=0.65),
    ),
    "quadx": (
        dict(dryden=0.45, domain_randomize={"mass": (1.00, 1.13)}),
        dict(dryden=0.75, domain_randomize={"mass": (1.13, 1.22)}),
    ),
    "quadx_track": (
        dict(camera_sigma_m=0.22, domain_randomize={"mass": (1.00, 1.12)}),
        dict(camera_sigma_m=0.40, domain_randomize={"mass": (1.12, 1.20)}),
    ),
    "fixedwing": (
        dict(dryden=0.50, domain_randomize={"thrust": (0.88, 0.98)}),
        dict(dryden=0.80, domain_randomize={"thrust": (0.79, 0.88)}),
    ),
    "quadplane": (
        dict(dryden=0.45, domain_randomize={"mass": (1.00, 1.10)}),
        dict(dryden=0.75, domain_randomize={"mass": (1.10, 1.18)}),
    ),
    "satellite": (
        dict(reset_noise_scale=0.018, domain_randomize={"rotor_h_dyn": (0.94, 1.0)}),
        dict(reset_noise_scale=0.033, domain_randomize={"rotor_h_dyn": (0.85, 0.94)}),
    ),
    "transfer": (
        dict(eclipse_thrust_fraction=0.50, domain_randomize={"f_max": (0.91, 0.97), "isp": (0.96, 1.0)}),
        dict(eclipse_thrust_fraction=0.10, domain_randomize={"f_max": (0.82, 0.91), "isp": (0.91, 0.96)}),
    ),
    "docking": (
        dict(init_range=(95., 130.), init_vmax=0.55, domain_randomize={"mass": (1.00, 1.05)}),
        dict(init_range=(130., 150.), init_vmax=0.62, domain_randomize={"mass": (1.05, 1.12)}),
    ),
    "rendezvous": (
        dict(init_range=(1150., 1430.), init_vmax=1.04, domain_randomize={"f_max": (0.94, 1.0)}),
        dict(init_range=(1430., 1560.), init_vmax=1.10, domain_randomize={"f_max": (0.88, 0.94)}),
    ),
}
CIVIL_TASKS = tuple(_LEVELS)
_WIND_SPEED = {"quadx": (1.0, 2.4), "fixedwing": (2.5, 5.0), "quadplane": (1.8, 3.5)}


def scenario_kwargs(task, profile, seed):
    if task not in _LEVELS:
        raise ValueError(f"No civil scenario defined for {task!r}")
    if profile == "navigation_outage":
        from .navigation_hold import OUTAGE_WINDOWS
        if task not in OUTAGE_WINDOWS:
            raise ValueError(f"No navigation outage specified for {task!r}")
        profile = "hard"
    if profile not in ("reference", "development", "hard"):
        raise ValueError(f"Unknown civil profile {profile!r}")
    if profile == "reference":
        return {}
    index = int(profile == "hard")
    options = deepcopy(_LEVELS[task][index])
    if task in _WIND_SPEED:
        speed = _WIND_SPEED[task][index]
        heading = Random(int(seed) + 81427).uniform(0, 2 * pi)
        options.update(steady_wind=(speed * cos(heading), speed * sin(heading), 0.), severity="moderate")
    return options
