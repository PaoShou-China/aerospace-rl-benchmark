"""Batched aerospace RL environments: five vehicle categories (multirotor,
fixedwing, rocket, satellite, missile) + shared machinery.

Unified entry point
-------------------
All tasks implement the :class:`~environments.common.base_env.BaseEnv`
contract (``step`` / ``reset`` / ``get_observations`` / ``close`` plus the
``action_space`` / ``observation_space`` descriptors) and can be constructed
through a single registry::

    from environments import make
    env = make('rocket', num_envs=1024, backend='gpu', seed=7)

The task ids below match the names used by the training / eval scripts.
Direct class imports (e.g. ``from environments.rocket import RocketLandingEnv``)
continue to work unchanged.
"""

from .common.base_env import BaseEnv
from .common.spaces import Box

from .multirotor import QuadXWaypointsEnv, QuadXTrackEnv
from .fixedwing import FixedwingWaypointsEnv
from .rocket import RocketLandingEnv
from .missile import MissileInterceptEnv
from .vtol import QuadplaneWaypointsEnv
from .satellite import (
    SatelliteAttitudeEnv,
    LowThrustTransferEnv,
    SpaceDockingEnv,
    SpaceRendezvousEnv,
)

# Registry: task id -> (env_cls, default_kwargs).  ``make`` forwards
# ``num_envs``, ``backend`` and ``seed`` positionally (the shared prefix of
# every constructor) and merges any task-specific kwargs on top of the
# registered defaults.
ENV_REGISTRY = {
    "quadx": (QuadXWaypointsEnv, {}),
    "quadx_track": (QuadXTrackEnv, {}),
    "fixedwing": (FixedwingWaypointsEnv, {}),
    "rocket": (RocketLandingEnv, {}),
    "missile": (MissileInterceptEnv, {}),
    "quadplane": (QuadplaneWaypointsEnv, {}),
    "satellite": (SatelliteAttitudeEnv, {}),
    "transfer": (LowThrustTransferEnv, {}),
    "docking": (SpaceDockingEnv, {}),
    "rendezvous": (SpaceRendezvousEnv, {}),
}

__all__ = [
    "BaseEnv",
    "Box",
    "make",
    "ENV_REGISTRY",
    "QuadXWaypointsEnv",
    "QuadXTrackEnv",
    "FixedwingWaypointsEnv",
    "RocketLandingEnv",
    "MissileInterceptEnv",
    "QuadplaneWaypointsEnv",
    "SatelliteAttitudeEnv",
    "LowThrustTransferEnv",
    "SpaceDockingEnv",
    "SpaceRendezvousEnv",
]


def make(task, num_envs=64, backend="cpu", seed=7, **kwargs):
    """Construct a registered task environment.

    Parameters
    ----------
    task : str
        One of the keys in :data:`ENV_REGISTRY`.
    num_envs, backend, seed
        The shared constructor prefix accepted by every environment.
    **kwargs
        Task-specific constructor arguments (e.g. ``benchmark_version`` for
        ``rocket``, ``num_cmgs`` for ``satellite``).  Unknown kwargs raise
        ``TypeError`` from the environment constructor, just like a direct
        instantiation.

    Returns
    -------
    BaseEnv
        A fully-initialised environment (``reset`` has been called).
    """
    if task not in ENV_REGISTRY:
        known = sorted(ENV_REGISTRY)
        raise KeyError(f"unknown task {task!r}; known: {known}")
    env_cls, defaults = ENV_REGISTRY[task]
    merged = dict(defaults)
    merged.update(kwargs)
    return env_cls(num_envs=num_envs, backend=backend, seed=seed, **merged)
