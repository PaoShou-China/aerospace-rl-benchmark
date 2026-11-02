"""Online gain-scheduling RL around a conventional controller.

The policy adjusts the base controller's tunable gains each step (instead of
adding a residual to the action).  See ``algorithms/gainsched/gain_sched_env.py``.
"""

from algorithms.gainsched.gain_sched_env import GainSchedulingVecEnv

__all__ = ["GainSchedulingVecEnv"]
