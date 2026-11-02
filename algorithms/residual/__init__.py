"""Residual reinforcement learning utilities (controller-prior experiments).

The package owns a conventional controller *outside* the task environment and
places it between the env and ``RslVecEnv`` so PPO keeps its original rollout /
log-probability semantics.  See ``algorithms/residual/residual_env.py``.
"""
from algorithms.residual.residual_env import (ResidualVecEnv, PolicyBaseController,
                                              zero_init_policy_output)

__all__ = ['ResidualVecEnv', 'PolicyBaseController', 'zero_init_policy_output']
