"""Optional action-space / auxiliary-objective wrappers (five-tuple protocol).

Both wrappers sit between the task env and ``RslVecEnv`` and keep the env's
own reward, success condition and termination untouched.

``DeltaActionVecEnv``
    The policy emits an INCREMENT ``d in [-1, 1]``; the executed action is
    ``a_t = clamp(a_{t-1} + scale * d_t, -bound, bound)`` and the increment
    state resets to zero at every episode boundary.  Motivation: satellite
    CMG gimbal-rate commands are bang-bang-like (the conventional controller
    saturates); a rate (delta) parameterization lets an MLP express that class
    with far smaller output magnitudes than absolute commands.  Evaluation
    MUST use the same wrapper (it is part of the action semantics, not an
    auxiliary loss), which is recorded in eval_config.json.

``ActionRatePenaltyVecEnv``
    Adds ``-coef * ||a_t - a_{t-1}||^2`` to the TRAINING reward only.  This is
    an auxiliary objective ablation and is disabled for evaluation (the
    official reward is never modified in the env).
"""
import torch

__all__ = ['DeltaActionVecEnv', 'ActionRatePenaltyVecEnv']


class DeltaActionVecEnv:
    """Five-tuple wrapper turning absolute commands into bounded increments."""

    def __init__(self, env, *, delta_scale=1.0, name='delta_action'):
        self.__dict__['env'] = env
        self.delta_scale = float(delta_scale)
        if self.delta_scale <= 0:
            raise ValueError('delta_scale must be positive')
        self.name = name
        bound = float(getattr(env, 'clip_actions', 1.0))
        self.action_bound = bound
        self.prev_action = torch.zeros(int(env.num_envs), int(env.num_actions),
                                       device=env.device, dtype=torch.float32)
        self.last_applied = self.prev_action.clone()
        self.last_delta = torch.zeros_like(self.prev_action)
        self.last_increment = torch.zeros_like(self.prev_action)

    def _reset_mask(self, mask):
        if mask is not None and bool(mask.any()):
            self.prev_action.masked_fill_(mask[:, None], 0.0)

    def step(self, policy_action):
        """Apply policy delta as bounded increment on the previous action and step the env."""
        policy_action = policy_action.to(self.env.device)
        if policy_action.shape != self.prev_action.shape:
            raise ValueError(f'delta action shape {tuple(policy_action.shape)} != '
                             f'{tuple(self.prev_action.shape)}')
        if not torch.isfinite(policy_action).all():
            raise ValueError('policy produced a non-finite delta action')
        delta = policy_action.clamp(-1.0, 1.0)
        increment = delta * self.delta_scale
        applied = (self.prev_action + increment).clamp(-self.action_bound,
                                                       self.action_bound)
        self.last_delta = delta
        self.last_increment = increment
        self.last_applied = applied
        obs, rew, term, trunc, extras = self.env.step(applied)
        self.prev_action.copy_(applied)
        self._reset_mask(term | trunc)
        return obs, rew, term, trunc, extras

    def reset(self):
        """Reset the wrapped env and zero the internal previous-action state."""
        obs = self.env.reset()
        self.prev_action.zero_()
        self.last_applied.zero_()
        self.last_delta.zero_()
        self.last_increment.zero_()
        return obs

    def close(self):
        """Close the wrapped environment."""
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRatePenaltyVecEnv:
    """Five-tuple wrapper adding a training-only action-rate penalty."""

    def __init__(self, env, *, coef=0.01, name='action_rate_penalty'):
        self.__dict__['env'] = env
        self.coef = float(coef)
        if self.coef < 0:
            raise ValueError('coef must be non-negative')
        self.name = name
        self.prev_action = torch.zeros(int(env.num_envs), int(env.num_actions),
                                       device=env.device, dtype=torch.float32)
        self.last_penalty = torch.zeros(int(env.num_envs), device=env.device)
        self.last_env_reward = torch.zeros_like(self.last_penalty)

    def step(self, action):
        """Step the wrapped env and subtract an action-change penalty from the reward."""
        action = action.to(self.env.device)
        if action.shape != self.prev_action.shape:
            raise ValueError(f'action shape {tuple(action.shape)} != '
                             f'{tuple(self.prev_action.shape)}')
        obs, rew, term, trunc, extras = self.env.step(action)
        penalty = self.coef * (action - self.prev_action).pow(2).sum(dim=-1)
        self.last_env_reward = rew
        self.last_penalty = penalty
        self.prev_action.copy_(action)
        done = term | trunc
        if bool(done.any()):
            self.prev_action.masked_fill_(done[:, None], 0.0)
        return obs, rew - penalty, term, trunc, extras

    def reset(self):
        """Reset the wrapped env and zero penalty tracking buffers."""
        obs = self.env.reset()
        self.prev_action.zero_()
        self.last_penalty.zero_()
        self.last_env_reward.zero_()
        return obs

    def close(self):
        """Close the wrapped environment."""
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
