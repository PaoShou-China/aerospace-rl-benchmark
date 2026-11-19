"""Reward-only ablations for a rocket policy that outputs actions directly.

This module does not initialize Genesis, call a controller, modify observations,
or reinterpret the benchmark's success predicate. Callers must provide the real
pre-autoreset next observation and an episode-persistent success latch.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


REWARD_CANDIDATE_VERSION = "direct-ppo-rewards-v1"
REWARD_CANDIDATE_MODES = ("legacy_scaled", "bounded_events")


def _number(name: str, value: float, *, minimum: float = 0.0,
            strictly_positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum or (strictly_positive and value <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else f'>= {minimum}'}")
    return value


def scaled_legacy_reward(legacy_reward: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
    """Multiply the supplied reward by one positive scalar, without clipping.

    The supplied reward can be from untouched upstream or a separately declared
    repaired environment. This function cannot turn one into the other.
    """
    scale = _number("scale", scale, strictly_positive=True)
    if not legacy_reward.is_floating_point():
        raise TypeError("legacy_reward must be floating point")
    return legacy_reward * scale


def bounded_landing_potential(observation: torch.Tensor, *, rest_height: float = 2.575,
                              reference_acceleration: float = 12.0,
                              reference_speed_limit: float = 80.0) -> torch.Tensor:
    """Observable landing heuristic in [-1, 0], evaluated at a policy boundary.

    Observation layout is body omega (0:3), quaternion xyzw (3:7), body
    velocity (7:10), and world position (10:13). Actuator fields and an optional
    appended policy context are ignored. The descent curve is reward guidance,
    not a controller or an assertion of physical recoverability.

    Quaternion normalization is local to reward calculation; it does not change
    the simulator quaternion or the original success/termination checks. Finite
    simulator observations are a caller precondition; NaNs propagate visibly.
    """
    if observation.ndim != 2 or observation.shape[1] < 13:
        raise ValueError("observation must have shape (num_envs, at least 13)")
    if not observation.is_floating_point():
        raise TypeError("observation must be floating point")
    rest_height = _number("rest_height", rest_height)
    reference_acceleration = _number("reference_acceleration", reference_acceleration,
                                     strictly_positive=True)
    reference_speed_limit = _number("reference_speed_limit", reference_speed_limit,
                                    strictly_positive=True)

    # Engine observations are normally float32. Double intermediates also keep
    # norms finite over the full representable range of finite float32 inputs.
    state = observation[:, :13].double()
    q = state[:, 3:7]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    xyz, w = q[:, :3], q[:, 3]
    velocity_body = state[:, 7:10]
    twice_cross = 2 * torch.linalg.cross(xyz, velocity_body)
    velocity = velocity_body + w[:, None] * twice_cross + torch.linalg.cross(xyz, twice_cross)
    x, y, z = xyz.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))

    height = (state[:, 12] - rest_height).clamp_min(0)
    desired_descent = -(2 * reference_acceleration * height).sqrt().clamp_max(reference_speed_limit)
    errors = torch.stack((
        height / 100.0,
        torch.linalg.vector_norm(state[:, 10:12], dim=-1) / 10.0,
        torch.linalg.vector_norm(velocity[:, :2], dim=-1) / 10.0,
        (velocity[:, 2] - desired_descent).abs() / 20.0,
        torch.linalg.vector_norm(torch.stack((roll, pitch), -1), dim=-1) / 0.3,
        torch.linalg.vector_norm(state[:, :3], dim=-1) / 0.3,
    ), dim=-1)
    # Six nonnegative terms, each in [0,1), with a convex combination.
    costs = errors / (1 + errors)
    weights = state.new_tensor((0.15, 0.20, 0.15, 0.25, 0.15, 0.10))
    return -(costs * weights).sum(-1).clamp(0, 1)


def compute_reward_candidate(
    previous_obs: torch.Tensor,
    next_obs: torch.Tensor,
    legacy_reward: torch.Tensor,
    outcome: Mapping[str, torch.Tensor],
    *,
    gamma: float,
    done: torch.Tensor,
    time_outs: torch.Tensor,
    seen_success: torch.Tensor,
    mode: str,
    options: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a reward and separate additive components/diagnostic potentials.

    ``next_obs`` MUST be the pre-reset terminal observation for done replicas.
    ``seen_success`` is the caller's latch before this transition. Update it as
    ``(seen_success | outcome['success_ever']) & ~done`` only after this call.
    ``time_outs`` is exactly the mask used for learner value bootstrapping:
    absorbing terminals have zero future potential; bootstrapped cutoffs retain
    the final state's potential. The function never gives a failure penalty
    merely because a bootstrapped cutoff occurred.

    For ``bounded_events``, the task surrogate pays first strict success once,
    and penalizes fatal/bounds termination even after earlier success. This
    changes the base reward objective; potential-based shaping is NOT evidence
    of equivalence to the old objective or of PPO convergence.

    No in-place mutation or host synchronization is performed. Tensor masks
    must be boolean, one-dimensional, and on the observation device. The caller
    remains responsible for coherent done/timeout/physical-failure flags.
    """
    if mode not in REWARD_CANDIDATE_MODES:
        raise ValueError(f"unknown reward candidate: {mode}")
    if not legacy_reward.is_floating_point():
        raise TypeError("legacy_reward must be floating point")
    gamma = _number("gamma", gamma, strictly_positive=True)
    if gamma > 1:
        raise ValueError("gamma must be in (0,1]")
    options = dict(options or {})
    if mode == "legacy_scaled":
        unknown = options.keys() - {"scale"}
        if unknown:
            raise ValueError(f"unknown legacy_scaled options: {sorted(unknown)}")
        reward = scaled_legacy_reward(legacy_reward, **options)
        return reward, {"scaled_legacy": reward}

    allowed = {"shaping_scale", "success_bonus", "failure_cost", "rest_height",
               "reference_acceleration", "reference_speed_limit"}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"unknown bounded_events options: {sorted(unknown)}")
    shaping_scale = _number("shaping_scale", options.get("shaping_scale", 1.0))
    success_bonus = _number("success_bonus", options.get("success_bonus", 10.0),
                            strictly_positive=True)
    failure_cost = _number("failure_cost", options.get("failure_cost", 20.0),
                           strictly_positive=True)
    if failure_cost <= success_bonus + shaping_scale:
        raise ValueError("failure_cost must exceed success_bonus + shaping_scale")
    if previous_obs.shape != next_obs.shape or previous_obs.device != next_obs.device:
        raise ValueError("previous_obs and next_obs must have matching shapes and devices")
    if legacy_reward.shape != previous_obs.shape[:1] or legacy_reward.device != previous_obs.device:
        raise ValueError("legacy_reward must match observation batch and device")
    for name, mask in {"done": done, "time_outs": time_outs, "seen_success": seen_success,
                       **{key: outcome[key] for key in
                          ("success_ever", "fatal_collision", "out_of_bounds")}}.items():
        if mask.dtype != torch.bool or mask.shape != previous_obs.shape[:1] or mask.device != previous_obs.device:
            raise ValueError(f"{name} must be a boolean mask matching observation batch and device")
    potential_options = {key: value for key, value in options.items() if key in
                         ("rest_height", "reference_acceleration", "reference_speed_limit")}
    phi_previous = bounded_landing_potential(previous_obs, **potential_options)
    phi_next = bounded_landing_potential(next_obs, **potential_options)
    absorbing_terminal = done & ~time_outs
    future_phi = torch.where(absorbing_terminal, torch.zeros_like(phi_next), phi_next)
    shaping = shaping_scale * (gamma * future_phi - phi_previous)
    new_success = outcome["success_ever"] & ~seen_success
    failed = done & (outcome["fatal_collision"] | outcome["out_of_bounds"])
    success_reward = new_success.to(shaping.dtype) * success_bonus
    failure_reward = -failed.to(shaping.dtype) * failure_cost
    reward = (shaping + success_reward + failure_reward).to(legacy_reward.dtype)
    return reward, {
        "potential_shaping": shaping,
        "success_event": success_reward,
        "failure_event": failure_reward,
        "phi_previous": phi_previous,
        "phi_next": phi_next,
        "absorbing_terminal": absorbing_terminal,
    }
