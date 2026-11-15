"""PPO with a frozen behavior-cloning anchor (BC-regularized policy update).

Motivation
----------
The shared PPO recipe occasionally destroys a good BC initialization
(docking / rendezvous pilot): the actor drifts away from the controller
prior within a few dozen updates.  A pinned exploration std is not enough
for rendezvous, so this alternative keeps the *policy distribution* close
to the frozen BC actor with an explicit KL penalty:

    L = L_PPO + beta * mean_{s ~ rollout} KL(pi_theta(.|s) || pi_BC(.|s))

The anchor is the BC actor itself (frozen, eval mode, its own frozen
observation normalizer), never the analytical controller.  ``beta`` can
decay linearly to zero so the policy is eventually free to improve.
This is an optimization-level ablation, not a knowledge mechanism: it
must be reported separately from BC -> PPO without the anchor.

Only the policy mean/scale are anchored; the critic, PPO ratio and
advantage computation are unchanged.  The anchor is enabled *after* the
BC initialization is loaded, so a warm start from a checkpoint and a
fitted BC actor use exactly the same target.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class KLAnchorPPO(PPO):
    """PPO plus an optional frozen BC policy KL penalty.

    Parameters
    ----------
    kl_anchor_coef
        Initial coefficient of the KL term.  0 disables the anchor.
    kl_anchor_decay_updates
        Linear decay horizon in ``update()`` calls.  0 keeps the
        coefficient constant.
    """

    def __init__(self, *args, kl_anchor_coef: float = 0.0,
                 kl_anchor_decay_updates: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_anchor_coef = float(kl_anchor_coef)
        self.kl_anchor_decay_updates = int(kl_anchor_decay_updates)
        self.bc_anchor = None
        self._anchor_calls = 0

    def set_bc_anchor(self, coef: float | None = None, decay_updates: int | None = None):
        """Freeze a copy of the current actor as the BC target."""
        if self.bc_anchor is not None:
            raise RuntimeError('BC anchor already set; it is frozen for the whole run')
        self.bc_anchor = copy.deepcopy(self.actor).eval()
        for p in self.bc_anchor.parameters():
            p.requires_grad_(False)
        if coef is not None:
            self.kl_anchor_coef = float(coef)
        if decay_updates is not None:
            self.kl_anchor_decay_updates = int(decay_updates)

    def _anchor_coef(self):
        if self.kl_anchor_coef <= 0.0:
            return 0.0
        if self.kl_anchor_decay_updates > 0:
            frac = max(0.0, 1.0 - self._anchor_calls / self.kl_anchor_decay_updates)
            return self.kl_anchor_coef * frac
        return self.kl_anchor_coef

    def update(self) -> dict[str, float]:
        """PPO update with the optional BC KL anchor (same PPO numerics)."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_anchor_kl = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        beta = self._anchor_coef()
        if self.bc_anchor is None:
            beta = 0.0
        with torch.no_grad():
            if self.bc_anchor is not None:
                self.bc_anchor.eval()

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = ((batch.advantages - batch.advantages.mean())
                                        / (batch.advantages.std() + 1e-8))
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(batch.observations, masks=batch.masks,
                       hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks,
                                 hidden_state=batch.hidden_states[1])
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params,
                                                      distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean())

            anchor_kl = None
            if beta > 0.0:
                obs = batch.observations
                with torch.no_grad():
                    self.bc_anchor(obs, stochastic_output=True)
                    bc_params = tuple(p[:original_batch_size]
                                      for p in self.bc_anchor.output_distribution_params)
                anchor_kl = self.actor.get_kl_divergence(
                    distribution_params, tuple(p.detach() for p in bc_params)).mean()
                loss = loss + beta * anchor_kl

            rnd_loss = (self.rnd.compute_loss(batch.observations[:original_batch_size])
                        if self.rnd else None)
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(
                    self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_anchor_kl += float(anchor_kl.detach()) if anchor_kl is not None else 0.0
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_anchor_kl /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self._anchor_calls += 1

        loss_dict = {"value": mean_value_loss, "surrogate": mean_surrogate_loss,
                     "entropy": mean_entropy}
        if beta > 0.0:
            loss_dict["bc_anchor_kl"] = mean_anchor_kl
            loss_dict["bc_anchor_beta"] = beta
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        self.storage.clear()
        return loss_dict
