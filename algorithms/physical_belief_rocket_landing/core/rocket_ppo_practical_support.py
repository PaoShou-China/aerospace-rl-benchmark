"""One practical direct-PPO route; upstream plant and task predicates stay intact."""
import copy

import torch
from rsl_rl.algorithms import PPO

from .rocket_direct_support import DirectRewardAdapter
from .rocket_flash_ppo import ProjectedFlashPPO


def tracking_terms(obs):
    """World-frame descent/lateral guidance, bounded to avoid runaway costs."""
    state = obs.double()
    q = state[:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz, w = q[:, :3], q[:, 3]
    cross = 2 * torch.linalg.cross(xyz, state[:, 7:10])
    velocity = state[:, 7:10] + w[:, None] * cross + torch.linalg.cross(xyz, cross)
    x, y, z = xyz.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    h = (state[:, 12] - 2.575).clamp_min(0)
    remaining = (2*h/12).sqrt().clamp(2., 12.)
    desired_xy = (-state[:, 10:12] / remaining[:, None]).clamp(-10., 10.)
    desired_z = -(2*12*h).sqrt().clamp_max(80.)
    lateral = ((velocity[:, :2] - desired_xy) / 8.).square().sum(-1)
    descent = ((velocity[:, 2] - desired_z) / 20.).square()
    attitude = (torch.stack((roll, pitch), -1) / .3).square().sum(-1)
    spin = (state[:, :3] / .3).square().sum(-1)
    costs = torch.stack((lateral, descent, attitude, spin), -1)
    bounded = costs / (1 + costs)
    tracking = (bounded * state.new_tensor([.3, .35, .2, .15])).sum(-1)
    distance = state[:, 10:12].norm(dim=-1)
    potential = -(.25*h/(100+h) + .25*distance/(10+distance) + .5*tracking)
    return potential, tracking


def landing_terms(obs, pad_contact):
    """Bounded approach potential and risk in the original observation units.

    The desired descent tends continuously to zero at the pad. The two
    velocity scales deliberately distinguish surviving contact (1 m/s) from
    the original settled criterion (0.02 m/s); neither predicate is changed.
    """
    state = obs.double()
    q = state[:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz, w = q[:, :3], q[:, 3]
    cross = 2 * torch.linalg.cross(xyz, state[:, 7:10])
    velocity = state[:, 7:10] + w[:, None] * cross + torch.linalg.cross(xyz, cross)
    x, y, z = xyz.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    tilt = torch.stack((roll, pitch), -1).norm(dim=-1)
    spin = state[:, :3].norm(dim=-1)
    speed = state[:, 7:10].norm(dim=-1)
    h = (state[:, 12] - 2.575).clamp_min(0)
    near = (1 - h / 10).clamp(0, 1)
    blend = ((h - 2) / 8).clamp(0, 1)
    blend = blend.square() * (3 - 2 * blend)
    flight_descent = (24*h).sqrt().clamp_max(80.)
    soft_descent = torch.minimum((24*h).sqrt(), .8*h)
    desired_z = -((1-blend)*soft_descent + blend*flight_descent)
    desired_xy = (-state[:, 10:12] / (2 + h)[:, None]).clamp(-2., 2.)
    costs = torch.stack((((velocity[:, :2]-desired_xy)/.25).square().sum(-1),
                         ((velocity[:, 2]-desired_z)/.5).square(),
                         (spin/.05).square(), (tilt/.1).square()), -1)
    bounded = costs / (1 + costs)
    near_cost = (bounded * state.new_tensor([.25, .4, .2, .15])).sum(-1)
    distance = state[:, 10:12].norm(dim=-1)
    near_phi = -(.15*h/(2+h) + .2*distance/(2+distance) + .65*near_cost)
    far_phi, tracking = tracking_terms(obs)
    phi = (1-near)*far_phi + near*near_phi
    # The risk remains small and bounded. Current contact activates the same
    # risk even if numerical contact occurs slightly above the nominal pad.
    risk_weight = torch.maximum(near, pad_contact.double())
    hazards = torch.stack(((speed/1.).square(), (spin/.35).square(),
                           (tilt/.1).square()), -1)
    risk = risk_weight * ((hazards/(1+hazards))*state.new_tensor([.55, .25, .2])).sum(-1)
    settled = pad_contact & (speed < .02) & (spin < .02) & (tilt < .1)
    return phi, tracking, risk, settled


class PracticalRewardAdapter(DirectRewardAdapter):
    def __init__(self, env, gamma=.999, ever_priority=False, success_terminal=False,
                 reward_variant="legacy", success_hold_steps=0):
        if reward_variant not in ("legacy", "touchdown", "settle"):
            raise ValueError(f"unknown practical reward variant: {reward_variant}")
        if success_hold_steps < 0 or int(success_hold_steps) != success_hold_steps:
            raise ValueError("success_hold_steps must be a nonnegative integer")
        super().__init__(env, "original", gamma)
        self.ever_priority = ever_priority
        self.reward_variant = reward_variant
        self.success_terminal = success_terminal or reward_variant != "legacy"
        self.success_hold_steps = (max(64, int(success_hold_steps)) if reward_variant == "settle"
                                   else int(success_hold_steps))
        self.early_success_counts = torch.zeros_like(env.completed_episodes)
        self._success_age = torch.full_like(env.completed_episodes, -1)
        self._stable_reward_steps = torch.zeros_like(env.completed_episodes)
        self._terminal_pad_contact = torch.zeros_like(self.seen_success)
        self.physical_failures = 0
        self.timeout_count = 0
        reset = env.reset_idx

        def record_contact_then_reset(mask):
            # The observation contact bit can lag the last physics substep.
            # Capture the current contact before the upstream reset overwrites
            # it, without taking another simulation step or changing RNG use.
            contact = self._pad_contact()
            self._terminal_pad_contact.copy_(torch.where(mask, contact, self._terminal_pad_contact))
            self._success_age.masked_fill_(mask, -1)
            self._stable_reward_steps.masked_fill_(mask, 0)
            reset(mask)

        env.reset_idx = record_contact_then_reset

    def _pad_contact(self):
        return self.env.plant.pad_contact

    def reset(self):
        self._success_age.fill_(-1)
        self._stable_reward_steps.zero_()
        return super().reset()

    @property
    def completed_episodes(self):
        return self.env.completed_episodes + self.early_success_counts

    @property
    def completed_successes(self):
        return self.env.completed_successes + self.early_success_counts

    def step(self, actions):
        previous = self.env.get_observations()["policy"]
        seen = self.seen_success.clone()
        previous_contact = self._pad_contact().clone()
        age = self._success_age.clone()
        stable_reward_steps = self._stable_reward_steps.clone()
        previous_learner_total = self.reward_sums.get("learner_reward")
        self._terminal_pad_contact.zero_()
        obs, raw, done, extras = super().step(actions)
        outcome = {name: torch.where(done, self._terminal[name], field)
                   for name, field in self._fields().items()}
        following = torch.where(done[:, None], extras["terminal_observations"]["policy"], obs["policy"])
        first_success = outcome["success_ever"] & ~seen
        contact = torch.where(done, self._terminal_pad_contact, self._pad_contact())
        current_settled = torch.zeros_like(done)
        stable_bonus = torch.zeros_like(raw, dtype=torch.float64)
        risk = torch.zeros_like(stable_bonus)
        age = torch.where(first_success, torch.zeros_like(age),
                          torch.where(seen, age + 1, torch.full_like(age, -1)))
        if self.reward_variant == "legacy":
            phi0, _ = tracking_terms(previous)
            phi1, cost = tracking_terms(following)
            early = first_success & ~done if self.success_terminal else torch.zeros_like(done)
        else:
            phi0, _, _, _ = landing_terms(previous, previous_contact)
            phi1, cost, risk, current_settled = landing_terms(following, contact)
            current_settled &= ~outcome["fatal_collision"] & ~outcome["out_of_bounds"]
            if self.reward_variant == "touchdown":
                early = first_success & ~done
            else:
                # Age zero is the first success. A 64-step hold therefore
                # collects 64 additional real transitions before termination.
                holding = seen & (age <= self.success_hold_steps)
                eligible = holding & current_settled & (stable_reward_steps < self.success_hold_steps)
                stable_bonus = (6.4/self.success_hold_steps) * eligible.double()
                stable_reward_steps += eligible.long()
                early = seen & (age >= self.success_hold_steps) & current_settled & ~done
        terminal = (done & ~extras["time_outs"]) | early
        future = torch.where(terminal, torch.zeros_like(phi1), phi1)
        physical_failure = done & (outcome["fatal_collision"] | outcome["out_of_bounds"])
        failure = (physical_failure & ~outcome["success_ever"]
                   if self.ever_priority and self.reward_variant == "legacy" else physical_failure)
        if self.reward_variant == "legacy":
            gain, time_cost, success_bonus, failure_cost = ((1., .001, 10., 10.) if self.ever_priority
                                                           else (10., .01, 50., 100.))
        else:
            gain, time_cost, success_bonus, failure_cost = 4., .002, 25., 30.
        terms = {"practical_shaping": gain*(self.gamma*future-phi0),
                 "practical_tracking_cost": -time_cost*cost,
                 "practical_success": success_bonus*first_success.double(),
                 "practical_failure": -failure_cost*failure.double(),
                 "practical_risk_cost": -.02*risk,
                 "practical_stable_bonus": stable_bonus,
                 "practical_settled_now": current_settled.double(),
                 "practical_post_success_fatal": (physical_failure & outcome["success_ever"]).double(),
                 "practical_early_success_terminals": early.double(),
                 "practical_timeouts": extras["time_outs"].double()}
        reward = sum(terms[name] for name in ("practical_shaping", "practical_tracking_cost",
                                             "practical_success", "practical_failure",
                                             "practical_risk_cost", "practical_stable_bonus")).to(raw.dtype)
        # DirectRewardAdapter accounted for the raw reward before this second
        # transformation. Keep legacy_raw, but make learner_reward truthful.
        reward_total = reward.detach().double().sum()
        self.reward_sums["learner_reward"] = (reward_total if previous_learner_total is None
                                               else previous_learner_total + reward_total)
        for name, value in {**terms, "practical_return": reward}.items():
            total = value.detach().double().sum()
            self.reward_sums[name] = self.reward_sums.get(name, torch.zeros_like(total)) + total
        if bool(early.any()):
            # This is an explicit training-only terminal. Preserve the actual
            # success state before the ordinary upstream reset mutates buffers.
            extras = dict(extras)
            extras["terminal_observations"] = extras["terminal_observations"].clone()
            extras["time_outs"] = extras["time_outs"] & ~early
            done = done | early
            self.early_success_counts += early.long()
            self.env.reset_idx(early)
            self.env.reset_buf = done
            self.env._update_observation()
            obs = self.env.get_observations()
        self._success_age.copy_(torch.where(done, torch.full_like(age, -1), age))
        self._stable_reward_steps.copy_(torch.where(done, torch.zeros_like(stable_reward_steps),
                                                    stable_reward_steps))
        if self.success_terminal:
            # Upstream's episode logger covers only natural terminations; do
            # not expose that subset as the modified training success rate.
            extras.pop("episode", None)
        return obs, reward, done, extras


class PracticalPPO(ProjectedFlashPPO):
    """Projected Flash PPO with explicit pre-reset terminal-state bootstrap."""
    def process_env_step(self, obs, rewards, dones, extras):
        timeouts = extras.get("time_outs")
        if timeouts is None:
            return super().process_env_step(obs, rewards, dones, extras)
        with torch.no_grad():
            values = self.critic(extras["terminal_observations"].to(self.device)).squeeze(-1)
        adjusted = rewards + self.gamma * torch.where(timeouts, values, torch.zeros_like(values))
        forwarded = dict(extras)
        forwarded["time_outs"] = torch.zeros_like(timeouts)
        return super().process_env_step(obs, adjusted, dones, forwarded)


class AnchoredPracticalPPO(PracticalPPO):
    """PPO with critic warmup, frozen observation statistics and a reference KL.

    The reference is a frozen learned DAgger actor, never a control fallback.
    No reference actions are executed and evaluation uses the learned actor only.
    """
    def __init__(self, *args, critic_learning_rate=3e-4, reference_kl_coef=.05,
                 critic_warmup_updates=8, max_update_kl=.01, hard_update_kl=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_actor = None
        self.reference_kl_coef = reference_kl_coef
        self.critic_warmup_updates = critic_warmup_updates
        self.max_update_kl = max_update_kl
        self.hard_update_kl = hard_update_kl
        self.update_index = 0
        # Preserve the projection hook on this optimizer, with separate rates.
        self.optimizer.param_groups[0]["params"] = list(self.actor.parameters())
        self.optimizer.add_param_group({"params": list(self.critic.parameters()), "lr": critic_learning_rate})

    def set_reference(self):
        self.reference_actor = copy.deepcopy(self.actor).eval()
        self.reference_actor.requires_grad_(False)

    def _masked_phase_mean(self, values, observations):
        """Measure trust-region movement only where trainable heads execute."""
        experts = getattr(self, "trainable_experts", None)
        if experts is None:
            return values.mean()
        h = observations["policy"][:, 12] - 2.575
        phase = torch.where(h > 10, 0, torch.where(h > 2, 1, 2))
        mask = torch.zeros_like(phase, dtype=torch.bool)
        for expert in experts:
            mask |= phase == expert
        per_sample = values.reshape(len(mask), -1).mean(-1)
        # A zero-active minibatch returns a differentiable zero, not NaN.
        return (per_sample * mask).sum() / mask.sum().clamp_min(1)

    def update(self):
        if self.reference_actor is None:
            raise RuntimeError("set_reference after loading the DAgger actor")
        warmup = self.update_index < self.critic_warmup_updates
        actor_stopped = warmup
        totals = {name: 0. for name in ("value", "surrogate", "reference_kl", "update_kl", "entropy", "offline_bc")}
        minibatches, actor_steps = 0, 0
        backtracks, rejected, maximum_accepted_kl = 0, 0, 0.
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            observations = batch.observations
            self.actor(observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            mean, std = self.actor.output_distribution_params
            with torch.no_grad():
                reference_mean = self.reference_actor(observations)
                update_kl = self._masked_phase_mean(
                    self.actor.get_kl_divergence(batch.old_distribution_params,
                                                self.actor.output_distribution_params), observations)
            if float(update_kl) > self.max_update_kl:
                actor_stopped = True
            reference_kl = self._masked_phase_mean(
                .5*((mean-reference_mean)/std).square().sum(-1), observations)
            ratio = (log_prob-batch.old_actions_log_prob.squeeze(-1)).exp()
            advantage = batch.advantages.squeeze(-1)
            surrogate = torch.maximum(-advantage*ratio,
                                      -advantage*ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
            values = self.critic(observations)
            value_loss = (values-batch.returns).square().mean()
            loss = self.value_loss_coef*value_loss
            bc_loss = torch.zeros((), device=self.device)
            if not actor_stopped:
                loss = loss + surrogate + self.reference_kl_coef*reference_kl
                if getattr(self, "offline_observations", None) is not None:
                    bc_loss = self.offline_bc_loss()
                    loss = loss + self.offline_bc_coef*bc_loss
                actor_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            before_actor = ([parameter.detach().clone() for parameter in self.actor.parameters()]
                            if self.hard_update_kl and not actor_stopped else None)
            self.optimizer.step()
            if before_actor is not None:
                # Check the actual post-Adam, post-projection policy. Merely
                # stopping later minibatches cannot prevent a large first step.
                with torch.no_grad():
                    proposed = [parameter.detach().clone() for parameter in self.actor.parameters()]
                    old_mean, old_std = batch.old_distribution_params
                    accepted = False
                    for attempt in range(13):
                        current_mean = self.actor(observations)
                        actual_kl = self._masked_phase_mean(
                            .5*((current_mean-old_mean)/old_std).square().sum(-1), observations)
                        if float(actual_kl) <= self.max_update_kl:
                            maximum_accepted_kl = max(maximum_accepted_kl, float(actual_kl))
                            accepted = True
                            break
                        backtracks += 1
                        fraction = .5 ** (attempt+1)
                        for parameter, previous, candidate in zip(self.actor.parameters(), before_actor, proposed):
                            parameter.copy_(previous + fraction*(candidate-previous))
                        project = getattr(self.actor, "project_parameters", None)
                        if project is not None:
                            project()
                    if not accepted:
                        for parameter, previous in zip(self.actor.parameters(), before_actor):
                            parameter.copy_(previous)
                        rejected += 1
                        actor_stopped = True
            for key, value in (("value", value_loss), ("surrogate", surrogate),
                               ("reference_kl", reference_kl), ("update_kl", update_kl),
                               ("entropy", self.actor.output_entropy.mean()), ("offline_bc", bc_loss)):
                totals[key] += float(value.detach())
            minibatches += 1
        # The inherited actor normalizer is fixed: updating it alone changes
        # policy outputs outside PPO clipping and would also move the reference.
        self.storage.clear()
        self.update_index += 1
        return {**{key: value/minibatches for key,value in totals.items()},
                "actor_optimizer_steps": actor_steps, "critic_warmup": float(warmup),
                "actor_backtracks": backtracks, "rejected_actor_steps": rejected,
                "offline_bc_samples_drawn": getattr(self, "offline_bc_samples_drawn", 0),
                "maximum_accepted_update_kl": maximum_accepted_kl}


class MoEAnchoredPracticalPPO(AnchoredPracticalPPO):
    """The same guarded update over standard MLP/MoE actors, without Flash hooks."""
    def __init__(self, *args, critic_learning_rate=3e-4, reference_kl_coef=.05,
                 critic_warmup_updates=8, max_update_kl=.01, hard_update_kl=True, **kwargs):
        # Deliberately bypass ProjectedFlashPPO.__init__: these models have no
        # unit-norm projection and no Flash optimizer hook should be installed.
        PPO.__init__(self, *args, **kwargs)
        self.reference_actor = None
        self.reference_kl_coef = reference_kl_coef
        self.critic_warmup_updates = critic_warmup_updates
        self.max_update_kl = max_update_kl
        self.hard_update_kl = hard_update_kl
        self.update_index = 0
        self.optimizer.param_groups[0]["params"] = list(self.actor.parameters())
        self.optimizer.add_param_group({"params": list(self.critic.parameters()), "lr": critic_learning_rate})

    def set_terminal_replay(self, observations, actions, *, batch_size=1024, coef=1., seed=7007,
                            trainable_experts=(2,)):
        """Freeze the shared flight representation and anchor selected heads.

        All rows are already-labeled training data, not evaluation examples.
        A private sampling generator avoids changing simulator/rollout RNG.
        """
        experts = tuple(trainable_experts)
        if (not experts or len(set(experts)) != len(experts)
                or any(not isinstance(index, int) or index < 0
                       or index >= len(self.actor.mlp.experts) for index in experts)):
            raise ValueError("trainable_experts must contain unique valid expert indices")
        self.trainable_experts = experts
        self.actor.mlp.requires_grad_(False)
        for index in experts:
            self.actor.mlp.experts[index].requires_grad_(True)
        # Keep parameter-group layout compatible with the ordinary frozen
        # evaluator; Adam skips the parameters whose gradients remain None.
        self.offline_observations = observations.to(self.device)
        self.offline_actions = actions.to(self.device)
        self.offline_bc_batch_size, self.offline_bc_coef = batch_size, coef
        self.offline_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.offline_bc_samples_drawn = 0

    def offline_bc_loss(self):
        idx = torch.randint(len(self.offline_observations), (self.offline_bc_batch_size,),
                            generator=self.offline_generator, device=self.device)
        predicted = self.actor({"policy": self.offline_observations[idx]})
        target = self.offline_actions[idx]
        continuous = [0, 1, 2, 4, 5, 6]
        weights = target.new_tensor([3., 3., 2., 1., 5., 5.])
        mse = ((predicted[:, continuous]-target[:, continuous]).square()*weights).mean()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(predicted[:, 3],
                                                                   (target[:, 3] > 0).float())
        self.offline_bc_samples_drawn += len(idx)
        return mse + .08*bce
