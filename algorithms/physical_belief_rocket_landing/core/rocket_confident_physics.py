"""Causal confidence, masked physical inference, and sequence control.

Confidence is an empirical reliability score, not a probability certificate.
Observed prediction innovations are the only source of its online statistics.
The same coordinate updater solves parameters, predicted states, and controls.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .rocket_physical_core import observation_state
from .rocket_phase_policy import features
from .rocket_unified_physics import UnifiedRocketActor, SharedPhysicalUpdate


class MaskedPhysicalUpdate(SharedPhysicalUpdate):
    def __init__(self):
        super().__init__()
        self.types = nn.Embedding(40, 8)

    def forward(self, value, gradient, context, mode):
        width = value.shape[-1]
        coordinate = torch.arange(width, device=value.device)
        if mode == 'control':
            coordinate = 12 + coordinate % 7
        elif mode == 'predict':
            coordinate = 19 + coordinate % 21
        mode_id = {'id': 0., 'predict': .5, 'control': 1.}[mode]
        x = torch.cat((value[..., None], gradient.tanh()[..., None],
                       torch.log1p(gradient.abs())[..., None],
                       value.new_full((*value.shape, 1), mode_id),
                       context[:, None].expand(-1, width, -1),
                       self.types(coordinate)[None].expand(len(value), -1, -1)), -1)
        step = .001 + .059*self.net(x).sigmoid().squeeze(-1)
        return value - step*gradient.tanh()


class ConfidenceSequenceActor(UnifiedRocketActor):
    """Physical coefficients stay factual; task gradients optimize controls."""
    def __init__(self, seed_policy, *, horizon=12, iterations=2, hypotheses=7,
                 knots=3, ablation='full', confidence=True):
        super().__init__(seed_policy, horizon=horizon, iterations=iterations)
        if hypotheses < 3 or hypotheses > 25 or hypotheses % 2 != 1:
            raise ValueError('hypotheses must be odd, between 3 and 25')
        if knots < 1 or knots > horizon:
            raise ValueError('knots must lie between one and horizon')
        if ablation not in ('full', 'fixed_belief', 'no_solver'):
            raise ValueError('unknown physical ablation')
        self.hypotheses, self.knots = hypotheses, knots
        self.ablation, self.use_confidence = ablation, confidence
        self.memory_width = 110 + knots*7
        self.shared_update = MaskedPhysicalUpdate()
        # Orthogonal DCT directions span more than the old single diagonal.
        # At 25 points all 12 coefficient directions are represented.
        axis = torch.arange(12, dtype=torch.float32)+.5
        directions = [torch.zeros(12)]
        for k in range((hypotheses-1)//2):
            direction = torch.cos(math.pi*k*axis/12)
            direction = direction/direction.norm()*math.sqrt(12)
            directions.extend((direction, -direction))
        self.hypothesis_directions = torch.stack(directions)
        self.sequence_head = nn.Sequential(nn.Linear(109, 128), nn.SiLU(),
                                           nn.Linear(128, 2*knots*6))
        nn.init.zeros_(self.sequence_head[-1].weight)
        nn.init.zeros_(self.sequence_head[-1].bias)
        self.noise_head = nn.Linear(64, 21)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.constant_(self.noise_head.bias, -2.)

    def initial_memory(self, count, device):
        memory = torch.zeros(count, self.memory_width, device=device)
        memory[:, 76:88] = -2.3
        memory[:, 88:109] = .25
        return memory

    def solve(self, value, energy, context, mode, mask, project, *, create_graph):
        """Same masked conditional inference loop for all three variable types."""
        if not value.requires_grad:
            value = value.requires_grad_(True)
        with torch.no_grad():
            initial = energy(value.detach())
        accepted = torch.zeros(len(value), device=value.device)
        for _ in range(self.iterations):
            objective = energy(value)
            gradient, = torch.autograd.grad(objective.sum(), value, create_graph=create_graph)
            proposal = self.shared_update(value, gradient, context, mode)
            proposal = project(value+(proposal-value)*mask)
            # Stop gradients through acceptance, not through accepted proposals.
            with torch.no_grad():
                trial = energy(proposal.detach())
                accept = torch.isfinite(trial) & (trial <= objective.detach()+1e-7)
            value = torch.where(accept[:, None], proposal, value)
            accepted = accepted+accept.float()
        return value, initial, energy(value), accepted/max(1, self.iterations)

    def infer_belief(self, obs, memory, previous_state, executed_action, history_valid, *, create_graph=True):
        state = observation_state(obs)
        valid = history_valid[:, None]
        previous = torch.where(valid, previous_state, state)
        evidence = torch.cat((previous/self.physics.encoder_scale, executed_action,
                              (state-previous)/self.physics.error_scale,
                              valid.to(state.dtype)), -1).clamp(-20, 20)
        hidden = self.history_encoder(torch.where(valid, evidence, 0.), memory[:, :64])
        context = self.context(hidden).tanh()
        prior, log_scale = memory[:, 64:76], memory[:, 76:88]
        raw = self.belief_head(hidden)
        mean = (prior+.03*raw[:, :12].tanh()).clamp(-.7, .7)
        log_scale = (log_scale+.02*raw[:, 12:].tanh()).clamp(-4.6, -.7)
        # Evaluate the previous belief BEFORE fitting the new observation.
        with torch.no_grad():
            predicted = self.physics(previous, executed_action, prior)
            innovation = self.physics.residual(predicted, state).square().clamp_max(100)
            ema = torch.where(valid, .95*memory[:, 88:109]+.05*innovation, memory[:, 88:109])
            age = memory[:, 109:110]+valid.float()
            relevant = torch.cat((ema[:, 3:13], ema[:, 17:20]), -1).mean(-1)
            confidence = age[:, 0]/(age[:, 0]+8.)/(1+relevant.sqrt())
        n, h = len(obs), self.hypotheses
        theta = (mean[:, None]+log_scale.exp()[:, None]*self.hypothesis_directions).clamp(-.7, .7)
        previous_h = previous[:, None].expand(-1, h, -1)
        actions_h = executed_action[:, None].expand(-1, h, -1)

        def likelihood(theta_flat):
            theta_h = theta_flat.reshape(n, h, 12)
            pred = self.physics(previous_h, actions_h, theta_h)
            error = self.physics.residual(pred, state[:, None])
            return (error.square().clamp_max(100)/(1+ema[:, None])).mean(-1)

        def energy(theta_flat):
            cost = likelihood(theta_flat)*history_valid[:, None]
            regularizer = .01*((theta_flat.reshape(n, h, 12)-prior[:, None])/log_scale.exp()[:, None]).square().mean(-1)
            return (cost+regularizer).flatten()

        if self.ablation == 'fixed_belief':
            theta = torch.zeros_like(theta)
            accepted = torch.zeros(n*h, device=obs.device)
        else:
            theta, _, _, accepted = self.solve(theta.reshape(-1, 12), energy,
                context[:, None].expand(-1, h, -1).reshape(-1, 16), 'id',
                history_valid[:, None, None].expand(-1, h, 12).reshape(-1, 12),
                lambda x: x.clamp(-.7, .7), create_graph=create_graph)
            theta = theta.reshape(n, h, 12)
        fit = likelihood(theta.reshape(-1, 12))
        weights = (-.5*fit).softmax(-1)
        posterior = (weights[..., None]*theta).sum(1)
        variance = (weights[..., None]*(theta-posterior[:, None]).square()).sum(1)+.0001
        # Poorly predicted transitions cannot collapse the belief uncertainty.
        proposed_scale = .95*log_scale+.025*variance.log()
        posterior_scale = torch.maximum(proposed_scale, log_scale-.01*confidence[:, None]).clamp(-4.6, -.7)
        new_memory = torch.cat((hidden, posterior, posterior_scale, ema, age, memory[:, 110:]), -1)
        return {'theta': theta, 'weights': weights, 'mean': posterior, 'log_scale': posterior_scale,
                'context': context, 'memory': new_memory, 'state': state,
                'fit': (weights*fit).sum(-1), 'confidence': confidence,
                'innovation': relevant, 'id_acceptance': accepted.reshape(n, h).mean(-1),
                'log_noise': self.noise_head(hidden).clamp(-6, 4)}

    def prediction_statistics(self, belief, action):
        h = self.hypotheses
        predicted = self.physics(belief['state'][:, None].expand(-1, h, -1),
                                 action[:, None].expand(-1, h, -1), belief['theta'])
        expectation = (belief['weights'][..., None]*predicted).sum(1)
        spread = ((predicted-expectation[:, None])/self.physics.error_scale).square()
        variance = (belief['weights'][..., None]*spread).sum(1)+belief['log_noise'].exp()+.01
        return expectation, variance

    def prediction_loss(self, belief, action, next_obs, valid):
        expectation, variance = self.prediction_statistics(belief, action)
        target = observation_state(next_obs)
        residual = self.physics.residual(expectation, target)
        mse = residual.square().clamp_max(100).mean(-1)
        nll = .5*(residual.square().clamp_max(100)/variance+variance.log()).mean(-1)
        # Prediction is the same conditional solve with state free and theta/action fixed.
        # A detached initializer makes the shared predictor update learn from evidence.
        normalized = expectation/self.physics.encoder_scale
        target_normalized = normalized
        def energy(value):
            return ((value-target_normalized)/(self.physics.error_scale/self.physics.encoder_scale)).square().mean(-1)
        initial = (belief['state']/self.physics.encoder_scale).detach().requires_grad_(True)
        predicted, _, _, _ = self.solve(initial, energy, belief['context'], 'predict',
            self.physics.error_scale/self.physics.encoder_scale, lambda x:x, create_graph=True)
        prediction_consistency = ((predicted-target/self.physics.encoder_scale).square()).mean(-1)
        loss = mse+.05*nll+.01*prediction_consistency+.001*belief['mean'].square().mean(-1)
        return (loss*valid.float()).sum()/valid.sum().clamp_min(1)

    def sequence_cost(self, flat, state, theta, weights, reference):
        n, h = len(state), self.hypotheses
        actions = flat.reshape(n, 2, self.knots, 7)
        s = state[:, None, None].expand(-1, 2, h, -1).reshape(-1, 21)
        th = theta[:, None].expand(-1, 2, -1, -1).reshape(-1, 12)
        accumulated = s.new_zeros(n*2*h)
        for step in range(self.horizon):
            position = step*(self.knots-1)/max(1, self.horizon-1)
            left = int(position)
            fraction = position-left
            u = actions[:, :, left]*(1-fraction)+actions[:, :, min(left+1, self.knots-1)]*fraction
            u = u[:, :, None].expand(-1, -1, h, -1).reshape(-1, 7)
            s = self.physics(s, u, th)
            height = (s[:, 2]-2.575).clamp_min(0)
            blend = ((height-2)/8).clamp(0, 1)
            blend = blend.square()*(3-2*blend)
            down = (1-blend)*torch.minimum((24*height+1e-6).sqrt(), .8*height)+blend*(24*height+1e-6).sqrt().clamp_max(80)
            xy = (-s[:, :2]/(2+height)[:, None]).clamp(-2, 2)
            velocity = torch.cat((xy, -down[:, None]), -1)
            near = torch.exp(-height/2)
            speed = ((s[:, 3:6]-velocity)/(.2+2*(1-near))[:, None]).square().mean(-1)
            task = speed+20*s[:, 6:8].square().sum(-1)+(.1+near)*s[:, 10:13].square().sum(-1)
            task = task+.02*s[:, :2].square().sum(-1)+10*nn.functional.relu(2.54-s[:, 2]).square()
            accumulated = accumulated+task.clamp_max(1000)/self.horizon
        costs = accumulated.reshape(n, 2, h)
        robust = (1-self.risk_logit.sigmoid())*(weights[:, None]*costs).sum(-1)+self.risk_logit.sigmoid()*costs.max(-1).values
        anchor = 3*(actions-reference).square().mean((-1, -2))
        smooth = .1*(actions[:, :, 1:]-actions[:, :, :-1]).square().mean((-1, -2)) if self.knots>1 else 0.
        return (robust+anchor+smooth).flatten()

    def forward(self, obs, memory, previous_state, executed_action, history_valid, *, create_graph=True):
        with torch.enable_grad():
            belief = self.infer_belief(obs, memory, previous_state, executed_action, history_valid, create_graph=create_graph)
            seed = self.seed_policy({'policy': obs})
            factual = belief['memory'][:, :64].detach()
            if self.ablation == 'fixed_belief':
                factual = torch.zeros_like(factual)
            descriptor = torch.cat((features(obs), factual, belief['mean'].detach()), -1)
            raw = self.action_head(descriptor)
            controls = seed[:, self.continuous].clamp(-1, 1)[:, None]+.1*raw[:, 1:].reshape(-1, 2, 6).tanh()
            sequence = controls[:, :, None]+.1*self.sequence_head(descriptor).reshape(-1, 2, self.knots, 6).tanh()
            bits = obs.new_tensor([-1., 1.])[None, :, None, None].expand(len(obs), -1, self.knots, -1)
            proposal = torch.cat((sequence[..., :3], bits, sequence[..., 3:]), -1).clamp(-1, 1)
            reference = proposal.detach()
            context = belief['context'].detach()[:, None].expand(-1, 2, -1).reshape(-1, 16)
            confidence = belief['confidence'].detach() if self.use_confidence else torch.ones(len(obs), device=obs.device)
            # Previous plans are causal warm starts, with one control tick shifted.
            prior_plan = memory[:, 110:].reshape(-1, self.knots, 7)
            shifted = torch.cat((prior_plan[:, 1:], prior_plan[:, -1:]), 1)
            rate = (self.knots-1)/max(1, self.horizon-1)
            warm = (1-rate)*prior_plan+rate*shifted
            mask = obs.new_tensor([1., 1., 1., 0., 1., 1., 1.]).repeat(self.knots)
            warm_weight = .1*confidence*history_valid
            proposal = proposal+warm_weight[:, None, None, None]*(warm[:, None]-proposal)*mask.reshape(1, 1, self.knots, 7)
            def energy(value):
                return self.sequence_cost(value, belief['state'], belief['theta'].detach(), belief['weights'].detach(), reference)
            if self.ablation == 'no_solver':
                initial = final = obs.new_zeros(len(obs), 2)
                accepted = obs.new_zeros(len(obs), 2)
                optimized = proposal
            else:
                optimized, initial, final, accepted = self.solve(proposal.reshape(-1, self.knots*7), energy,
                    context, 'control', mask, lambda x:x.clamp(-1, 1), create_graph=create_graph)
                optimized = optimized.reshape_as(proposal)
                initial, final, accepted = [x.reshape(-1, 2) for x in (initial, final, accepted)]
            improvement = ((initial-final)/(1+initial.abs())).clamp(0, 1).detach()
            authority = self.physical_mix_logit.sigmoid()*confidence
            authority = authority*(.25+.75*torch.sigmoid((60-(obs[:, 12]-2.575))/20))
            authority = authority[:, None]*(.25+.75*improvement)
            if self.ablation == 'no_solver':
                authority = torch.zeros_like(authority)
            means = controls+authority[..., None]*(optimized[:, :, 0, self.continuous]-controls)
            logits = (12*seed[:, 3]).clamp(-8, 8)+raw[:, 0]+authority.mean(-1)*(final[:, 0]-final[:, 1]).clamp(-4, 4)
            scales = self.log_std.clamp(-7, -1).exp().expand_as(means)
            chosen = (logits.detach()>0).long()
            next_plan = optimized[torch.arange(len(obs), device=obs.device), chosen].detach().flatten(1)
            belief['memory'] = torch.cat((belief['memory'][:, :110], next_plan), -1)
            return {'logits': logits, 'means': means, 'scales': scales, 'belief': belief,
                    'model_control_cost': final.mean(-1), 'confidence': confidence,
                    'solver_improvement': improvement.mean(-1), 'control_acceptance': accepted.mean(-1),
                    'physical_action_delta': (means-controls).norm(dim=-1).mean(-1)}


def build_physical_actor(seed_policy, config):
    if config.get('architecture', 'published') == 'published':
        return UnifiedRocketActor(seed_policy, horizon=config['horizon'], iterations=config['iterations'])
    return ConfidenceSequenceActor(seed_policy, horizon=config['horizon'], iterations=config['iterations'],
        hypotheses=config.get('hypotheses', 7), knots=config.get('knots', 3),
        ablation=config.get('ablation', 'full'), confidence=not config.get('disable_confidence', False))


def initialize_from_published(actor, state):
    """Explicit weight transfer; a new optimizer is used after architecture changes."""
    current = actor.state_dict()
    copied = []
    for name, value in state.items():
        if name == 'hypothesis_directions':
            continue
        if name == 'shared_update.types.weight':
            current[name][:len(value)] = value
        elif name in current and current[name].shape == value.shape:
            current[name] = value
        else:
            raise ValueError(f'unexpected source checkpoint key/shape: {name}')
        copied.append(name)
    actor.load_state_dict(current, strict=True)
    return copied
