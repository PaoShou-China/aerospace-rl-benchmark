"""Observable-belief / survival-tail extension of the repository's state actor.

Research prototype, NOT a VLA model or a flight-certified controller. This file
is copied into an isolated snapshot's common package by prepare_experiment.py.
Known physics is reused, not replaced. Neither benchmark dynamics nor its
original success predicate is changed. Tail costs are heuristics, not a proof
of safe landing. Use matched ablations before claiming performance gains.
"""
from __future__ import annotations
import torch
from torch import nn
from .rocket_confident_physics import ConfidenceSequenceActor
from .rocket_physical_core import observation_state
from .rocket_phase_policy import features
from .observable_belief import joseph_update, cubature_points, batch_jacobian


class ObservableSurvivalActor(ConfidenceSequenceActor):
    """Causal full-covariance evidence + engine-off continuation cost.

    12 effective coefficients are retained for checkpoint compatibility.
    Airborne evidence estimates flight coefficients only. Contact coefficients
    are NOT identified in this prototype and retain their prior uncertainty.
    """
    def __init__(self, seed_policy, *, horizon=8, iterations=2, hypotheses=25,
                 knots=3, ablation='full', confidence=True, tail_steps=40,
                 tail_weight=.15, observable_update=True):
        if hypotheses != 25:
            raise ValueError('full 12-dimensional cubature requires 25 hypotheses')
        if tail_steps < 0 or tail_weight < 0:
            raise ValueError('nonnegative tail settings required')
        super().__init__(seed_policy, horizon=horizon, iterations=iterations,
                         hypotheses=hypotheses, knots=knots, ablation=ablation,
                         confidence=confidence)
        self.base_width = 110 + knots * 7
        self.memory_width = self.base_width + 144
        self.tail_steps, self.tail_weight = int(tail_steps), float(tail_weight)
        self.observable_update = bool(observable_update)
        self.register_buffer('measurement_indices', torch.tensor([3,4,5,10,11,12,17,18,19]))

    def initial_memory(self, count, device):
        memory = super().initial_memory(count, device)
        covariance = torch.eye(12, device=device)[None].expand(count, -1, -1) * .01
        memory[:, self.base_width:] = covariance.flatten(1)
        return memory

    def infer_belief(self, obs, memory, previous_state, executed_action,
                     history_valid, *, create_graph=True):
        state = observation_state(obs)
        valid = history_valid.bool()
        previous = torch.where(valid[:, None], previous_state, state).detach()
        applied = executed_action.detach()
        prior = memory[:, 64:76].detach()
        covariance = memory[:, self.base_width:].reshape(-1, 12, 12).detach()
        evidence = torch.cat((previous/self.physics.encoder_scale, applied,
                              (state-previous)/self.physics.error_scale,
                              valid[:, None].to(state.dtype)), -1).clamp(-20, 20)
        new_hidden = self.history_encoder(torch.where(valid[:, None], evidence, 0.), memory[:, :64])
        hidden = torch.where(valid[:, None], new_hidden, memory[:, :64])
        context = self.context(hidden).tanh()
        # This is pre-update, out-of-sample prediction error, never fitted error.
        with torch.enable_grad():
            parameter = prior.clone().requires_grad_(True)
            prediction = self.physics(previous, applied, parameter)
            measurement = prediction[:, self.measurement_indices] / self.physics.error_scale[self.measurement_indices]
            jacobian = batch_jacobian(measurement, parameter).detach()
        innovation_full = self.physics.residual(prediction.detach(), state.detach())
        innovation = -innovation_full[:, self.measurement_indices]
        # State estimation and contact impulses are not silently called changes
        # in mass or thrust. Reject reset transitions and both sides of contact.
        flight = (valid & (previous[:, 2] > 3.2) & (state[:, 2] > 3.2)
                  & (obs[:, -1] < .5))
        if self.ablation == 'fixed_belief' or not self.observable_update:
            flight = torch.zeros_like(flight)
        jacobian[:, :, 9:] = 0.  # no airborne evidence about contact coefficients
        # Noise is predicted from PAST context, before seeing the new outcome.
        prior_noise = self.noise_head(memory[:, :64]).detach().clamp(-6, 4).exp()
        variance = prior_noise[:, self.measurement_indices] + .05
        update = joseph_update(prior, covariance, innovation, jacobian, variance, flight)
        mean = update.mean.clamp(-.65, .65).detach()
        covariance = update.covariance.detach()
        # Projection onto bounds can make Gaussian intervals inaccurate; expose
        # uncertainty as an approximate belief, not a confidence certificate.
        with torch.no_grad():
            squared = innovation_full.square().clamp_max(100)
            ema = torch.where(valid[:, None], .95*memory[:, 88:109]+.05*squared,
                              memory[:, 88:109])
            age = memory[:, 109:110] + valid[:, None].to(state.dtype)
            relevant = ema[:, self.measurement_indices].mean(-1)
            reliability = age[:, 0]/(age[:, 0]+8.)/(1+relevant.sqrt())
            theta, weights = cubature_points(mean, covariance)
            log_scale = .5 * covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log()
        new_memory = torch.cat((hidden, mean, log_scale, ema, age,
                                memory[:, 110:self.base_width], covariance.flatten(1)), -1)
        return {'theta':theta, 'weights':weights, 'mean':mean, 'log_scale':log_scale,
                'context':context, 'memory':new_memory, 'state':state,
                'fit':squared.mean(-1), 'confidence':reliability,
                'innovation':relevant, 'id_acceptance':update.accepted.float(),
                'log_noise':self.noise_head(hidden).clamp(-6, 4),
                'observable_rank':update.observable_rank,
                'innovation_nis':update.innovation_nis, 'covariance':covariance}

    def sequence_cost(self, flat, state, theta, weights, reference):
        base = super().sequence_cost(flat, state, theta, weights, reference)
        if not self.tail_steps or not self.tail_weight:
            return base
        n, h = len(state), self.hypotheses
        actions = flat.reshape(n, 2, self.knots, 7)
        s = state[:, None, None].expand(-1, 2, h, -1).reshape(-1, 21)
        th = theta[:, None].expand(-1, 2, -1, -1).reshape(-1, 12)
        for step in range(self.horizon):
            position = step*(self.knots-1)/max(1, self.horizon-1)
            left, fraction = int(position), position-int(position)
            a = actions[:, :, left]*(1-fraction) + actions[:, :, min(left+1,self.knots-1)]*fraction
            s = self.physics(s, a[:, :, None].expand(-1,-1,h,-1).reshape(-1,7), th)
        # A shared engine-off continuation in every uncertainty hypothesis.
        # This does not assume the controller already knows which world is real.
        cutoff = s.new_zeros((len(s), 7))
        cutoff[:, 3:5] = -1.
        tail = s.new_zeros(len(s))
        for _ in range(self.tail_steps):
            s = self.physics(s, cutoff, th)
            velocity = s[:, 3:6].square().sum(-1)
            angular = s[:, 10:13].square().sum(-1)
            tilt = s[:, 6:8].square().sum(-1)
            height = (s[:, 2]-2.575).square()
            off_pad = nn.functional.relu(s[:, :2].norm(dim=-1)-1.4).square()
            penalty = velocity/.09 + angular/.01 + 80*tilt + height/.04 + 10*off_pad
            tail = tail + penalty.clamp_max(1000)/self.tail_steps
        tail = tail.reshape(n, 2, h)
        risk = self.risk_logit.sigmoid()
        robust = (1-risk)*(weights[:, None]*tail).sum(-1) + risk*tail.max(-1).values
        # Explicit task prior: settling-tail relevance increases near the pad.
        # It must be ablated, and does not prove anything about an unseen surface.
        near = torch.sigmoid((2.0-(state[:,2]-2.575))/.4)
        return base + self.tail_weight*(near[:,None]*robust).flatten()

    def forward(self, obs, memory, previous_state, executed_action, history_valid, *, create_graph=True):
        # The distribution logic is the same as ConfidenceSequenceActor. Only
        # factual inference, memory layout and the sequence cost differ.
        with torch.enable_grad():
            belief = self.infer_belief(obs, memory, previous_state, executed_action,
                                       history_valid, create_graph=create_graph)
            seed = self.seed_policy({'policy':obs})
            factual = belief['memory'][:,:64].detach()
            if self.ablation == 'fixed_belief':
                factual = torch.zeros_like(factual)
            descriptor = torch.cat((features(obs), factual, belief['mean'].detach()), -1)
            raw = self.action_head(descriptor)
            controls = seed[:,self.continuous].clamp(-1,1)[:,None] + .1*raw[:,1:].reshape(-1,2,6).tanh()
            sequence = controls[:,:,None] + .1*self.sequence_head(descriptor).reshape(-1,2,self.knots,6).tanh()
            bits = obs.new_tensor([-1.,1.])[None,:,None,None].expand(len(obs),-1,self.knots,-1)
            proposal = torch.cat((sequence[...,:3],bits,sequence[...,3:]),-1).clamp(-1,1)
            reference = proposal.detach()
            context = belief['context'].detach()[:,None].expand(-1,2,-1).reshape(-1,16)
            confidence = belief['confidence'].detach() if self.use_confidence else obs.new_ones(len(obs))
            prior_plan = memory[:,110:self.base_width].reshape(-1,self.knots,7)
            shifted = torch.cat((prior_plan[:,1:],prior_plan[:,-1:]),1)
            rate = (self.knots-1)/max(1,self.horizon-1)
            warm = (1-rate)*prior_plan + rate*shifted
            mask = obs.new_tensor([1.,1.,1.,0.,1.,1.,1.]).repeat(self.knots)
            warm_weight = .1*confidence*history_valid
            proposal = proposal + warm_weight[:,None,None,None]*(warm[:,None]-proposal)*mask.reshape(1,1,self.knots,7)
            def energy(value):
                return self.sequence_cost(value,belief['state'],belief['theta'].detach(),
                                          belief['weights'].detach(),reference)
            if self.ablation == 'no_solver':
                initial = final = obs.new_zeros(len(obs),2)
                accepted = obs.new_zeros(len(obs),2)
                optimized = proposal
            else:
                optimized,initial,final,accepted = self.solve(proposal.reshape(-1,self.knots*7),energy,
                    context,'control',mask,lambda x:x.clamp(-1,1),create_graph=create_graph)
                optimized = optimized.reshape_as(proposal)
                initial,final,accepted = [x.reshape(-1,2) for x in (initial,final,accepted)]
            improvement = ((initial-final)/(1+initial.abs())).clamp(0,1).detach()
            authority = self.physical_mix_logit.sigmoid()*confidence
            authority = authority*(.25+.75*torch.sigmoid((60-(obs[:,12]-2.575))/20))
            authority = authority[:,None]*(.25+.75*improvement)
            if self.ablation == 'no_solver':
                authority = torch.zeros_like(authority)
            means = controls+authority[...,None]*(optimized[:,:,0,self.continuous]-controls)
            logits = (12*seed[:,3]).clamp(-8,8)+raw[:,0]+authority.mean(-1)*(final[:,0]-final[:,1]).clamp(-4,4)
            scales = self.log_std.clamp(-7,-1).exp().expand_as(means)
            chosen = (logits.detach()>0).long()
            next_plan = optimized[torch.arange(len(obs),device=obs.device),chosen].detach().flatten(1)
            belief['memory'] = torch.cat((belief['memory'][:,:110],next_plan,
                                         belief['covariance'].flatten(1)), -1)
            return {'logits':logits,'means':means,'scales':scales,'belief':belief,
                    'model_control_cost':final.mean(-1),'confidence':confidence,
                    'solver_improvement':improvement.mean(-1),'control_acceptance':accepted.mean(-1),
                    'physical_action_delta':(means-controls).norm(dim=-1).mean(-1)}
