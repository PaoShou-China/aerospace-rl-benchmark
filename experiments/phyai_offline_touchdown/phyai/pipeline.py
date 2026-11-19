"""Offline candidate selection; returns a checked prefix, never calls an environment."""
from dataclasses import dataclass
import math
import torch
from frozen.core.phi_rocket_physics import state_plus
from frozen.core.phi_rocket_types import STATE_SCALE
from .action_uncertainty import physical_sensitivity, local_projection
from .touchdown_decoder import rollout, prefix_margins
from .execution_assessment import ActionPacket


@dataclass
class Decision:
    plan: object
    assessments: list
    selected: int
    command: torch.Tensor
    packet: ActionPacket


def select_candidate(assessments):
    finite = [i for i,a in enumerate(assessments) if math.isfinite(a['score'])]
    if not finite:
        raise ValueError('no finite candidate assessment')
    supported = [i for i in finite if assessments[i]['status'] in ('empirical_support','conditional_pass')]
    return min(supported or finite,key=lambda i:assessments[i]['score'])


class ResearchController:
    def __init__(self, head, decoder, *, tail_tolerance=.05, max_rank=6):
        self.head, self.decoder = head, decoder
        if not 1 <= max_rank <= 6 or tail_tolerance < 0:
            raise ValueError("invalid local projection budget")
        self.tail_tolerance, self.max_rank = tail_tolerance, max_rank

    def propose(self, observation, task, snapshot):
        if not snapshot.factual_valid:
            raise ValueError('invalid or unavailable factual assumptions')
        expected_target = task.new_tensor([0.,0.,2.575,0.,0.,0.])
        if task.shape != (7,) or not torch.isfinite(task).all() or not torch.equal(task[:6],expected_target) or task[6]<=0:
            raise ValueError('prototype target is the frozen benchmark pad and zero velocity')
        if snapshot.modes != ('flight',):
            raise ValueError('prefix prototype requires an unambiguous flight snapshot')
        intent = self.head(observation, task, snapshot)
        plan = self.decoder.decode(snapshot, intent)
        assessments, scores = [], []
        # Decoding has used estimated means only. The exact same resulting
        # sequence is now propagated through all uncertain worlds.
        with torch.no_grad():
            for k in range(len(plan.controls)):
                J = physical_sensitivity(self.decoder.physics, snapshot, plan.controls[k], plan.arrival[k])
                for rank in range(1, self.max_rank+1):
                    proj = local_projection(J, snapshot.covariance, rank)
                    if float(proj.tail_norm.max()) <= self.tail_tolerance:
                        break
                directions = torch.cat((torch.zeros(1,87,dtype=J.dtype,device=J.device), proj.basis.T, -proj.basis.T))
                delta = directions @ proj.factor.T
                x = state_plus(snapshot.state.expand(len(delta),-1), delta[:,:20]*J.new_tensor(STATE_SCALE))
                theta = snapshot.theta[None] + delta[:,20:]
                controls = plan.controls[k].detach()[None].expand(len(delta),-1,-1)
                states = rollout(self.decoder.physics, x, controls, theta, plan.arrival[k].detach().expand(len(delta)))
                empirical_min = prefix_margins(states).amin(0)
                # Tail penalty makes the omitted local sensitivity affect the
                # decision. Higher-order/model errors remain explicitly unknown.
                local_lower = empirical_min - proj.tail_norm
                score = plan.cost[k].detach() + 10*torch.relu(-local_lower).square().sum()
                valid = bool(torch.isfinite(states).all() & torch.isfinite(local_lower).all())
                status = 'empirical_support' if valid and bool((local_lower>=0).all()) else 'insufficient_evidence'
                assessments.append(dict(candidate=k, snapshot_id=snapshot.fingerprint, belief_dimension=87,
                                        projection_rank=rank, scenario_count=len(delta),
                                        singular_values=proj.singular_values.tolist(),
                                        empirical_minimum=empirical_min.tolist(), omitted_linear_norm=proj.tail_norm.tolist(),
                                        local_penalized_minimum=local_lower.tolist(), status=status,
                                        conditional_status='insufficient_evidence',
                                        hessian_region_bound=None, model_bound=None, numerical_bound=None,
                                        mode_coverage_qualified=False, projected_minimum_qualified=False,
                                        scope='fixed-command prefix only; no 2-second support qualification', score=float(score)))
                scores.append(score if valid else score.new_tensor(float('inf')))
        if not torch.isfinite(torch.stack(scores)).any():
            raise ValueError('no finite candidate assessment')
        selected = select_candidate(assessments)
        command = plan.controls[selected,0].detach().clone()
        packet = ActionPacket(snapshot.fingerprint,snapshot.version,snapshot.modes,snapshot.clock+.025,
                              tuple(command.tolist()),assessments[selected]['status'])
        return Decision(plan,assessments,selected,command,packet)
