"""Causal structure and local timing checks; zero new environment interactions."""
from pathlib import Path
import json
import hashlib
import platform
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phyai.belief_contract import BeliefSnapshot
from phyai.touchdown_intent import IntentHead, IntentBatch
from phyai.touchdown_decoder import TouchdownDecoder
from phyai.pipeline import ResearchController
from frozen.core.phi_rocket_types import PARAMETERS
from frozen.core.rocket_physical_core import observation_state


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    source = Path(sys.argv[1])
    data = torch.load(source,map_location='cpu',weights_only=False)
    eligible = data['first_episode'][:,0] & (data['obs'][:,0,12] < 5.) & (data['obs'][:,0,29] < .5)
    row = int(eligible.nonzero()[0])
    obs = data['obs'][row,0].double()
    x = observation_state(obs)
    theta = torch.tensor(json.loads((ROOT/'frozen/candidate/reference_theta.json').read_text())['reference_theta'],dtype=torch.float64,requires_grad=True)
    cov = (torch.eye(87,dtype=torch.float64)*.0225).requires_grad_()
    b = BeliefSnapshot(x,theta,cov,version=1,clock=row/40,modes=('flight',),model_id='SOURCE_MANIFEST.json',history_id=f'{source.name}:env0:{row}')
    raw = torch.zeros(1,25,dtype=torch.float64,requires_grad=True)
    decoder = TouchdownDecoder()
    initial = decoder.decode(b,IntentBatch(raw,b.fingerprint))
    effect_first, effect_all = [], []
    for j in range(25):
        perturbed = raw.detach().clone(); perturbed[0,j] += .1
        result = decoder.decode(b,IntentBatch(perturbed,b.fingerprint))
        effect_first.append(float((result.controls[:,0]-initial.controls[:,0]).detach().norm()))
        effect_all.append(float((result.controls-initial.controls).detach().norm()))
    grad = torch.autograd.grad(initial.cost.sum(),raw)[0]
    head = IntentHead(candidates=4).double()
    task = obs.new_tensor([0.,0.,2.575,0.,0.,0.,30-row/40])
    proposal = head(obs,task,b)
    decoded = decoder.decode(b,proposal)
    decoded.cost.mean().backward()
    parameter_grad = float(sum(p.grad.abs().sum() for p in head.parameters() if p.grad is not None))
    changed_theta = b.theta; changed_theta[32] += .1
    b2 = BeliefSnapshot(x,changed_theta,b.covariance,version=2,clock=b.clock,modes=b.modes,model_id=b.model_id,history_id=b.history_id)
    physical_change = decoder.decode(b2,IntentBatch(raw.detach(),b2.fingerprint))
    controller = ResearchController(head,decoder)
    timings=[]
    for _ in range(8):
        start=time.perf_counter()
        decision=controller.propose(obs,task,b)
        timings.append(time.perf_counter()-start)
    report = dict(evidence_class='offline structure and model computations, no performance evaluation',
                  new_real_interactions=0, new_training_updates=0,
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  state_index=dict(episode=0,step=row),
                  belief_origin='frozen calibrated reference and explicit diagonal structural-test prior; NOT a reconstructed historical posterior',
                  physical_parameters=len(PARAMETERS), disturbance_dimensions=6, joint_covariance_shape=list(b.covariance.shape),
                  raw_intent_dimensions=25, tested_first_action_dimensions=sum(e>1e-9 for e in effect_first),
                  tested_plan_dimensions=sum(e>1e-9 for e in effect_all),
                  per_dimension_first_action_change=effect_first, per_dimension_plan_change=effect_all,
                  local_surrogate_gradient_norm_by_dimension=grad[0].abs().tolist(),
                  neural_head_gradient_l1=parameter_grad, factual_theta_grad_is_none=theta.grad is None,
                  factual_covariance_grad_is_none=cov.grad is None,
                  changed_belief_same_intent_first_action_change=float((physical_change.controls[:,0]-initial.controls[:,0]).detach().norm()),
                  selected_candidate=decision.selected, assessment=decision.assessments,
                  local_cpu_timing=dict(samples=len(timings),seconds=timings,median=float(torch.tensor(timings).median()),
                                        p95=float(torch.quantile(torch.tensor(timings),.95)),
                                        note='8 local CPU samples on one retained state; no real-time/GPU claim'),
                  runtime=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),threads=1),
                  gates=dict(G0='contact/angular model qualification unresolved',G1='offline typed neural-to-physical prefix implemented',
                             G2='numerical local projection only; no qualified hybrid remainder',G3='not trained or compared',G4='not started'))
    (ROOT/'reports/structure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['tested_first_action_dimensions','tested_plan_dimensions','neural_head_gradient_l1','changed_belief_same_intent_first_action_change','local_cpu_timing']}))


if __name__=='__main__':main()
