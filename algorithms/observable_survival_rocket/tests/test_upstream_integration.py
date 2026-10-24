"""Run on the actual checkout with its dependencies; explicitly skip otherwise."""
from pathlib import Path
import importlib.util
import sys
import pytest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[2]/'physical_belief_rocket_landing/core'
if not (SOURCE/'rocket_confident_physics.py').exists():
    pytest.skip('Actual upstream checkout is not mounted; Genesis/actor integration NOT tested', allow_module_level=True)
if importlib.util.find_spec('rsl_rl') is None:
    pytest.skip('rsl_rl dependency absent; actual actor integration NOT tested', allow_module_level=True)

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
prefix='algorithms.physical_belief_rocket_landing.core'
import importlib
parent=importlib.import_module(prefix+'.rocket_confident_physics')
physics=importlib.import_module(prefix+'.rocket_physical_core')
for name in ('observable_belief','rocket_observable_survival'):
    path=Path(__file__).resolve().parents[1]/f'{name}.py'
    spec=importlib.util.spec_from_file_location(prefix+'.'+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
Actor=module.ObservableSurvivalActor

class Seed(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(30,7)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, obs): return self.linear(obs['policy']).tanh()


def test_actual_actor_forward_backward_and_memory_shape():
    torch.set_num_threads(1); torch.manual_seed(1)
    actor=Actor(Seed(),horizon=2,iterations=1,knots=1,tail_steps=1)
    obs=torch.zeros(2,30); obs[:,6]=1.; obs[:,12]=6.; obs[:,25]=.05; obs[:,26]=.4
    state=physics.observation_state(obs)
    memory=actor.initial_memory(2,'cpu')
    action=torch.zeros(2,7); action[:,3]=1.; action[:,4]=-.4
    future=actor.physics(state,action,torch.zeros(2,12)).detach()
    nxt=obs.clone()
    nxt[:,10:13]=future[:,:3]; nxt[:,3:7]=future[:,6:10]
    nxt[:,7:10]=physics.inverse_rotate(future[:,6:10],future[:,3:6])
    nxt[:,:3]=future[:,10:13]; nxt[:,20:24]=future[:,13:17]
    nxt[:,26:29]=future[:,17:20]; nxt[:,25]=future[:,20]
    output=actor(nxt,memory,state,action,torch.ones(2,dtype=torch.bool),create_graph=True)
    assert output['belief']['memory'].shape==(2,actor.memory_width)
    assert torch.isfinite(output['means']).all()
    loss=output['means'].square().mean()+output['logits'].square().mean()*.01
    loss.backward()
    grads=[p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_published_checkpoint_transfer_and_auxiliary_loss():
    checkpoint_path=SOURCE.parent/'artifacts/policy.pt'
    if not checkpoint_path.exists():
        pytest.skip('published weights missing')
    data=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    import json,copy
    phase=importlib.import_module(prefix+'.rocket_phase_policy')
    obs=torch.zeros(2,30); obs[:,6]=1.; obs[:,12]=6.; obs[:,25]=.05
    cfg=json.loads((SOURCE.parent/'artifacts/parent/config.json').read_text())
    params=copy.deepcopy(cfg['actor']); params.pop('class_name')
    std=params.get('distribution_cfg',{}).get('init_std')
    if isinstance(std,list): params['distribution_cfg']['init_std']=torch.tensor(std)
    policy=phase.RocketPhaseModel({'policy':obs},cfg['obs_groups'],'actor',7,**params)
    actor=Actor(policy,horizon=2,iterations=1,knots=1,tail_steps=1)
    copied=parent.initialize_from_published(actor,data['actor'])
    assert copied
    state=physics.observation_state(obs)
    memory=actor.initial_memory(2,'cpu')
    action=torch.zeros(2,7); action[:,3]=1.; action[:,4]=-.4
    output=actor(obs,memory,state,action,torch.ones(2,dtype=torch.bool),create_graph=True)
    loss=actor.prediction_loss(output['belief'],action,obs,torch.ones(2,dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
