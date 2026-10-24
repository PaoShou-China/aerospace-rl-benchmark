"""Regression for finite higher-order gradients at the landing-pad center.

This numerical test is NOT a proof of contact-model accuracy or safe landing.
"""
import pytest
import torch
from test_upstream_integration import Actor, Seed, physics


@pytest.mark.parametrize('height', [2.55, 2.575, 2.7, 3.0])
def test_default_40_step_tail_has_finite_second_order_training_gradients(height):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    actor=Actor(Seed(),horizon=6,iterations=1,knots=3,tail_steps=40)
    obs=torch.zeros(2,30)
    obs[:,6]=1.; obs[:,12]=height; obs[:,25]=.05
    obs[:,9]=-.1; obs[:,26]=.1
    state=physics.observation_state(obs)
    memory=actor.initial_memory(2,'cpu')
    action=torch.zeros(2,7)
    output=actor(obs,memory,state,action,torch.ones(2,dtype=torch.bool),create_graph=True)
    loss=output['means'].square().mean()+.01*output['logits'].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(output['model_control_cost']).all()
    gradients=[p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
