import torch
from phyai.offline_audit import event_windows, truth_coordinates
from frozen.core.phi_rocket_physics import PhiPhysics


def test_contact_selector_uses_terminal_next_obs_without_reset_contamination():
    obs = torch.zeros(5, 1, 30)
    nxt = torch.zeros_like(obs)
    nxt[2, 0, 29] = 1             # terminal contact, captured before automatic reset
    obs[3:, 0, 29] = 1           # next episode is excluded
    data = dict(obs=obs, next_obs=nxt, first_episode=torch.tensor([[1],[1],[1],[0],[0]], dtype=torch.bool),
                done=torch.tensor([[0],[0],[1],[0],[0]],dtype=torch.bool))
    windows, events = event_windows(data, before=2, after=2)
    assert windows["pre_contact"] == [(0,0),(1,0)]
    assert windows["contact_crossing"] == [(2,0)]
    assert windows["post_contact"] == []
    assert events == [{"episode":0,"contact_transition":2,"pre_contact_observation":2}]


def test_truth_mapping_records_unrepresentable_actuator_lag():
    physical = PhiPhysics().double()
    tp = dict(thrust=[1.],fin=[1.],booster_tau=[.03],gimbal_tau=[.03],
              gimbal_bias=[[0.,0.]],wind=[[0.,0.,0.]])
    theta, clipping = truth_coordinates(physical, torch.zeros(67,dtype=torch.float64), tp)
    assert theta.shape == (1,67)
    assert clipping["engine_lag_gain"] == 1
    assert clipping["gimbal_lag_gain"] == 1
    assert .01*physical.values(theta)[0,7] < .021
