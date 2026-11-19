import torch
import pytest
from dataclasses import replace
from phyai.touchdown_intent import IntentHead, IntentBatch
from phyai.touchdown_decoder import TouchdownDecoder
from phyai.execution_assessment import ActionPacket, consume


def intent(snapshot, raw=None):
    if raw is None:
        raw = torch.zeros(1, 25, dtype=snapshot.state.dtype, requires_grad=True)
    return IntentBatch(raw, snapshot.fingerprint)


def test_full_joint_log_probability_includes_future_nodes_and_arrival(snapshot):
    torch.manual_seed(11)
    head = IntentHead(candidates=4).double()
    distribution = head.distribution(torch.zeros(30, dtype=torch.float64), torch.zeros(7, dtype=torch.float64), snapshot)
    z = distribution.base_dist.loc.detach().clone()
    reference = distribution.log_prob(z)
    for k in range(100):
        changed = z.clone(); changed[k] += .1
        assert distribution.log_prob(changed) < reference


def test_covariance_cross_terms_reach_neural_proposals(snapshot):
    from phyai.belief_contract import BeliefSnapshot
    torch.manual_seed(12)
    p = snapshot.covariance; p[0, 23] = p[23, 0] = .001
    b = BeliefSnapshot(snapshot.state, snapshot.theta, p, version=8, clock=3, modes=snapshot.modes,
                       model_id=snapshot.model_id, history_id=snapshot.history_id)
    h = IntentHead(candidates=4).double()
    o, t = torch.zeros(30, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)
    assert not torch.equal(h(o, t, snapshot).raw, h(o, t, b).raw)


def test_future_intent_nodes_change_current_decoded_action(snapshot):
    decoder = TouchdownDecoder()
    raw = torch.zeros(1, 25, dtype=torch.float64)
    initial = decoder.decode(snapshot, intent(snapshot, raw))
    for node in range(4):
        changed = raw.clone(); changed[0, node * 6] = .15
        result = decoder.decode(snapshot, intent(snapshot, changed))
        assert (result.controls[:, 0] - initial.controls[:, 0]).abs().max() > 1e-7
    assert initial.controls.shape == (1, 4, 7)
    assert initial.controls.abs().max() <= 1
    assert initial.snapshot_id == snapshot.fingerprint


def test_terminal_loss_reaches_neural_head_but_not_factual_inputs(state):
    from phyai.belief_contract import BeliefSnapshot
    theta = torch.zeros(67, dtype=torch.float64, requires_grad=True)
    cov = (torch.eye(87, dtype=torch.float64) * .0025).requires_grad_()
    snapshot = BeliefSnapshot(state, theta, cov, version=1, clock=0, modes=("flight",), model_id="m", history_id="h")
    torch.manual_seed(3)
    head = IntentHead(candidates=1).double()
    output = head(torch.zeros(30, dtype=torch.float64), torch.zeros(7, dtype=torch.float64), snapshot)
    result = TouchdownDecoder().decode(snapshot, output)
    # A terminal-state loss has no direct intent-matching term: it must pass
    # through decoded actuator commands and the shared physical rollout.
    terminal_loss = result.states[:,-1,:6].square().sum() + result.states[:,-1,10:13].square().sum()
    terminal_loss.backward()
    assert head.mean.weight.grad is not None and torch.isfinite(head.mean.weight.grad).all()
    assert head.mean.weight.grad.abs().sum() > 1e-8
    assert theta.grad is None and cov.grad is None


def test_snapshot_mismatch_rejected_before_decoding(snapshot):
    with pytest.raises(ValueError, match="snapshot"):
        TouchdownDecoder().decode(snapshot, IntentBatch(torch.zeros(1,25,dtype=torch.float64), "other"))


def test_execution_checks_actual_command_version_mode_and_expiry(snapshot):
    u = torch.zeros(7, dtype=torch.float64)
    p = ActionPacket(snapshot.fingerprint, snapshot.version, snapshot.modes, snapshot.clock + .025, tuple(u.tolist()), "empirical_support")
    assert torch.equal(consume(p, snapshot, u, snapshot.clock), u)
    with pytest.raises(ValueError, match="expired"):
        consume(p, snapshot, u, snapshot.clock + .025)
    changed = u.clone(); changed[4] = .1
    with pytest.raises(ValueError, match="command"):
        consume(p, snapshot, changed, snapshot.clock)
    with pytest.raises(ValueError, match="version"):
        consume(replace(p, version=6), snapshot, u, snapshot.clock)


def test_integrated_candidate_selection_keeps_actual_decoded_command(snapshot):
    from phyai.pipeline import ResearchController
    torch.manual_seed(4)
    controller = ResearchController(IntentHead(candidates=2).double(), TouchdownDecoder())
    d = controller.propose(torch.zeros(30,dtype=torch.float64), torch.tensor([0,0,2.575,0,0,0,20],dtype=torch.float64), snapshot)
    assert len(d.assessments) == 2
    assert torch.equal(d.command, d.plan.controls[d.selected,0].detach())
    assert all(a['belief_dimension'] == 87 and a['conditional_status'] == 'insufficient_evidence' for a in d.assessments)
    assert all(a['projection_rank'] <= 6 and a['projection_rank'] >= 1 for a in d.assessments)
    assert d.packet.snapshot_id == snapshot.fingerprint
    if d.packet.status == 'empirical_support':
        assert torch.equal(consume(d.packet,snapshot,d.command,snapshot.clock),d.command)
    else:
        with pytest.raises(ValueError, match='assessment'):
            consume(d.packet,snapshot,d.command,snapshot.clock)


def test_command_binding_cannot_truncate_through_integer_or_lower_precision(snapshot):
    p = ActionPacket(snapshot.fingerprint,snapshot.version,snapshot.modes,snapshot.clock+.025,
                     (.4,)*7,'empirical_support')
    with pytest.raises(ValueError,match='command'):
        consume(p,snapshot,torch.zeros(7,dtype=torch.int64),snapshot.clock)
    with pytest.raises(ValueError,match='command'):
        consume(p,snapshot,torch.full((7,),.4,dtype=torch.float32),snapshot.clock)
    assert torch.equal(consume(p,snapshot,torch.full((7,),.4,dtype=torch.float64),snapshot.clock),
                       torch.full((7,),.4,dtype=torch.float64))


def test_selection_prefers_supported_action_over_cheaper_rejected_one():
    from phyai.pipeline import select_candidate
    rows = [dict(status='empirical_support',score=1.),dict(status='insufficient_evidence',score=.001)]
    assert select_candidate(rows) == 0
    rows[0]['status'] = 'insufficient_evidence'
    assert select_candidate(rows) == 1    # diagnostic choice, still not executable


def test_controller_rejects_goals_and_modes_outside_frozen_prefix_scope(snapshot):
    from phyai.pipeline import ResearchController
    from phyai.belief_contract import BeliefSnapshot
    controller = ResearchController(IntentHead(candidates=1).double(), TouchdownDecoder())
    with pytest.raises(ValueError,match='target'):
        controller.propose(torch.zeros(30,dtype=torch.float64),torch.tensor([10,0,2.575,0,0,0,20],dtype=torch.float64),snapshot)
    support = BeliefSnapshot(snapshot.state,snapshot.theta,snapshot.covariance,version=8,clock=3,
                            modes=('support',),model_id='m',history_id='h')
    with pytest.raises(ValueError,match='flight'):
        controller.propose(torch.zeros(30,dtype=torch.float64),torch.tensor([0,0,2.575,0,0,0,20],dtype=torch.float64),support)


def test_invalid_factual_evidence_cannot_become_a_consumable_packet(snapshot):
    from phyai.pipeline import ResearchController
    from phyai.belief_contract import BeliefSnapshot
    b=BeliefSnapshot(snapshot.state,snapshot.theta,snapshot.covariance,version=8,clock=3,modes=('flight',),
                     model_id='m',history_id='h',evidence={'assumption_invalid':True})
    controller=ResearchController(IntentHead(candidates=1).double(),TouchdownDecoder())
    with pytest.raises(ValueError,match='factual'):
        controller.propose(torch.zeros(30,dtype=torch.float64),torch.tensor([0,0,2.575,0,0,0,20],dtype=torch.float64),b)
    packet=ActionPacket(b.fingerprint,b.version,b.modes,b.clock+.025,(0.,)*7,'empirical_support')
    with pytest.raises(ValueError,match='factual'):
        consume(packet,b,torch.zeros(7,dtype=torch.float64),b.clock)


def test_same_candidate_physics_does_not_change_with_batch_companion(state):
    from phyai.touchdown_decoder import interval
    from frozen.core.phi_rocket_physics import PhiPhysics
    physics=PhiPhysics().double()
    u=torch.tensor([[.1,.2,.1,1.,-.4,.2,-.1]],dtype=torch.float64)
    c=torch.zeros(1,67,dtype=torch.float64)
    alone=interval(physics,state[None],u,c,torch.tensor([.1],dtype=torch.float64))
    batched=interval(physics,state[None].expand(2,-1),u.expand(2,-1),c.expand(2,-1),
                     torch.tensor([.1,.3],dtype=torch.float64))
    assert torch.allclose(alone[0],batched[0],atol=1e-12,rtol=1e-12)
