import torch
import pytest
from phyai.belief_contract import BeliefSnapshot, TransitionRecord, TransitionLedger


def test_snapshot_is_detached_and_cannot_alias_live_filter(state):
    theta = torch.zeros(67, dtype=state.dtype, requires_grad=True)
    cov = torch.eye(87, dtype=state.dtype)
    cov[0, 23] = cov[23, 0] = .2
    b = BeliefSnapshot(state, theta, cov, version=1, clock=0, modes=("flight",),
                       model_id="frozen", history_id="h0")
    state[2] = 100
    cov[0, 23] = 0
    exposed = b.theta
    exposed[0] = 1
    assert b.state[2] == 4
    assert b.theta[0] == 0
    assert b.covariance[0, 23] == .2
    assert not b.features().requires_grad
    assert b.covariance.shape == (87, 87)


def test_snapshot_rejects_wrong_layout_and_indefinite_covariance(state):
    kw = dict(version=1, clock=0, modes=("flight",), model_id="m", history_id="h")
    with pytest.raises(ValueError, match="67"):
        BeliefSnapshot(state, torch.zeros(61), torch.eye(87), **kw)
    cov = torch.eye(87, dtype=state.dtype)
    cov[0, 1] = cov[1, 0] = 2
    with pytest.raises(ValueError, match="positive semidefinite"):
        BeliefSnapshot(state, torch.zeros(67, dtype=state.dtype), cov, **kw)


def test_transition_preserves_executed_action_and_deduplicates():
    proposed, sent, applied = (torch.ones(7) * v for v in (.3, .2, .1))
    record = TransitionRecord("ep1:3", 7, 0., .025, proposed, sent, applied,
                              torch.arange(7, dtype=torch.float32), torch.zeros(30), torch.zeros(30))
    ledger = TransitionLedger()
    ledger.accept(record)
    assert torch.equal(ledger.records[0].applied, applied)
    assert not torch.equal(record.proposed, record.applied)
    applied[0] = 9
    assert record.applied[0] == .1
    with pytest.raises(ValueError, match="duplicate"):
        ledger.accept(record)
    assert ledger.real_transition_count == 1


def test_filter_adapter_keeps_full_covariance_and_explicit_evidence(snapshot):
    b = dict(state=snapshot.state[None], mean=snapshot.theta[:61][None],
             disturbance=snapshot.theta[61:][None], covariance=snapshot.covariance[None],
             lower=-torch.ones(1, 87), upper=torch.ones(1, 87),
             evidence_matrix=torch.zeros(1, 20, 87), evidence_radius=torch.ones(1, 20),
             evidence_center=torch.zeros(1, 87))
    s = BeliefSnapshot.from_filter(b, index=0, version=8, clock=3.025, modes=("flight",),
                                  model_id="m", history_id="ep1:121")
    assert s.theta.shape == (67,)
    assert s.evidence["qualified"] is False
    assert len(s.evidence["lower"]) == 87
    b["mean"].fill_(1)
    assert not s.theta[:61].any()


def test_filter_adapter_preserves_actual_evidence_and_derives_modes(snapshot):
    b = dict(state=snapshot.state[None],mean=snapshot.theta[:61][None],disturbance=snapshot.theta[61:][None],
             covariance=snapshot.covariance[None],evidence_window=torch.zeros(1,4,50),
             evidence_valid=torch.ones(1,4),assumption_invalid=torch.tensor([False]),
             numerical_valid=torch.tensor([True]),history_valid=torch.tensor([True]),
             possible_modes=torch.tensor([[True,False,False,False,False]]),
             mode_probability=torch.tensor([[1.,0,0,0,0]]),version=torch.tensor([7]),clock=torch.tensor([3.]))
    kw=dict(model_id='m',history_id='h')
    s=BeliefSnapshot.from_filter(b,**kw)
    assert s.modes == ('flight',) and s.version==7
    assert s.evidence['evidence_window'] == [[0.]*50]*4
    assert s.evidence['evidence_center'] is None
    b['evidence_window'][0,0,0] = .2
    assert BeliefSnapshot.from_filter(b,**kw).fingerprint != s.fingerprint
    b['assumption_invalid'][0] = True
    invalid=BeliefSnapshot.from_filter(b,**kw)
    assert invalid.evidence['assumption_invalid'] is True
    with pytest.raises(ValueError,match='modes'):
        BeliefSnapshot.from_filter(b,modes=('support',),**kw)
    with pytest.raises(ValueError,match='version'):
        BeliefSnapshot.from_filter(b,version=8,**kw)
