import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

belief = load('observable_belief')
metrics = load('survival_metrics')
patch = load('prepare_experiment')


def prior(batch=3,p=4,d=2):
    mean=torch.zeros(batch,p,dtype=torch.float64)
    cov=torch.eye(p,dtype=torch.float64)[None].repeat(batch,1,1)
    h=torch.zeros(batch,d,p,dtype=torch.float64)
    h[:,:,:d]=torch.eye(d,dtype=torch.float64)
    innovation=torch.ones(batch,d,dtype=torch.float64)
    noise=torch.full((batch,d),.1,dtype=torch.float64)
    valid=torch.ones(batch,dtype=torch.bool)
    return mean,cov,innovation,h,noise,valid


def test_no_history_does_not_change_belief():
    args=list(prior()); args[-1][:]=False
    result=belief.joseph_update(*args)
    torch.testing.assert_close(result.mean,args[0],rtol=0,atol=0)
    torch.testing.assert_close(result.covariance,args[1],rtol=0,atol=0)
    assert not result.accepted.any()


def test_unexcited_directions_do_not_get_false_certainty():
    args=prior(); result=belief.joseph_update(*args,process_variance=0)
    torch.testing.assert_close(result.covariance[:,2:,2:],args[1][:,2:,2:],rtol=0,atol=0)
    assert bool((result.observable_rank==2).all())
    assert bool((result.covariance[:,:2,:2].diagonal(dim1=-2,dim2=-1)<1).all())


def test_zero_jacobian_cannot_identify_parameters():
    args=list(prior()); args[3].zero_()
    result=belief.joseph_update(*args,process_variance=0)
    torch.testing.assert_close(result.mean,args[0])
    torch.testing.assert_close(result.covariance,args[1])


def test_nonfinite_innovation_is_rejected_per_environment():
    args=list(prior()); args[2][0,0]=float('nan')
    result=belief.joseph_update(*args)
    assert result.accepted.tolist()==[False,True,True]
    assert torch.isfinite(result.mean).all()
    torch.testing.assert_close(result.covariance[0],args[1][0])


def test_covariance_stays_symmetric_positive_definite():
    torch.manual_seed(13)
    args=list(prior(batch=5,p=12,d=9))
    args[3]=torch.randn(5,9,12,dtype=torch.float64)
    for _ in range(25):
        r=belief.joseph_update(*args)
        args[0],args[1]=r.mean,r.covariance
    torch.testing.assert_close(r.covariance,r.covariance.transpose(-1,-2))
    assert bool((torch.linalg.eigvalsh(r.covariance)>0).all())


def test_cubature_matches_all_covariance_directions():
    torch.manual_seed(12)
    a=torch.randn(2,12,12,dtype=torch.float64)
    cov=a@a.transpose(-1,-2)+.01*torch.eye(12,dtype=torch.float64)
    mean=torch.randn(2,12,dtype=torch.float64)
    points,weights=belief.cubature_points(mean,cov)
    reconstructed_mean=(weights[...,None]*points).sum(1)
    delta=points-reconstructed_mean[:,None]
    reconstructed_cov=torch.einsum('bi,bij,bik->bjk',weights,delta,delta)
    torch.testing.assert_close(mean,reconstructed_mean)
    torch.testing.assert_close(cov,reconstructed_cov)
    assert bool((torch.linalg.matrix_rank(reconstructed_cov)==12).all())


def test_batch_jacobian_matches_known_physics():
    x=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    y=torch.stack((x[:,0]*x[:,1],x[:,0].square()),-1)
    j=belief.batch_jacobian(y,x)
    torch.testing.assert_close(j,torch.tensor([[[2.,1.],[2.,0.]],[[4.,3.],[6.,0.]]]))


def test_single_contact_is_not_durable_success():
    m=metrics.StableLandingMonitor(1)
    m.update([True],[True],[False],[False])
    record=m.finish(0)
    assert record['official_success_ever']
    assert not record['durable_success']


def test_crash_after_two_seconds_is_not_success():
    m=metrics.StableLandingMonitor(1)
    for _ in range(80): m.update([True],[True],[False],[False])
    m.update([False],[True],[True],[False])
    row=m.finish(0)
    assert row['sampled_longest_dwell_seconds']==2.
    assert not row['durable_success']
    assert row['official_then_fatal_or_out']


def test_terminal_streak_and_no_future_episode_contamination():
    m=metrics.StableLandingMonitor(2)
    for _ in range(80): m.update([True,False],[True,False],[False,False],[False,False])
    row=m.finish(0)
    assert row['durable_success']
    m.update([False,True],[True,True],[True,False],[False,False])
    assert not m.ever_fatal[0]
    with pytest.raises(ValueError): m.finish(0)


def test_dwell_must_be_consecutive():
    m=metrics.StableLandingMonitor(1)
    for _ in range(50): m.update([True],[True],[False],[False])
    m.update([False],[True],[False],[False])
    for _ in range(50): m.update([True],[True],[False],[False])
    assert not m.finish(0)['durable_success']


def test_patch_rejects_source_drift():
    with pytest.raises(RuntimeError): patch.patch_factory('unknown source')


def test_resume_clears_conflicting_initialization():
    code="""def f(args, config):
    if args.from_parent:
        config.pop('resume', None)
    if args.resume is not None:
        config['resume'] = str(args.resume)
    return config
"""
    patched=patch.patch_run(code); env={}; exec(patched,env)
    from types import SimpleNamespace
    out=env['f'](SimpleNamespace(from_parent=False,resume='new'),{'initialize_from':'old','resume':'old'})
    assert out=={'resume':'new'}
    out=env['f'](SimpleNamespace(from_parent=True,resume=None),{'initialize_from':'old','resume':'old'})
    assert out=={}


def test_all_delivered_python_parses():
    for path in ROOT.rglob('*.py'): ast.parse(path.read_text(),filename=str(path))
