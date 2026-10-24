"""CPU/GPU algebra probe only. NOT the Genesis benchmark; no learned policy.

A simplified vertical acceleration sensor observes g0*u*thrust_gain + wind.
The estimator sees only applied input and noisy measured acceleration. True
parameters are used by the plant and final metric calculation, never the update.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
try:
    from .observable_belief import joseph_update
except ImportError:
    from observable_belief import joseph_update


def run(device='cpu',count=128,steps=80,seed=930101):
    torch.set_num_threads(1)
    generator=torch.Generator(device=device).manual_seed(seed)
    start=time.perf_counter()
    dtype=torch.float64
    truth=torch.stack((.7+.6*torch.rand(count,generator=generator,device=device,dtype=dtype),
                       -2+4*torch.rand(count,generator=generator,device=device,dtype=dtype)),-1)
    mean=torch.zeros_like(truth); mean[:,0]=1.
    covariance=torch.diag(torch.tensor([.2**2,2.**2],device=device,dtype=dtype))[None].repeat(count,1,1)
    valid=torch.ones(count,dtype=torch.bool,device=device)
    variance=torch.full((count,1),.2**2,device=device,dtype=dtype)
    errors=[]
    for step in range(steps):
        # All channels and episodes receive the same information budget.
        u=.15+.7*torch.rand(count,generator=generator,device=device,dtype=dtype)
        h=torch.stack((40*u,torch.ones_like(u)),-1)[:,None]
        measured=(h[:,0]*truth).sum(-1)+.2*torch.randn(count,generator=generator,device=device,dtype=dtype)
        innovation=measured[:,None]-(h@mean[...,None]).squeeze(-1)
        update=joseph_update(mean,covariance,innovation,h,variance,valid,process_variance=0.)
        mean,covariance=update.mean,update.covariance
        errors.append(float((mean-truth).square().mean()))
    heldout_u=.15+.7*torch.rand(count,32,generator=generator,device=device,dtype=dtype)
    actual=40*heldout_u*truth[:,:1]+truth[:,1:]
    adaptive=40*heldout_u*mean[:,:1]+mean[:,1:]
    nominal=40*heldout_u
    sd=covariance.diagonal(dim1=-2,dim2=-1).sqrt()
    result={
        'scope':'synthetic linear acceleration identification only',
        'genesis_benchmark_run':False,'neural_network_trained':False,
        'seed':seed,'episodes':count,'observations_per_episode':steps,'device':device,
        'nominal_acceleration_rmse_m_s2':float((nominal-actual).square().mean().sqrt()),
        'adaptive_acceleration_rmse_m_s2':float((adaptive-actual).square().mean().sqrt()),
        'thrust_gain_mae':float((mean[:,0]-truth[:,0]).abs().mean()),
        'wind_acceleration_mae_m_s2':float((mean[:,1]-truth[:,1]).abs().mean()),
        'nominal_95_parameter_interval_coverage':((mean-truth).abs()<=1.96*sd).double().mean(0).tolist(),
        'covariance_min_eigenvalue':float(torch.linalg.eigvalsh(covariance).min()),
        'wall_seconds':time.perf_counter()-start,
        'limitations':['Correct linear measurement model; ideal known input.',
                      'No contact, attitude, fuel, latency or state-observation error.',
                      'Not a landing success rate or evidence of superiority to adaptive MPC.']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=p.parse_args()
    result=run(args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
