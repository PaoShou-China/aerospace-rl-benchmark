"""Batched PyFlyt lifting-surface model, including positive/negative stall.

Formula attribution: PyFlyt LiftingSurface (MIT), source commit documented in README.
"""
import math
import numpy as np
import torch


class LiftingSurfaceBatch:
    def __init__(self, params, lifts, forwards, n, device, dt):
        self.dt=dt
        self.lift=torch.tensor(lifts,device=device,dtype=torch.float64)
        self.forward=torch.tensor(forwards,device=device,dtype=torch.float64)
        self.torque_axis=torch.linalg.cross(self.lift,self.forward)
        for key in params[0]:
            setattr(self,key,torch.tensor([p[key] for p in params],device=device,dtype=torch.float64))
        if (self.tau<dt).any():raise ValueError('Surface dt must not exceed actuator tau')
        self.aspect=self.span/self.chord
        self.area=self.span*self.chord
        # Precompute using the same NumPy float64 operations/order as upstream.
        def constant(fn):return torch.tensor([fn(p) for p in params],device=device,dtype=torch.float64)
        self.slope=constant(lambda p:p['Cl_alpha_2D']*((p['span']/p['chord'])/((p['span']/p['chord'])+2*((p['span']/p['chord'])+4)/((p['span']/p['chord'])+2))))
        self.aero_tau=constant(lambda p:1-(np.arccos(2*p['flap_to_chord']-1)-np.sin(np.arccos(2*p['flap_to_chord']-1)))/np.pi)
        self.alpha0=constant(lambda p:np.deg2rad(p['alpha_0_base']))
        self.stall_p=constant(lambda p:np.deg2rad(p['alpha_stall_P_base']))
        self.stall_n=constant(lambda p:np.deg2rad(p['alpha_stall_N_base']))
        self.actuation=torch.zeros((n,len(params)),device=device,dtype=torch.float64)

    def reset(self,ids):self.actuation[ids]=0

    def coefficients(self,alpha,actuation):
        delta=torch.deg2rad(actuation*self.deflection_limit)
        dcl=self.slope*self.aero_tau*self.eta*delta
        clmaxp=self.slope*(self.stall_p-self.alpha0)+self.flap_to_chord*dcl
        clmaxn=self.slope*(self.stall_n-self.alpha0)+self.flap_to_chord*dcl
        a0=self.alpha0-dcl/self.slope
        ap=a0+clmaxp/self.slope
        an=a0+clmaxn/self.slope
        linear=(alpha>an)&(alpha<ap)
        cl=self.slope*(alpha-a0)
        ai=cl/(math.pi*self.aspect)
        effective=alpha-a0-ai
        ct=self.Cd_0*torch.cos(effective)
        cn=(cl+ct*torch.sin(effective))/torch.cos(effective)
        cd=cn*torch.sin(effective)+ct*torch.cos(effective)
        cm=-cn*(.25-.175*(1-2*effective/math.pi))
        # np.interp clips outside the stall-to-90-degree interval.
        aipos=clmaxp/(math.pi*self.aspect)*(1-((alpha-ap)/(math.pi/2-ap)).clamp(0,1))
        aineg=clmaxn/(math.pi*self.aspect)*((alpha+math.pi/2)/(an+math.pi/2)).clamp(0,1)
        ai_stall=torch.where(alpha>0,aipos,aineg)
        effective_stall=alpha-a0-ai_stall
        cd90=-.0426*delta**2+.21*delta+1.98
        sn,cs=torch.sin(effective_stall),torch.cos(effective_stall)
        cns=cd90*sn*(1/(.56+.44*sn.abs())-.41*(1-torch.exp(-17/self.aspect)))
        cts=.5*self.Cd_0*cs
        cls=cns*cs-cts*sn
        cds=cns*sn+cts*cs
        cms=-cns*(.25-.175*(1-2*effective_stall.abs()/math.pi))
        return tuple(torch.where(linear,a,b) for a,b in [(cl,cls),(cd,cds),(cm,cms)])

    def step(self,cmd,velocity):
        velocity=velocity.to(torch.float64)
        self.actuation += self.dt/self.tau*(cmd-self.actuation)
        alpha=torch.atan2(-(velocity*self.lift).sum(-1),(velocity*self.forward).sum(-1))
        cl,cd,cm=self.coefficients(alpha,self.actuation)
        qa=.5*1.225*velocity.square().sum(-1)*self.area
        lift,drag=cl*qa,cd*qa
        normal=lift*torch.cos(alpha)+drag*torch.sin(alpha)
        parallel=lift*torch.sin(alpha)-drag*torch.cos(alpha)
        force=normal[...,None]*self.lift+parallel[...,None]*self.forward
        torque=(qa*cm*self.chord)[...,None]*self.torque_axis
        return force,torque
