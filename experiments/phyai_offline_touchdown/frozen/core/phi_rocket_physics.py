"""Full parameterized rigid body, fuel, actuators, aerodynamics, contact, sensor.

The internal model is an approximation of the unchanged Genesis environment.
It retains every registered physical parameter. Contact uses rotational
effective masses and implicit compliance; it is not a certified contact solver.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .rocket_physical_core import RocketFlightOperator,rotate,inverse_rotate,observation_state
from .rocket_geometry import rotation_error
from .phi_rocket_types import PARAMETERS,NP,NX,NJ,STATE_SCALE,WRENCH_BOUND,RocketTask


def quat_product(a,b):
    return torch.cat((a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1),
        a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)),-1)


def rotation_increment(v):
    n=v.norm(dim=-1,keepdim=True)
    return torch.cat((v*.5*torch.sinc(n/(2*math.pi)),torch.cos(n*.5)),-1)


def state_plus(state,error):
    q=nn.functional.normalize(quat_product(state[...,6:10],rotation_increment(error[...,6:9])),dim=-1)
    return torch.cat((state[...,:6]+error[...,:6],q,state[...,10:]+error[...,9:]),-1)


def state_minus(state,reference):
    return torch.cat((state[...,:6]-reference[...,:6],rotation_error(state[...,6:10],reference[...,6:10]),
        state[...,10:]-reference[...,10:]),-1)


class PhiPhysics(RocketFlightOperator):
    parameter_names=tuple(p[0] for p in PARAMETERS)
    def __init__(self,task=None):
        super().__init__()
        self.task=task or RocketTask();self.substeps=1
        self.register_buffer('parameter_scale',torch.tensor([p[1] for p in PARAMETERS]))
        self.register_buffer('log_parameters',torch.tensor([p[2]=='log' for p in PARAMETERS]))
        self.register_buffer('state_scale',torch.tensor(STATE_SCALE))
        self.register_buffer('wrench_bound',torch.tensor(WRENCH_BOUND))
        feet=[]
        for x,y,yaw in ((0.,.35,0.),(.3031,-.175,4.188),(-.3031,-.175,-4.188)):
            for dx in (-.025,.025):
                for dy in (-.25,.25):
                    feet.append([x+math.cos(yaw)*dx-math.sin(yaw)*dy,y+math.sin(yaw)*dx+math.cos(yaw)*dy,-2.425])
        self.register_buffer('feet',torch.tensor(feet))

    def values(self,c):
        raw=c[...,:NP].clamp(-1,1)*self.parameter_scale
        return torch.where(self.log_parameters,raw.exp(),raw)

    def mass_properties(self,fuel,c):
        a=self.values(c)
        mass=138.2*a[:,12:13]+410.9*fuel.clamp_min(0)*a[:,13:14]
        com=torch.cat((a[:,15:17],(-94.+.2*2.051)/mass+a[:,17:18]),-1)
        base_xy=(753.8835702*a[:,12:13]+1678*fuel*a[:,13:14]-mass*com[:,2:3].square()).clamp_min(30)
        diag=torch.cat((base_xy*a[:,4:5],base_xy*a[:,4:5]*a[:,18:19],
            (2.3845*a[:,12:13]+7.01*fuel*a[:,13:14])*a[:,5:6]),-1).clamp_min(.1)
        L=torch.diag_embed(diag.sqrt())
        L[:,1,0]=a[:,19]*diag[:,1].sqrt();L[:,2,0]=a[:,20]*diag[:,2].sqrt();L[:,2,1]=a[:,21]*diag[:,2].sqrt()
        inertia=L@L.transpose(-1,-2)
        inverse=torch.cholesky_inverse(L)
        return mass,com,inertia,inverse

    @staticmethod
    def actuator(value,target,tau,dt):
        # Exact continuous first-order endpoint and interval mean; stable at dt=0.
        rate=dt/tau.clamp_min(.001);decay=torch.exp(-rate)
        average=-torch.expm1(-rate)/rate.clamp_min(1e-8)
        return target+(value-target)*decay,target+(value-target)*average

    def flight(self,state,u,c,dt):
        a=self.values(c);dt=torch.as_tensor(dt,device=state.device,dtype=state.dtype)
        if dt.ndim:dt=dt.reshape(-1,1)
        p,v,q,w=state[:,:3],state[:,3:6],state[:,6:10],state[:,10:13]
        fins,engine,gimbal,fuel=state[:,13:17],state[:,17:18],state[:,18:20],state[:,20:21].clamp_min(0)
        command=torch.stack((u[:,1]+u[:,2],u[:,1]-u[:,2],u[:,0]-u[:,2],u[:,0]+u[:,2]),-1)
        bias=a[:,23:24]+torch.stack((a[:,54]+a[:,55],a[:,54]-a[:,55],a[:,53]-a[:,55],a[:,53]+a[:,55]),-1)
        fins,fin_mean=self.actuator(fins,(command+bias).clamp(-1,1),.05*a[:,22:23],dt)
        ignition=(u[:,3:4]>0).to(state.dtype)
        target=ignition*(2966.7/7607+(u[:,4:5]+1)*.5*(1-2966.7/7607)+a[:,29:30]).clamp(0,1)
        tau=.01*a[:,7:8]*torch.where(ignition>0,torch.ones_like(ignition),a[:,60:61])
        engine,eng_mean=self.actuator(engine,target,tau,dt)
        engine=engine*(fuel>0);eng_mean=eng_mean*(fuel>0)
        gimbal,gim_mean=self.actuator(gimbal,u[:,5:7],.01*a[:,8:9],dt)
        next_fuel=(fuel-eng_mean*(1.451/410.9)*a[:,14:15]*dt).clamp_min(0)
        mass,com,inertia,inv=self.mass_properties((fuel+next_fuel)*.5,c)
        vb=inverse_rotate(q,v-a[:,32:35])
        # v is the authored BODY ORIGIN velocity, not COM velocity. Use the
        # origin-to-fin arm for local flow; COM arms belong only in torques.
        vf=vb[:,None]+torch.cross(w[:,None].expand(-1,4,-1),self.fin_positions[None].expand(len(state),-1,-1),dim=-1)
        vl=(vf*self.lift).sum(-1);va=(vf*self.forward_axis).sum(-1)
        alpha=torch.atan2(-vl,va+((vl.square()+va.square())<1e-10).to(va.dtype)*1e-5)
        cl,cd,cm=self.coefficients(alpha,fin_mean)
        cl=cl*a[:,24:25];cd=cd.clamp_min(0)*a[:,25:26];cm=cm*a[:,26:27]
        qa=.5*1.225*a[:,35:36]*.25*vf.square().sum(-1)
        normal=qa*(cl*torch.cos(alpha)+cd*torch.sin(alpha));parallel=qa*(cl*torch.sin(alpha)-cd*torch.cos(alpha))
        ff=(normal[...,None]*self.lift+parallel[...,None]*self.forward_axis)*a[:,6:7,None]
        ft=(qa*cm*.5)[...,None]*torch.cross(self.lift,self.forward_axis,dim=-1)*a[:,6:7,None]
        x=gim_mean[:,0]*math.radians(5)+a[:,27]+a[:,30]
        y=gim_mean[:,1]*math.radians(5)+a[:,28]+a[:,31]
        direction=torch.stack((torch.sin(y),-torch.sin(x)*torch.cos(y),torch.cos(x)*torch.cos(y)),-1)
        thrust=direction*eng_mean*7607*a[:,:1]
        drag=-self.drag*a[:,35:36]*vb.abs()*vb*a[:,1:4]
        disturbance=c[:,NP:NP+6].clamp(-1,1)*self.wrench_bound if c.shape[-1]>NP else torch.zeros_like(state[:,:6])
        force=ff.sum(1)+thrust+drag+disturbance[:,:3]
        arm=torch.zeros_like(com);arm[:,2]=-2.
        torque=(torch.cross(self.fin_positions[None]-com[:,None],ff,dim=-1)+ft).sum(1)
        torque+=torch.cross(arm-com,thrust,dim=-1)-torch.cross(com,drag+disturbance[:,:3],dim=-1)+disturbance[:,3:]
        alpha_b=(inv@(torque-torch.cross(w,(inertia@w[...,None])[...,0],dim=-1))[...,None])[...,0]
        transport=torch.cross(alpha_b,com,dim=-1)+torch.cross(w,torch.cross(w,com,dim=-1),dim=-1)
        acceleration=rotate(q,force/mass-transport)+self.gravity*a[:,57:58]
        next_v=v+dt*acceleration;next_w=w+dt*alpha_b
        next_q=nn.functional.normalize(quat_product(q,rotation_increment((w+next_w)*.5*dt)),dim=-1)
        next_p=p+dt*(v+next_v)*.5
        return torch.cat((next_p,next_v,next_q,next_w,fins,engine,gimbal,next_fuel),-1)

    def contact(self,state,c,dt):
        """Coupled rotational effective mass, unilateral normal/friction impulses."""
        a=self.values(c);mass,com,inertia,inv=self.mass_properties(state[:,20:21],c)
        p,v,q,w=state[:,:3],state[:,3:6],state[:,6:10],state[:,10:13]
        arm=self.feet[None].expand(len(state),-1,-1).clone();arm[:,:,2]+=a[:,38:39]
        world_arm=rotate(q[:,None].expand(-1,len(self.feet),-1),arm)
        points=p[:,None]+world_arm
        pad=((points[...,:2]-points.new_tensor(self.task.pad_center)).square().sum(-1)<=self.task.pad_radius**2)
        height=torch.where(pad,self.task.pad_height+a[:,39:40],a[:,56:57])
        gap=points[...,2]-height
        point_v=v[:,None]+rotate(q[:,None].expand(-1,len(self.feet),-1),torch.cross(w[:,None].expand_as(arm),arm,dim=-1))
        nworld=torch.zeros_like(arm);nworld[...,2]=1
        nb=inverse_rotate(q[:,None].expand(-1,len(self.feet),-1),nworld)
        r=arm-com[:,None];rn=torch.cross(r,nb,dim=-1)
        rotational=torch.einsum('bfi,bij,bfj->bf',rn,inv,rn)
        effective=1/(1/mass+rotational).clamp_min(1e-8)
        contact_weight=torch.sigmoid(-gap/.0005)
        count=contact_weight.sum(-1,keepdim=True).clamp_min(1)
        penetration=nn.functional.softplus(-gap/.0005)*.0005
        stiffness=80000*a[:,9:10];damping=500*a[:,10:11]
        dt=torch.as_tensor(dt,device=state.device,dtype=state.dtype)
        if dt.ndim:dt=dt.reshape(-1,1)
        force=(stiffness*penetration+damping*contact_weight*nn.functional.relu(-point_v[...,2]))/(1+(dt*damping+dt.square()*stiffness)*count/effective)
        restitution=(.05*a[:,36:37]).clamp(0,.5)
        impulse=(dt*force+restitution*effective*nn.functional.relu(-point_v[...,2])*contact_weight/count).clamp_min(0)
        # Each tangential direction also uses rotational effective mass.
        tangential=[]
        speed=point_v[...,:2].norm(dim=-1)
        mu=.6*a[:,11:12]*torch.where(pad,a[:,40:41],torch.ones_like(gap))
        static=mu*a[:,37:38]
        cap=torch.where(speed<.02,static,mu)*impulse
        for k in range(2):
            axis=torch.zeros_like(arm);axis[...,k]=1
            axis_b=inverse_rotate(q[:,None].expand(-1,len(self.feet),-1),axis)
            cross=torch.cross(r,axis_b,dim=-1)
            mt=1/(1/mass+torch.einsum('bfi,bij,bfj->bf',cross,inv,cross)).clamp_min(1e-8)
            stop=-point_v[...,k]*mt/(count*(1+.05*a[:,58:59]))
            tangential.append(stop)
        tangent=torch.stack(tangential,-1)
        tangent=tangent*(cap/(tangent.norm(dim=-1)+1e-8)).clamp_max(1)[...,None]
        impulse_world=torch.cat((tangent,impulse[...,None]),-1)
        ib=inverse_rotate(q[:,None].expand(-1,len(self.feet),-1),impulse_world)
        torque=torch.cross(r,ib,dim=-1).sum(1)
        contact_w=w+(inv@torque[...,None])[...,0]
        damping=.02*a[:,59:60]*impulse.sum(-1,keepdim=True)
        identity=torch.eye(3,device=w.device,dtype=w.dtype)[None]
        next_w=torch.linalg.solve(inertia+damping[...,None]*identity,(inertia@contact_w[...,None]))[...,0]
        dw=next_w-w
        dv=impulse_world.sum(1)/mass-rotate(q,torch.cross(dw,com,dim=-1))
        return torch.cat((p,v+dv,q,w+dw,state[:,13:]),-1)

    def forward(self,state,raw_action,theta,dt=.025):
        shape=state.shape;s=state.reshape(-1,21);u=raw_action.reshape(-1,7).clamp(-1,1)
        c=theta.reshape(-1,theta.shape[-1])
        # The old 0.1 s rigid-body/contact update diverged on retained failure
        # states. Subdivide time, never parameters/operators. The 1.4 s plan
        # and 2 s continuation durations remain exactly unchanged.
        if not isinstance(dt,(int,float)):
            if dt.numel()!=1:raise ValueError('physical interval must be scalar')
            dt=float(dt)
        steps=max(1,math.ceil(round(float(dt)/.0125,8)))
        for _ in range(steps):
            s=self.contact(self.flight(s,u,c,float(dt)/steps),c,float(dt)/steps)
        return s.reshape(shape)

    def sensor(self,state,theta):
        a=self.values(theta)
        q=nn.functional.normalize(quat_product(state[...,6:10],rotation_increment(a[...,50:53])),dim=-1)
        return torch.cat((state[...,:3]+a[...,41:44],state[...,3:6]+a[...,44:47],q,
            state[...,10:13]+a[...,47:50],state[...,13:]),-1)

    def measurement_error(self,predicted,actual):return state_minus(predicted,actual)

    def residual(self,predicted,actual):
        # Compatibility with the common evaluator's 21-coordinate diagnostic.
        e=state_minus(predicted,actual.expand_as(predicted))/self.state_scale
        return torch.cat((e[...,:9],torch.zeros_like(e[...,:1]),e[...,9:]),-1)

    def rollout_grid(self,state,controls,theta,grid):
        states=[]
        for k,dt in enumerate(grid):
            state=self(state,controls[:,k],theta,dt);states.append(state)
        return torch.stack(states,1)
