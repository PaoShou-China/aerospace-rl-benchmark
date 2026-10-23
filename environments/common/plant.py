"""Shared device-resident vehicle plant for the three original-task RL environments.

Geometry, actuation (rate-PID mixer, lifting surfaces, booster/fuel), physics
stepping, contact checks and reset/warmup primitives live here. Task rewards,
terminations, observations and spawn distributions live in
{multirotor,fixedwing,rocket}.env. All buffers stay on one device;
masked operations never consume random numbers or advance state of inactive
replicas.
"""
from pathlib import Path
import math
import torch
import genesis as gs
import yaml
from .dynamics import inverse_rotate
from .aerodynamics import LiftingSurfaceBatch


def rotate(q,v):
    return inverse_rotate(torch.cat((q[...,:1],-q[...,1:]),-1),v)


class TensorRandom:
    """Independent per-replica counter streams; RNG state also lives on device.

    Distribution-compatible with upstream, not NumPy PCG64 bitstream-compatible.
    Masked resets/steps do not consume random numbers in another replica.
    """
    def __init__(self,n,device,seed):
        self.seed=torch.arange(n,device=device,dtype=torch.int64)+seed
        self.counter=torch.zeros(n,device=device,dtype=torch.int64)
    def uniform(self,width,mask):
        k=self.counter[:,None]+torch.arange(width,device=mask.device,dtype=torch.int64)
        x=(self.seed[:,None]+(k+1)*1664525+1013904223)&0xffffffff
        x=((x^(x>>16))*2246822519)&0xffffffff
        x=((x^(x>>13))*3266489917)&0xffffffff
        x=x^(x>>16)
        self.counter.add_(mask.to(torch.int64)*width)
        return (x.to(torch.float64)+.5)/4294967296.
    def normal(self,mask,mean):
        u=self.uniform(2,mask)
        return mean+torch.sqrt(-2*torch.log(u[:,0]))*torch.cos(2*math.pi*u[:,1])


class TensorRatePID:
    def __init__(self,p,n,device):
        self.kp=torch.tensor(p['kp'],device=device,dtype=torch.float64)
        self.ki=torch.tensor(p['ki'],device=device,dtype=torch.float64)
        self.kd=torch.tensor(p['kd'],device=device,dtype=torch.float64)
        self.limit=torch.tensor(p['lim'],device=device,dtype=torch.float64)
        self.integral=torch.zeros((n,3),device=device,dtype=torch.float64)
        self.previous=torch.zeros_like(self.integral)
    def reset(self,mask):
        self.integral.copy_(torch.where(mask[:,None],0.,self.integral))
        self.previous.copy_(torch.where(mask[:,None],0.,self.previous))
    def step(self,state,target,mask):
        error=target-state
        integral=(self.integral+self.ki*error/120).clamp(-self.limit,self.limit)
        out=(self.kp*error+integral+self.kd*(error-self.previous)*120).clamp(-self.limit,self.limit)
        self.integral.copy_(torch.where(mask[:,None],integral,self.integral))
        self.previous.copy_(torch.where(mask[:,None],error,self.previous))
        return out


class TensorPlant:
    def __init__(self,vehicle,n,backend='cpu',seed=7,vel_clamp=None):
        if vehicle not in ('quadx','fixedwing','rocket'):raise ValueError(vehicle)
        if n<1:raise ValueError('n must be positive')
        self.vehicle=vehicle;self.n=n
        # Optional solver guard (None = disabled): cap world speed and body spin
        # norm before scene.step so exploration-driven tumbling cannot push the
        # rigid contact solver into NaN ("invalid accelerations"). Scale factor is
        # 1 whenever the state is already inside the cap, so nominal trajectories
        # (hover |v|<2 m/s, |w|<10 rad/s) are untouched.
        self.vel_clamp=None if vel_clamp is None else (float(vel_clamp[0]),float(vel_clamp[1]))
        # Geometry/metadata loading is construction-only. No compatibility API is
        # called by physics_step, auxiliary, or the reset primitives.
        from .original_backend import GenesisBulletClient,engine_configuration
        from .original_batch import ConfiguredBulletClient
        with engine_configuration(backend):self._geometry=ConfiguredBulletClient()
        c=self._geometry;c._num_envs=n;c.resetSimulation()
        self.ground_id=c.loadURDF('plane.urdf',useFixedBase=True)
        model={'quadx':'cf2x','fixedwing':'fixedwing','rocket':'rocket'}[vehicle]
        source=Path(__file__).parent/'original_pyflyt/models/vehicles'/model
        self.params=yaml.safe_load((source/f'{model}.yaml').read_text())
        self.body_id=c.loadURDF(str(source/f'{model}.urdf'),basePosition=(0,0,1 if vehicle=='quadx' else 10 if vehicle=='fixedwing' else 450))
        c._build()
        self.pad=None
        if vehicle=='rocket':
            pad=Path(__file__).parent/'original_pyflyt/models/landing_pad.urdf'
            self.pad_id=c.loadURDF(str(pad),basePosition=(0,0,.1),useFixedBase=True)
            self.pad=c._bodies[self.pad_id]['entity']
        self.scene=c._scene;self.drone=c._bodies[self.body_id]['entity'];self.ground=c._bodies[0]['entity']
        self.device=gs.device;self.dtype=torch.float64;d=self.device
        self.links={link.name:link.idx_local for link in self.drone.links}
        self.force=torch.zeros((n,self.drone.n_links,3),device=d,dtype=torch.float64)
        self.torque=torch.zeros_like(self.force)
        self.rng=TensorRandom(n,d,seed)
        self.command_width={'quadx':4,'fixedwing':4,'rocket':7}[vehicle]
        self.physics_steps=0
        self.throttle=torch.zeros((n,4 if vehicle=='quadx' else 1),device=d,dtype=torch.float64)
        self.ground_contact=torch.zeros(n,device=d,dtype=torch.bool)
        self.pad_contact=torch.zeros(n,device=d,dtype=torch.bool)
        self.last_contact=torch.zeros(n,device=d,dtype=torch.bool)
        if vehicle=='quadx':
            self.pid=TensorRatePID(self.params['control_params']['ang_vel'],n,d)
            self.motor_map=torch.tensor([[-1,-1,-1,1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],device=d,dtype=torch.float64)
            self.prop_ids=torch.tensor([self.links[f'prop{i}_link'] for i in range(1,5)],device=d,dtype=torch.long)
            self.spin=torch.tensor([-1,-1,1,1],device=d,dtype=torch.float64)
            self.body_drag_id=self.links['center_of_mass_link'] if 'center_of_mass_link' in self.links else self.links['body_link']
            self.pwm=torch.zeros((n,4),device=d,dtype=torch.float64)
        else:
            if vehicle=='fixedwing':
                names=['ail_left_link','ail_right_link','horizontal_tail_link','vertical_tail_link','main_wing_link']
                keys=['left_wing_flapped_params','right_wing_flapped_params','horizontal_tail_params','vertical_tail_params','main_wing_params']
                lifts=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,0,1]];forwards=[[1,0,0]]*5
            else:
                # Upstream finlet_map (core/drones/rocket.py L152-159) maps
                # (finlet_x, finlet_y, finlet_yaw) onto FOUR URDF fin links:
                #   fin_pos_x <- (y+yaw), fin_neg_x <- (y-yaw),
                #   fin_pos_y <- (x-yaw), fin_neg_y <- (x+yaw)
                # with lifting axes [0,1,0] for the x-fins and [1,0,0] for the
                # y-fins (rocket.py L129/L143). The step() cmd stack below
                # already produces exactly that 4-vector; an earlier mapping
                # here wrongly attached the first two slots to fueltank/booster
                # links (near-zero moment arm) and never touched fin_pos_y /
                # fin_neg_y -- fixed 2026-10-20 to the upstream assignment.
                names=['fin_pos_x','fin_neg_x','fin_pos_y','fin_neg_y'];keys=['finlet_params']*4
                lifts=[[0,1,0],[0,1,0],[1,0,0],[1,0,0]];forwards=[[0,0,-1]]*4
            self.surface_ids=torch.tensor([self.links[name] for name in names],device=d,dtype=torch.long)
            self.surfaces=LiftingSurfaceBatch([self.params[k] for k in keys],lifts,forwards,n,d,1/240)
            if vehicle=='fixedwing':
                self.fixed_index=torch.tensor([0,0,1,1,2],device=d,dtype=torch.long)
                self.fixed_sign=torch.tensor([1,-1,1,-1,0],device=d,dtype=torch.float64)
            else:
                from .consistency import inertial_basis
                self.fuel=torch.full((n,),.05,device=d,dtype=torch.float64)
                self.ignited=torch.zeros(n,device=d,dtype=torch.bool)
                self.gimbal=torch.zeros((n,2),device=d,dtype=torch.float64)
                self.fuel_id=self.links['fueltank_link']
                self.fuel_basis=torch.as_tensor(inertial_basis(self.drone.links[self.fuel_id]),device=d,dtype=torch.float64)
                self.fuel_inertia=torch.tensor([self.params['booster_params'][f'inertia_i{x}{x}'] for x in 'xyz'],device=d,dtype=torch.float64)
                body=self.params['body_params']
                self.body_drag_constant=torch.tensor([.5*1.225*body[f'drag_coef_{x}']*body[f'area_{x}'] for x in 'xyz'],device=d,dtype=torch.float64)
                self.drag_id=self.links['base_link']

    def state(self):
        return tuple(x.to(torch.float64) for x in (self.drone.get_pos(),self.drone.get_quat(),self.drone.get_vel(),self.drone.get_ang()))

    def set_state(self,p,q,v,w):
        self.drone.set_pos(p.to(gs.tc_float),zero_velocity=False)
        self.drone.set_quat(q.to(gs.tc_float),zero_velocity=False)
        self.drone.set_dofs_velocity(torch.cat((v,inverse_rotate(q,w)),-1).to(gs.tc_float))

    def _air_velocity(self,ids,q):
        v=self.drone.get_links_vel(links_idx_local=ids,ref=gs.link_ref_frame.link_COM).to(torch.float64)
        return inverse_rotate(q[:,None,:].expand(-1,len(ids),-1),v)

    def controller(self,command,mask):
        if self.vehicle=='quadx':
            _,q,_,w=self.state();rates=inverse_rotate(q,w)
            torque=self.pid.step(rates,command[:,:3],mask)
            mixed=torch.cat((torque,command[:,3:4].clamp(0,1)),-1)@self.motor_map.T
            high=mixed.amax(-1,keepdim=True);low=mixed.amin(-1,keepdim=True)
            hi=high.clamp_max(1);lo=low.clamp_min(.05)
            add=(lo-low)/(hi-low)*(hi-mixed);sub=(high-hi)/(high-lo)*(mixed-lo)
            pwm=torch.where(high!=low,mixed+add-sub,mixed).clamp(.05,1)
            self.pwm.copy_(torch.where(mask[:,None],pwm,self.pwm))

    def clear_contacts(self):
        self.ground_contact.zero_();self.pad_contact.zero_()

    def physics_step(self,command,mask):
        p,q,v,w=self.state();vb=inverse_rotate(q,v);wb=inverse_rotate(q,w)
        self.force.zero_();self.torque.zero_()
        if self.vehicle in ('quadx','fixedwing'):
            m=self.params['motor_params'];target=self.pwm if self.vehicle=='quadx' else command[:,3:4]
            throttle=self.throttle+(1/240/m['tau'])*(target-self.throttle)
            # Upstream normal(*shape) yields one scalar per replica, shared across its motors.
            noise=self.rng.normal(mask,4 if self.vehicle=='quadx' else 1)
            throttle=throttle*(1+m['noise_ratio']*noise[:,None])
            self.throttle.copy_(torch.where(mask[:,None],throttle,self.throttle))
        if self.vehicle=='quadx':
            m=self.params['motor_params'];thrust=self.throttle.abs()*self.throttle*m['total_thrust']/4
            self.force[:,self.prop_ids,2]=thrust
            self.torque[:,self.prop_ids,2]=thrust*(m['torque_coef']/m['thrust_coef'])*self.spin
            drag=self.params['drag_params']
            vb_air=vb
            self.force[:,self.body_drag_id]=-.5*1.225*drag['drag_coef_xyz']*drag['drag_area_xyz']*vb_air.abs()*vb_air
            self.torque[:,0]=-drag['drag_coef_pqr']*wb.abs()*wb*(~self.last_contact)[:,None]
        else:
            if self.vehicle=='fixedwing':
                cmd=command.index_select(1,self.fixed_index)*self.fixed_sign
            else:
                a=command;cmd=torch.stack((a[:,1]+a[:,2],a[:,1]-a[:,2],a[:,0]-a[:,2],a[:,0]+a[:,2]),-1).clamp(-1,1)
            previous=self.surfaces.actuation.clone()
            f,t=self.surfaces.step(cmd,self._air_velocity(self.surface_ids,q))
            self.surfaces.actuation.copy_(torch.where(mask[:,None],self.surfaces.actuation,previous))
            self.force[:,self.surface_ids]=f;self.torque[:,self.surface_ids]=t
            if self.vehicle=='fixedwing':
                m=self.params['motor_params'];thrust=self.throttle[:,0].abs()*self.throttle[:,0]*m['total_thrust']
                self.force[:,self.links['motor_link'],0]=thrust
                self.torque[:,self.links['motor_link'],0]=thrust*m['torque_coef']/m['thrust_coef']
            else:
                b=self.params['booster_params'];body=self.params['body_params']
                # Upstream body drag acts on body_ids=[0] == base_link
                # (core/drones/rocket.py L92-116), not the fueltank.
                vel=self._air_velocity([self.drag_id],q)[:,0]
                self.force[:,self.drag_id]-=self.body_drag_constant*vel.abs()*vel
                gimbal=self.gimbal+(1/240/b['gimbal_tau'])*(command[:,5:7]-self.gimbal)
                self.gimbal.copy_(torch.where(mask[:,None],gimbal,self.gimbal))
                ignition=(self.ignited if not b['reignitable'] else torch.zeros_like(self.ignited))|(command[:,3]>.5)
                self.ignited.copy_(torch.where(mask,ignition,self.ignited))
                minimum=b['min_thrust']/b['max_thrust']
                target=self.ignited*(minimum+command[:,4]*(1-minimum))
                throttle=self.throttle[:,0]+(1/240/b['booster_tau'])*(target-self.throttle[:,0])
                throttle=throttle*(1+b['noise_ratio']*self.rng.normal(mask,1))*(self.fuel>0)
                self.throttle[:,0].copy_(torch.where(mask,throttle,self.throttle[:,0]))
                fuel=(self.fuel-self.throttle[:,0]*b['max_fuel_rate']/b['total_fuel']/240).clamp(0,1)
                self.fuel.copy_(torch.where(mask,fuel,self.fuel))
                x,y=(self.gimbal*math.radians(b['gimbal_range_degrees'])).unbind(-1)
                direction=torch.stack((torch.sin(y),-torch.sin(x)*torch.cos(y),torch.cos(x)*torch.cos(y)),-1)
                self.force[:,self.links['booster_link']]+=direction*(self.throttle[:,0]*b['max_thrust'])[:,None]
                mass=(self.fuel*b['total_fuel']).clamp_min(1e-8)
                inertia=self.fuel_basis.T@torch.diag_embed((self.fuel[:,None]*self.fuel_inertia).clamp_min(1e-8))@self.fuel_basis
                # unconditional mass write: set_links_mass/inertia are idempotent,
                # so the update runs every physics step -- while no fuel burns it
                # writes back identical values, keeping the hot path free of the
                # bool((throttle>1e-6).any()) host sync (mass depends only on
                # fuel, and fuel only changes while throttle>0, so skipping the
                # write when fuel is constant was a pure performance shortcut)
                self.drone.set_links_mass(mass[:,None].to(gs.tc_float),links_idx_local=[self.fuel_id])
                self.drone.set_links_inertia(inertia[:,None].to(gs.tc_float),links_idx_local=[self.fuel_id])
        self.force*=mask[:,None,None];self.torque*=mask[:,None,None]
        qlinks=self.drone.get_links_quat().to(torch.float64)
        force=rotate(qlinks,self.force);torque=rotate(qlinks,self.torque)
        self.drone.apply_links_external_wrench(force=force.to(gs.tc_float),torque=torque.to(gs.tc_float),ref=gs.link_ref_frame.link_COM,local=False)
        if self.vel_clamp is not None:
            v_cap,w_cap=self.vel_clamp
            cv=self.drone.get_vel().to(torch.float64);cw=self.drone.get_ang().to(torch.float64)
            cv=cv*(v_cap/cv.norm(dim=-1,keepdim=True).clamp_min(v_cap))
            cw=cw*(w_cap/cw.norm(dim=-1,keepdim=True).clamp_min(w_cap))
            self.drone.set_dofs_velocity(torch.cat((cv,inverse_rotate(self.drone.get_quat().to(torch.float64),cw)),-1).to(gs.tc_float))
        self.scene.step(update_visualizer=False,refresh_visualizer=False)
        self.physics_steps+=1
        ground=self.drone.get_contacts(with_entity=self.ground,is_padded=True)['valid_mask'].any(-1)
        pad=torch.zeros_like(ground) if self.pad is None else self.drone.get_contacts(with_entity=self.pad,is_padded=True)['valid_mask'].any(-1)
        self.ground_contact|=ground&mask;self.pad_contact|=pad&mask
        self.last_contact.copy_(torch.where(mask,ground|pad,self.last_contact))
        # Keep inactive replicas paused during early termination and reset warmup.
        # The restore runs UNCONDITIONALLY: for a fully-active batch the
        # write-back is a lossless float32 round-trip (state() upcasts to
        # float64, set_state casts the identical values back to float32), and
        # for an all-inactive batch it now pauses them correctly too (the old
        # `uniform` shortcut skipped the restore there and let the whole scene
        # drift ballistically). This removes the per-physics-step
        # bool(mask.all()) / bool(mask.any()) host-device syncs.
        pn,qn,vn,wn=self.state()
        self.set_state(torch.where(mask[:,None],pn,p),torch.where(mask[:,None],qn,q),torch.where(mask[:,None],vn,v),torch.where(mask[:,None],wn,w))

    def auxiliary(self):
        if self.vehicle=='quadx':return self.throttle
        if self.vehicle=='fixedwing':return torch.cat((self.surfaces.actuation,self.throttle),-1)
        return torch.cat((self.surfaces.actuation,self.ignited[:,None].to(torch.float64),self.fuel[:,None],self.throttle,self.gimbal),-1)

    def zero(self,mask):
        """Plant-side reset zeroing; spawn placement and task buffers stay with the env."""
        self.ground_contact.masked_fill_(mask,False)
        self.pad_contact.masked_fill_(mask,False)
        self.last_contact.masked_fill_(mask,False)
        self.throttle.masked_fill_(mask[:,None],0)
        if self.vehicle=='quadx':
            self.pid.reset(mask);self.pwm.masked_fill_(mask[:,None],0)
        else:
            self.surfaces.actuation.masked_fill_(mask[:,None],0)
            if self.vehicle=='rocket':
                self.fuel.masked_fill_(mask,.05);self.ignited.masked_fill_(mask,False);self.gimbal.masked_fill_(mask[:,None],0)

    def warmup(self,mask,iterations=10):
        """Original 10-Aviary-step warmup with zero command on the masked slots."""
        # An empty warmup must not advance the global scene: physics_step runs
        # scene.step() for every replica, and with an all-False mask the
        # pause-restore below is skipped (uniform shortcut), so the caller would
        # absorb 20 extra ballistic substeps per agent step (~4.3x effective
        # gravity for the 6-substep quadx agent step). Same root cause as the
        # fixedwing reset_idx bug fixed on 2026-10-20.
        if not bool(mask.any()):return
        command=torch.zeros((self.n,self.command_width),device=self.device,dtype=torch.float64)
        for _ in range(iterations):
            self.controller(command,mask)
            for _ in range(2):self.physics_step(command,mask)

    def close(self):self._geometry.disconnect()
