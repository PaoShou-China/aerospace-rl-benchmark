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


def _mix32(x):
    """32-bit avalanche finalizer (murmur3 fmix32); bijective modulo 2**32."""
    x=(x^(x>>16))&0xffffffff
    x=(x*0x85ebca6b)&0xffffffff
    x=(x^(x>>13))&0xffffffff
    x=(x*0xc2b2ae35)&0xffffffff
    return x^(x>>16)


class TensorRandom:
    """Independent per-replica counter streams; RNG state also lives on device.

    Distribution-compatible with upstream, not NumPy PCG64 bitstream-compatible.
    Masked resets/steps do not consume random numbers in another replica.

    Replica streams are keyed by mix32(base_seed ^ (i*GOLDEN)), not by
    base_seed+i. The contiguous form made base seed s+1 hand replica i the
    exact stream of replica i+1 under seed s, so two adjacent seeds sampled a
    1024-stream population that overlapped in 1023 of 1024 replicas (and the
    whole batch was a permutation of the previous one, hence identical
    aggregate metrics). Hashing the (base_seed, replica) pair makes every
    (seed, replica) stream distinct and breaks that shift relation.
    """
    def __init__(self,n,device,seed):
        i=torch.arange(n,device=device,dtype=torch.int64)
        self.seed=_mix32((int(seed)&0xffffffff)^((i*0x9e3779b1)&0xffffffff))
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
        if vehicle not in ('quadx','fixedwing','rocket','missile'):raise ValueError(vehicle)
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
        model={'quadx':'cf2x','fixedwing':'fixedwing','rocket':'rocket','missile':'missile'}[vehicle]
        source=Path(__file__).parent/'original_pyflyt/models/vehicles'/model
        self.params=yaml.safe_load((source/f'{model}.yaml').read_text())
        self.body_id=c.loadURDF(str(source/f'{model}.urdf'),basePosition=(0,0,1 if vehicle=='quadx' else 10 if vehicle=='fixedwing' else 4000 if vehicle=='missile' else 450))
        self.pad=None;self.target=None
        if vehicle=='rocket':
            pad=Path(__file__).parent/'original_pyflyt/models/landing_pad.urdf'
            self.pad_id=c.loadURDF(str(pad),basePosition=(0,0,.1),useFixedBase=True)
            self.pad=c._bodies[self.pad_id]['entity']
        if vehicle=='missile':
            # Kinematic engagement target: pure visual entity (collision=False),
            # its state is overwritten every physics step from the env's
            # analytic 3-DOF buffers, exactly like the pause-restore pattern.
            # Loaded before c._build() like every other scene body.
            from .consistency import verify_missile_assets
            verify_missile_assets()
            tgt=Path(__file__).parent/'original_pyflyt/models/target.urdf'
            self.target_id=c.loadURDF(str(tgt),basePosition=(0,0,4500),collision=False)
            self.target=c._bodies[self.target_id]['entity']
        c._build()
        self.scene=c._scene;self.drone=c._bodies[self.body_id]['entity'];self.ground=c._bodies[0]['entity']
        self.device=gs.device;self.dtype=torch.float64;d=self.device
        self.links={link.name:link.idx_local for link in self.drone.links}
        self.force=torch.zeros((n,self.drone.n_links,3),device=d,dtype=torch.float64)
        self.torque=torch.zeros_like(self.force)
        self.rng=TensorRandom(n,d,seed)
        self.command_width={'quadx':4,'fixedwing':4,'rocket':7,'missile':3}[vehicle]
        self.physics_steps=0
        # world-frame wind velocity (n,3) float64; None = calm air. Aerodynamic
        # terms (lifting surfaces, quadx body drag, rocket body drag) evaluate
        # air-relative velocity v_air = v - wind, so None/wind=0 reproduces the
        # calm-air dynamics bit-for-bit. The owning env updates the buffer once
        # per control step (gusts are frozen across the physics substeps).
        self.wind=None
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
            elif vehicle=='missile':
                # Cross tail: +-y fins (lift axis z) -> pitch/roll, +-z fins
                # (lift axis y) -> yaw/roll; nose along +x like fixedwing.
                names=['fin_left','fin_right','fin_top','fin_bottom'];keys=['fin_params']*4
                lifts=[[0,0,1],[0,0,1],[0,1,0],[0,1,0]];forwards=[[1,0,0]]*4
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
            elif vehicle=='rocket':
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
            else:
                # missile: boost-coast engine + rate-PID steering (TensorRatePID
                # on body rates, fin_cmd = normalized (roll, pitch, yaw) surface
                # command). No fuel/throttle channel -- the burn clock below is
                # the only propulsion state.
                self.pid=TensorRatePID(self.params['control_params']['ang_vel'],n,d)
                self.ignited=torch.zeros(n,device=d,dtype=torch.bool)
                self.burn_clock=torch.zeros(n,device=d,dtype=torch.float64)
                self.fin_cmd=torch.zeros((n,3),device=d,dtype=torch.float64)
                b=self.params['boost_params'];body=self.params['body_params']
                self.max_thrust=torch.tensor(b['max_thrust'],device=d,dtype=torch.float64)
                self.burn_time=torch.tensor(b['burn_time'],device=d,dtype=torch.float64)
                self.boost_noise_ratio=torch.tensor(b['noise_ratio'],device=d,dtype=torch.float64)
                self.body_drag_constant=torch.tensor(.5*1.225*body['drag_coef_x']*body['area_x'],device=d,dtype=torch.float64)
                self.cn_k=torch.tensor(.5*1.225*body['area_x']*body['cn_alpha'],device=d,dtype=torch.float64)
                self.cp_x=torch.tensor(body['cp_offset_x'],device=d,dtype=torch.float64)
                self.cmq_coef=torch.tensor(body['cmq_coef'],device=d,dtype=torch.float64)
                self.roll_gear=torch.tensor(self.params['control_params']['roll_gear'],device=d,dtype=torch.float64)
                self.drag_id=self.links['base_link']

    def state(self):
        return tuple(x.to(torch.float64) for x in (self.drone.get_pos(),self.drone.get_quat(),self.drone.get_vel(),self.drone.get_ang()))

    def set_state(self,p,q,v,w):
        self.drone.set_pos(p.to(gs.tc_float),zero_velocity=False)
        self.drone.set_quat(q.to(gs.tc_float),zero_velocity=False)
        self.drone.set_dofs_velocity(torch.cat((v,inverse_rotate(q,w)),-1).to(gs.tc_float))

    def _air_velocity(self,ids,q):
        v=self.drone.get_links_vel(links_idx_local=ids,ref=gs.link_ref_frame.link_COM).to(torch.float64)
        if self.wind is not None:v=v-self.wind[:,None,:]
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
        elif self.vehicle=='missile':
            # Rate command -> normalized (roll, pitch, yaw) fin command; the
            # skid-to-turn mixer expands it onto the four tail fins in
            # physics_step.
            _,q,_,w=self.state();rates=inverse_rotate(q,w)
            fin_cmd=self.pid.step(rates,command[:,:3],mask)
            self.fin_cmd.copy_(torch.where(mask[:,None],fin_cmd,self.fin_cmd))

    def clear_contacts(self):
        self.ground_contact.zero_();self.pad_contact.zero_()

    def physics_step(self,command,mask,target_state=None):
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
            vb_air=vb-(inverse_rotate(q,self.wind) if self.wind is not None else 0.)
            self.force[:,self.body_drag_id]=-.5*1.225*drag['drag_coef_xyz']*drag['drag_area_xyz']*vb_air.abs()*vb_air
            self.torque[:,0]=-drag['drag_coef_pqr']*wb.abs()*wb*(~self.last_contact)[:,None]
        else:
            if self.vehicle=='fixedwing':
                cmd=command.index_select(1,self.fixed_index)*self.fixed_sign
            elif self.vehicle=='missile':
                # skid-to-turn mixer. Torque signs from the fin geometry
                # (nose +x, y-fins: lift z at y=+-0.13, z-fins: lift y at
                # z=+-0.13, all at x=-1.3 behind the CG):
                #   pitch: tau_y = +1.3*(F_left+F_right)
                #   yaw:   tau_z = -1.3*(F_top+F_bottom)  -> yaw channel negated
                #   roll:  tau_x = +0.13*(F_left-F_right) - 0.13*(F_top-F_bottom)
                #          -> the z-fin pair carries the ROLLED-BACK differential
                # order matches surface_ids [fin_left,fin_right,fin_top,fin_bottom]
                roll,pitch,yaw=self.fin_cmd.unbind(-1)
                g=self.roll_gear
                cmd=torch.stack((pitch+g*roll,pitch-g*roll,-yaw-g*roll,-yaw+g*roll),-1).clamp(-1,1)
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
            elif self.vehicle=='missile':
                # Axial body drag (rocket-style), then the slender-body normal
                # force F = -0.5*rho*|v|*A*Cn*v_lat applied at the aft CP:
                # it is both the turning force and the weathercock restoring
                # moment (tau = r x F with r = (cp_x, 0, 0)).
                # Boost-coast engine: ignites on the first active physics step
                # and burns along body +x until burn_time elapses. One normal
                # sample per physics step regardless of burn state (RNG count
                # invariant).
                vel=self._air_velocity([self.drag_id],q)[:,0]
                self.force[:,self.drag_id,0]-=self.body_drag_constant*vel[:,0].abs()*vel[:,0]
                k=self.cn_k*vel.norm(dim=-1)
                # Transverse flow at the CP includes the rotational contribution
                # (v_cp = v_cg + omega x r, r = (cp_x,0,0)), which turns the
                # slender-body force into BOTH the weathercock restoring moment
                # AND a rotational damping moment (-k*|v|*cp_x^2 * omega_lat).
                # Without the omega term the pitch/yaw channels are undamped
                # oscillators (verified open-loop: constant full-fin deflection
                # produced a sustained ~0.6 s-period limit cycle).
                fy=-k*(vel[:,1]+self.cp_x*wb[:,2])
                fz=-k*(vel[:,2]-self.cp_x*wb[:,1])
                # Cmq-style synthetic pitch/yaw damping (see missile.yaml).
                cmq=self.cmq_coef*vel.norm(dim=-1)
                self.force[:,self.drag_id,1]+=fy
                self.force[:,self.drag_id,2]+=fz
                self.torque[:,self.drag_id,1]+=(-self.cp_x)*fz-cmq*wb[:,1]
                self.torque[:,self.drag_id,2]+=self.cp_x*fy-cmq*wb[:,2]
                self.ignited.copy_(torch.where(mask,self.ignited|mask,self.ignited))
                noise=self.rng.normal(mask,1)
                burning=self.ignited&(self.burn_clock<self.burn_time)
                thrust=self.max_thrust*(1.+self.boost_noise_ratio*noise)*burning.to(torch.float64)
                self.force[:,self.drag_id,0]+=thrust
                self.burn_clock.copy_(torch.where(mask,self.burn_clock+1/240,self.burn_clock))
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
        if target_state is not None and self.target is not None:
            # Kinematic engagement target (missile): overwrite the visual
            # entity's state from the env's analytic 3-DOF buffers before the
            # solver runs (pause-restore pattern; collision=False, no forces).
            tp,tq,tv,tw=target_state
            self.target.set_pos(tp.to(gs.tc_float),zero_velocity=False)
            self.target.set_quat(tq.to(gs.tc_float),zero_velocity=False)
            self.target.set_dofs_velocity(torch.cat((tv,inverse_rotate(tq,tw)),-1).to(gs.tc_float))
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
        if self.vehicle=='missile':
            burn_left=(self.burn_time-self.burn_clock).clamp_min(0.)/self.burn_time
            burning=(self.ignited&(self.burn_clock<self.burn_time)).to(torch.float64)
            return torch.cat((self.surfaces.actuation,self.ignited[:,None].to(torch.float64),
                              burning[:,None],burn_left[:,None]),-1)
        return torch.cat((self.surfaces.actuation,self.ignited[:,None].to(torch.float64),self.fuel[:,None],self.throttle,self.gimbal),-1)

    def zero(self,mask):
        """Plant-side reset zeroing; spawn placement and task buffers stay with the env."""
        self.ground_contact.masked_fill_(mask,False)
        self.pad_contact.masked_fill_(mask,False)
        self.last_contact.masked_fill_(mask,False)
        self.throttle.masked_fill_(mask[:,None],0)
        if self.vehicle=='quadx':
            self.pid.reset(mask);self.pwm.masked_fill_(mask[:,None],0)
        elif self.vehicle=='missile':
            self.pid.reset(mask);self.surfaces.actuation.masked_fill_(mask[:,None],0)
            self.ignited.masked_fill_(mask,False)
            self.burn_clock.masked_fill_(mask,0)
            self.fin_cmd.masked_fill_(mask[:,None],0)
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
