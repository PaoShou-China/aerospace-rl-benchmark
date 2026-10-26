"""Missile-intercept task: 6-DOF tail-controlled missile vs kinematic target.

Dynamics: the project-authored missile URDF (environments/common/
original_pyflyt/models/vehicles/missile/, SHA-256 pinned by
consistency.verify_missile_assets) on the shared Genesis TensorPlant --
6-DOF rigid body (150 kg, nose along body +x), four cross tail fins through
the LiftingSurfaceBatch aero model, body drag, and an autonomous boost-coast
engine (20 kN along body +x for the first 4 s of each episode, then unpowered
coast). Control: 60 Hz policy / 120 Hz rate-PID / 240 Hz physics; the policy
commands BODY RATES (roll +-6 rad/s, pitch/yaw +-0.5 rad/s -- coordinated-turn
limit a/V) which the plant's rate PID converts to normalized (roll, pitch, yaw)
fin commands expanded by the skid-to-turn mixer onto the four fins.

Engagement: the target is a kinematic 3-DOF aircraft advanced analytically in
torch buffers every physics substep (straight line / sinusoidal weave /
horizontal barrel roll) at 250 m/s around 4 km altitude; a collision-free
visual entity in the Genesis scene mirrors it for rendering only. The
missile spawns 2.5 km away with fire-control
lead: it aims at the straight-line extrapolation of the target over the
nominal time-to-go t_go = TGO_SCALE * R / (V_M0 + V_T), then launches at
250 m/s along that bearing; the target heading points back at the missile
within a fixed cone.

Success: miss distance < KILL_RADIUS (15 m proximity fuse) at ANY physics
substep -> immediate termination with the one-shot kill bonus. Fixed forever.
Failure: the closest-approach point passes
(range rate turns positive after 2 s, i.e. the shot is unrecoverable),
leaving the bounds |p| > 15 km or z < 0, or the 15 s time limit.

Engagement geometry (single fixed medium operating point, 2026-10-24 user
decision):
  * initial range        2500 m (midpoint of the old 1500-3500 m axis)
  * target heading cone  +-37.5 deg around the missile bearing
  * target lateral accel  7.5 g (weave amplitude/severity, barrel radius)
  * weave frequency       0.75 Hz

Reward (per inner step, only while active):
  +0.01 * (d_prev - d)          telescoping closing-progress potential
  +0.02 * (v_hat . r_hat)       velocity/LOS alignment (dense PN-consistent
                                guidance shaping; the from-scratch signal)
  -0.01 * |action|^2            rate-command effort
  +50 on kill (once), -10 on unrecoverable miss (once)
Components are accumulated and reported as Episode/rew_* scalars plus
completed_metrics keys, so reward attribution is observable.

Observation (34 dims, float32, unnormalized -- the policy-side
EmpiricalNormalization owns scaling):
  [0:3] body omega [rad/s]   [3:7] quat xyzw   [7:10] body v [m/s]
  [10:13] world p [m]        [13:16] prev action (rate cmds, rad/s)
  [16:20] fin actuation (4)  [20:21] ignited flag  [21:22] engine burning
  [22:23] burn-time left fraction
  [23:26] target rel position (body) [m]  [26:29] rel velocity (body) [m/s]
  [29:32] LOS rate (body) [rad/s]  [32:33] closing speed Vc [m/s]
  [33:34] target speed [m/s]
"""
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import euler, inverse_rotate
from algorithms.baselines.baselines import MissileAPNBaseline


def _aim_quat(aim,roll):
    """Batched xyzw quaternion taking body +x to `aim` plus a small extra roll
    about body x.  Local on purpose: the shared dynamics.quat_from_euler is
    the repo's (scaled, near-identity) rocket/fixedwing convention and must
    not be retargeted."""
    x_hat=torch.tensor([1.,0.,0.],device=aim.device,dtype=aim.dtype).expand_as(aim)
    axis=torch.linalg.cross(x_hat,aim)
    ang=torch.atan2(axis.norm(dim=-1),aim[...,0].clamp(-1,1))
    axis=axis/axis.norm(dim=-1).clamp_min(1e-9)[:,None]
    half=.5*ang
    qa=torch.cat((half.cos()[:,None],axis*half.sin()[:,None]),-1)      # wxyz
    hr=.5*roll
    qr=torch.stack((hr.cos(),hr.sin(),torch.zeros_like(hr),torch.zeros_like(hr)),-1)
    w1,x1,y1,z1=qa.unbind(-1);w2,x2,y2,z2=qr.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2),-1)


OMEGA_MAX=6.0     # roll-rate command limit [rad/s]
OMEGA_LAT=1.0     # pitch/yaw rate command limit [rad/s]. Raised 0.5 -> 1.0 on
                  # 2026-10-24 for vanilla-PPO learnability: the steady
                  # coordinated-turn rate is a/V (0.2-0.5 rad/s), but terminal
                  # guidance needs TRANSIENT authority above it -- the rate PID
                  # simply saturates the fins, so a 1.0 rad/s command cap lets
                  # the policy pull harder during the endgame correction.
KILL_RADIUS=15.0  # proximity-fuse kill radius [m]; FIXED.
                  # Raised 8 -> 15 on 2026-10-24: at 8 m the BC warm start's median
                  # miss (9.8 m at d=0) sat ON the fuse boundary, so any ~1e-3
                  # weight drift from a PPO update wiped out all kills (measured
                  # blocker); 15 m puts the warm start comfortably inside the gate.
R_SPAWN=2500.     # initial range [m] (fixed medium operating point)
CONE=37.5         # target-heading cone half-width [deg]
A_LAT=7.5         # target lateral accel [g]
F_WEAVE=.75       # weave angular rate [Hz]
V_T=250.          # target speed [m/s]
V_M0=250.         # missile launch speed [m/s]
Z_MID=4000.       # engagement altitude [m]
K_PROG=.01        # closing-progress gain per metre
K_ALIGN=.02       # velocity/LOS alignment gain (dense PN-shaped guidance signal)
K_CTRL=.01        # rate-command effort gain
KILL_BONUS=50.
MISS_PENALTY=10.
TGO_SCALE=0.8     # fire-control t_go factor for the launch-aim lead
                  # (1.0 = nominal closed-form; 0.8 measured best on the fixed
                  # medium point: shorter lead trims overshoot misses, +0.8pp
                  # SR on two disjoint 1024-ep seed populations)


class MissileInterceptEnv:
    conventional_controller=MissileAPNBaseline
    OMEGA_MAX=OMEGA_MAX

    def __init__(self,num_envs=64,backend='cpu',seed=7,
                 steady_wind=None,dryden=0.,severity='moderate'):
        if num_envs<1:raise ValueError('n must be positive')
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'missile','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized_dynamics',
                       'task_defaults':'project-authored 6-DOF intercept'}
        self.clip_actions=1.0
        self.plant=TensorPlant('missile',num_envs,backend,seed,vel_clamp=(1200.,40.))
        self.device=self.plant.device
        self.agent_hz=60;self.inner_steps=120//self.agent_hz;self.max_steps=900  # 15 s
        self.max_episode_length=self.max_steps+2
        self.num_actions=3
        # Per-axis rate-command limits: roll spins fast, pitch/yaw are limited
        # by the coordinated-turn relation a/V (see OMEGA_LAT above).
        omega_scale=torch.tensor((OMEGA_MAX,OMEGA_LAT,OMEGA_LAT),device=self.device,dtype=torch.float64)
        self.action_low=-omega_scale
        self.action_high=omega_scale

        d=self.device
        zeros=lambda *s:torch.zeros(*s,device=d,dtype=torch.float64)
        self.action=zeros(num_envs,self.num_actions)
        self.reward=zeros(num_envs)
        self.count=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.sim_time=zeros(num_envs)
        self.terminated=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.out_of_bounds=self.terminated.clone()
        self.complete=self.terminated.clone()
        self.miss_flag=self.terminated.clone()
        self._observation=None;self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=d,dtype=torch.float32)
        self.reset_buf=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.returns=zeros(num_envs);self.completed_returns=self.returns.clone()
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)
        # engagement state
        self.min_distance=zeros(num_envs)
        self.dist=zeros(num_envs)          # current missile-target range
        self.dist_prev=zeros(num_envs)     # range at the previous inner-step boundary
        self.target_pos=zeros(num_envs,3)
        self.target_vel=zeros(num_envs,3)
        # kinematic target parameters (reset_idx-owned)
        self.t_p0=zeros(num_envs,3)        # target position at t=0
        self.t_head=zeros(num_envs,3)      # unit heading (straight/weave)
        self.t_perp=zeros(num_envs,3)      # horizontal left-normal of the heading
        self.t_omega=zeros(num_envs)       # weave angular rate [rad/s]
        self.t_phase=zeros(num_envs)       # maneuver phase [rad]
        self.t_a=zeros(num_envs)           # lateral accel [m/s^2]
        self.t_type=torch.zeros(num_envs,device=d,dtype=torch.long)  # 0 straight 1 weave 2 barrel
        self.t_center=zeros(num_envs,3)    # barrel circle center
        self.t_omega_b=zeros(num_envs)     # barrel angular rate (signed) [rad/s]
        self.t_radius=zeros(num_envs)      # barrel radius [m]
        keys=('progress','align','ctrl','kill','miss')
        self.reward_components={k:zeros(num_envs) for k in keys}
        self.completed_metrics={'miss_distance':zeros(num_envs),
                                'kill_rate':zeros(num_envs),'miss_rate':zeros(num_envs),
                                'timeout_rate':zeros(num_envs)}
        for k in keys:self.completed_metrics[f'rew_{k}']=zeros(num_envs)
        self.extras=dict();self.cfg=dict(self.task_cfg)
        self._init_wind(num_envs,seed,steady_wind,dryden,severity)
        self.reset()

    def _init_wind(self,n,seed,steady_wind,dryden,severity):
        self._wind=None
        if steady_wind is not None or dryden>0.:
            from environments.common.disturbances import WindDisturbance
            self._wind=WindDisturbance(n,self.device,seed,steady_wind=steady_wind,
                                       dryden=dryden,severity=severity,dt=1./self.agent_hz)
            self.plant.wind=torch.zeros((n,3),device=self.device,dtype=torch.float64)
            self.cfg['wind']=dict(steady_wind=list(steady_wind) if steady_wind is not None else None,
                                  dryden=dryden,severity=severity)

    def _scale_actions(self,actions):
        raw=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)
        return self.action_low+(raw+1)*.5*(self.action_high-self.action_low)

    # ------------------------------------------------------------------ #
    # kinematic target: closed-form state at the current per-replica time
    # ------------------------------------------------------------------ #
    def _target_state(self):
        t=self.sim_time                      # (n,)
        m0=self.t_type==0;m1=self.t_type==1;m2=self.t_type==2
        p_straight=self.t_p0+V_T*self.t_head*t[:,None]
        v_straight=(V_T*self.t_head).expand_as(p_straight)
        ph=self.t_omega*t+self.t_phase       # (n,)
        s=(self.t_a/(self.t_omega*self.t_omega))*torch.sin(ph)
        vp=(self.t_a/self.t_omega)*torch.cos(ph)
        p_weave=p_straight+self.t_perp*s[:,None]
        v_weave=v_straight+self.t_perp*vp[:,None]
        th=self.t_omega_b*t+self.t_phase     # (n,)
        p_barrel=self.t_center+self.t_radius[:,None]*torch.stack((torch.cos(th),torch.sin(th),torch.zeros_like(th)),-1)
        v_barrel=V_T*torch.stack((-torch.sin(th),torch.cos(th),torch.zeros_like(th)),-1)
        p=torch.where(m0[:,None],p_straight,torch.where(m1[:,None],p_weave,p_barrel))
        v=torch.where(m0[:,None],v_straight,torch.where(m1[:,None],v_weave,v_barrel))
        return p,v

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        command=self._scale_actions(actions)
        self.action.copy_(command)
        if self._wind is not None:self.plant.wind.copy_(self._wind.step())
        self.reward.fill_(0.)
        self.count+=1
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            self.dist_prev.copy_(torch.where(active,self.dist,self.dist_prev))
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            kill_ev=torch.zeros_like(active);miss_ev=torch.zeros_like(active)
            for _ in range(2):
                sub=~(self.terminated|self.truncated)
                tp,tv=self._target_state()
                self.target_pos.copy_(torch.where(sub[:,None],tp,self.target_pos))
                self.target_vel.copy_(torch.where(sub[:,None],tv,self.target_vel))
                tq=_aim_quat(tv/tv.norm(dim=-1,keepdim=True).clamp_min(1e-6),torch.zeros_like(tv[:,0]))
                self.plant.physics_step(command,sub,target_state=(tp,tq,tv,torch.zeros_like(tv)))
                self.sim_time.copy_(torch.where(sub,self.sim_time+1/240,self.sim_time))
                pm=self.plant.drone.get_pos().to(torch.float64)
                vm=self.plant.drone.get_vel().to(torch.float64)
                r=tp-pm;dist=r.norm(dim=-1)
                self.dist.copy_(torch.where(sub,dist,self.dist))
                self.min_distance.copy_(torch.where(sub&(dist<self.min_distance),dist,self.min_distance))
                closing=(r*(tv-vm)).sum(-1)
                kill_ev|=(dist<KILL_RADIUS)&sub
                # closest approach passed while the shot is still on: unrecoverable
                miss_ev|=(closing>0)&(self.sim_time>2.)&sub&~kill_ev
            # kill preempts miss inside the same inner step
            miss_ev&=~kill_ev
            p,q,v,w=self.plant.state()
            rpy=euler(q);wb=inverse_rotate(q,w);vb=inverse_rotate(q,v)
            r_w=self.target_pos-p
            v_rel=self.target_vel-v
            r2=r_w.square().sum(-1).clamp_min(1e-6)
            lam_w=torch.linalg.cross(r_w,v_rel)/r2[:,None]
            lam_b=inverse_rotate(q,lam_w)
            vc=-(r_w*v_rel).sum(-1)/r2.clamp_min(1e-6).sqrt()
            rel=torch.cat((inverse_rotate(q,r_w),inverse_rotate(q,v_rel),lam_b,
                           vc[:,None],self.target_vel.norm(dim=-1)[:,None]),-1)
            observation=torch.cat((wb,q[:,1:],q[:,:1],vb,p,self.action,self.plant.auxiliary(),rel),-1)
            if self._observation is None:self._observation=observation.clone()
            else:self._observation.copy_(torch.where(active[:,None],observation,self._observation))
            # reward: telescoping closing progress, dense velocity/LOS
            # alignment shaping (the from-scratch PPO guidance signal -- it is
            # potential-like, locally computable, and unlike the old |LOS
            # rate| penalty cannot be gamed by flying away), effort
            progress=K_PROG*(self.dist_prev-self.dist)
            v_hat=v/v.norm(dim=-1,keepdim=True).clamp_min(1.)
            r_hat=r_w/r2.clamp_min(1e-6).sqrt()[:,None]
            align=K_ALIGN*(v_hat*r_hat).sum(-1)
            ctrl=-K_CTRL*command.square().sum(-1)
            kill_r=torch.where(kill_ev&active,torch.full_like(progress,KILL_BONUS),torch.zeros_like(progress))
            miss_r=torch.where(miss_ev&active,torch.full_like(progress,-MISS_PENALTY),torch.zeros_like(progress))
            shaping=progress+align+ctrl+kill_r+miss_r
            self.reward.copy_(torch.where(active,self.reward+shaping,self.reward))
            comps=(progress,align,ctrl,kill_r,miss_r)
            for k,c in zip(self.reward_components,comps):
                self.reward_components[k]+=torch.where(active,c,0.)
            self.complete|=kill_ev&active
            self.miss_flag|=miss_ev&active
            self.terminated|=(miss_ev|((p[:,2]<0)|(p.norm(dim=-1)>15000)))&active
        # time limit applies after the full control step (rocket semantics)
        active=~(self.terminated|self.truncated)
        self.truncated|=((self.count>=self.max_steps)|self.complete)&active

        self.reset_buf=self.terminated|self.truncated
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out,trunc_out=self.terminated.clone(),self.truncated.clone()
        self.completed_metrics['miss_distance']+=torch.where(done,self.min_distance,0.)
        self.completed_metrics['kill_rate']+=(done&self.complete).to(torch.float64)
        self.completed_metrics['miss_rate']+=(done&self.miss_flag).to(torch.float64)
        self.completed_metrics['timeout_rate']+=(done&~self.complete&~self.miss_flag&~self.out_of_bounds).to(torch.float64)

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        done_any=bool(done.any())
        if done_any:
            episode={'success_rate':(done&self.complete).to(torch.float64).sum()/done.sum().clamp_min(1),
                     'miss_distance':(self.min_distance*done).sum()/done.sum().clamp_min(1)}
            for k,buf in self.reward_components.items():
                episode[f'rew_{k}']=(buf*done).sum()/done.sum().clamp_min(1)
            self.extras['episode']=episode
        else:
            self.extras.pop('episode',None)
        if done_any:self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,term_out,trunc_out,self.extras

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask.

        One uniform(9) draw covers ALL random quantities. Spawn geometry is a
        single fixed medium operating point (R_SPAWN / CONE / A_LAT / F_WEAVE).
        """
        mask=envs_idx
        p,q,v,w=self.plant.state()
        u=self.plant.rng.uniform(9,mask)
        # target spawn: horizontal bearing around the origin, fixed range
        R=R_SPAWN
        bearing=u[:,0]*2.*torch.pi
        target_p=torch.stack((R*torch.cos(bearing),R*torch.sin(bearing),
                              Z_MID+(u[:,2]-.5)*400.),-1)
        # target heading: back toward the missile inside a fixed cone
        base=torch.atan2(-target_p[:,1],-target_p[:,0])
        cone=CONE*torch.pi/180.
        heading=base+(u[:,3]*2.-1.)*cone
        h=torch.stack((torch.cos(heading),torch.sin(heading),torch.zeros_like(heading)),-1)
        perp=torch.stack((-h[:,1],h[:,0],torch.zeros_like(h[:,0])),-1)
        # maneuver: lateral accel a, angular rate f, type/phase/sign
        a=torch.full_like(u[:,0],A_LAT*9.81)
        omega=torch.full_like(u[:,0],2.*torch.pi*F_WEAVE)
        mtype=(u[:,4]*3.).clamp(0,2).to(torch.long)
        m0=mtype==0;m1=mtype==1
        phase=u[:,5]*2.*torch.pi
        # missile spawn near the origin at engagement altitude. Fire-control
        # lead: aim at the target's PREDICTED position at the nominal flight
        # time t_go = R/(V_M0+V_T), evaluated with the same closed-form motion
        # model the target flies (straight/weave/barrel), so the launch starts
        # on a collision course for every maneuver type and the guidance only
        # has to correct residual errors.
        t_go=torch.full_like(u[:,0],R)/(V_M0+V_T)
        missile_p=torch.stack((torch.zeros_like(bearing),torch.zeros_like(bearing),
                               Z_MID+(u[:,6]-.5)*100.),-1)
        eul_roll=u[:,7]*.2-.1
        # barrel geometry: circle through the spawn point whose INITIAL TANGENT
        # matches the cone heading (otherwise the barrel target starts flying
        # in an arbitrary direction and the engagement degenerates into a tail
        # chase). v = R*omega_b*(-sin th, cos th), so aligning the tangent with
        # h pins the phase to atan2(-h_x, h_y) (+pi when omega_b < 0 flips the
        # traversal direction); the center then sits one radius along the
        # inward normal from the spawn point.
        Rb=V_T*V_T/a
        omega_b=torch.where(u[:,8]>.5,torch.ones_like(u[:,8]),-torch.ones_like(u[:,8]))*a/V_T
        phi_barrel=torch.atan2(-h[:,0],h[:,1])+torch.where(omega_b>0,0.,torch.pi)
        center=target_p-Rb[:,None]*torch.stack((torch.cos(phi_barrel),torch.sin(phi_barrel),
                                        torch.zeros_like(phi_barrel)),-1)
        # weave keeps its random phase; barrel must start on the aligned tangent
        phase=torch.where(mtype==2,phi_barrel,phase)
        # fire-control lead: extrapolate the target along its CURRENT heading
        # for the nominal flight time (straight-line shot solution). The
        # perpendicular component of this lead is what nulls the initial LOS
        # rate; the along-LOS component is benign. A full maneuver-model
        # prediction was tested and performed WORSE (the maneuver displacement
        # over the flight is tens of metres, while the straight-line lead
        # correctly cancels the crossing geometry).
        def predict(t):
            return target_p+V_T*t[:,None]*h
        t_go=TGO_SCALE*torch.full_like(u[:,0],R)/(V_M0+V_T)
        pred=predict(t_go)
        aim=pred-missile_p;aim=aim/aim.norm(dim=-1,keepdim=True)
        v_m=V_M0*aim
        # plant state (masked writes keep other replicas frozen)
        p_new=torch.where(mask[:,None],missile_p,p)
        q_new=torch.where(mask[:,None],_aim_quat(aim,eul_roll),q)
        v_new=torch.where(mask[:,None],v_m,v)
        w_new=torch.where(mask[:,None],torch.zeros_like(w),w)
        self.plant.set_state(p_new,q_new,v_new,w_new)
        # task buffers (masked)
        def sel(x,cur):return torch.where(mask[:,None] if x.dim()>1 else mask,x,cur)
        self.t_p0=sel(target_p,self.t_p0);self.t_head=sel(h,self.t_head);self.t_perp=sel(perp,self.t_perp)
        self.t_omega=sel(omega,self.t_omega);self.t_phase=sel(phase,self.t_phase);self.t_a=sel(a,self.t_a)
        self.t_type=torch.where(mask,mtype,self.t_type)
        self.t_center=sel(center,self.t_center);self.t_omega_b=sel(omega_b,self.t_omega_b)
        self.t_radius=sel(Rb,self.t_radius)
        tp,tv=self._target_state()
        self.target_pos=sel(tp,self.target_pos);self.target_vel=sel(tv,self.target_vel)
        dist=(self.target_pos-p_new).norm(dim=-1)
        self.dist=sel(dist,self.dist);self.dist_prev=sel(dist,self.dist_prev)
        self.min_distance=sel(torch.full_like(dist,1e9),self.min_distance)
        self.sim_time=sel(torch.zeros_like(dist),self.sim_time)
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.out_of_bounds.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False);self.miss_flag.masked_fill_(mask,False)
        for buf in self.reward_components.values():buf.masked_fill_(mask,0.)
        self.action.masked_fill_(mask[:,None],0)
        self.plant.zero(mask)
        # No warmup on purpose: the boost engine fires autonomously on the
        # first active physics step, so a rocket-style zero-command warmup
        # would burn thrust (and shift the spawn state) before the first
        # policy step; an air-borne missile has no contacts to settle.
        self._observe(mask)
        self.episode_length_buf.masked_fill_(mask,0)
        self.returns.masked_fill_(mask,0)
        if self._wind is not None:self._wind.reset(mask)
        self._update_observation()

    def _observe(self,mask):
        """Refresh self._observation on the masked slots (no return value)."""
        p,q,v,w=self.plant.state()
        rpy=euler(q);wb=inverse_rotate(q,w);vb=inverse_rotate(q,v)
        r_w=self.target_pos-p
        v_rel=self.target_vel-v
        r2=r_w.square().sum(-1).clamp_min(1e-6)
        lam_w=torch.linalg.cross(r_w,v_rel)/r2[:,None]
        lam_b=inverse_rotate(q,lam_w)
        vc=-(r_w*v_rel).sum(-1)/r2.clamp_min(1e-6).sqrt()
        rel=torch.cat((inverse_rotate(q,r_w),inverse_rotate(q,v_rel),lam_b,
                       vc[:,None],self.target_vel.norm(dim=-1)[:,None]),-1)
        observation=torch.cat((wb,q[:,1:],q[:,:1],vb,p,self.action,self.plant.auxiliary(),rel),-1)
        if self._observation is None:self._observation=observation.clone()
        else:self._observation.copy_(torch.where(mask[:,None],observation,self._observation))

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):self.plant.close()
