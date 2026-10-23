"""QuadX-Hover original task in the official Genesis RL environment format.

Task: hover at the spawn point (0, 0, 1) for 400 agent steps. Failure
(termination): ground contact or |p| > 3 m; success: surviving to the step
limit without either (the original PyFlyt hover task defines no success
condition, which would make the reported success rate structurally 0).

cfg dicts and buffers in __init__, step() returning
(get_observations(), rew_buf, reset_buf, extras), get_observations() wrapping
self.obs_buf in a TensorDict under the 'policy' group, and reset_idx() for
episodic bookkeeping. Rewards, terminations, observations and spawn
distribution are task-local; physics/actuation live in common.plant.
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import euler, inverse_rotate, quat_from_euler
from algorithms.baselines.baselines import QuadXSE3Baseline


def hover_shaping(position,rpy,body_omega):
    delta=position.clone();delta[:,2]-=1
    return 1-delta.norm(dim=-1)-rpy[:,:2].norm(dim=-1)-.01*body_omega[:,2].square()


class QuadXHoverEnv:
    # Spawn-noise attrs are manual domain-randomization knobs (all zero in the
    # benchmark, where no random numbers are drawn and the reset matches the
    # legacy deterministic spawn exactly). Wind is optional via the
    # steady_wind/dryden ctor kwargs; calm air reproduces the benchmark
    # bit-for-bit.
    spawn_xy_noise=0.        # spawn xy-offset radius [m] (manual DR knob)
    spawn_z_noise=0.         # spawn z-offset half-range [m] (manual DR knob)
    spawn_vel_noise=0.       # spawn |v0| per-axis half-range [m/s] (manual DR knob)
    spawn_att_noise=0.       # spawn per-axis euler half-range [rad] (manual DR knob)
    bounds_radius=3.         # out-of-bounds radius [m]
    # conventional controller: SE(3) geometric hover control, Lee et al. 2010
    # (baselines.py; reuses the env's rate+collective action convention)
    conventional_controller=QuadXSE3Baseline

    def __init__(self,num_envs=64,backend='cpu',seed=7,steady_wind=None,dryden=0.,severity='moderate'):
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'quadx','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized','task_defaults':'PyFlyt 0.29.0'}
        # Raw policy actions live in [-1, 1] and are mapped onto the original
        # physical action bounds by _scale_actions, mirroring the official
        # clip_actions convention.
        self.clip_actions=1.0

        # vel_clamp guards the rigid contact solver: exploration bursts (32-step
        # correlated max-rate commands) can spin the quad to |w| in the hundreds
        # of rad/s and NaN the Genesis solver on propeller-ground impact
        # ("invalid accelerations"). Caps sit far above nominal trajectories
        # (hover |v|<2 m/s, |w|<10 rad/s), so only already-unphysical crash
        # states are affected. Disabled (None) unless this task opts in.
        self.plant=TensorPlant('quadx',num_envs,backend,seed,vel_clamp=(15.,30.))
        self.device=self.plant.device
        self.agent_hz=40;self.inner_steps=120//self.agent_hz;self.max_steps=400
        self.max_episode_length=self.max_steps+2
        self.num_actions=4
        self.action_low=torch.tensor([-math.pi,-math.pi,-math.pi,0.],device=self.device,dtype=torch.float64)
        self.action_high=torch.tensor([math.pi,math.pi,math.pi,.8],device=self.device,dtype=torch.float64)
        self.hover_target=torch.tensor((0.,0.,1.),device=self.device,dtype=torch.float64)

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=self.device,dtype=torch.float64)
        self.reward=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.count=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.terminated=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.collision=self.terminated.clone();self.out_of_bounds=self.terminated.clone()
        self.complete=self.terminated.clone();self.fatal_collision=self.terminated.clone()
        self._observation=None
        self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=self.device,dtype=torch.float32)
        self.reset_buf=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.returns=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.completed_returns=torch.zeros_like(self.returns)
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)
        self.completed_metrics={'terminal_pos_err_m':torch.zeros_like(self.returns),
                                'terminal_att_err_rad':torch.zeros_like(self.returns)}

        self.extras=dict()
        self.cfg=dict(self.task_cfg)
        self._init_wind(num_envs,seed,steady_wind,dryden,severity)

        self.reset()

    def _init_wind(self,n,seed,steady_wind,dryden,severity):
        """Optional wind disturbance (environments/common/disturbances.py).

        Calm air (default) leaves the plant dynamics bit-for-bit unchanged;
        with wind configured, the gust filters advance once per control step
        and the plant aerodynamic terms evaluate v_air = v - wind.
        """
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

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        physical=self._scale_actions(actions)
        self.action.copy_(physical)
        command=physical.clone()
        if self._wind is not None:self.plant.wind.copy_(self._wind.step())
        self.reward.fill_(-.1)
        self.count+=1
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            for _ in range(2):self.plant.physics_step(command,active)
            p,rpy,v,wb,vb=self._observe(active)
            collision=self.plant.ground_contact;bounds=p.norm(dim=-1)>self.bounds_radius
            reward=torch.where(collision|bounds,-100.,self.reward)+hover_shaping(p,rpy,wb)
            term=collision|bounds
            self.reward.copy_(torch.where(active,reward,self.reward))
            self.terminated|=term&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents still
        # terminate inside the substep loop above
        active=~(self.terminated|self.truncated)
        timeout=(self.count>=self.max_steps)&active
        self.truncated|=timeout
        self.complete|=timeout&~self.collision&~self.out_of_bounds

        self.reset_buf=self.terminated|self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf
        n_done=done.sum().clamp_min(1)
        pos_err=torch.where(done,(p-self.hover_target).norm(dim=-1),0.)
        att_err=torch.where(done,rpy[:,:2].norm(dim=-1),0.)
        self.completed_metrics['terminal_pos_err_m']+=pos_err
        self.completed_metrics['terminal_att_err_rad']+=att_err

        # 'complete' marks eval success (survived to the limit) -- it is NOT an
        # MDP termination. Normal step-limit expiry must stay a timeout so
        # FlashSAC bootstraps from it; only true terminations (crash /
        # out-of-bounds) cut the bootstrap.
        self.extras['time_outs']=self.truncated&~self.terminated
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any=bool(done.any())
        if done_any:
            self.extras['episode']={
                'success_rate':(done&self.complete).to(torch.float64).sum()/n_done,
                'terminal_pos_err_m':pos_err.sum()/n_done,
                'terminal_att_err_rad':att_err.sum()/n_done}
        else:
            self.extras.pop('episode',None)

        # reset_idx on an empty mask is a no-op for every buffer and plant.warmup
        # early-exits on empty masks (see plant.warmup), so gating it with the
        # already-synced done_any is purely a cost guard, not a correctness one.
        if done_any:self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def _observe(self,mask):
        p,q,v,w=self.plant.state()
        rpy=euler(q);wb=inverse_rotate(q,w);vb=inverse_rotate(q,v)
        attitude=torch.cat((wb,q[:,1:],q[:,:1],vb,p,self.action,self.plant.auxiliary()),-1)
        if self._observation is None:self._observation=attitude.clone()
        else:self._observation.copy_(torch.where(mask[:,None],attitude,self._observation))
        return p,rpy,v,wb,vb

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask=envs_idx
        p,q,v,w=self.plant.state()
        spawn=torch.zeros_like(p);spawn[:,2]=1.
        rotation=torch.zeros_like(q);rotation[:,0]=1
        velocity=torch.zeros_like(v);omega=torch.zeros_like(w)
        # spawn noise attrs are manual domain-randomization knobs (all zero in
        # the benchmark, where no random numbers are drawn and the reset
        # matches the legacy deterministic spawn exactly)
        if self.spawn_xy_noise>0. or self.spawn_z_noise>0.:
            u=self.plant.rng.uniform(3,mask)
            rad=self.spawn_xy_noise*u[:,0];ang=2*math.pi*u[:,1]
            off=torch.stack((rad*ang.cos(),rad*ang.sin(),self.spawn_z_noise*(2*u[:,2]-1.)),-1)
            spawn=spawn+torch.where(mask[:,None],off,0.)
        if self.spawn_vel_noise>0.:
            u=self.plant.rng.uniform(3,mask)
            velocity=velocity+torch.where(mask[:,None],self.spawn_vel_noise*(2*u-1.),0.)
        if self.spawn_att_noise>0.:
            u=self.plant.rng.uniform(3,mask)
            rot=quat_from_euler(self.spawn_att_noise*(2*u-1.))
            rotation=torch.where(mask[:,None],rot,rotation)
        self.plant.set_state(torch.where(mask[:,None],spawn,p),torch.where(mask[:,None],rotation,q),
                             torch.where(mask[:,None],velocity,v),torch.where(mask[:,None],omega,w))
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.collision.masked_fill_(mask,False);self.out_of_bounds.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False);self.fatal_collision.masked_fill_(mask,False)
        self.action.masked_fill_(mask[:,None],0)
        self.plant.zero(mask)
        self.plant.warmup(mask)
        self._observe(mask)
        self.episode_length_buf.masked_fill_(mask,0)
        self.returns.masked_fill_(mask,0)
        if self._wind is not None:self._wind.reset(mask)

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):self.plant.close()
