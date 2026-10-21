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
from ..common.plant import TensorPlant
from ..common.dynamics import euler, inverse_rotate, quat_from_euler
from ..common.curriculum import SuccessRateCurriculum


def hover_shaping(position,rpy,body_omega):
    delta=position.clone();delta[:,2]-=1
    return 1-delta.norm(dim=-1)-rpy[:,:2].norm(dim=-1)-.01*body_omega[:,2].square()


class QuadXHoverEnv:
    # Difficulty axis (0 easiest = the deterministic-spawn benchmark -> 1
    # hardest): reset perturbations grow with d (classic hover curriculum /
    # domain randomization). At d=0 no random numbers are drawn, so the reset
    # state and RNG stream match the legacy benchmark bit-for-bit. Success
    # criteria (ground contact, bounds, step limit) are untouched.
    spawn_xy_noise=0.        # spawn xy-offset radius [m] (axis 0 -> .5)
    spawn_z_noise=0.         # spawn z-offset half-range [m] (axis 0 -> .25)
    spawn_vel_noise=0.       # spawn |v0| per-axis half-range [m/s] (axis 0 -> .5)
    spawn_att_noise=0.       # spawn per-axis euler half-range [rad] (axis 0 -> .35)
    benchmark_difficulty=0.
    difficulty=None          # None = benchmark behavior (no cfg record)
    curriculum=False
    curriculum_window=1024   # completed episodes per advance decision
    curriculum_threshold=.6  # rolling success rate that advances the stage

    def __init__(self,num_envs=64,backend='cpu',seed=7,difficulty=None,curriculum=None):
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
        self._resolve_difficulty(difficulty,curriculum)

        self.reset()

    def _resolve_difficulty(self,difficulty,curriculum):
        """Shared difficulty/curriculum resolution (every task env, same shape)."""
        if difficulty is not None:
            if not 0.<=difficulty<=1.:raise ValueError('difficulty must be in [0,1]')
            self.difficulty=float(difficulty)
        if curriculum is not None:self.curriculum=bool(curriculum)
        self._tracker=None
        if self.curriculum:
            if self.difficulty is None:self.difficulty=0.
            self._tracker=SuccessRateCurriculum(self.difficulty,self.curriculum_window,self.curriculum_threshold)
            self.cfg['curriculum']=True
        if self.difficulty is not None:self._apply_difficulty(self.difficulty)

    def _apply_difficulty(self,d):
        """Map difficulty in [0,1] onto the spawn-perturbation knobs."""
        self.difficulty=d
        self.spawn_xy_noise=.5*d
        self.spawn_z_noise=.25*d
        self.spawn_vel_noise=.5*d
        self.spawn_att_noise=.35*d
        self.cfg['difficulty']=d

    def _scale_actions(self,actions):
        raw=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)
        return self.action_low+(raw+1)*.5*(self.action_high-self.action_low)

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        physical=self._scale_actions(actions)
        self.action.copy_(physical)
        command=physical.clone()
        self.reward.fill_(-.1)
        self.count+=1
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            for _ in range(2):self.plant.physics_step(command,active)
            p,rpy,v,wb,vb=self._observe(active)
            timeout=self.count>=self.max_steps
            collision=self.plant.ground_contact;bounds=p.norm(dim=-1)>3
            reward=torch.where(collision|bounds,-100.,self.reward)+hover_shaping(p,rpy,wb)
            term=collision|bounds;trunc=timeout
            self.reward.copy_(torch.where(active,reward,self.reward))
            self.terminated|=term&active;self.truncated|=trunc&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active
            self.complete|=(timeout&~collision&~bounds)&active

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

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        if bool(done.any()):
            self.extras['episode']={
                'success_rate':(done&self.complete).to(torch.float64).sum()/n_done,
                'terminal_pos_err_m':pos_err.sum()/n_done,
                'terminal_att_err_rad':att_err.sum()/n_done}
            if self._tracker is not None:
                new=self._tracker.update(done,self.complete)
                if new is not None:
                    self._apply_difficulty(new)
                    print(f'[curriculum] success {self._tracker.last_rate:.2f} -> difficulty {new:.2f}',flush=True)
        else:
            self.extras.pop('episode',None)

        # Guard the unconditional reset call (fixedwing-precedent): reset_idx on
        # an empty mask is a no-op for every buffer but used to run warmup, which
        # advanced the global scene 20 ballistic substeps every agent step
        # (~4.3x effective gravity). plant.warmup now early-exits on empty masks;
        # this guard additionally skips the whole reset path.
        if bool(self.reset_buf.any()):self.reset_idx(self.reset_buf)
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
        # difficulty>0 perturbs the spawn (position ring + z, velocity, attitude);
        # all knobs are 0 at the benchmark difficulty, where no random numbers
        # are drawn and the reset matches the legacy deterministic spawn exactly
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

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):self.plant.close()
