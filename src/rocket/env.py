"""Rocket-Landing task: upstream-faithful port of PyFlyt 0.29.0.

Task definition (reward, dynamics, episode logic, initial conditions) follows
gym_envs/rocket_envs/rocket_landing_env.py + rocket_base_env.py of the bundled
original_pyflyt reference verbatim; only the physics engine differs (Genesis
rigid-body solver vs PyBullet), per the 2026-10-20 alignment decision: "the
physics engine may differ, the reward function and dynamics model must match".

Concretely, relative to the previous staged port (all measured/documented
there), these upstream behaviours are RESTORED:
  * Dense upstream reward, every inner (120 Hz) step:
      -0.3                                      stay-airborne cost
      +0.3 / (||p_xy|| + 0.1)                   over-pad holding
      +10 * (||p_xy_prev|| - ||p_xy||)          lateral progress
      +0.2 * (z_prev - z)                       descent progress
      +4 * (vz - vz_prev + 1) / exp(z)          deceleration, scaled by
            * (1 if vz<0 else -1)               altitude, up-drift penalized
      -|omega_yaw|  -  ||rpy[:2]||
    pad contact: +5 - 0.3*|vz| on EVERY contact inner step; success +3.
    The earlier v1-v3 reward redesigns (hover exploits, brake-distance
    potential, exponential terminal corridor) are DROPPED: v1 reproduced this
    reward but was paired with a non-upstream "any touch ends the episode"
    rule that made +5/step farmable; with the upstream touch semantics
    restored (below), the farming loop terminates itself via the success or
    fatal checks, and the upstream reward is used as-is.
  * Pad contact does NOT terminate the episode (upstream L231-263): on a
    contact inner step the rocket is fatal iff the PREVIOUS inner-step speed
    exceeds 1.0 m/s or 0.35 rad/s; it "lands" (success, truncation) iff that
    speed is <0.02 m/s and <0.02 rad/s with tilt <0.1; otherwise the episode
    CONTINUES and the rocket may settle on the pad. Success is therefore an
    on-pad settling process, not a one-shot quantum touch.
  * Initial conditions follow upstream begin_reset with the options the
    landing task passes (randomize_drop=True, accelerate_drop=True):
    xy in +/-20 m (max_displacement*0.1), z in [400, 450] (ceiling*0.8..0.9),
    euler orn in +/-0.3 rad per axis, vz = -100 m/s, zero angular velocity.
    Upstream quirk reproduced: the random linear/angular velocities in
    begin_reset never fire because the option key is misspelled
    ("randoimize_drop" vs "randomize_drop"), so only the -100 m/s drop speed
    applies. The earlier staged drop (at-rest z=150) was tied to the dropped
    v3 reward and is reverted.
  * Action box matches upstream action_space: dims 0-2 and 5-6 in [-1,1],
    ignition (dim 3) and throttle (dim 4) in [0,1] -- uniform exploration
    fires the engine on ~50% of steps, exactly like upstream. The earlier
    identity-mapped ignition [-1,1] and dead-zoned throttle clamp are
    dropped.

Unchanged port details: 40 Hz policy / 120 Hz control / 240 Hz physics, 1200
policy steps (30 s) truncation, bounds ||p_xy||>200 or z>500, any non-pad
contact or z<0 is fatal, obs = [body omega, quat, body v, p, action, aux(9),
pad contact flag] (30 dims), 5% starting fuel, reignitable engine with the
per-physics-step ignition>0.5 rule.

Success: pad contact with |v|<0.02, |omega|<0.02, tilt<0.1 (upstream values).
"""
import math
import torch
from tensordict import TensorDict
from ..common.plant import TensorPlant
from ..common.dynamics import euler, inverse_rotate


def quat_from_euler(rpy):
    """Roll-pitch-yaw (XYZ intrinsic, PyBullet convention) to quaternion [w,x,y,z]."""
    roll,pitch,yaw=rpy.unbind(-1)
    cr,sr=.5*roll.cos(),.5*roll.sin()
    cp,sp=.5*pitch.cos(),.5*pitch.sin()
    cy,sy=.5*yaw.cos(),.5*yaw.sin()
    return torch.stack((cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy),-1)


def rocket_shaping(position,previous_position,vel_z,previous_vel_z,rpy,body_omega):
    """Upstream dense reward (rocket_landing_env.py L197-229), per inner step.

    vel_z is the WORLD-frame vertical velocity ("ground_lin_vel" upstream,
    which their code computes as body-frame v rotated to world). exp(z) uses
    the absolute altitude, so deceleration only pays near the ground.
    """
    lateral_progress=previous_position[:,:2].norm(dim=-1)-position[:,:2].norm(dim=-1)
    vertical_progress=previous_position[:,2]-position[:,2]
    deceleration_progress=((vel_z-previous_vel_z+1.)/torch.exp(position[:,2])
                           *torch.where(vel_z<0,1.,-1.))
    return (-0.3
            +0.3/(position[:,:2].norm(dim=-1)+.1)
            +10.*lateral_progress
            +0.2*vertical_progress
            +4.*deceleration_progress
            -body_omega[:,2].abs()
            -rpy[:,:2].square().sum(-1).sqrt())


class RocketLandingEnv:
    def __init__(self,num_envs=64,backend='cpu',seed=7,randomization=None):
        self.num_envs=num_envs
        # `randomization` stays in the signature for run_training API
        # compatibility but is unused: the task is upstream-nominal, and the
        # plant no longer carries eval-side wind/thrust-scale knobs.
        self.task_cfg={'vehicle':'rocket','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized','task_defaults':'PyFlyt 0.29.0'}
        self.clip_actions=1.0

        self.plant=TensorPlant('rocket',num_envs,backend,seed)
        self.device=self.plant.device
        self.agent_hz=40;self.inner_steps=120//self.agent_hz;self.max_steps=1200
        self.max_episode_length=self.max_steps+2
        self.num_actions=7
        # Upstream action_space Box (rocket_base_env.py L85-107): finlet x/y/yaw
        # and both gimbals in [-1,1]; ignition and throttle in [0,1]. The
        # affine map below sends raw [-1,1] onto that box, so a zero-mean
        # Gaussian explores ignition at 50% firing rate -- upstream behaviour,
        # deliberate (the earlier [-1,1] identity mapping at 0.6%/step and the
        # throttle dead-zone were non-upstream aids and are dropped).
        self.action_low=torch.tensor([-1.,-1.,-1.,0.,0.,-1.,-1.],device=self.device,dtype=torch.float64)
        self.action_high=torch.ones(self.num_actions,device=self.device,dtype=torch.float64)

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=self.device,dtype=torch.float64)
        self.reward=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.count=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.terminated=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.collision=self.terminated.clone();self.out_of_bounds=self.terminated.clone()
        self.complete=self.terminated.clone();self.fatal_collision=self.terminated.clone()
        self.obs_pad=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self._observation=None
        self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=self.device,dtype=torch.float32)
        self.reset_buf=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.returns=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.completed_returns=torch.zeros_like(self.returns)
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)
        # first-touch telemetry (monitoring only; NOT part of episode logic --
        # upstream keeps simulating after a touch so the rocket can settle)
        self.touch_flag=self.terminated.clone()
        self.touch_speed=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.touch_ang_vel=torch.zeros_like(self.touch_speed)
        self.touch_tilt=torch.zeros_like(self.touch_speed)
        self.completed_metrics={'pad_touch_rate':torch.zeros_like(self.touch_speed),
                                'touch_speed':torch.zeros_like(self.touch_speed),
                                'touch_ang_vel':torch.zeros_like(self.touch_speed),
                                'touch_tilt':torch.zeros_like(self.touch_speed)}

        self.extras=dict()
        self.cfg=dict(self.task_cfg)

        self.reset()

    def _scale_actions(self,actions):
        raw=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)
        return self.action_low+(raw+1)*.5*(self.action_high-self.action_low)

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        physical=self._scale_actions(actions)
        # Upstream passes the gym action straight to set_setpoint; the plant
        # applies the ignition>0.5 rule per physics step (240 Hz, reignitable)
        # and the throttle -> [0.39, 1.0] min/max thrust remap (boosters.py
        # L231-233). No extra env-level reshaping.
        command=physical.clone()
        self.action.copy_(command)
        self.reward.fill_(0.)
        self.touch_flag.fill_(False)
        self.count+=1
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            p0,q0,v0,w0=self.plant.state();vb0=inverse_rotate(q0,v0);wb0=inverse_rotate(q0,w0)
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            for _ in range(2):self.plant.physics_step(command,active)
            p,rpy,v,wb,vb=self._observe(active)
            timeout=self.count>=self.max_steps
            # Upstream compute_base_term_trunc_reward: any contact other than
            # (rocket, pad) is a fatal collision, as is dropping below ground.
            collision=self.plant.ground_contact|(p[:,2]<0)
            bounds=(p[:,:2].norm(dim=-1)>200)|(p[:,2]>500)
            reward=self.reward+rocket_shaping(p,p0,v[:,2],v0[:,2],rpy,wb)
            touch=self.plant.pad_contact&active
            speed0=vb0.norm(dim=-1);spin0=wb0.norm(dim=-1);tilt0=rpy[:,:2].norm(dim=-1)
            # Upstream L231-263: on a pad-contact step, pay +5-0.3|vz|, then
            # judge the PREVIOUS inner-step speed magnitude (= speed0/spin0
            # here, the state at the start of this inner step): >1.0 m/s or
            # >0.35 rad/s is fatal; <0.02/<0.02 with tilt<0.1 is a landed
            # success; anything in between keeps simulating (on-pad settling).
            fatal=collision|(touch&((speed0>1.)|(spin0>.35)))
            complete=touch&~fatal&(spin0<.02)&(speed0<.02)&(tilt0<.1)
            reward+=torch.where(touch,5.-0.3*v[:,2].abs(),0.)
            reward+=complete.to(torch.float64)*3
            new_touch=touch&~self.touch_flag
            self.touch_flag|=new_touch
            self.touch_speed.copy_(torch.where(new_touch,speed0,self.touch_speed))
            self.touch_ang_vel.copy_(torch.where(new_touch,spin0,self.touch_ang_vel))
            self.touch_tilt.copy_(torch.where(new_touch,tilt0,self.touch_tilt))
            self.complete|=complete&active;self.fatal_collision|=fatal&active
            self.obs_pad.copy_(torch.where(active,self.plant.pad_contact.to(torch.float64),self.obs_pad))
            term=fatal|bounds;trunc=timeout|complete
            self.reward.copy_(torch.where(active,reward,self.reward))
            self.terminated|=term&active;self.truncated|=trunc&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active

        self.reset_buf=self.terminated|self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        landed=self.reset_buf&self.touch_flag
        self.completed_metrics['pad_touch_rate']+=landed.to(torch.float64)
        self.completed_metrics['touch_speed']+=torch.where(landed,self.touch_speed,0.)
        self.completed_metrics['touch_ang_vel']+=torch.where(landed,self.touch_ang_vel,0.)
        self.completed_metrics['touch_tilt']+=torch.where(landed,self.touch_tilt,0.)
        done=self.reset_buf

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        if bool(done.any()):
            episode={'success_rate':(done&self.complete).to(torch.float64).sum()/done.sum().clamp_min(1),
                     'pad_touch_rate':landed.to(torch.float64).sum()/done.sum().clamp_min(1)}
            touched=done&self.touch_flag
            if bool(touched.any()):
                n=touched.to(torch.float64).sum()
                episode['touch_speed']=(self.touch_speed*touched).sum()/n
                episode['touch_ang_vel']=(self.touch_ang_vel*touched).sum()/n
                episode['touch_tilt']=(self.touch_tilt*touched).sum()/n
            self.extras['episode']=episode
        else:
            self.extras.pop('episode',None)

        if bool(self.reset_buf.any()):self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def _observe(self,mask):
        p,q,v,w=self.plant.state()
        rpy=euler(q);wb=inverse_rotate(q,w);vb=inverse_rotate(q,v)
        attitude=torch.cat((wb,q[:,1:],q[:,:1],vb,p,self.action,self.plant.auxiliary()),-1)
        observation=torch.cat((attitude,self.obs_pad[:,None]),-1)
        if self._observation is None:self._observation=observation.clone()
        else:self._observation.copy_(torch.where(mask[:,None],observation,self._observation))
        return p,rpy,v,wb,vb

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask.

        Upstream ICs (rocket_landing_env.reset -> begin_reset with
        randomize_drop/accelerate_drop): xy uniform in +/-20 m, z uniform in
        [400, 450], euler orn uniform in +/-0.3 rad/axis, vz = -100 m/s, no
        angular velocity. The upstream random linear/angular velocities are
        dead code there ("randoimize_drop" typo) and are not reproduced.
        """
        mask=envs_idx
        p,q,v,w=self.plant.state()
        u=self.plant.rng.uniform(6,mask)
        spawn=torch.zeros_like(p)
        spawn[:,0]=(u[:,0]*2.-1.)*20.
        spawn[:,1]=(u[:,1]*2.-1.)*20.
        spawn[:,2]=400.+u[:,2]*50.
        rotation=quat_from_euler((u[:,3:]*2.-1.)*0.3)
        velocity=torch.zeros_like(v);velocity[:,2]=-100.
        omega=torch.zeros_like(w)
        self.plant.set_state(torch.where(mask[:,None],spawn,p),torch.where(mask[:,None],rotation,q),
                             torch.where(mask[:,None],velocity,v),torch.where(mask[:,None],omega,w))
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.collision.masked_fill_(mask,False);self.out_of_bounds.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False);self.fatal_collision.masked_fill_(mask,False)
        self.touch_flag.masked_fill_(mask,False)
        self.touch_speed.masked_fill_(mask,0);self.touch_ang_vel.masked_fill_(mask,0);self.touch_tilt.masked_fill_(mask,0)
        self.action.masked_fill_(mask[:,None],0)
        self.plant.zero(mask)
        self.obs_pad.masked_fill_(mask,0)
        if bool(mask.any()):self.plant.warmup(mask)
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
