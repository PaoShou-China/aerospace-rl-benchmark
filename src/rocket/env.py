"""Rocket-Landing task: upstream-faithful port of PyFlyt 0.29.0.

Task definition (reward, dynamics, episode logic, initial conditions) follows
gym_envs/rocket_envs/rocket_landing_env.py + rocket_base_env.py of the bundled
original_pyflyt reference verbatim; only the physics engine differs (Genesis
rigid-body solver vs PyBullet), per the 2026-10-20 alignment decision: "the
physics engine may differ, the reward function and dynamics model must match".

Concretely, relative to the previous staged port (all measured/documented
there), these upstream behaviours are KEPT:
  * Pad contact does NOT terminate the episode (upstream L231-263): on a
    contact inner step the rocket is fatal iff the PREVIOUS inner-step speed
    exceeds 1.0 m/s or 0.35 rad/s; it "lands" (success) iff that
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

Reward (2026-10-22: the 2026-10-21 'v2' redesign is now the ONLY arm; the
verbatim-upstream shaping was deleted after a full 1000-update run converged
to a reward-farming local optimum on it -- hover on the pad axis farming the
absolute holding term, fuel out at ~step 305, pad hit at ~40 m/s, success 0).
Two of the upstream dense terms are replaced and the stay-airborne cost is
raised (see the reward formula below for the 2026-10-22 rationale); the
remaining terms are untouched:
  * deceleration 4*(dvz+1)/exp(z)*sign(vz) is numerically dead above ~10 m
    (the suicide burn must start at z ~150-300 m), pays free fall through the
    +1 offset, and has a sign-flip cliff on bounce. Replaced by an
    impact-speed potential  brake = KB*(Phi_prev - Phi),
    Phi = sqrt(vz^2 + 2*9.81*z): the speed the booster would hit the pad with
    if it braked at g from here -- at A_REF=g, Phi^2 IS the mechanical energy
    per unit mass, so free fall conserves Phi exactly and pays zero (the real
    usable net deceleration is ~17-46 m/s^2 with drag, so Phi overestimates
    the danger; as a potential only the gradient direction matters).
    Delta-Phi telescopes, is alive at every altitude, rewards braking and
    descending alike, and has no sign-flip cliff.
  * holding +0.3/(||p_xy||+0.1) is absolute and farmable by hovering on the
    pad axis. Replaced by an off-axis cost -KR*||p_xy|| (non-farmable
    centring pressure).
  * terminal: the per-contact-step +5-0.3|vz| becomes a FIRST-touch-only
    quality bonus TOUCH_BONUS*max(0, 1 - v_first) (continuous gradient on
    touch speed, nothing to farm by sitting on the pad), and success
    +3 -> SUCCESS_BONUS=+10 per inner step WHILE SETTLED on the pad: a
    success no longer truncates -- the episode keeps running to the
    1200-step time limit and the settled stream (up to 10*3600 inner steps)
    dominates the dense shaping, so landing softly EARLY is strictly the
    best action. This replaces the earlier +100 one-shot design.
Every dense component is accumulated per replica and reported as
Episode/rew_* tensorboard scalars plus run.json completed_metrics keys, so
reward attribution is directly observable.

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
from ..common.dynamics import euler, inverse_rotate, quat_from_euler
from ..common.baselines import RocketPDGBaseline


A_REF=9.81      # reference braking decel = g exactly: free fall conserves Phi (pays zero)
KB=10.          # impact-speed potential gain (episode total caps ~ 10*Phi_spawn ~ 1350)
KR=0.1          # off-axis cost per metre (replaces the farmable holding term)
TOUCH_BONUS=50. # first-touch quality bonus 50*max(0,1-v_first)
SUCCESS_BONUS=10.  # per-inner-step reward while settled on the pad (paid continuously)


def rocket_shaping(position,previous_position,vel_z,previous_vel_z,rpy,body_omega,landed):
    """Dense reward per inner step, returned as (total, per-component dict).

    vel_z is the WORLD-frame vertical velocity ("ground_lin_vel" upstream,
    which their code computes as body-frame v rotated to world).

    The holding and deceleration terms of the verbatim upstream shaping are
    replaced (rationale in the module docstring) by an impact-speed potential
    and an off-axis cost; the remaining terms are identical to upstream.
    While `landed` (success already achieved, still on the pad) the stay-
    airborne cost is waived: the episode keeps running to the 1200-step time
    limit after success and the settled SUCCESS_BONUS stream takes over.
    """
    lateral_progress=previous_position[:,:2].norm(dim=-1)-position[:,:2].norm(dim=-1)
    vertical_progress=previous_position[:,2]-position[:,2]
    attitude=-body_omega[:,2].abs()-rpy[:,:2].square().sum(-1).sqrt()
    airborne=torch.where(landed,torch.zeros_like(lateral_progress),
                         torch.full_like(lateral_progress,-3.))
    phi=(vel_z.square()+2.*A_REF*position[:,2]).clamp_min(0.).sqrt()
    phi_prev=(previous_vel_z.square()+2.*A_REF*previous_position[:,2]).clamp_min(0.).sqrt()
    brake=KB*(phi_prev-phi)
    off_axis=-KR*position[:,:2].norm(dim=-1)
    total=airborne+brake+10.*lateral_progress+off_axis+.2*vertical_progress+attitude
    components={'airborne':airborne,'brake':brake,'lateral':10.*lateral_progress,
                'off_axis':off_axis,'descent':.2*vertical_progress,'attitude':attitude}
    return total,components


class RocketLandingEnv:
    # Difficulty axis (0 easiest -> 1 hardest = the upstream ICs): the reset
    # ranges narrow toward the nominal drop as d drops; at d=1 the upstream
    # xy +/-20 m, z [400,450], euler +/-0.3 rad are restored bit-for-bit.
    # Success criteria (pad contact thresholds) are untouched.
    ic_xy_half=20.           # spawn xy half-range [m] (axis 20 -> 2)
    ic_z_min=400.            # spawn z lower bound [m] (axis 400 -> 440, span 50 fixed)
    ic_euler_half=.3         # spawn euler per-axis half-range [rad] (axis .3 -> .05)
    benchmark_difficulty=1.
    difficulty=None          # None = benchmark behavior (no cfg record)
    # nominal controller: ZEM/ZEV powered-descent guidance + TVC/fin attitude
    # PD (baselines.py; reuses the env's raw 7-dim action convention)
    nominal_controller=RocketPDGBaseline

    def __init__(self,num_envs=64,backend='cpu',seed=7,difficulty=None):
        self.num_envs=num_envs
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
        # per-component reward accounting: Episode/rew_* tensorboard scalars and
        # run.json completed_metrics keys, so reward attribution is observable
        keys=('airborne','brake','lateral','off_axis','descent','attitude','touch','success')
        self.reward_components={k:torch.zeros(num_envs,device=self.device,dtype=torch.float64) for k in keys}
        for k in keys:self.completed_metrics[f'rew_{k}']=torch.zeros_like(self.touch_speed)

        self.extras=dict()
        self.cfg=dict(self.task_cfg)
        self._resolve_difficulty(difficulty)

        self.reset()

    def _resolve_difficulty(self,difficulty):
        """Shared difficulty resolution (every task env, same shape)."""
        if difficulty is not None:
            if not 0.<=difficulty<=1.:raise ValueError('difficulty must be in [0,1]')
            self.difficulty=float(difficulty)
        if self.difficulty is not None:self._apply_difficulty(self.difficulty)

    def _apply_difficulty(self,d):
        """Map difficulty in [0,1] onto the initial-condition knobs."""
        self.difficulty=d
        self.ic_xy_half=2.+18.*d
        self.ic_z_min=440.-40.*d
        self.ic_euler_half=.05+.25*d
        self.cfg['difficulty']=d

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
            # landed mask from PREVIOUS inner steps: once a success has been
            # achieved the episode keeps running to the time limit, the
            # stay-airborne cost is waived and LANDED_BONUS is paid per step
            landed=self.complete&active
            shaping,comps=rocket_shaping(p,p0,v[:,2],v0[:,2],rpy,wb,landed)
            reward=self.reward+shaping
            for k in comps:self.reward_components[k]+=torch.where(active,comps[k],0.)
            touch=self.plant.pad_contact&active
            speed0=vb0.norm(dim=-1);spin0=wb0.norm(dim=-1);tilt0=rpy[:,:2].norm(dim=-1)
            # Upstream L231-263: on a pad-contact step, pay +5-0.3|vz| (upstream)
            # or a first-touch-only quality bonus (v2), then judge the PREVIOUS
            # inner-step speed magnitude (= speed0/spin0 here, the state at the
            # start of this inner step): >1.0 m/s or >0.35 rad/s is fatal;
            # <0.02/<0.02 with tilt<0.1 is a landed success; anything in between
            # keeps simulating (on-pad settling).
            fatal=collision|(touch&((speed0>1.)|(spin0>.35)))
            complete=touch&~fatal&(spin0<.02)&(speed0<.02)&(tilt0<.1)
            new_touch=touch&~self.touch_flag
            # first-touch-only bonus, continuous in the touch speed
            # (speed0 = pre-contact inner-step speed, the same value the
            # fatal/settle judgement uses); unfarmable by sitting on the pad
            touch_r=torch.where(new_touch,(TOUCH_BONUS*(1.-speed0)).clamp_min(0.),0.)
            # success stream: +SUCCESS_BONUS per inner step while settled on
            # the pad (complete stays true at rest; the episode runs to the
            # time limit, so landing softly early maximizes this stream)
            succ_r=complete.to(torch.float64)*SUCCESS_BONUS
            reward+=touch_r
            reward+=succ_r
            self.reward_components['touch']+=torch.where(active,touch_r,0.)
            self.reward_components['success']+=torch.where(active,succ_r,0.)
            self.touch_flag|=new_touch
            self.touch_speed.copy_(torch.where(new_touch,speed0,self.touch_speed))
            self.touch_ang_vel.copy_(torch.where(new_touch,spin0,self.touch_ang_vel))
            self.touch_tilt.copy_(torch.where(new_touch,tilt0,self.touch_tilt))
            self.complete|=complete&active;self.fatal_collision|=fatal&active
            self.obs_pad.copy_(torch.where(active,self.plant.pad_contact.to(torch.float64),self.obs_pad))
            term=fatal|bounds
            # success does NOT truncate (2026-10-22): the rocket keeps earning
            # the settled SUCCESS_BONUS stream until the 1200-step time limit
            # below
            trunc=torch.zeros_like(fatal)
            self.reward.copy_(torch.where(active,reward,self.reward))
            self.terminated|=term&active;self.truncated|=trunc&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents still
        # terminate inside the substep loop above. This is the ONLY truncation:
        # a landed rocket keeps simulating (and earning) until here.
        active=~(self.terminated|self.truncated)
        self.truncated|=(self.count>=self.max_steps)&active

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
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any=bool(done.any())
        if done_any:
            episode={'success_rate':(done&self.complete).to(torch.float64).sum()/done.sum().clamp_min(1),
                     'pad_touch_rate':landed.to(torch.float64).sum()/done.sum().clamp_min(1)}
            touched=done&self.touch_flag
            n=touched.to(torch.float64).sum()
            # torch.where instead of a bool(touched.any()) sync: when no done
            # replica has touched the pad the masked sums are 0, so report 0
            zero=torch.zeros((),device=self.device,dtype=torch.float64)
            episode['touch_speed']=torch.where(n>0,(self.touch_speed*touched).sum()/n,zero)
            episode['touch_ang_vel']=torch.where(n>0,(self.touch_ang_vel*touched).sum()/n,zero)
            episode['touch_tilt']=torch.where(n>0,(self.touch_tilt*touched).sum()/n,zero)
            for k,buf in self.reward_components.items():
                episode[f'rew_{k}']=(buf*done).sum()/done.sum().clamp_min(1)
            self.extras['episode']=episode
        else:
            self.extras.pop('episode',None)

        # empty-mask reset is fully safe (warmup early-exits; pause-restore is
        # unconditional); the gate reuses the already-synced done_any for cost
        if done_any:self.reset_idx(self.reset_buf)
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
        The difficulty axis (see the class block) shrinks the same ranges
        toward the nominal drop: xy +/-2 m, z [440, 450], euler +/-0.05 rad.
        """
        mask=envs_idx
        p,q,v,w=self.plant.state()
        u=self.plant.rng.uniform(6,mask)
        spawn=torch.zeros_like(p)
        spawn[:,0]=(u[:,0]*2.-1.)*self.ic_xy_half
        spawn[:,1]=(u[:,1]*2.-1.)*self.ic_xy_half
        spawn[:,2]=self.ic_z_min+u[:,2]*(450.-self.ic_z_min)
        rotation=quat_from_euler((u[:,3:]*2.-1.)*self.ic_euler_half)
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
        for buf in self.reward_components.values():buf.masked_fill_(mask,0.)
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
