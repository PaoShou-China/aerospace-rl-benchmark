"""Rocket-Landing task: six-spherical-pad contact model with a sustained-landing
(2 s) primary success metric; see docs/rocket_benchmark_v2_zh.md.

The instantaneous touch event below is still used for reward/diagnostics.
Contact model and scoring do not change reward, fatal thresholds, initial
distribution or interfaces.
"""
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
  * Action box is the upstream 7-dim finlet mixer convention: dims 0-2 are
    the finlet (x, y, yaw) demands in [-1,1] and the plant mixes them onto
    the four grid-fin hinge servos; ignition (dim 3) and throttle (dim 4)
    in [0,1] -- uniform exploration fires the engine on ~50% of steps,
    exactly like upstream; gimbals (dims 5-6) in [-1,1].

Reward (2026-10-24 'v4', retimed 2026-10-27 to the 40 Hz policy): dense
per-POLICY-step shaping replacing the v2 potential design, which a 1000-update
run farmed by minimum-speed descent on the brake potential -- brake +1238 of
reward, pad_touch_rate 0. The v4 coefficients come from a discounted-return
analysis at dt=0.02 s (50 Hz), gamma 0.995, B=5000. The analysed 50 Hz does
NOT divide the 120 Hz control loop (upstream rejects non-divisor agent rates),
so the policy runs at the upstream rocket_landing default 40 Hz (dt 0.025 s):
the high-speed-dive bound admits DESCENT_COEF=0.1 up to
u = HOVER/(DESCENT*dt) = 120 m/s, still above the 100 m/s drop speed:
  r = -HOVER_COST                                time pressure, always
      + HOLD_COEF/(||p_xy||+0.1)                 <= 0.05, cannot farm hover
      + LATERAL_COEF*(d_xy_prev - d_xy)          telescoping pad approach
      + DESCENT_COEF*(z_prev - z)                0.1: a pure dive stays
                                                 net-negative to u<=150 m/s
      + DECEL_COEF*(dvz+1)/e^z*sign(vz)          upstream decel term, 4->0.1
      - YAW_COEF*|w_yaw| - TILT_COEF*||rpy_xy||
      + contact: CONTACT_BASE - CONTACT_VEL_COEF*|vz| per step with any
        pad contact (quasi-static settle nets ~+0.1..0.35/step)
      + SUCCESS_BONUS once on first settle (5000 >> gamma*c/(1-gamma)~70,
        so delaying a reachable success is never profitable)
Success still does NOT truncate; the settled contact stream keeps paying.

Port details: 40 Hz policy / 120 Hz control / 240 Hz physics (all rates read
from the plant, none hardcoded), 1200 policy steps (30 s) truncation, bounds
||p_xy||>200 or z>500, any
non-pad contact or z<0 is fatal, obs = [body omega, quat, body v, p,
action (7), aux(9), pad contact flag] (30 dims), 5% starting fuel, reignitable
engine with the per-physics-step ignition>0.5 rule.

Success: pad contact with |v|<0.02, |omega|<0.02, tilt<0.1 (upstream values).
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import euler, inverse_rotate, quat_from_euler
from algorithms.baselines.baselines import RocketPDGBaseline


HOVER_COST=0.3      # per policy step, always (time pressure)
HOLD_COEF=0.005     # holding reward HOLD_COEF/(||p_xy||+0.1), max 0.05 -- too
                    # small to farm against the hover cost
LATERAL_COEF=10.    # telescoping lateral progress toward the pad axis
DESCENT_COEF=0.1    # descent progress (0.2 -> 0.1 by the dive analysis: a pure
                    # vertical dive then stays net-negative up to 150 m/s)
DECEL_COEF=0.1      # upstream deceleration term (dvz+1)/e^z*sign(vz), 4 -> 0.1
YAW_COEF=0.5        # -YAW_COEF*|body yaw rate|
TILT_COEF=0.5       # -TILT_COEF*||roll/pitch||
CONTACT_BASE=0.5    # per policy step with any pad contact
CONTACT_VEL_COEF=1. # -CONTACT_VEL_COEF*|vz| on contact steps
SUCCESS_BONUS=5000. # one-shot on first settle-success (B >> gamma*c/(1-gamma))

REWARD_KEYS=('hover','hold','lateral','descent','decel','attitude','contact','success')


def rocket_shaping(cfg,p,previous_position,vel_z,previous_vel_z,rpy,body_omega,
                   contact,contact_vz,newly_complete):
    """Dense reward per POLICY step, returned as (total, per-component dict).

    vel_z is the WORLD-frame vertical velocity. `contact` marks any pad
    contact during this policy step, `contact_vz` the world descent speed at
    the first contact inner step; `newly_complete` fires once per episode on
    the settle-success event. All coefficients come from cfg (see the module
    constants); computed once per policy step over the whole-step deltas.
    """
    dxy=p[:,:2].norm(dim=-1)
    dxy_prev=previous_position[:,:2].norm(dim=-1)
    dvz=vel_z-previous_vel_z
    decel=(dvz+1.)*torch.exp(-p[:,2].clamp_min(0.))*torch.where(vel_z<0.,1.,-1.)
    comps={
        'hover':torch.full_like(dxy,-cfg['hover']),
        'hold':cfg['hold']/(dxy+0.1),
        'lateral':cfg['lateral']*(dxy_prev-dxy),
        'descent':cfg['descent']*(previous_position[:,2]-p[:,2]),
        'decel':cfg['decel']*decel,
        'attitude':-cfg['yaw']*body_omega[:,2].abs()-cfg['tilt']*rpy[:,:2].square().sum(-1).sqrt(),
        'contact':torch.where(contact,cfg['contact_base']-cfg['contact_vel']*contact_vz.abs(),
                              torch.zeros_like(dxy)),
        'success':newly_complete.to(dxy.dtype)*cfg['success'],
    }
    total=sum(comps.values())
    return total,comps


class RocketLandingEnv:
    # Upstream ICs (unchanged): xy +/-20 m, z [400,450], euler +/-0.3 rad.
    # Success criteria (pad contact thresholds) are untouched. Wind is
    # optional via the steady_wind/dryden ctor kwargs; calm air reproduces
    # the benchmark bit-for-bit.
    ic_xy_half=20.           # spawn xy half-range [m]
    ic_z_min=400.            # spawn z lower bound [m] (span 50 fixed)
    ic_euler_half=.3         # spawn euler per-axis half-range [rad]
    # conventional controller: ZEM/ZEV powered-descent guidance + TVC/fin attitude
    # PD (baselines.py; reuses the env's unified 7-dim finlet mixer convention)
    conventional_controller=RocketPDGBaseline

    def __init__(self,num_envs=64,backend='cpu',seed=7,steady_wind=None,dryden=0.,severity='moderate',
                 reward=None,variation='nominal',agent_hz=None,benchmark_version='v2'):
        if benchmark_version not in ('v1','v2'):raise ValueError(benchmark_version)
        from environments.common.rocket_benchmark import VARIATIONS, LandingMetrics
        if variation not in VARIATIONS:raise ValueError(variation)
        self.benchmark_version=benchmark_version
        # Default agent_hz per benchmark: v2=40 Hz (current Falcon9 upstream),
        # v1=60 Hz (external benchmark). Explicit agent_hz overrides the default.
        if agent_hz is None:agent_hz=60 if benchmark_version=='v1' else 40
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'rocket','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized','task_defaults':'PyFlyt 0.29.0'}
        self.clip_actions=1.0
        # reward coefficients (module defaults; `reward` overrides per-key for
        # the evolution sweep arms, recorded into cfg -> run.json)
        self.reward_cfg=dict(hover=HOVER_COST,hold=HOLD_COEF,lateral=LATERAL_COEF,
                             descent=DESCENT_COEF,decel=DECEL_COEF,yaw=YAW_COEF,
                             tilt=TILT_COEF,contact_base=CONTACT_BASE,
                             contact_vel=CONTACT_VEL_COEF,success=SUCCESS_BONUS)
        if reward:self.reward_cfg.update(reward)

        self.plant=TensorPlant('rocket',num_envs,backend,seed,
                               rocket_variation=variation,rocket_benchmark=benchmark_version)
        self.device=self.plant.device
        # agent_hz validation: must divide the plant control loop. All rates
        # are read from the plant, never hardcoded here.
        self.agent_hz=agent_hz
        if agent_hz<=0 or agent_hz!=int(agent_hz) or self.plant.control_hz%agent_hz!=0:
            raise ValueError(f'agent_hz must be a positive divisor of the '
                             f'{self.plant.control_hz} Hz control loop; 50 Hz is not '
                             f'(use 40 or 60)')
        self.inner_steps=int(self.plant.control_hz//agent_hz);self.max_steps=30*agent_hz
        self.landing_metrics=LandingMetrics(num_envs,self.device,agent_hz)
        self.max_episode_length=self.max_steps+2
        # Unified 7-dim upstream finlet mixer convention (matches external benchmark):
        #   finlet x/y/yaw [-1,1], ignition [0,1], throttle [0,1], gimbals [-1,1]
        self.num_actions=7
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
        for key in self.landing_metrics.values(self.complete,self.terminated):
            self.completed_metrics[key]=torch.zeros_like(self.touch_speed)
        # per-component reward accounting: Episode/rew_* tensorboard scalars and
        # run.json completed_metrics keys, so reward attribution is observable
        keys=REWARD_KEYS
        self.reward_components={k:torch.zeros(num_envs,device=self.device,dtype=torch.float64) for k in keys}
        for k in keys:self.completed_metrics[f'rew_{k}']=torch.zeros_like(self.touch_speed)

        self.extras=dict()
        self.cfg=dict(self.task_cfg);self.cfg['reward']=dict(self.reward_cfg)
        self._init_wind(num_envs,seed,steady_wind,dryden,severity)
        self.benchmark_env_kwargs=dict(benchmark_version=benchmark_version,variation=variation,agent_hz=agent_hz)
        from pathlib import Path
        import hashlib
        rocket_model_dir=Path(__file__).parents[1]/'common/original_pyflyt/models/vehicles/rocket'
        model=rocket_model_dir/('rocket_v2.urdf' if benchmark_version=='v2' else 'rocket.urdf')
        primary_metric='ever_stable_2s' if benchmark_version=='v2' else 'ever_success'
        self.cfg['benchmark']=dict(version=benchmark_version,variation=variation,
            variation_ranges=VARIATIONS[variation],variation_seed_offset=104729,policy_hz=agent_hz,
            control_hz=self.plant.control_hz,physics_hz=self.plant.physics_hz,
            horizon_seconds=30,stable_samples=self.landing_metrics.required_samples,
            primary_metric=primary_metric,
            collision_asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            physics_and_success_predicate_modified=(benchmark_version=='v2'),
            integration='native-main-v2',external_wind=self.cfg.get('wind'))

        self.reset()

    def _init_wind(self,n,seed,steady_wind,dryden,severity):
        """Optional wind disturbance (environments/common/disturbances.py).

        Calm air (default) leaves the plant dynamics bit-for-bit unchanged;
        with wind configured, the gust filters advance once per control step
        and the fin/body aerodynamic terms evaluate v_air = v - wind.
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
        # Upstream passes the gym action straight to set_setpoint; the plant
        # applies the ignition>0.5 rule per physics step (240 Hz, reignitable)
        # and the throttle -> [0.39, 1.0] min/max thrust remap (boosters.py
        # L231-233). No extra env-level reshaping.
        command=physical.clone()
        self.action.copy_(command)
        if self._wind is not None:self.plant.wind.copy_(self._wind.step())
        self.touch_flag.fill_(False)
        self.count+=1
        # policy-step snapshots for the per-policy-step reward deltas
        p_start,_,v_start,_=self.plant.state()
        p_start=p_start.clone();vz_start=v_start[:,2].clone()
        complete_start=self.complete.clone()
        contact_any=self.terminated.clone().fill_(False)
        contact_vz=torch.zeros(self.num_envs,device=self.device,dtype=torch.float64)
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            p0,q0,v0,w0=self.plant.state();vb0=inverse_rotate(q0,v0);wb0=inverse_rotate(q0,w0)
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            for _ in range(self.plant.physics_per_control):self.plant.physics_step(command,active)
            p,rpy,v,wb,vb=self._observe(active)
            # Upstream compute_base_term_trunc_reward: any contact other than
            # (rocket, pad) is a fatal collision, as is dropping below ground.
            collision=self.plant.ground_contact|(p[:,2]<0)
            bounds=(p[:,:2].norm(dim=-1)>200)|(p[:,2]>500)
            touch=self.plant.pad_contact&active
            speed0=vb0.norm(dim=-1);spin0=wb0.norm(dim=-1);tilt0=rpy[:,:2].norm(dim=-1)
            # Upstream L231-263: on a pad-contact step judge the PREVIOUS
            # inner-step speed magnitude (= speed0/spin0 here, the state at the
            # start of this inner step): >1.0 m/s or >0.35 rad/s is fatal;
            # <0.02/<0.02 with tilt<0.1 is a landed success; anything in between
            # keeps simulating (on-pad settling).
            fatal=collision|(touch&((speed0>1.)|(spin0>.35)))
            complete=touch&~fatal&(spin0<.02)&(speed0<.02)&(tilt0<.1)
            new_touch=touch&~self.touch_flag
            self.touch_flag|=new_touch
            self.touch_speed.copy_(torch.where(new_touch,speed0,self.touch_speed))
            self.touch_ang_vel.copy_(torch.where(new_touch,spin0,self.touch_ang_vel))
            self.touch_tilt.copy_(torch.where(new_touch,tilt0,self.touch_tilt))
            # per-policy-step contact bookkeeping: any contact while active,
            # and the world descent speed at the FIRST contact inner step
            contact_vz.copy_(torch.where(touch&~contact_any,v0[:,2],contact_vz))
            contact_any|=touch
            self.complete|=complete&active;self.fatal_collision|=fatal&active
            self.obs_pad.copy_(torch.where(active,self.plant.pad_contact.to(torch.float64),self.obs_pad))
            term=fatal|bounds
            # success does NOT truncate: the rocket keeps earning the settled
            # contact stream until the time limit below
            trunc=torch.zeros_like(fatal)
            self.terminated|=term&active;self.truncated|=trunc&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active
        # one dense reward per POLICY step over the whole-step state deltas
        shaping,comps=rocket_shaping(self.reward_cfg,p,p_start,v[:,2],vz_start,rpy,wb,
                                     contact_any,contact_vz,self.complete&~complete_start)
        self.reward.copy_(shaping)
        for k in comps:self.reward_components[k]+=comps[k]
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
        self.landing_metrics.update(self.obs_pad.bool(),v,wb,rpy[:,:2].norm(dim=-1),self.terminated)
        metrics=self.landing_metrics.values(self.complete,self.terminated)
        primary=metrics['ever_stable_2s']
        self.completed_successes+=(self.reset_buf&primary).to(torch.long)
        for key,value in metrics.items():
            self.completed_metrics[key]+=(self.reset_buf&value).to(torch.float64)
        landed=self.reset_buf&self.touch_flag
        self.completed_metrics['pad_touch_rate']+=landed.to(torch.float64)
        self.completed_metrics['touch_speed']+=torch.where(landed,self.touch_speed,0.)
        self.completed_metrics['touch_ang_vel']+=torch.where(landed,self.touch_ang_vel,0.)
        self.completed_metrics['touch_tilt']+=torch.where(landed,self.touch_tilt,0.)
        done=self.reset_buf
        # reward attribution at done: accumulate the per-episode component sums
        for k,buf in self.reward_components.items():
            self.completed_metrics[f'rew_{k}']+=torch.where(done,buf,0.)
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out,trunc_out=self.terminated.clone(),self.truncated.clone()

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any=bool(done.any())
        if done_any:
            episode={'success_rate':(done&primary).to(torch.float64).sum()/done.sum().clamp_min(1),
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
        return self.get_observations(),self.rew_buf,term_out,trunc_out,self.extras

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
        self.landing_metrics.reset(mask)
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
        if self._wind is not None:self._wind.reset(mask)

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):self.plant.close()
