"""Spacecraft rendezvous & docking RL environments in the official Genesis
RL environment format.

Task reference: the SpaceGym docking / rendezvous benchmarks (Supreeth &
Mehta, "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement
Learning in Spacecraft Proximity Operations", CVPR 2026 Workshop on
AI4Space, CC BY 4.0). No public reference implementation exists, so both
tasks are reproduced parameter-for-parameter from the paper's Section 3.

Dynamics: Clohessy-Wiltshire-Hill linearized relative motion in the LVLH
frame of a circular 400 km target orbit (a = 6778 km, m = 500 kg fixed,
mu = 3.986e14, n ~ 1.13e-3 rad/s), integrated with the paper's symplectic
Euler scheme at dt = 1 s (Algorithm 1), fully tensorized on one device
(no engine, no gym).

Docking: final approach from a 50-100 m separation; success requires
< 1 m and < 0.1 m/s; failure beyond 200 m range or fuel exhaustion
(2 kg budget: the paper's 50 kg cannot bind at Isp=300 s, where a
full-thrust episode consumes < 3 kg); 500-step limit. Rendezvous:
medium-range approach from 800-1200 m to a < 10 m & < 0.5 m/s goal while
avoiding a 5 m radius keep-out sphere; failure on keep-out violation,
fuel depletion (4 kg budget, same rescaling), or 2 km range; 1000-step
limit. Masked (inactive) replicas are frozen and consume no random
numbers.

Both tasks use the telescoping potential-shaping reward
(`reward_mode='progress'`) instead of the paper's Eq. 4/5 dense shaping:
the dense penalty is not telescoping, which makes early termination
cheaper than a careful approach (for docking on the 500-step horizon an
early range-fail outscores hovering to timeout; for rendezvous on the
1000-step horizon it also stars the terminal bonus of any discounted
credit, gamma^1000 ~ 4e-5). The progress mode rewards the stepwise
decrease of a brake-distance-augmented range potential
Phi = -(d + |v|^2 / (2 a_ref)) / d0 -- the episode sum telescopes to
Phi_T - Phi_0 regardless of path length, and the quadratic velocity term
steers the gradient along the maximum-brake approach profile instead of
crashing into the target. Violations then cost a one-time terminal
penalty; successes keep the 100-point terminal bonus.
"""
import math
import torch
from tensordict import TensorDict
from ..common.plant import TensorRandom
from ..common.baselines import (DockingCWHLQRBaseline,
                                RendezvousTwoImpulseBaseline)


class _CWHBase:
    """Shared CWH relative-motion machinery; subclasses set the task table."""

    # Task table (SpaceGym Table 1); overridden per task.
    max_steps=500
    init_range=(50.,100.)     # initial |p0| sampling range [m]
    init_vmax=.5              # initial |v0| upper bound [m/s]
    pos_goal=1.               # success: |p| < pos_goal [m]
    vel_goal=.1               # success: |v| < vel_goal [m/s]
    range_fail=200.           # failure: |p| > range_fail [m]
    fuel0=50.                 # fuel budget [kg]; subclasses rescale so it can bind
    keep_out=None             # failure+penalty sphere radius [m] (rendezvous)
    keep_out_penalty=10.
    pos_norm=100.             # reward position normalizer [m]
    vel_norm=1.               # reward velocity normalizer [m/s]
    reward_mode='dense'       # 'dense' (paper Eq. 4/5) | 'progress' (telescoping potential)
    progress_weight=100.      # progress-mode shaping scale
    viol_penalty=100.         # progress-mode one-time terminal violation cost
    brake_ref=None            # progress-mode reference decel [m/s^2]; resolved in __init__
    # Difficulty axis (0 easiest -> 1 hardest = the benchmark table): the
    # |p0| sampling range interpolates linearly from init_range_start (d=0)
    # to init_range_final (d=1; the benchmark table when unset) and |v0| from
    # init_vmax_start when the task defines one. Success criteria (goal, keep-
    # out, horizon) are untouched.
    init_range_start=None     # easy |p0| sampling range [m] (difficulty 0)
    init_range_final=None     # benchmark |p0| sampling range [m] (difficulty 1)
    init_vmax_start=None      # easy |v0| bound [m/s]; None leaves init_vmax fixed
    benchmark_difficulty=1.
    difficulty=None           # None = benchmark behavior (no cfg record)

    def __init__(self,num_envs=64,backend='cpu',seed=7,difficulty=None):
        if num_envs<1:raise ValueError('n must be positive')
        if backend not in ('cpu','gpu'):raise ValueError(backend)
        self.num_envs=num_envs
        self.task_cfg={'vehicle':self.vehicle,'num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized',
                       'init_range':list(self.init_range),
                       'task_defaults':'SpaceGym benchmark (CVPR 2026 W AI4Space), Section 3'}
        # Resolve the concrete device (with index) so action-device checks
        # compare equal against tensors created on the default device.
        self.device=str(torch.empty(0,device='cuda' if backend=='gpu' else 'cpu').device)
        d=self.device
        # Shared dynamics constants (paper Section 3.1).
        self.mu=3.986e14         # Earth gravitational parameter [m^3/s^2]
        self.sma=6.778e6         # 400 km circular target orbit [m]
        self.mass=500.           # chaser mass [kg]
        self.f_max=10.           # per-axis thrust limit [N]
        self.isp=300.            # RCS specific impulse [s] (paper leaves Isp symbolic)
        self.g0=9.80665           # standard gravity [m/s^2]
        self.n=math.sqrt(self.mu/self.sma**3)
        self.dt=1.                # integration step [s], Algorithm 1
        # progress-mode reference decel: 60% of the per-axis thrust authority,
        # matching the brake profile proven feasible by the scripted-guidance check
        self.brake_ref=.6*self.f_max/self.mass if self.brake_ref is None else self.brake_ref
        self.max_episode_length=self.max_steps+2
        self.num_actions=3
        self.clip_actions=1.
        # Reward weights (paper Eq. 4/5).
        self.alpha=1.;self.beta=.1;self.gamma=.01;self.terminal_bonus=100.

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=d,dtype=torch.float64)
        self.p=torch.zeros((num_envs,3),device=d,dtype=torch.float64)
        self.v=torch.zeros_like(self.p)
        self.fuel=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.d0=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.reward=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.count=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.terminated=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.complete=self.terminated.clone()
        self._observation=None
        self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=d,dtype=torch.float32)
        self.reset_buf=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.returns=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.completed_returns=torch.zeros_like(self.returns)
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)

        self.extras=dict()
        self.cfg=dict(self.task_cfg)
        self.rng=TensorRandom(num_envs,d,seed)
        # difficulty resolution; init_range_final falls back to the benchmark
        # class table when the task doesn't pin one
        self.init_range_benchmark=tuple(self.init_range)
        self.init_vmax_benchmark=self.init_vmax
        if difficulty is not None:
            if not 0.<=difficulty<=1.:raise ValueError('difficulty must be in [0,1]')
            self.difficulty=float(difficulty)
        if self.difficulty is not None:self._apply_difficulty(self.difficulty)

        self.reset()

    def _apply_difficulty(self,d):
        """Map difficulty in [0,1] onto the spawn distribution (range + vmax)."""
        self.difficulty=d
        lo0,hi0=self.init_range_start if self.init_range_start is not None else self.init_range_benchmark
        lo1,hi1=self.init_range_final if self.init_range_final is not None else self.init_range_benchmark
        self.init_range=(lo0+(lo1-lo0)*d,hi0+(hi1-hi0)*d)
        if self.init_vmax_start is not None:
            self.init_vmax=self.init_vmax_start+(self.init_vmax_benchmark-self.init_vmax_start)*d
        self.cfg['init_range']=list(self.init_range)
        self.cfg['difficulty']=d

    def _accel(self,p,v,thrust):
        """CWH accelerations (paper Eqs. 1-3): x radial, y along-track, z cross-track."""
        n2=self.n*self.n
        return torch.stack((3*n2*p[:,0]+2*self.n*v[:,1]+thrust[:,0]/self.mass,
                            -2*self.n*v[:,0]+thrust[:,1]/self.mass,
                            -n2*p[:,2]+thrust[:,2]/self.mass),-1)

    def _integrate(self,thrust,mask):
        """Symplectic Euler exactly as the paper's Algorithm 1."""
        acc=self._accel(self.p,self.v,thrust)
        vn=self.v+acc*self.dt
        pn=self.p+vn*self.dt
        fuel=thrust.norm(dim=-1)*self.dt/(self.isp*self.g0)
        self.v.copy_(torch.where(mask[:,None],vn,self.v))
        self.p.copy_(torch.where(mask[:,None],pn,self.p))
        self.fuel.add_(torch.where(mask,fuel,torch.zeros_like(fuel)))

    def _observe(self,mask):
        obs=torch.cat((self.p,self.v),-1)
        if self._observation is None:self._observation=obs.clone()
        else:self._observation.copy_(torch.where(mask[:,None],obs,self._observation))

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        thrust=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)*self.f_max
        self.action.copy_(thrust)
        active=~(self.terminated|self.truncated)
        dist_pre=self.p.norm(dim=-1)
        v2_pre=(self.v*self.v).sum(-1)
        self._integrate(thrust,active)
        self._observe(active)

        dist=self.p.norm(dim=-1);speed=self.v.norm(dim=-1);f_norm=thrust.norm(dim=-1)
        pos_norm=self.pos_norm if self.pos_norm is not None else self.d0
        if self.reward_mode=='progress':
            # telescoping shaping on the brake-distance-augmented range; the
            # |v|^2 term is unchanged by full-authority closing thrust, so the
            # reward gradient follows the maximum-brake approach profile
            deff_pre=dist_pre+v2_pre/(2*self.brake_ref)
            deff_post=dist+(self.v*self.v).sum(-1)/(2*self.brake_ref)
            reward=self.progress_weight*(deff_pre-deff_post)/pos_norm-self.gamma*f_norm/self.f_max
        else:
            reward=-(self.alpha*dist/pos_norm+self.beta*speed/self.vel_norm
                     +self.gamma*f_norm/self.f_max)
            if self.keep_out is not None:
                inside=dist<self.keep_out
                reward=reward-self.keep_out_penalty*inside.to(torch.float64)
        violation=(self.fuel>=self.fuel0)|(dist>self.range_fail)
        if self.keep_out is not None:
            violation=violation|(dist<self.keep_out)
        # Termination priority (frozen 2026-10-22): a constraint violation on
        # the terminal step is a failure, never a success. This matters for
        # rendezvous because the 5 m keep-out sphere lies INSIDE the 10 m goal
        # ball -- a capture-speed penetration of the no-fly zone would
        # otherwise satisfy (dist, speed) and be counted as a success while
        # also flagging terminated; the violation now wins and the episode is
        # a keep-out failure (no terminal bonus, no completed_successes).
        success=(dist<self.pos_goal)&(speed<self.vel_goal)&~violation
        self.count+=1
        timeout=self.count>=self.max_steps
        reward=reward+self.terminal_bonus*success.to(torch.float64)
        if self.reward_mode=='progress':
            # one-time terminal cost replaces the dense-mode per-step keep-out
            # penalty; since success requires a violation-free terminal state,
            # every violating trajectory (including keep-out penetration) pays it
            reward=reward-self.viol_penalty*(violation&~success).to(torch.float64)
        self.reward.copy_(torch.where(active,reward,self.reward))
        self.complete|=success&active
        self.terminated|=violation&active
        self.truncated|=(timeout|success)&active

        self.reset_buf=self.terminated|self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        if bool(done.any()):
            self.extras['episode']={'success_rate':(done&self.complete).to(torch.float64).sum()/done.sum().clamp_min(1),
                                    'init_range_hi':torch.tensor(self.init_range[1],device=self.device,dtype=torch.float64)}
        else:
            self.extras.pop('episode',None)

        self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask=envs_idx
        # Uniform sphere direction for p0, then |p0| in the task range;
        # v0 gets an independent uniform direction and |v0| <= init_vmax.
        u=self.rng.uniform(6,mask)
        z_p=1-2*u[:,0];r_p=torch.sqrt((1-z_p*z_p).clamp_min(0.))
        ang_p=2*math.pi*u[:,1]
        p_dir=torch.stack((r_p*ang_p.cos(),r_p*ang_p.sin(),z_p),-1)
        z_v=1-2*u[:,3];r_v=torch.sqrt((1-z_v*z_v).clamp_min(0.))
        ang_v=2*math.pi*u[:,4]
        v_dir=torch.stack((r_v*ang_v.cos(),r_v*ang_v.sin(),z_v),-1)
        p0=p_dir*(self.init_range[0]+(self.init_range[1]-self.init_range[0])*u[:,2,None])
        v0=v_dir*(self.init_vmax*u[:,5,None])
        self.p.copy_(torch.where(mask[:,None],p0,self.p))
        self.v.copy_(torch.where(mask[:,None],v0,self.v))
        self.d0.copy_(torch.where(mask,p0.norm(dim=-1),self.d0))
        self.fuel.masked_fill_(mask,0)
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False)
        self.action.masked_fill_(mask[:,None],0)
        self.reward.masked_fill_(mask,0)
        self._observe(mask)
        self.episode_length_buf.masked_fill_(mask,0)
        self.returns.masked_fill_(mask,0)

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):
        """No engine resources to release; kept for API parity."""


class SpaceDockingEnv(_CWHBase):
    """SpaceGym docking: close-range final approach (Table 1, Section 3.1).

    Uses the telescoping progress reward (see module docstring). The paper's
    dense shaping was diagnosed as untrainable here: the absolute distance
    term makes an early range-fail (suicide) outscore hovering to timeout
    (~ -150 vs ~ -500 per episode), pinning PPO at that degenerate optimum,
    and the KL-adaptive lr raced to its 1e-2 ceiling while the policy
    collapsed. The progress potential removes the time pathology (hover ~0,
    closing +, retreating -, round trips net 0) and its brake-distance term
    steers the approach along a maximum-brake profile whose envelope lands
    near the vel_goal at the goal distance.
    """
    vehicle='docking'
    max_steps=500
    init_range=(50.,100.)
    init_vmax=.5
    # difficulty-0 easy endpoints: successes dense enough to bootstrap the
    # terminal braking maneuver before the benchmark 50-100 m / 0.5 m/s ramp
    init_range_start=(15.,30.)
    init_vmax_start=.1
    pos_goal=1.
    vel_goal=.1
    range_fail=200.
    fuel0=2.                  # paper's 50 kg never binds at Isp=300 s (max ~2.9 kg over 500 steps)
    keep_out=None
    pos_norm=100.
    vel_norm=1.
    reward_mode='progress'
    # nominal controller: CWH-LQR terminal approach with range shield and
    # capture net (baselines.py; exact symplectic discretization of the plant)
    nominal_controller=DockingCWHLQRBaseline


class SpaceRendezvousEnv(_CWHBase):
    """SpaceGym rendezvous: medium-range approach with keep-out zone
    (Table 1, Section 3.2); the reward normalizer is the per-episode
    initial separation d0. Uses the telescoping progress reward (see
    module docstring): the paper's dense shaping structurally rewards
    early termination over a 1000-step horizon and never trains."""
    vehicle='rendezvous'
    max_steps=1000
    init_range=(800.,1200.)          # benchmark range; eval always uses this
    init_range_final=(800.,1200.)
    # Difficulty-0 easy endpoints: 150-300 m is the docking-scale regime where
    # the terminal braking maneuver is reachable by exploration; difficulty 1
    # restores the 800-1200 m benchmark range.
    init_range_start=(150.,300.)
    init_vmax=1.
    pos_goal=10.
    vel_goal=.5
    range_fail=2000.
    fuel0=4.                  # ~5.9 kg max over 1000 steps; same ~68% full-thrust rescaling
    keep_out=5.
    keep_out_penalty=10.
    pos_norm=None   # per-episode d0
    vel_norm=1.
    reward_mode='progress'
    # brake-distance reference decel for the progress potential. The base-class
    # default (0.6 * per-axis authority) reproduces the scripted-guidance brake
    # profile but makes d_eff flat under full-authority closing thrust: the
    # accelerate-toward-target phase earns zero shaping and GAE(lambda) can
    # never bridge the gap, so PPO stalls in a slow-creep local optimum.
    # 0.1 m/s^2 keeps 80% progress credit during acceleration (soft dead-zone)
    # while still penalizing unrecoverable approach speeds near the target.
    brake_ref=.1
    # 20-point terminal bonus/violation spread (base class uses 100). The
    # critic cannot predict success-vs-timeout outcomes near the decision
    # boundary, so its uncertainty enters the advantages scaled by the bonus
    # magnitude; with a 100-point spread PPO booms to ~0.3-0.9 success then
    # the advantage noise slowly erodes the maneuver. A 20-point spread keeps
    # the same incentive ordering (approach shaping dominates) at 1/5 of the
    # churn, giving monotonic learning curves.
    terminal_bonus=20.
    # nominal controller: receding-horizon CWH two-impulse guidance to the
    # hold point on the initial line of sight (baselines.py; exact symplectic
    # STM, online transfer-time ratchet)
    nominal_controller=RendezvousTwoImpulseBaseline
    viol_penalty=20.
