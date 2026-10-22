"""Satellite attitude-control (control moment gyros) RL environment in the
official Genesis RL environment format.

Task reference: torchrl ``SatelliteEnv`` (pytorch/rl, MIT) -- the agent
commands gimbal rates of a 4-CMG pyramid (skew 54.7356 deg) or 6-CMG
orthogonal cluster to slew the bus to a target attitude sampled uniformly
on SO(3) at reset, while avoiding internal cluster singularities via the
manipulability penalty ``-w / (m_norm + eps)`` with ``m_norm = sqrt(det(J J^T)) / h^3``
(Wie, Bong 2008, "Space Vehicle Dynamics and Control", CMG steering chapter).
The soft denominator bounds the penalty (max ``-w/eps``) unlike the torchrl
reference form ``-w / m_norm``, whose -1e5-scale spikes near singular
surfaces destabilize PPO value learning while remaining unavoidable in
optimal trajectories (CMG slews routinely pass near singularities).

Reward shaping (docking/rendezvous ``progress`` convention): the reward is a
telescoping potential-based progress bonus ``Phi_prev - Phi`` with the positive
distance-to-go potential
``Phi = k1*att_err + k1b*att_err*g_near(att_err) + k2*|omega|^2/(2*alpha_ref)*g(att_err)
+ w_sing/(m_norm + eps)``: approaching the target, braking near it, and
escaping low-manipulability gimbal configurations all DECREASE Phi and earn a
positive bonus, while the episode sum telescopes to ``Phi_0 - Phi_T``
regardless of path (Ng et al. 1999; no cross-episode leakage since the bonus
is evaluated on pre-reset states). The ``g`` gates are smooth sigmoids: the
far-field signal is the proven error-progress term alone; the near boost
(k1b) steepens the error gradient inside 0.5 rad -- a run-2 assist diagnostic
showed the capture maneuver is worth +560 return but earned only +0.2/step
shaping under k1=30, invisible under PPO advantage noise -- and the brake
term (k2, active inside 0.8 rad) surcharges arriving fast and rewards
braking (it also carries the dense near-target damping gradient). Termination
(docking convention): entering the success region (att/omega tolerances)
terminates the episode immediately with a one-time settle bonus; a runaway
spin (|omega| > fail_omega) terminates as a failure with a one-time penalty
(the telescoping reward alone is failure-neutral). Earlier revisions ran
every episode the full 1200 steps and paid a per-step Gaussian settle bonus,
which tied the return to the remaining horizon and let a policy hovering
just outside the tolerance farm reward without ever succeeding.

BUG FIX 2026-10-20: the original implementation stored the NEGATIVE potential
``Phi = -(k1*att_err + ...)`` but kept the ``Phi_prev - Phi`` subtraction, so
the shaping sign was inverted -- approaching the target and braking were
PENALIZED, actively teaching fly-throughs
(training plateaued at att_err~0.3-0.6, <=5% success across all 7 prior runs;
the physics itself is correct -- inertial angular momentum conserves to 0.0
drift and single-CMG torque response matches the analytic CMG equation).

Dynamics: reduced-order rigid-body + CMG model, fully tensorized on one
device (no engine, no gym). With rotor speeds pinned and gimbal rates
commanded, the bus obeys ``I wdot = -w x (I w) - h w x sum(r_i) - h sum
(dtheta_i (g_i x r_i))`` integrated with RK4 at 2.5 ms over 10 substeps
per control step (40 Hz control, matching the sibling tasks' agent rate;
the torchrl reference runs the same physics at 100 Hz). Masked (inactive)
replicas are frozen and consume no random numbers.
"""
import math
import torch
from tensordict import TensorDict
from ..common.plant import TensorRandom
from ..common.baselines import NominalActionMixin,SatelliteCMGBaseline


def quat_mul(q1,q2):
    """Hamilton product, (...,4) with (w,x,y,z) layout."""
    w1,x1,y1,z1=q1.unbind(-1);w2,x2,y2,z2=q2.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2,
                        w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2,
                        w1*z2+x1*y2-y1*x2+z1*w2),-1)


def quat_conj(q):
    return torch.cat((q[...,:1],-q[...,1:]),-1)


def quat_log(q,eps=1e-8):
    """Log map of a unit quaternion to a 3-vector axis-angle (shortest arc)."""
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(eps)
    q=q*torch.where(q[...,:1]<0,-1.,1.)
    w=q[...,:1].clamp(-1.,1.);v=q[...,1:]
    v_norm=v.norm(dim=-1,keepdim=True).clamp_min(eps)
    return v/v_norm*(2*torch.atan2(v_norm,w))


def quat_slerp(q0,q1,t):
    """Slerp from q0 to q1 at fraction t (scalar); t >= 1 short-circuits to q1."""
    if t>=1.:return q1
    d=(q0*q1).sum(-1,keepdim=True)
    q1=q1*torch.where(d<0,-1.,1.)
    d=(q0*q1).sum(-1,keepdim=True).clamp(-1.,1.)
    ang=torch.acos(d)
    s=torch.sin(ang).clamp_min(1e-12)
    return q0*(torch.sin((1-t)*ang)/s)+q1*(torch.sin(t*ang)/s)


# Cluster geometries: gimbal axes g (fixed, unit), rows aligned per CMG.
# 4: pyramid, one gimbal per bus face, skew 54.7356 deg = arctan(sqrt 2);
# 6: orthogonal cluster, one CMG per bus face, full rank at theta=0.
_PYRAMID_SKEW=math.radians(54.7356)
_CMG_GIMBAL_AXES={
    4:torch.tensor([[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[1.,0.,0.]]),
    6:torch.tensor([[0.,1.,0.],[0.,0.,1.],[0.,0.,1.],[1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])}

def _rotor_axes_ref(num_cmgs):
    if num_cmgs==4:
        sb,cb=math.sin(_PYRAMID_SKEW),math.cos(_PYRAMID_SKEW)
        return torch.tensor([[sb,0.,cb],[0.,sb,cb],[-sb,0.,cb],[0.,-sb,cb]])
    return torch.tensor([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])


class SatelliteAttitudeEnv(NominalActionMixin):
    # Difficulty axis (0 easiest -> 1 hardest = the torchrl reference): the
    # target attitude is pulled toward the start attitude as d drops (rotation
    # angle cap 0.3 rad -> pi via slerp; at cap=pi the legacy uniform-SO(3)
    # sampling code path runs untouched) and reset noise shrinks to zero.
    # Success criteria (att/omega tolerances) are untouched.
    target_angle_cap=math.pi # target rotation-angle cap [rad] (axis .3 -> pi)
    benchmark_difficulty=1.
    difficulty=None          # None = benchmark behavior (no cfg record)
    # nominal controller: quaternion PD + singularity-robust damped CMG
    # steering, Schaub & Junkins (baselines.py; gimbal-rate commands)
    nominal_controller=SatelliteCMGBaseline

    def __init__(self,num_envs=64,backend='cpu',seed=7,num_cmgs=4,difficulty=None):
        if num_cmgs not in (4,6):raise ValueError('num_cmgs must be 4 or 6')
        if num_envs<1:raise ValueError('n must be positive')
        if backend not in ('cpu','gpu'):raise ValueError(backend)
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'satellite','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized','num_cmgs':num_cmgs,
                       'task_defaults':'torchrl SatelliteEnv reference (Wie 2008 CMG steering)'}
        # Resolve the concrete device (with index) so action-device checks
        # compare equal against tensors created on the default device.
        self.device=str(torch.empty(0,device='cuda' if backend=='gpu' else 'cpu').device)
        d=self.device
        # Reference-model constants (torchrl SatelliteEnv defaults).
        self.clip_actions=1.
        self.action_scale=3.
        self.singularity_weight=.5  # telescoped singularity potential w_sing/(m_norm+soft), bounded in [w/2.05, w/soft]
        self.singularity_soft=.5   # soft denominator keeps the potential bounded (max w_sing/soft)
        self.ctrl_cost_weight=.0005  # full-scale command costs .018/step
        self.progress_shaping_gain=30.  # k1: far-field error-progress potential
        self.near_boost_gain=70.     # k1b: near-field error boost (x100/0.5 total slope
        self.near_gate=.5            #   inside 0.5 rad): run-2 diagnostic showed the
        self.near_gate_soft=.15      #   capture maneuver earns only +0.2/step shaping
                                     #   under k1=30, too weak vs advantage noise
                                     #   (assist differential +560 proved capture is
                                     #   worth ~6x the park equilibrium)
        self.reset_noise_scale=.001
        self.success_att_tol=.05;self.success_omega_tol=.02
        self.settle_bonus=1.         # one-time bonus on the success-terminating step.
                                     # Was a per-step Gaussian state reward: with no
                                     # early termination the episode sum scaled with the
                                     # remaining horizon (~+1195 for an instant settle),
                                     # dominating the telescoped shaping, and a policy
                                     # hovering just OUTSIDE the hard tolerance farmed
                                     # ~+0.5/step for 1200 steps without ever terminating.
                                     # The dense damping gradient near the target is now
                                     # carried by the telescoped brake term (k2), so the
                                     # per-step surrogate is unnecessary.
        self.fail_omega=10.          # failure termination: divergence catch. Legit slews
                                     # peak ~1-3 rad/s (CMG torque authority), so only a
                                     # runaway spin crosses this; failing fast stops bad
                                     # trajectories from polluting the GAE chain.
        self.fail_penalty=10.        # one-time terminal cost on failure; without it the
                                     # telescoping reward is failure-neutral (hover and
                                     # tumble to timeout score the same Phi0-Phi_T).
        self.brake_gain=30.          # k2: near-target brake-distance term, gated by
        self.brake_gate=.8           #   g = sigmoid((brake_gate - att_err)/brake_gate_soft)
        self.brake_gate_soft=.15     #   wide gate: brake shaping across the whole
                                     #   approach (the narrow 0.2 gate let the policy park
                                     #   dead just outside it; greedy eval att_err~0.36)
        self.alpha_ref=1.0           # reference angular accel (rad/s^2) for the brake-distance term
        self.dt=2.5e-3;self.substeps=10  # 40 Hz control, as the sibling tasks
        self.agent_hz=int(round(1/(self.dt*self.substeps)))
        self.max_steps=1200;self.max_episode_length=self.max_steps+2
        # Per-variant physical parameters estimated from the reference
        # satellite_large.xml / satellite_small.xml composite bodies.
        if num_cmgs==4:
            self.rotor_speed=100.;self.rotor_h_dyn=20.  # 0.2 kg m^2 rotor at 100 rad/s
            self.inertia=torch.tensor((75.,60.,113.),device=d,dtype=torch.float64)
        else:
            self.rotor_speed=200.;self.rotor_h_dyn=1.25e-2  # 6.25e-5 kg m^2 rotor at 200 rad/s
            self.inertia=torch.tensor((.055,.055,.085),device=d,dtype=torch.float64)
        self.rotor_h=self.rotor_speed  # proxy unit for the singularity metric
        self.num_gimbals=num_cmgs;self.num_actions=num_cmgs
        self.gimbal_axes=_CMG_GIMBAL_AXES[num_cmgs].to(device=d,dtype=torch.float64)
        self.rotor_axes_ref=_rotor_axes_ref(num_cmgs).to(device=d,dtype=torch.float64)
        self.g_cross_r0=torch.linalg.cross(self.gimbal_axes,self.rotor_axes_ref,dim=-1)
        # Momentum-neutral reset home (2026-10-20 bug fix): the previous reset
        # parked gimbals at theta0~0, where the pyramid has NET rotor momentum
        # Sum r_i(0) = [0,0,4cos(54.7356deg)] (|H0| = h_dyn*2.309 = 46.2 for the
        # 4-CMG). That momentum is exactly conserved (probe-verified to 4e-10),
        # so the bus could rest at the target attitude only if the gimbals ended
        # on Sum r_i(theta) = R(target)^T Sum r_i(0) -- unreachable for ~20% of
        # random targets (206/256 seed-99 inits feasible => hard 80.5% success
        # ceiling) and the cause of the capture-then-drift failure of every
        # controller. A quiescent spacecraft carries ~zero stored momentum:
        # park the gimbals where Sum r_i = 0 exactly -- 4-CMG (pi,0,pi,0):
        # -r1(0)+r2(0)-r3(0)+r4(0) = 0; 6-CMG (pi,pi,pi,0,0,0): the second
        # triplet of axes duplicates the first, so antipodal pairing cancels.
        self.gimbal_home=torch.zeros(self.num_gimbals,device=d,dtype=torch.float64)
        if num_cmgs==4:
            self.gimbal_home[0]=math.pi;self.gimbal_home[2]=math.pi
        elif num_cmgs==6:
            self.gimbal_home[:3]=math.pi

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=d,dtype=torch.float64)
        self.q=torch.zeros((num_envs,4),device=d,dtype=torch.float64);self.q[:,0]=1
        self.omega=torch.zeros((num_envs,3),device=d,dtype=torch.float64)
        self.gimbal=torch.zeros((num_envs,self.num_gimbals),device=d,dtype=torch.float64)
        self.gimbal_rate=torch.zeros_like(self.gimbal)
        self.target_quat=torch.zeros_like(self.q);self.target_quat[:,0]=1
        self.reward=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self._prev_phi=torch.zeros(num_envs,device=d,dtype=torch.float64)
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
        self._resolve_difficulty(difficulty)

        self._init_nominal()
        self.reset()

    def _resolve_difficulty(self,difficulty):
        """Shared difficulty resolution (every task env, same shape)."""
        if difficulty is not None:
            if not 0.<=difficulty<=1.:raise ValueError('difficulty must be in [0,1]')
            self.difficulty=float(difficulty)
        if self.difficulty is not None:self._apply_difficulty(self.difficulty)

    def _apply_difficulty(self,d):
        """Map difficulty in [0,1] onto the target-attitude/noise knobs."""
        self.difficulty=d
        self.target_angle_cap=.3+(math.pi-.3)*d
        self.reset_noise_scale=.001*d
        self.cfg['difficulty']=d

    def _rotor_axes(self,theta):
        """Rotor momentum directions r_i(theta) via Rodrigues rotation; g.r0=0."""
        cos=theta.cos().unsqueeze(-1);sin=theta.sin().unsqueeze(-1)
        return cos*self.rotor_axes_ref+sin*self.g_cross_r0

    def _dynamics(self,q,w,theta,delta):
        n=q.shape[0]
        r=self._rotor_axes(theta)                                   # (n,N,3)
        h_momentum=self.rotor_h_dyn*r.sum(1)                        # stored rotor momentum
        gr=torch.linalg.cross(self.gimbal_axes.expand(n,-1,-1),r,dim=-1)
        tau=-(self.rotor_h_dyn*delta.unsqueeze(-1)*gr).sum(1)       # CMG reaction torque
        wdot=(-torch.linalg.cross(w,self.inertia*w)
              -torch.linalg.cross(w,h_momentum)+tau)/self.inertia
        w_quat=torch.zeros_like(q);w_quat[:,1:]=w
        return .5*quat_mul(q,w_quat),wdot

    def _integrate(self,delta,mask):
        """RK4 the (q,omega,gimbal) state with constant commanded gimbal rates."""
        q,w,th=self.q,self.omega,self.gimbal;dt=self.dt
        for _ in range(self.substeps):
            qd1,wd1=self._dynamics(q,w,th,delta)
            qd2,wd2=self._dynamics(q+dt*.5*qd1,w+dt*.5*wd1,th+dt*.5*delta,delta)
            qd3,wd3=self._dynamics(q+dt*.5*qd2,w+dt*.5*wd2,th+dt*.5*delta,delta)
            qd4,wd4=self._dynamics(q+dt*qd3,w+dt*wd3,th+dt*delta,delta)
            qn=q+dt/6*(qd1+2*qd2+2*qd3+qd4)
            wn=w+dt/6*(wd1+2*wd2+2*wd3+wd4)
            qn=qn/qn.norm(dim=-1,keepdim=True)
            thn=(th+dt*delta+math.pi)%(2*math.pi)-math.pi
            q=torch.where(mask[:,None],qn,q);w=torch.where(mask[:,None],wn,w)
            th=torch.where(mask[:,None],thn,th)
        self.q.copy_(q);self.omega.copy_(w);self.gimbal.copy_(th)
        self.gimbal_rate.copy_(torch.where(mask[:,None],delta,self.gimbal_rate))

    def _attitude_error(self):
        # q_err = current^-1 * target, as the 3-vector log map.
        return quat_log(quat_mul(quat_conj(self.q),self.target_quat))

    def _manipulability(self):
        """(m_norm, m_raw): singularity metric sqrt(det(J J^T)) with J columns
        h*(g_i x r_i); m_norm is rotor-speed invariant and bounded in [0, ~1.55],
        used for both the reward penalty and the observation (the raw metric
        spans 10 orders of magnitude [1e-4, 1.5e6], which affine obs
        normalization collapses into an indistinguishable near-singular band)."""
        n=self.num_envs
        r=self._rotor_axes(self.gimbal)
        gr=torch.linalg.cross(self.gimbal_axes.expand(n,-1,-1),r,dim=-1)*self.rotor_h
        jjt=gr.transpose(1,2)@gr
        raw=torch.sqrt(torch.linalg.det(jjt).clamp_min(0.)+1e-8)
        return raw/self.rotor_h**3,raw

    def _potential(self,att_err,m_norm):
        """POSITIVE distance-to-go potential (docking 'progress' convention):
        Phi = k1*e + k1b*e*g_near(e) + k2*|w|^2/(2a)*g(e) + w_sing/(m+soft),
        g = sigmoid((gate - e)/soft), g_near = sigmoid((near_gate - e)/near_soft).
        The k1 term teaches slewing everywhere, the k1b term steepens the
        error gradient inside 0.5 rad (the capture maneuver needs ~3x the
        per-step advantage to rise above PPO advantage noise), the k2 term
        teaches braking/capture, and the singularity term penalizes losing
        momentum authority. Progress (approach / brake / escape low m)
        decreases Phi and earns the telescoped bonus Phi_prev - Phi > 0."""
        gate=torch.sigmoid((self.brake_gate-att_err)/self.brake_gate_soft)
        gate_near=torch.sigmoid((self.near_gate-att_err)/self.near_gate_soft)
        return (self.progress_shaping_gain*att_err
                +self.near_boost_gain*att_err*gate_near
                +self.brake_gain*self.omega.norm(dim=-1)**2/(2.*self.alpha_ref)*gate
                +self.singularity_weight/(m_norm+self.singularity_soft))

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def _observe(self,mask):
        quat_err=self._attitude_error()
        r=self._rotor_axes(self.gimbal)
        gr=torch.linalg.cross(self.gimbal_axes.expand(self.num_envs,-1,-1),r,dim=-1)  # (n,N,3)
        J=gr.transpose(1,2)/self.rotor_h                       # (n,3,N) unit columns
        I3=torch.eye(3,device=self.device,dtype=torch.float64)
        # unit-torque allocation direction v = (J/h)^T ((J/h)(J/h)^T + I)^-1 e_hat:
        # the min-norm gimbal-rate direction that would produce unit torque along the
        # attitude-error axis, with fixed unit damping so the feature stays bounded and
        # gracefully collapses onto feasible torque directions near singularities.
        # This exposes the fixed CMG geometry (a bilinear map of the raw obs the
        # policy MLP cannot represent, verified by a supervised probe) without
        # giving away the control law (gains, damping, timing, avoidance).
        alloc=(J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+I3,quat_err.unsqueeze(-1))).squeeze(-1)
        m_norm,_=self._manipulability()
        obs=torch.cat((self.q,quat_err,self.omega,self.gimbal.sin(),self.gimbal.cos(),
                       self.gimbal_rate,m_norm[:,None],alloc),-1)
        if self._observation is None:self._observation=obs.clone()
        else:self._observation.copy_(torch.where(mask[:,None],obs,self._observation))
        return quat_err

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        # nominal (classical-controller) action for this step's observation
        self.extras['nominal_action']=self._get_nominal_action()
        rates=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)*self.action_scale
        self.action.copy_(rates)
        active=~(self.terminated|self.truncated)
        self._integrate(rates,active)
        quat_err=self._observe(active)
        att_err=quat_err.norm(dim=-1)
        omega_norm=self.omega.norm(dim=-1)
        m_norm,_=self._manipulability()
        # telescoping gated brake-distance potential (Phi_prev - Phi), evaluated
        # on pre-reset states: far from the target this is the proven error-progress
        # signal; inside the gate it additionally rewards braking (|omega| shrinking)
        # and surcharges speeding up, which is exactly the capture skill the
        # velocity-blind potential never taught (policies parked at att_err~0.3).
        # The k1/k2 gains are baked into _potential -- do not rescale here.
        phi=self._potential(att_err,m_norm)
        shaping=self._prev_phi-phi
        # termination/reward set (docking convention): success terminates the
        # episode immediately (marked truncated but NOT time_out -- rsl_rl and
        # FlashSAC treat it as terminal and skip bootstrapping) and pays a
        # one-time settle bonus; runaway spin (|omega| > fail_omega) terminates
        # as a failure with a one-time penalty. Before this change episodes
        # always ran the full 1200 steps, the per-step settle bonus scaled the
        # return with the remaining horizon, and success was only checkable at
        # the timeout step (mid-episode arrivals that later drifted out were
        # scored as failures).
        success=(att_err<self.success_att_tol)&(omega_norm<self.success_omega_tol)
        fail=omega_norm>self.fail_omega
        # pure telescoping reward + one-time terminal bonus/penalty + tiny
        # control regularizer: the former absolute per-step state costs
        # (-att_err, -w_sing/(m+soft), -0.1|w|^2) swamped the shaping signal
        # (-2.5/step vs +0.2/step) and are now folded into the potential
        # (telescoped) instead.
        terminal=(success.double()*self.settle_bonus-fail.double()*self.fail_penalty)
        reward=(torch.where(active,shaping,0.)
                +torch.where(active,terminal,0.)
                -self.ctrl_cost_weight*(rates**2).sum(-1))
        self._prev_phi=torch.where(active,phi,self._prev_phi)
        self.count+=1
        timeout=self.count>=self.max_steps
        self.reward.copy_(torch.where(active,reward,self.reward))
        self.complete|=success&active
        self.terminated|=fail&active
        self.truncated|=(timeout|success)&active

        self.reset_buf=self.terminated|self.truncated
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf

        self.extras['time_outs']=self.truncated&~self.terminated&~self.complete
        # snapshot BEFORE reset_idx: reset_idx calls _observe(mask), which would
        # otherwise overwrite _observation with the post-reset initial state and
        # hand downstream bootstrapping the wrong transition target.
        terminal_obs=self._observation.to(torch.float32)
        self.extras['terminal_observations']=TensorDict(
            {'policy':terminal_obs},batch_size=[self.num_envs])
        if bool(done.any()):
            self.extras['episode']={'success_rate':(done&self.complete).to(torch.float64).sum()/done.sum().clamp_min(1)}
        else:
            self.extras.pop('episode',None)

        self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask=envs_idx
        u=self.rng.uniform(3,mask)
        s1=torch.sqrt(1-u[:,0]);s2=torch.sqrt(u[:,0])
        ang1=2*math.pi*u[:,1];ang2=2*math.pi*u[:,2]
        target=torch.stack((s1*ang1.sin(),s1*ang1.cos(),s2*ang2.sin(),s2*ang2.cos()),-1)
        if self.target_angle_cap<math.pi:
            # difficulty<1: pull the target toward the start attitude (identity
            # quaternion [1,0,0,0]); at the benchmark cap=pi the legacy
            # uniform-SO(3) sample passes through untouched
            home=torch.zeros_like(target);home[:,0]=1.
            target=quat_slerp(home,target,self.target_angle_cap/math.pi)
        self.target_quat.copy_(torch.where(mask[:,None],target,self.target_quat))
        noise=2*self.rng.uniform(7+self.num_gimbals,mask)-1
        qn=torch.zeros_like(self.q);qn[:,0]=1;qn=qn+self.reset_noise_scale*noise[:,:4]
        qn=qn/qn.norm(dim=-1,keepdim=True)
        wn=self.reset_noise_scale*noise[:,4:7]
        thn=self.reset_noise_scale*noise[:,7:]+self.gimbal_home
        self.q.copy_(torch.where(mask[:,None],qn,self.q))
        self.omega.copy_(torch.where(mask[:,None],wn,self.omega))
        self.gimbal.copy_(torch.where(mask[:,None],thn,self.gimbal))
        self._prev_phi.copy_(torch.where(mask,self._potential(
            self._attitude_error().norm(dim=-1),self._manipulability()[0]),self._prev_phi))
        self.gimbal_rate.masked_fill_(mask[:,None],0)
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False)
        self.action.masked_fill_(mask[:,None],0)
        self._observe(mask)
        self.episode_length_buf.masked_fill_(mask,0)
        self.returns.masked_fill_(mask,0)
        self._nominal_reset_idx(mask)

    def reset(self):
        mask=torch.ones((self.num_envs,),device=self.device,dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf=mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):
        """No engine resources to release; kept for API parity."""
