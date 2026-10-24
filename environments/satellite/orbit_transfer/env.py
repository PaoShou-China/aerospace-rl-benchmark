"""Low-thrust perigee-raising orbital transfer RL environment in the
official Genesis RL environment format.

Task reference: ``PerigeeRaisingEnv`` from gym-satellite-trajectory
(github.com/zampanteymedio/gym-satellite-trajectory), the environment
published with Casas, Carro & Sanchez-Esguevillas, "Low-Thrust Orbital
Transfer using Dynamics-Agnostic Reinforcement Learning" (arXiv:2211.08272,
IEEE). A 100 kg medium-Earth-orbit satellite with three orthogonal thruster
pairs (10 mN per pair, Isp 4000 s) fires for 166 control steps of 5 minutes
(~5 revolutions) to raise the pericenter radius of an initial a=10000 km,
e=0.1, i=pi/3 orbit while keeping the apocenter radius and propellant use
low; the initial mean anomaly is re-sampled uniformly at every reset.

Dynamics: two-body point-mass gravity (WGS84 mu) + constant thrust over
each control step, fully tensorized on one device (no engine, no gym).
The reference integrates the same ODE with Orekit's adaptive Dormand-Prince
8(5,3); here it is integrated with fixed-step RK4 (10 s substeps), whose
per-revolution phase error stays far below the reward granularity.
The reference reward is sparse and terminal only,
R = 1e-4 * (-|dra| + drp + 2e5 * dm), radii in meters, mass in kg
(== +0.1/km pericenter raise, -0.1/km |apocenter drift|, -20/kg propellant,
the paper's Table I weights), granted once when the step limit is reached.
That terminal R is kept as the task-level success criterion (R > 0 at the
final orbit; the reference's is_success=True-at-any-done is vacuous under
timeout-only termination) and as the Episode/completed metrics, but the
training reward is its exact potential-based telescoping decomposition:
Phi(s) = 1e-4 * (-|ra-ra0| + (rp-rp0) + 2e5 * (m-mass0)) is zero at every
initial condition, so the per-step reward F = Phi(s') - Phi(s) sums over an
episode to exactly the terminal R while giving every step an immediate,
action-attributable credit (burning prograde near apogee raises rp -> +F;
burning elsewhere raises |ra-ra0| or drains m -> -F). This matters at the
standard 64-env batch, where a sparse terminal grant reaches the learner
through only ~12 episode boundaries per PPO update and does not train
(diagnosed 2026-10-20: random-policy terminal R = -0.29+-0.07 with 0%
positive under the nominal reward; a 1024-env run learned to R=1.47 only
through ~197 boundaries per update). With gamma = 1.0 (fixed-horizon task)
the PBRS return is undiscounted and exactly invariant; timeouts are true
terminals for this task, so extras['time_outs'] is all-False -- the rsl_rl
gamma * V timeout bootstrap would corrupt returns and, at gamma = 1, leave
the critic without a fixed point.
The observation is the reference's raw 8-dim state [elapsed time, r, v, m];
the reference observation_space merely declares +-1.1x characteristic
bounds and never normalizes.
Defensive terminations (impact, escape, mass depletion) are unreachable
at the reference thrust but guard the metric against hyperparameter
changes. Masked (inactive) replicas are frozen and consume no random
numbers.
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorRandom
from algorithms.baselines.baselines import TransferGaussBaseline


class LowThrustTransferEnv:
    # conventional controller: Gauss planetary-equation tangential-thrust perigee
    # raising (baselines.py; greedy coast/burn decision on the policy obs)
    conventional_controller=TransferGaussBaseline

    def __init__(self,num_envs=64,backend='cpu',seed=7):
        if num_envs<1:raise ValueError('n must be positive')
        if backend not in ('cpu','gpu'):raise ValueError(backend)
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'transfer','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized',
                       'task_defaults':'gym-satellite-trajectory PerigeeRaisingEnv (arXiv:2211.08272, IEEE)'}
        # Resolve the concrete device (with index) so action-device checks
        # compare equal against tensors created on the default device.
        self.device=str(torch.empty(0,device='cuda' if backend=='gpu' else 'cpu').device)
        d=self.device
        # Reference-model constants (PerigeeRaisingEnv defaults, Table I).
        self.mu=3.986004418e14   # WGS84_EARTH_MU [m^3/s^2]
        self.f_max=0.01          # thruster-pair force [N]
        self.isp=4000.           # specific impulse [s]
        self.g0=9.80665          # standard gravity [m/s^2]
        self.mass0=100.          # initial mass [kg]
        self.sma0=1.0e7            # initial semi-major axis [m]
        self.ecc0=0.1
        self.inc0=math.pi/3
        self.raan0=2*math.pi/3
        self.argp0=4*math.pi/3
        self.dt=300.               # control step [s]
        self.substeps=30           # 10 s RK4 substeps per control step
        self.max_steps=166         # ~5 revolutions
        self.max_episode_length=self.max_steps+2
        self.num_actions=3
        self.clip_actions=1.
        # Reference observation_space bounds (declared only; values are
        # returned raw): [dt*max_steps, ra, ra, ra, vmax, vmax, vmax, m0]*1.1.
        self.ra_ref=self.sma0*(1+self.ecc0)
        self.rp_ref=self.sma0*(1-self.ecc0)
        self.vel_ref=math.sqrt(self.mu*(2/self.rp_ref-1/self.sma0))

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=d,dtype=torch.float64)
        self.r=torch.zeros((num_envs,3),device=d,dtype=torch.float64)
        self.v=torch.zeros_like(self.r)
        self.m=torch.full((num_envs,),self.mass0,device=d,dtype=torch.float64)
        self.ra0=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.rp0=torch.zeros_like(self.ra0)
        self._phi=torch.zeros(num_envs,device=d,dtype=torch.float64)  # PBRS potential at the current state
        self.reward=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.terminal_reward=torch.zeros(num_envs,device=d,dtype=torch.float64)  # terminal R, metrics only
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
        self.completed_metrics={'perigee_raise_km':torch.zeros_like(self.returns),
                                'apo_drift_km':torch.zeros_like(self.returns),
                                'propellant_kg':torch.zeros_like(self.returns),
                                'terminal_reward':torch.zeros_like(self.returns)}

        self.extras=dict()
        self.cfg=dict(self.task_cfg)
        self.rng=TensorRandom(num_envs,d,seed)

        self.reset()

    def _kepler_to_cartesian(self,mean_anomaly):
        """Perifocal -> GCRF state for the fixed (a,e,i,raan,argp) and given M."""
        e,a,mu=self.ecc0,self.sma0,self.mu
        E=mean_anomaly.clone()
        for _ in range(8):
            E=E-(E-e*E.sin()-mean_anomaly)/(1-e*E.cos())
        b=a*math.sqrt(1-e*e);n_a=math.sqrt(mu/a**3)
        den=1-e*E.cos()
        r_pf=torch.stack((a*(E.cos()-e),b*E.sin(),torch.zeros_like(E)),-1)
        v_pf=torch.stack((-a*n_a*E.sin()/den,b*n_a*E.cos()/den,torch.zeros_like(E)),-1)
        cO,sO=math.cos(self.raan0),math.sin(self.raan0)
        ci,si=math.cos(self.inc0),math.sin(self.inc0)
        cw,sw=math.cos(self.argp0),math.sin(self.argp0)
        rot=torch.tensor([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                          [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],
                          [sw*si,cw*si,ci]],device=self.device,dtype=torch.float64)
        return r_pf@rot.T,v_pf@rot.T

    @staticmethod
    def _radii(r,v,mu):
        """(ra, rp) from a Cartesian state: a(1+e), a(1-e)."""
        rm=r.norm(dim=-1,keepdim=True)
        a=1./(2./rm-(v*v).sum(-1,keepdim=True)/mu)
        h=torch.linalg.cross(r,v,dim=-1)
        e_vec=torch.linalg.cross(v,h,dim=-1)/mu-r/rm
        e=e_vec.norm(dim=-1,keepdim=True)
        return (a*(1+e)).squeeze(-1),(a*(1-e)).squeeze(-1)

    def _deriv(self,s,action):
        """Point-mass gravity + constant thrust ODE for state [r,v,m]."""
        r,v,m=s[:,:3],s[:,3:6],s[:,6]
        d=torch.zeros_like(s)
        d[:,:3]=v
        d[:,3:6]=-self.mu*r/r.norm(dim=-1,keepdim=True)**3+self.f_max*action/m[:,None]
        d[:,6]=-self.f_max*action.abs().sum(-1)/(self.isp*self.g0)
        return d

    def _integrate(self,action,mask):
        """RK4 the [r,v,m] state under constant thrust for one control step."""
        s=torch.cat((self.r,self.v,self.m[:,None]),-1)
        h=self.dt/self.substeps
        for _ in range(self.substeps):
            k1=self._deriv(s,action)
            k2=self._deriv(s+h*.5*k1,action)
            k3=self._deriv(s+h*.5*k2,action)
            k4=self._deriv(s+h*k3,action)
            s=torch.where(mask[:,None],s+h/6*(k1+2*k2+2*k3+k4),s)
        self.r.copy_(s[:,:3]);self.v.copy_(s[:,3:6]);self.m.copy_(s[:,6])

    def _observe(self,mask):
        t=self.count.to(torch.float64)*self.dt
        obs=torch.cat((t[:,None],self.r,self.v,self.m[:,None]),-1)
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
        rates=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)
        self.action.copy_(rates)
        active=~(self.terminated|self.truncated)
        self._integrate(rates,active)
        self.count+=1
        self._observe(active)
        ra,rp=self._radii(self.r,self.v,self.mu)
        # Potential-based telescoping reward. Phi equals the reference terminal
        # reward functional and is zero at every reset, so the per-step reward
        # F = Phi(s') - Phi(s) sums over an episode to exactly the terminal R
        # (gamma = 1.0, fixed horizon) while crediting each step immediately.
        phi=1.e-4*(-(ra-self.ra0).abs()+(rp-self.rp0)+2.e5*(self.m-self.mass0))
        dense=torch.where(active,phi-self._phi,self.reward)
        # Defensive terminations (unreachable at the reference 10 mN thrust):
        # impact below the WGS84 equatorial radius, escape (non-positive
        # orbital energy -> negative semi-major axis), mass depletion.
        violation=(rp<6.378137e6)|(ra<=0)|(self.m<=1.)
        self.terminated|=violation&active
        ok=active&~violation
        timeout=self.count>=self.max_steps
        grant=ok&timeout
        self.reward.copy_(dense)
        # Terminal R is metrics-only now (success criterion + Episode logging):
        # the training reward already contains it via telescoping.
        self.terminal_reward.copy_(torch.where(grant,phi,self.terminal_reward))
        # Task-level success: reference objective R > 0 at the final orbit.
        # (The reference reports is_success=True at any done, vacuous here.)
        self.complete|=grant&(phi>0)
        self.truncated|=grant
        self._phi.copy_(phi)

        self.reset_buf=self.terminated|self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf
        # snapshot for the step return: reset_idx below clears terminated/truncated
        term_out,trunc_out=self.terminated.clone(),self.truncated.clone()
        n_done=done.sum().clamp_min(1)
        drp_km=torch.where(done,(rp-self.rp0)*1e-3,0.)
        dra_km=torch.where(done,(ra-self.ra0).abs()*1e-3,0.)
        dm_kg=torch.where(done,self.mass0-self.m,0.)
        self.completed_metrics['perigee_raise_km']+=drp_km
        self.completed_metrics['apo_drift_km']+=dra_km
        self.completed_metrics['propellant_kg']+=dm_kg
        self.completed_metrics['terminal_reward']+=torch.where(done,self.terminal_reward,0.)

        # Fixed-horizon task: the step limit is a true terminal, so the rsl_rl
        # gamma*V timeout bootstrap is disabled (all-False key kept for API
        # parity with the other environments).
        self.extras['time_outs']=torch.zeros_like(self.truncated)
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
        if bool(done.any()):
            self.extras['episode']={
                'success_rate':(done&self.complete).to(torch.float64).sum()/n_done,
                'perigee_raise_km':drp_km.sum()/n_done,
                'apo_drift_km':dra_km.sum()/n_done,
                'propellant_kg':dm_kg.sum()/n_done}
        else:
            self.extras.pop('episode',None)

        self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,term_out,trunc_out,self.extras

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask=envs_idx
        mean_anomaly=(2*self.rng.uniform(1,mask)-1).reshape(-1)*math.pi
        r0,v0=self._kepler_to_cartesian(mean_anomaly)
        self.r.copy_(torch.where(mask[:,None],r0,self.r))
        self.v.copy_(torch.where(mask[:,None],v0,self.v))
        self.m.copy_(torch.where(mask,torch.full_like(self.m,self.mass0),self.m))
        ra,rp=self._radii(self.r,self.v,self.mu)
        self.ra0.copy_(torch.where(mask,ra,self.ra0))
        self.rp0.copy_(torch.where(mask,rp,self.rp0))
        # Phi is exactly zero at every fresh initial condition.
        self._phi.copy_(torch.where(mask,0.,self._phi))
        self.terminal_reward.masked_fill_(mask,0)
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
