"""Toy pendulum swing-up for RL-algorithm sanity checks, in the official
Genesis RL environment format.

Task reference: the gym-classic Pendulum-v1 spec (point-mass pendulum,
Barto/Sutton swing-up). Fully tensorized: all replicas advance on one
device, no engine, no gym.

Dynamics: th'' = -3g/(2l) sin(th + pi) + 3u/(m l^2) with g = 10, m = 1,
l = 1, torque u = max_torque * clip(a, -1, 1), |th'| <= 8 rad/s, explicit
Euler exactly as gym (the position integrates the *unclipped* derivative
before the velocity clamp), dt = 0.05. th = 0 is the upright position.

Reward: r = -(th_norm^2 + 0.1 th'^2 + 0.001 u^2) with th_norm the angle
wrapped to [-pi, pi]. Reset: th ~ U[-pi, pi], th' ~ U[-1, 1]. Episodes
never terminate -- they truncate at 200 steps, and the runner bootstraps
the timeout values through extras['time_outs'] / terminal_observations,
so the toy also exercises the truncation-bootstrap path of the training
stack. Success metric: completed-episode return > -200 (do-nothing
baseline ~ -680, a policy that swings up within ~40 steps and holds the
upright position scores ~ -160). Masked (inactive) replicas are frozen
and consume no random numbers.
"""
import math
import torch
from tensordict import TensorDict
from ..common.plant import TensorRandom


class PendulumEnv:
    """Classic single-DOF pendulum swing-up, gym Pendulum-v1 spec."""

    # gym Pendulum-v1 constants
    max_steps=200            # episode length limit [steps]
    max_speed=8.             # |th'| clamp [rad/s]
    max_torque=2.            # torque at action saturation [N m]
    g=10.                    # gravity [m/s^2]
    m=1.                     # point mass [kg]
    l=1.                     # rod length [m]
    dt=.05                   # integration step [s]
    success_return=-200.     # toy success bar on the completed-episode return

    def __init__(self,num_envs=64,backend='cpu',seed=7):
        if num_envs<1:raise ValueError('n must be positive')
        if backend not in ('cpu','gpu'):raise ValueError(backend)
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'pendulum','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized',
                       'task_defaults':'gym-classic Pendulum-v1 spec'}
        # Resolve the concrete device (with index) so action-device checks
        # compare equal against tensors created on the default device.
        self.device=str(torch.empty(0,device='cuda' if backend=='gpu' else 'cpu').device)
        d=self.device
        self.max_episode_length=self.max_steps+2
        self.num_actions=1
        self.clip_actions=1.

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=d,dtype=torch.float64)
        self.th=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.thdot=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.count=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.terminated=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.complete=self.terminated.clone()
        self._observation=None
        self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=d,dtype=torch.float32)
        self.reward=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.reset_buf=torch.zeros(num_envs,device=d,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=d,dtype=torch.long)
        self.returns=torch.zeros(num_envs,device=d,dtype=torch.float64)
        self.completed_returns=torch.zeros_like(self.returns)
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)

        self.extras=dict()
        self.cfg=dict(self.task_cfg)
        self.rng=TensorRandom(num_envs,d,seed)
        # run.json reports the mean completed-episode return (key toy signal)
        self.completed_metrics={'return':self.completed_returns}

        self.reset()

    def _observe(self,mask):
        obs=torch.stack((self.th.cos(),self.th.sin(),self.thdot),-1)
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
        torque=actions.to(torch.float64).reshape(self.num_envs,self.num_actions).clamp(-self.clip_actions,self.clip_actions)*self.max_torque
        self.action.copy_(torque)
        active=~(self.terminated|self.truncated)
        # gym Pendulum-v1 Euler step: position integrates the unclipped derivative
        th_pre=self.th;thdot_pre=self.thdot
        thdot_new=thdot_pre+(-3*self.g/(2*self.l)*torch.sin(th_pre+math.pi)
                             +3./(self.m*self.l**2)*torque[:,0])*self.dt
        th_new=th_pre+thdot_new*self.dt
        thdot_new=thdot_new.clamp(-self.max_speed,self.max_speed)
        # gym reward: evaluated on the *pre-integration* state, like gym's step()
        th_norm=torch.atan2(torch.sin(th_pre),torch.cos(th_pre))
        reward=-(th_norm*th_norm+.1*thdot_pre*thdot_pre+.001*torque[:,0]*torque[:,0])
        self.th.copy_(torch.where(active,th_new,self.th))
        self.thdot.copy_(torch.where(active,thdot_new,self.thdot))
        self._observe(active)

        self.count+=1
        timeout=self.count>=self.max_steps
        # success bar on the prospective episode return at truncation
        success=timeout&active&((self.returns+reward)>self.success_return)
        self.reward.copy_(torch.where(active,reward,self.reward))
        self.complete|=success
        self.truncated|=timeout&active   # never terminates early; only timeout truncations

        self.reset_buf=self.terminated|self.truncated
        # update buffers
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf+=1
        self.returns+=self.reward
        self.completed_returns+=torch.where(self.reset_buf,self.returns,0.)
        self.completed_episodes+=self.reset_buf.to(torch.long)
        self.completed_successes+=(self.reset_buf&self.complete).to(torch.long)
        done=self.reset_buf

        # everything ends by timeout, so the values of the terminal states are
        # bootstrapped (the toy exercises the truncation-bootstrap path)
        self.extras['time_outs']=self.truncated&~self.terminated
        self.extras['terminal_observations']=TensorDict(
            {'policy':self._observation.to(torch.float32)},batch_size=[self.num_envs])
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
        u=self.rng.uniform(2,mask)
        th0=(2*u[:,0]-1)*math.pi
        thdot0=2*u[:,1]-1.
        self.th.copy_(torch.where(mask,th0,self.th))
        self.thdot.copy_(torch.where(mask,thdot0,self.thdot))
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
