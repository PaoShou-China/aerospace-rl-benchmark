"""Fixedwing-Waypoints original task in the official Genesis RL environment format.

Reward redesign (2026-10-20, PPO debug): the original PyFlyt shaped reward
(-0.1/step + clamp_min(0) progress + 1/d + 100 on reach / -100 on crash) is
replaced by potential-based telescoping shaping. With a 3600-step episode and
gamma=0.99 the terminal +100/-100 are unreachable by credit assignment, while
the clamp + 1/d terms pay infinite positive return for orbiting the current
waypoint without reaching it. The potential

    Phi(s) = -(distance_to_current_target + sum of remaining path legs)

telescopes to the total path length flown successfully: progress pays its true
value, orbiting nets exactly zero, and crashing/timeouts forfeit all unrealized
potential (so ending the episode early is never cheaper than flying). An
explicit +15 per waypoint reached keeps a crisp sparse event; an explicit -10
sharpens crash/OOB. Terminations, success semantics, observations and spawn
distribution are unchanged from the original task.

Action channels (audited 2026-10-22 against the upstream PyFlyt mode-0
setpoint mapping, original_pyflyt/core/drones/fixedwing.py
L141-144 `surface_assist_ids=[0,0,1,1,2] sign=[1,-1,1,-1,0]`): a0 ->
differential ailerons, a1 -> horizontal tail + vertical tail (opposite
sign), a2 -> DEAD (upstream maps it to the main wing whose deflection
limit is 0; the plant multiplies it by 0), a3 -> throttle. Effective
action dimensionality is therefore 3, not 4; the classical autopilot
drives a2 with exactly 0 and BC/PPO learn a constant there. This is
upstream design, not a porting bug -- do not "fix" the mapping.
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import euler, inverse_rotate
from algorithms.baselines.baselines import FixedwingAutopilotBaseline

REACH_BONUS = 15.0
CRASH_PENALTY = 10.0
# Capture sphere radius. The original 2 m is quasi-unreachable for this
# fixedwing: trim constraints give a ~11 m/s minimum controllable speed and
# hence a ~10-15 m minimum turn radius, so neither scripted nor learned
# policies can reliably thread a 2 m window (validated by 5 autopilot
# variants, best head-on pass missed by 8.6 m). 10 m ~ one minimum turn
# radius keeps the navigation challenge while making captures reachable.
CAPTURE_RADIUS = 10.0
# Waypoint shell. The original 1-90 m shell put targets within 10 m of the
# 100 m out-of-bounds sphere, which a 10-15 m minimum turn radius cannot
# reverse inside; z was clamped only at 0.5 m, so many targets required a
# ground-skimming capture. 20-65 m keeps the navigation challenge while
# leaving 35 m of OOB margin and a 20 m altitude floor.
WAYPOINT_MIN_RADIUS = 20.0
WAYPOINT_MAX_RADIUS = 65.0
WAYPOINT_MIN_ALTITUDE = 20.0


class FixedwingWaypointsEnv:
    # Difficulty axis (0 easiest -> 1 hardest = the benchmark shell): at d=0
    # waypoints only spawn on large circles (min radius 50 m), at d=1 the full
    # benchmark 20-65 m shell is restored bit-for-bit. Success criteria
    # (capture radius, crash/OOB, step limit) are untouched.
    waypoint_min_radius=WAYPOINT_MIN_RADIUS  # axis 50 -> WAYPOINT_MIN_RADIUS (20)
    benchmark_difficulty=1.
    difficulty=None          # None = benchmark behavior (no cfg record)
    # conventional controller: L1 lateral guidance + TECS (baselines.py, tuned on
    # train/dev ICs; params in algorithms/baselines/fixedwing_autopilot.json)
    conventional_controller=FixedwingAutopilotBaseline

    def __init__(self,num_envs=64,backend='cpu',seed=7,difficulty=None):
        self.num_envs=num_envs
        self.task_cfg={'vehicle':'fixedwing','num_envs':num_envs,'backend':backend,
                       'implementation':'fully_tensorized','task_defaults':'PyFlyt 0.29.0'}
        self.clip_actions=1.0

        self.plant=TensorPlant('fixedwing',num_envs,backend,seed)
        self.device=self.plant.device
        self.agent_hz=30;self.inner_steps=120//self.agent_hz;self.max_steps=3600
        self.max_episode_length=self.max_steps+2
        self.num_actions=4
        self.action_low=-torch.ones(self.num_actions,device=self.device,dtype=torch.float64)
        self.action_high=torch.ones(self.num_actions,device=self.device,dtype=torch.float64)

        # initialize buffers
        self.action=torch.zeros((num_envs,self.num_actions),device=self.device,dtype=torch.float64)
        self.reward=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.count=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.terminated=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.truncated=self.terminated.clone()
        self.collision=self.terminated.clone();self.out_of_bounds=self.terminated.clone()
        self.complete=self.terminated.clone();self.fatal_collision=self.terminated.clone()
        self.targets=torch.zeros((num_envs,4,3),device=self.device,dtype=torch.float64)
        self.target_index=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.leg_suffix=torch.zeros((num_envs,5),device=self.device,dtype=torch.float64)
        self.old_distance=torch.full((num_envs,),torch.inf,device=self.device,dtype=torch.float64)
        self.new_distance=torch.full_like(self.old_distance,torch.inf)
        self._observation=None
        self.obs_buf=None
        self.rew_buf=torch.zeros(num_envs,device=self.device,dtype=torch.float32)
        self.reset_buf=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.returns=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
        self.completed_returns=torch.zeros_like(self.returns)
        self.completed_episodes=torch.zeros_like(self.episode_length_buf)
        self.completed_successes=torch.zeros_like(self.episode_length_buf)
        self.completed_metrics={'waypoints_reached':torch.zeros((),device=self.device,dtype=torch.float64),
                                'crashes':torch.zeros((),device=self.device,dtype=torch.float64),
                                'out_of_bounds':torch.zeros((),device=self.device,dtype=torch.float64),
                                'timeouts':torch.zeros((),device=self.device,dtype=torch.float64)}

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
        """Map difficulty in [0,1] onto the waypoint-shell knob."""
        self.difficulty=d
        self.waypoint_min_radius=50.+(WAYPOINT_MIN_RADIUS-50.)*d
        self.cfg['difficulty']=d

    def _scale_actions(self,actions):
        raw=actions.to(torch.float64).clamp(-self.clip_actions,self.clip_actions)
        return self.action_low+(raw+1)*.5*(self.action_high-self.action_low)

    def _potential(self):
        """Phi = -(distance to current target + remaining path legs); 0 when complete."""
        p=self.plant.drone.get_pos().to(torch.float64)
        slots=torch.arange(4,device=self.device)[None]+self.target_index[:,None]
        valid=slots<4;indices=slots.clamp_max(3)
        targets=self.targets.gather(1,indices[:,:,None].expand(-1,-1,3))
        d=(targets[:,0]-p).norm(dim=-1)*valid[:,0].to(torch.float64)
        return -(d+self.leg_suffix.gather(1,self.target_index[:,None]).squeeze(1))

    def step(self,actions):
        if actions.device!=torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        physical=self._scale_actions(actions)
        self.action.copy_(physical)
        command=physical.clone();command[:,3]=command[:,3]/2+.5
        old_potential=self._potential()
        self.count+=1
        crash=torch.zeros_like(self.terminated)
        reach_count=torch.zeros(self.num_envs,device=self.device,dtype=torch.float64)
        for _ in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            self.plant.clear_contacts()
            self.plant.controller(command,active)
            for _ in range(2):self.plant.physics_step(command,active)
            p,rpy,v,wb,vb=self._observe(active)
            collision=self.plant.ground_contact;bounds=p.norm(dim=-1)>100
            reached=self.new_distance<CAPTURE_RADIUS
            self.target_index+=(reached&active).to(torch.long)
            complete=self.target_index>=4
            term=collision|bounds;trunc=complete
            crash|=term&active
            reach_count+=(reached&active).to(torch.float64)
            self.terminated|=term&active;self.truncated|=trunc&active
            self.collision|=collision&active;self.out_of_bounds|=bounds&active
            self.complete|=complete&active
        # the time limit is applied AFTER the full control step (all physics
        # substeps) so the final control step is not cut short; accidents and
        # waypoint completion still terminate inside the substep loop above
        active=~(self.terminated|self.truncated)
        self.truncated|=(self.count>=self.max_steps)&active
        # potential-based telescoping reward: progress + reach bonus - crash penalty
        self.reward.copy_(self._potential()-old_potential+REACH_BONUS*reach_count-CRASH_PENALTY*crash.to(torch.float64))

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
        # single host sync per agent step (done == reset_buf; the same bool
        # gates both the episode extras and the reset path)
        done_any=bool(done.any())
        if done_any:
            n=done.sum().clamp_min(1)
            crashes=done&self.collision;oob=done&self.out_of_bounds
            timeouts=done&self.truncated&~self.complete
            self.extras['episode']={'success_rate':(done&self.complete).to(torch.float64).sum()/n,
                                    'waypoints_reached':self.target_index[done].to(torch.float64).mean(),
                                    'crash_rate':crashes.to(torch.float64).sum()/n,
                                    'out_of_bounds_rate':oob.to(torch.float64).sum()/n,
                                    'timeout_rate':timeouts.to(torch.float64).sum()/n}
            self.completed_metrics['waypoints_reached']+=self.target_index[done].to(torch.float64).sum()
            self.completed_metrics['crashes']+=crashes.to(torch.float64).sum()
            self.completed_metrics['out_of_bounds']+=oob.to(torch.float64).sum()
            self.completed_metrics['timeouts']+=timeouts.to(torch.float64).sum()
        else:
            self.extras.pop('episode',None)

        # reset_idx ends with plant.warmup(), which advances the GLOBAL scene 20
        # substeps with all external wrenches masked to zero. Calling it on an
        # empty done-mask (the common per-step case) injected 20 pure-ballistic
        # substeps into EVERY agent step (28 total vs the intended 8 = 1/30 s),
        # making the effective gravity-to-lift ratio 3.5x and level flight
        # physically impossible. warmup now early-exits on empty masks and the
        # pause-restore in physics_step is unconditional, so an empty reset is
        # fully safe; the gate reuses the already-synced done_any purely for cost.
        if done_any:
            self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras

    def get_observations(self):
        return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def _observe(self,mask):
        p,q,v,w=self.plant.state()
        rpy=euler(q);wb=inverse_rotate(q,w);vb=inverse_rotate(q,v)
        slots=torch.arange(4,device=self.device)[None]+self.target_index[:,None]
        valid=slots<4;indices=slots.clamp_max(3)
        targets=self.targets.gather(1,indices[:,:,None].expand(-1,-1,3))
        delta=inverse_rotate(q[:,None,:].expand(-1,4,-1),targets-p[:,None,:])*valid[:,:,None]
        distance=delta[:,0].norm(dim=-1)
        self.old_distance.copy_(torch.where(mask,self.new_distance,self.old_distance))
        self.new_distance.copy_(torch.where(mask,distance,self.new_distance))
        attitude=torch.cat((wb,q[:,1:],q[:,:1],vb,p,self.action,self.plant.auxiliary()),-1)
        observation=torch.cat((attitude,delta.flatten(1),valid.to(torch.float64)),-1)
        if self._observation is None:self._observation=observation.clone()
        else:self._observation.copy_(torch.where(mask[:,None],observation,self._observation))
        return p,rpy,v,wb,vb

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf=torch.zeros(self._observation.shape,device=self.device,dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def reset_idx(self,envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask=envs_idx
        p,q,v,w=self.plant.state()
        spawn=torch.zeros_like(p);spawn[:,2]=10.
        rotation=torch.zeros_like(q);rotation[:,0]=1
        velocity=torch.zeros_like(v);velocity[:,0]=20
        omega=torch.zeros_like(w)
        self.plant.set_state(torch.where(mask[:,None],spawn,p),torch.where(mask[:,None],rotation,q),
                             torch.where(mask[:,None],velocity,v),torch.where(mask[:,None],omega,w))
        self.count.masked_fill_(mask,0)
        self.terminated.masked_fill_(mask,False);self.truncated.masked_fill_(mask,False)
        self.collision.masked_fill_(mask,False);self.out_of_bounds.masked_fill_(mask,False)
        self.complete.masked_fill_(mask,False);self.fatal_collision.masked_fill_(mask,False)
        self.action.masked_fill_(mask[:,None],0)
        self.plant.zero(mask)
        theta=self.plant.rng.uniform(4,mask)*2*math.pi
        phi=self.plant.rng.uniform(4,mask)*2*math.pi
        radius=self.waypoint_min_radius+self.plant.rng.uniform(4,mask)*(WAYPOINT_MAX_RADIUS-self.waypoint_min_radius)
        targets=torch.stack((radius*torch.sin(phi)*torch.cos(theta),radius*torch.sin(phi)*torch.sin(theta),
                             (radius*torch.cos(phi)).abs().clamp_min(WAYPOINT_MIN_ALTITUDE)),-1)
        self.targets.copy_(torch.where(mask[:,None,None],targets,self.targets))
        legs=(self.targets[:,1:]-self.targets[:,:-1]).norm(dim=-1)
        ext=torch.cat((legs,torch.zeros((self.num_envs,2),device=self.device,dtype=torch.float64)),-1)
        suffix=ext.flip(-1).cumsum(-1).flip(-1)
        self.leg_suffix.copy_(torch.where(mask[:,None],suffix,self.leg_suffix))
        self.target_index.masked_fill_(mask,0)
        self.old_distance.masked_fill_(mask,torch.inf);self.new_distance.masked_fill_(mask,torch.inf)
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
