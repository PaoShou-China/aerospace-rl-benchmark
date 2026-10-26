"""Split-S racing task on the unchanged QuadX plant.

Task layer: directional gates, progress/rate reward, 36-D two-gate observations
and training-only reset curriculum. Dynamics, actuation and 40/120/240 Hz timing
remain those of the existing QuadX model. See README.md for local choices.
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant
from environments.common.dynamics import inverse_rotate, quat_from_euler
from algorithms.baselines.baselines import QuadXSE3Baseline
from .split_s import BOUNDS_RADIUS, NUM_GATES, DEFAULT_TRAIN_GATE_COUNT, PAPER_LOOKAHEAD_GATES, sample_gate_indices, full_track_indices, sequence_indices
from .gate_geometry import GateGeometry
from .initial_state_buffer import InitialStateBuffer


class SplitsSE3Baseline(QuadXSE3Baseline):
    """Legacy controller adapter, not a racing expert or a guaranteed gate passer."""
    def actions(self, obs):
        return super().actions(self.env.baseline_observation)


class QuadXSplitsEnv:
    conventional_controller=SplitsSE3Baseline
    bounds_radius=BOUNDS_RADIUS

    def __init__(self,num_envs=64,backend='cpu',seed=7,steady_wind=None,dryden=0.,severity='moderate',
                 full_track=False,laps=1,vehicle_margin=.05,frame_width=.10,
                 initialization='curriculum',buffer_cfg=None,max_episode_seconds=20.):
        if not isinstance(laps,int) or isinstance(laps,bool) or laps<1:
            raise ValueError('laps must be a positive integer')
        if not full_track and laps!=1:
            raise ValueError('multiple laps require full_track=True')
        if initialization not in ('fixed','segment','curriculum'):
            raise ValueError('initialization must be fixed, segment or curriculum')
        if not math.isfinite(max_episode_seconds) or max_episode_seconds<=0:
            raise ValueError('max_episode_seconds must be finite and positive')
        self.num_envs,self.full_track=num_envs,full_track
        self.initialization='fixed' if full_track else initialization
        self.num_targets=NUM_GATES*laps if full_track else DEFAULT_TRAIN_GATE_COUNT
        self.plant=TensorPlant('quadx',num_envs,backend,seed,vel_clamp=(15.,30.))
        self.device=self.plant.device
        self.gate_geometry=GateGeometry(self.device,vehicle_margin,frame_width)
        self.initial_states=InitialStateBuffer(self.device,**(buffer_cfg or {})) if self.initialization=='curriculum' else None
        self.agent_hz=40;self.inner_steps=3;self.max_steps=max(1,round(max_episode_seconds*self.agent_hz))
        self.max_episode_length=self.max_steps
        self.num_actions=4;self.clip_actions=1.
        self.action_low=torch.tensor([-math.pi,-math.pi,-math.pi,0.],device=self.device,dtype=torch.float64)
        self.action_high=torch.tensor([math.pi,math.pi,math.pi,.8],device=self.device,dtype=torch.float64)
        self.cfg=dict(vehicle='quadx',num_envs=num_envs,backend=backend,task='splits_racing_v1',
                      full_track=full_track,laps=laps,initialization=self.initialization,
                      observation='world_velocity_rotation_body_gate_corners_36',
                      reward='progress_minus_0.01_body_rate_crash_minus10_finish_plus10',
                      vehicle_margin=vehicle_margin,frame_width=frame_width,
                      physics='unchanged_quadx',policy_hz=40,pid_hz=120,physics_hz=240,
                      max_episode_seconds=max_episode_seconds,buffer_cfg=dict(buffer_cfg or {}))
        self.task_cfg=self.cfg
        def zeros(*shape,dtype=torch.float64):return torch.zeros(*shape,device=self.device,dtype=dtype)
        self.action=zeros(num_envs,4);self.reward=zeros(num_envs)
        self.count=zeros(num_envs,dtype=torch.long)
        self.terminated=zeros(num_envs,dtype=torch.bool);self.truncated=self.terminated.clone()
        self.complete=self.terminated.clone();self.collision=self.terminated.clone()
        self.gate_collision=self.terminated.clone();self.out_of_bounds=self.terminated.clone()
        self.targets=zeros(num_envs,self.num_targets,3)
        self.gate_ids=zeros(num_envs,self.num_targets,dtype=torch.long)
        self.target_index=zeros(num_envs,dtype=torch.long)
        self.obs_buf=zeros(num_envs,36,dtype=torch.float32);self._observation=zeros(num_envs,36)
        self.rew_buf=zeros(num_envs,dtype=torch.float32);self.reset_buf=self.terminated.clone()
        self.episode_length_buf=self.count.clone();self.returns=zeros(num_envs)
        self.completed_episodes=self.count.clone();self.completed_successes=self.count.clone()
        self.completed_returns=zeros(num_envs)
        self.completed_metrics={k:zeros(()) for k in ('waypoints_reached','crashes','gate_collisions','out_of_bounds','timeouts')}
        self.speed_sum=zeros(num_envs);self.rate_sum=zeros(num_envs);self.physical_count=zeros(num_envs)
        self.max_speed=zeros(num_envs);self.min_clearance=torch.full((num_envs,),torch.inf,device=self.device,dtype=torch.float64)
        self.lap_times=torch.full((num_envs,laps),torch.nan,device=self.device,dtype=torch.float64)
        self.last_lap_time=zeros(num_envs)
        self.extras={};self._wind=None
        if steady_wind is not None or dryden>0:
            from environments.common.disturbances import WindDisturbance
            self._wind=WindDisturbance(num_envs,self.device,seed,steady_wind=steady_wind,dryden=dryden,severity=severity,dt=1/40)
            self.plant.wind=zeros(num_envs,3)
            self.cfg['wind']=dict(steady_wind=steady_wind,dryden=dryden,severity=severity)
        self.reset()

    @property
    def gate_index(self):
        return self.gate_ids.gather(1,self.target_index.clamp_max(self.num_targets-1)[:,None]).squeeze(1)

    @property
    def current_target(self):return self.gate_geometry.centres[self.gate_index]
    @property
    def gates_passed(self):return self.target_index
    @property
    def lap_count(self):return self.target_index//NUM_GATES

    def _capture_state(self):
        p,q,v,w=self.plant.state()
        return torch.cat((p,q,v,w,self.plant.throttle,self.plant.pwm,
                          self.plant.pid.integral,self.plant.pid.previous,self.action),-1)

    def _observe(self,mask):
        p,q,v,w=self.plant.state()
        qw,x,y,z=q.unbind(-1)
        rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-z*qw),2*(x*z+y*qw),
                              2*(x*y+z*qw),1-2*(x*x+z*z),2*(y*z-x*qw),
                              2*(x*z-y*qw),2*(y*z+x*qw),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
        slots=self.target_index[:,None]+torch.arange(PAPER_LOOKAHEAD_GATES,device=self.device)
        valid=slots<self.num_targets
        ids=self.gate_ids.gather(1,slots.clamp_max(self.num_targets-1))
        corners=self.gate_geometry.corners()[ids]
        local=torch.einsum('nkj,n...k->n...j',rotation,corners-p[:,None,None])
        local*=valid[:,:,None,None]
        observation=torch.cat((v,rotation.flatten(1),local.flatten(1)),-1)
        self._observation.copy_(torch.where(mask[:,None],observation,self._observation))
        self.obs_buf.copy_(self._observation)
        self.baseline_observation=torch.cat((inverse_rotate(q,w),q[:,1:],q[:,:1],inverse_rotate(q,v),p),-1)

    def get_observations(self):return TensorDict({'policy':self.obs_buf.clone()},batch_size=[self.num_envs])

    def step(self,actions):
        if actions.device!=torch.device(self.device):raise ValueError('actions must be on env.device')
        physical=self.action_low+(actions.to(torch.float64).clamp(-1,1)+1)*.5*(self.action_high-self.action_low)
        self.action.copy_(physical)
        if self._wind is not None:self.plant.wind.copy_(self._wind.step())
        self.count+=1
        progress=torch.zeros_like(self.reward);rate_cost=torch.zeros_like(self.reward)
        crash=torch.zeros_like(self.terminated);events=[]
        for inner in range(self.inner_steps):
            active=~(self.terminated|self.truncated)
            self.plant.controller(physical,active)
            for sub in range(2):
                active=~(self.terminated|self.truncated)
                before=self.plant.drone.get_pos().to(torch.float64).clone()
                self.plant.clear_contacts();self.plant.physics_step(physical,active)
                p,q,v,w=self.plant.state()
                passed,fractions,frame_hit=self.gate_geometry.sweep(before,p)
                collision=self.plant.ground_contact|frame_hit
                bounds=p.norm(dim=-1)>self.bounds_radius;failed=collision|bounds
                segment=before.clone();previous_fraction=torch.full_like(progress,-1.)
                passed_any=torch.zeros_like(active);next_gate=self.gate_index.clone()
                for _ in range(self.num_targets):
                    ids=self.gate_index;target=self.gate_geometry.centres[ids]
                    fraction=fractions.gather(1,ids[:,None]).squeeze(1)
                    reached=(passed.gather(1,ids[:,None]).squeeze(1)&active&~failed
                             &(self.target_index<self.num_targets)&(fraction>previous_fraction+1e-9))
                    point=before+fraction[:,None]*(p-before)
                    progress+=((target-segment).norm(dim=-1)-(target-point).norm(dim=-1))*reached
                    segment=torch.where(reached[:,None],point,segment)
                    local=torch.einsum('nij,nj->ni',self.gate_geometry.rotations[ids].transpose(1,2),point-target)
                    clearance=(self.gate_geometry.sizes[ids,1:]/2-local[:,1:].abs()).amin(-1)-self.gate_geometry.margin
                    self.min_clearance.copy_(torch.where(reached,torch.minimum(self.min_clearance,clearance),self.min_clearance))
                    self.target_index+=reached.long();passed_any|=reached
                    next_gate=torch.where(reached,(ids+1)%NUM_GATES,next_gate)
                    previous_fraction=torch.where(reached,fraction,previous_fraction)
                    if self.full_track:
                        lap_done=reached&(self.target_index%NUM_GATES==0)
                        elapsed=(self.count-1)/self.agent_hz+(inner*2+sub+fraction)/240
                        lap_slot=(self.target_index//NUM_GATES-1).clamp(0,self.lap_times.shape[1]-1)
                        old=self.lap_times.gather(1,lap_slot[:,None]).squeeze(1)
                        value=torch.where(lap_done,elapsed-self.last_lap_time,old)
                        self.lap_times.scatter_(1,lap_slot[:,None],value[:,None])
                        self.last_lap_time.copy_(torch.where(lap_done,elapsed,self.last_lap_time))
                remainder=active&~failed&(self.target_index<self.num_targets)
                progress+=((self.current_target-segment).norm(dim=-1)-(self.current_target-p).norm(dim=-1))*remainder
                rate=inverse_rotate(q,w).norm(dim=-1)
                rate_cost+=rate*active/6
                self.speed_sum+=v.norm(dim=-1)*active;self.rate_sum+=rate*active;self.physical_count+=active
                self.max_speed.copy_(torch.where(active,torch.maximum(self.max_speed,v.norm(dim=-1)),self.max_speed))
                complete=(self.target_index>=self.num_targets)&~failed
                crash|=failed&active
                self.terminated|=(failed|complete)&active
                self.collision|=collision&active;self.gate_collision|=frame_hit&active
                self.out_of_bounds|=bounds&active;self.complete|=complete&active
                if self.initial_states is not None:
                    normal=self.gate_geometry.rotations[next_gate,:,0]
                    on_entry=((p-self.gate_geometry.centres[next_gate])*normal).sum(-1)<0
                    events.append((self._capture_state().clone(),next_gate.clone(),passed_any&on_entry))
        if self.initial_states is not None:
            self.initial_states.add(*(torch.cat([e[i] for e in events]) for i in range(3)))
        self.truncated|=(self.count>=self.max_steps)&~self.terminated
        self.reward.copy_(progress-.01*rate_cost)
        self.reward.copy_(torch.where(self.complete,10.,self.reward))
        self.reward.copy_(torch.where(crash,-10.,self.reward))
        self.rew_buf.copy_(self.reward);self.returns+=self.reward;self.episode_length_buf+=1
        self._observe(torch.ones_like(self.terminated))
        self.reset_buf=self.terminated|self.truncated;done=self.reset_buf
        self.extras['time_outs']=self.truncated.clone()
        self.extras['terminal_observations']=self.get_observations()
        term,trunc=self.terminated.clone(),self.truncated.clone()
        self.completed_episodes+=done.long();self.completed_successes+=(done&self.complete).long()
        self.completed_returns+=torch.where(done,self.returns,0.)
        if bool(done.any()):
            denom=self.physical_count.clamp_min(1)
            outcome=dict(done=done.clone(),success=self.complete.clone(),gates_passed=self.target_index.clone(),
                         laps=self.lap_count.clone(),time_s=self.physical_count.clone()/240,
                         crash=self.collision.clone(),out_of_bounds=self.out_of_bounds.clone(),
                         timeout=trunc.clone(),gate_collision=self.gate_collision.clone(),
                         min_clearance_m=self.min_clearance.clone(),mean_speed=self.speed_sum/denom,
                         max_speed=self.max_speed.clone(),mean_body_rate=self.rate_sum/denom,lap_times=self.lap_times.clone())
            self.extras['outcome']=outcome
            self.extras['terminal_state']=dict(zip(('p','q','v','w'),(t.clone() for t in self.plant.state())))
            self.extras['episode']={k:outcome[key][done].to(torch.float64).mean() for k,key in
                (('success_rate','success'),('waypoints_reached','gates_passed'),('laps_completed','laps'),
                 ('crash_rate','crash'),('out_of_bounds_rate','out_of_bounds'),('timeout_rate','timeout'),
                 ('gate_collision_rate','gate_collision'),('episode_time_s','time_s'),('body_rate_rad_s','mean_body_rate'))}
            for key,value in (('waypoints_reached',self.target_index),('crashes',self.collision),
                              ('gate_collisions',self.gate_collision),('out_of_bounds',self.out_of_bounds),('timeouts',trunc)):
                self.completed_metrics[key]+=value[done].to(torch.float64).sum()
            if self.initial_states is not None:
                self.initial_states.update(done,self.complete)
                self.extras['episode'].update(initial_state_buffer_size=self.initial_states.count,
                                              reset_curriculum_ready=float(self.initial_states.ready))
            self.reset_idx(done)
        else:
            for key in ('outcome','terminal_state','episode'):self.extras.pop(key,None)
        return self.get_observations(),self.rew_buf,term,trunc,self.extras

    def _hover_states(self,start,mask):
        centres=self.gate_geometry.centres;normal=self.gate_geometry.rotations[start,:,0]
        target=centres[start];mid=(centres[(start-1)%NUM_GATES]+target)/2
        spawn=target-normal
        pending=mask.clone()
        for _ in range(8):
            candidate=mid+.5*(self.plant.rng.uniform(3,pending)*2-1)
            signed=((candidate-target)*normal).sum(-1)
            candidate-=(signed+.5).clamp_min(0)[:,None]*normal
            candidate[:,2].clamp_min_(.4)
            _,_,hit=self.gate_geometry.sweep(candidate,candidate)
            valid=~hit&(candidate.norm(dim=-1)<self.bounds_radius-.1)
            spawn=torch.where((pending&valid)[:,None],candidate,spawn)
            pending&=~valid
        yaw=torch.atan2(target[:,1]-spawn[:,1],target[:,0]-spawn[:,0])
        rpy=torch.zeros_like(spawn);rpy[:,2]=yaw
        return spawn,quat_from_euler(rpy)

    def reset_idx(self,mask):
        p,q,v,w=self.plant.state()
        spawn=torch.zeros_like(p);spawn[:,2]=1.
        rotation=torch.zeros_like(q);rotation[:,0]=1.
        velocity=torch.zeros_like(v);omega=torch.zeros_like(w)
        replay=torch.zeros(self.num_envs,31,device=self.device,dtype=torch.float64)
        use=torch.zeros_like(mask)
        if self.full_track:
            ids=full_track_indices(self.device,self.num_targets//NUM_GATES)[None].expand(self.num_envs,-1)
        else:
            ids=sample_gate_indices(self.plant.rng,mask,DEFAULT_TRAIN_GATE_COUNT)
            start=ids[:,0]
            if self.initialization!='fixed':spawn,rotation=self._hover_states(start,mask)
            if self.initial_states is not None:
                replay,gate,use=self.initial_states.sample(self.plant.rng,mask)
                ids=sequence_indices(torch.where(use,gate,start),self.num_targets)
                spawn=torch.where(use[:,None],replay[:,:3],spawn)
                rotation=torch.where(use[:,None],replay[:,3:7],rotation)
                velocity=torch.where(use[:,None],replay[:,7:10],velocity)
                omega=torch.where(use[:,None],replay[:,10:13],omega)
        self.plant.set_state(torch.where(mask[:,None],spawn,p),torch.where(mask[:,None],rotation,q),
                             torch.where(mask[:,None],velocity,v),torch.where(mask[:,None],omega,w))
        self.plant.zero(mask)
        self.action.masked_fill_(mask[:,None],0.)
        for dest,section in ((self.plant.throttle,slice(13,17)),(self.plant.pwm,slice(17,21)),
                             (self.plant.pid.integral,slice(21,24)),(self.plant.pid.previous,slice(24,27)),
                             (self.action,slice(27,31))):
            dest.copy_(torch.where(use[:,None],replay[:,section],dest))
        self.gate_ids.copy_(torch.where(mask[:,None],ids,self.gate_ids))
        self.targets.copy_(torch.where(mask[:,None,None],self.gate_geometry.centres[ids],self.targets))
        for tensor in (self.count,self.target_index,self.episode_length_buf,self.returns,self.speed_sum,
                       self.rate_sum,self.physical_count,self.max_speed,self.last_lap_time):tensor.masked_fill_(mask,0)
        for tensor in (self.terminated,self.truncated,self.complete,self.collision,self.gate_collision,self.out_of_bounds):tensor.masked_fill_(mask,False)
        self.min_clearance.masked_fill_(mask,torch.inf);self.lap_times.masked_fill_(mask[:,None],torch.nan)
        # Do not advance zero-command physics after restoring an initial state.
        if self._wind is not None:self._wind.reset(mask)
        self._observe(mask)

    def reset(self):
        mask=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.reset_idx(mask);self.reset_buf=mask.clone()
        return self.get_observations()

    def training_state_dict(self):
        return dict(buffer=self.initial_states.state_dict() if self.initial_states is not None else None,
                    rng_seed=self.plant.rng.seed.cpu().clone(),rng_counter=self.plant.rng.counter.cpu().clone())

    def load_training_state_dict(self,state):
        if self.full_track:return  # Evaluation never restores training reset states.
        if state['rng_seed'].shape!=self.plant.rng.seed.shape:raise ValueError('resume requires the same environment count')
        if self.initial_states is not None and state['buffer'] is not None:self.initial_states.load_state_dict(state['buffer'])
        self.plant.rng.seed.copy_(state['rng_seed']);self.plant.rng.counter.copy_(state['rng_counter'])
        self.reset()  # Restart trajectories with the restored curriculum, not exact in-flight replay.

    def close(self):self.plant.close()
