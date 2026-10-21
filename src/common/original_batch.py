"""Shared-scene batching of the unchanged PyFlyt task/control/component code.

One Genesis scene owns N independent replicas. Python task/controller objects are
per-replica; physics advances once for the whole batch. Selected resets preserve
other replicas, including their physical time as observed by their Aviary objects.
This correctness-first bridge still runs original NumPy code on the host.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import numpy as np
import torch
import genesis as gs
from .original_backend import GenesisBulletClient


@dataclass(frozen=True)
class EngineConfiguration:
    backend: str = 'cpu'
    pool: object = None
    slot: int = 0

_configuration=ContextVar('pyflyt_genesis_engine',default=EngineConfiguration())

@contextmanager
def engine_configuration(backend='cpu',pool=None,slot=0):
    token=_configuration.set(EngineConfiguration(backend,pool,slot))
    try:yield
    finally:_configuration.reset(token)


class ConfiguredBulletClient(GenesisBulletClient):
    """Select a scene replica without changing the frozen Aviary implementation."""
    def __init__(self,*args,**kwargs):
        cfg=_configuration.get()
        self._backend=cfg.backend;self._pool=cfg.pool;self._env_idx=cfg.slot
        self._num_envs=1 if cfg.pool is None else cfg.pool.num_envs
        super().__init__(*args,**kwargs)
        if self._pool is not None:self._pool.clients[self._env_idx]=self

    def resetSimulation(self):
        if self._pool is None or self._pool.owner is None:
            super().resetSimulation()
            if self._pool is not None:self._pool.owner=self
        else:
            owner=self._pool.owner
            self._scene=owner._scene;self._built=owner._built;self._bodies=[];self._contacts=[]
            self._pending_pad=None

    def loadURDF(self,fileName,*args,**kwargs):
        if self._pool is None or self._pool.owner is self:
            return super().loadURDF(fileName,*args,**kwargs)
        owner=self._pool.owner;body=len(self._bodies)
        template=owner._bodies[body]
        from pathlib import Path
        if Path(fileName).name!=Path(template['file']).name:
            raise ValueError('All replicas must use the same task and body topology')
        self._bodies.append(template)
        return body

    def stepSimulation(self):
        if self._pool is None:return super().stepSimulation()
        # Used only during an individual task's original reset/warmup.
        self._pool.physics_step([self._env_idx])

    def disconnect(self):
        if getattr(self,'_pool',None) is None:super().disconnect()
        # A per-task reset must not destroy the shared scene.


class OriginalTaskBatch:
    """Original Gym semantics with a leading batch dimension and one shared scene.

    step takes physical actions in the original action spaces. It does NOT reset
    terminal observations. Call reset(ids) explicitly or use the RSL-RL adapter.
    reset_kwargs are passed unchanged to the original task reset (options omitted
    versus options=None retain their original, different meanings).
    """
    def __init__(self,vehicle,num_envs=16,backend='cpu',seed=7,task_kwargs=None):
        if num_envs<1:raise ValueError('num_envs must be positive')
        self.vehicle=vehicle;self.num_envs=num_envs;self.backend=backend
        self.owner=None;self.clients=[None]*num_envs;self.tasks=[];self.physics_steps=0
        self.seed=seed;self._closed=False
        from .original_backend import _make_original_task
        for i in range(num_envs):
            with engine_configuration(backend,self,i):
                task=_make_original_task(vehicle,**(task_kwargs or {}))
                task.reset(seed=seed+i)
            self.tasks.append(task)
        self.device=gs.device
        self.action_space=self.tasks[0].action_space
        self.observation_space=self.tasks[0].observation_space
        self.num_actions=len(self.action_space.low)
        self.agent_hz=120/self.tasks[0].env_step_ratio
        self.max_episode_length=self.tasks[0].max_steps+2
        self.done=np.zeros(num_envs,dtype=bool)

    @property
    def scene(self):return self.owner._scene

    def observations(self):return [task.state for task in self.tasks]

    def reset(self,ids=None,seeds=None,**reset_kwargs):
        ids=list(range(self.num_envs)) if ids is None else [int(i) for i in ids]
        if seeds is not None and len(seeds)!=len(ids):raise ValueError('One seed required per reset replica')
        for j,i in enumerate(ids):
            if not 0<=i<self.num_envs:raise IndexError(i)
            with engine_configuration(self.backend,self,i):
                self.tasks[i].reset(seed=None if seeds is None else seeds[j],**reset_kwargs)
            self.done[i]=False
        return self.observations()

    def physics_step(self,active):
        """Advance requested replicas together; pause nonparticipants exactly.

        Genesis advances all replicas. Its public selected-state reset restores
        nonparticipants after a partial warmup/early stop. NumPy/controller/RNG
        state of nonparticipants is never updated by this method.
        """
        active=list(active)
        if not active:return
        owner=self.owner;owner._build()
        inactive=sorted(set(range(self.num_envs))-set(active))
        snapshot=self.scene.get_state() if inactive else None
        for body in owner._bodies:
            if not body['fixed']:
                body['entity'].apply_links_external_wrench(force=body['force'],torque=body['torque'],
                    ref=gs.link_ref_frame.link_COM,local=False)
        self.scene.step(update_visualizer=False,refresh_visualizer=False)
        # Read active contacts before a selected state restoration clears caches.
        contacts={i:[] for i in active}
        for a,body in enumerate(owner._bodies):
            if body['fixed']:continue
            for b,other in enumerate(owner._bodies):
                if a==b:continue
                mask=body['entity'].get_contacts(with_entity=other['entity'])['valid_mask']
                for i in active:
                    if mask[i].any().item():contacts[i].append((0,a,b))
        for body in owner._bodies:
            body['force'].zero_();body['torque'].zero_()
        if inactive:self.scene.reset(state=snapshot,envs_idx=inactive)
        for i in active:self.clients[i]._contacts=contacts[i]
        self.physics_steps+=1

    def _aviary_step(self,active):
        """Same update_control/physics/state/contact order as original Aviary.step."""
        for i in active:self.clients[i].contact_array[:]=False
        ratio=self.clients[active[0]].updates_per_step
        for _ in range(ratio):
            for i in active:
                aviary=self.clients[i]
                for drone in aviary.armed_drones:drone.update_control(aviary.physics_steps)
                for drone in aviary.armed_drones:drone.update_physics()
            self.physics_step(active)
            for i in active:
                aviary=self.clients[i]
                for drone in aviary.armed_drones:drone.update_state();drone.update_last(aviary.physics_steps)
                for contact in aviary.getContactPoints():
                    a,b=contact[1:3];aviary.contact_array[a,b]=aviary.contact_array[b,a]=True
                aviary.physics_steps+=1;aviary.elapsed_time=aviary.physics_steps/aviary.physics_hz
        for i in active:self.clients[i].aviary_steps+=1

    def step(self,actions):
        if self.done.any():raise RuntimeError('Reset completed replicas before the next batch step')
        actions=np.asarray(actions,dtype=np.float64)
        if actions.shape!=(self.num_envs,self.num_actions):raise ValueError('Incorrect original action batch shape')
        if not np.isfinite(actions).all():raise ValueError('Nonfinite actions')
        for i,task in enumerate(self.tasks):
            task.action=actions[i].copy()
            task.reward=0. if self.vehicle=='rocket' else -.1
            command=actions[i].copy()
            if self.vehicle=='fixedwing':command[-1]=command[-1]/2+.5
            task.env.set_setpoint(0,command)
        ratio=self.tasks[0].env_step_ratio
        for _ in range(ratio):
            active=[i for i,t in enumerate(self.tasks) if not(t.termination or t.truncation)]
            if not active:break
            self._aviary_step(active)
            for i in active:
                task=self.tasks[i];task.compute_state();task.compute_term_trunc_reward()
        for task in self.tasks:task.step_count+=1
        rewards=np.array([t.reward for t in self.tasks],dtype=np.float64)
        terminated=np.array([t.termination for t in self.tasks],dtype=bool)
        truncated=np.array([t.truncation for t in self.tasks],dtype=bool)
        self.done=terminated|truncated
        return self.observations(),rewards,terminated,truncated,[t.info.copy() for t in self.tasks]

    def close(self):
        if not self._closed and self.owner is not None:
            GenesisBulletClient.disconnect(self.owner);self._closed=True
