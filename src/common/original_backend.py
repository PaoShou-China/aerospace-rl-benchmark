"""Headless Genesis implementation of the BulletClient calls used by three PyFlyt tasks.

Task/control/aerodynamic source is imported from the frozen original_pyflyt package.
This is an API compatibility backend, not a claim of bitwise physics equivalence.
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import genesis as gs
from .consistency import inertial_basis


def initialize_genesis(backend):
    if backend not in ('cpu','gpu'):raise ValueError('backend must be cpu or gpu')
    if backend=='gpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA backend requested, but CUDA-enabled PyTorch and a CUDA device are unavailable')
    if gs._initialized:
        expected='cuda' if backend=='gpu' else 'cpu'
        if torch.device(gs.device).type!=expected:
            raise RuntimeError('Genesis backend is process-global; select CPU or GPU before constructing the first environment')
    if not gs._initialized:
        gs.init(backend=gs.cpu if backend=='cpu' else gs.gpu,logging_level='warning')


def getQuaternionFromEuler(rpy):return tuple(Rotation.from_euler('xyz',rpy).as_quat())
def getEulerFromQuaternion(q):return tuple(Rotation.from_quat(q).as_euler('xyz'))
def getMatrixFromQuaternion(q):return tuple(Rotation.from_quat(q).as_matrix().ravel())

bullet_api=SimpleNamespace(DIRECT=0,GUI=1,LINK_FRAME=1,WORLD_FRAME=2,URDF_USE_INERTIA_FROM_FILE=2,
    getQuaternionFromEuler=getQuaternionFromEuler,getEulerFromQuaternion=getEulerFromQuaternion,
    getMatrixFromQuaternion=getMatrixFromQuaternion,isNumpyEnabled=lambda:True)
bullet_data=SimpleNamespace(getDataPath=lambda:'')


class GenesisBulletClient:
    LINK_FRAME=1;WORLD_FRAME=2;URDF_USE_INERTIA_FROM_FILE=2
    getQuaternionFromEuler=staticmethod(getQuaternionFromEuler)
    getEulerFromQuaternion=staticmethod(getEulerFromQuaternion)
    getMatrixFromQuaternion=staticmethod(getMatrixFromQuaternion)

    def __init__(self,connection_mode=0,**kwargs):
        if connection_mode!=0:raise NotImplementedError('Original-task backend currently supports headless operation only')
        initialize_genesis(getattr(self,'_backend','cpu'));self._scene=None;self._built=False;self._bodies=[];self._pending_pad=None
        self._dt=1/240;self._gravity=(0,0,-9.81);self._contacts=[];self._client=0
        self._temp=tempfile.TemporaryDirectory(prefix='pyflyt-genesis-')
    def resetSimulation(self):
        self.disconnect();self._bodies=[];self._built=False;self._pending_pad=None;self._contacts=[]
        # integrator left at the Genesis default (approximate_implicitfast):
        # an explicit Euler override here made pad contacts jitter at ~0.1 m/s
        # (measured 2026-10-20, controlled upright drop probe), which keeps
        # |v| above the 0.02 m/s settle threshold and makes landing successes
        # unreachable. iterations/ls_iterations stay at defaults (25/50).
        self._scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=self._dt,gravity=self._gravity),
            rigid_options=gs.options.RigidOptions(batch_links_info=True,enable_self_collision=False))
    def disconnect(self):
        if self._scene is not None:self._scene.destroy();self._scene=None
    def setGravity(self,x,y,z):
        if (x,y,z)!=(0,0,-9.81):raise NotImplementedError('Only original default gravity supported')
    def setAdditionalSearchPath(self,*a,**k):pass
    def addUserDebugText(self,*a,**k):return -1
    def resetDebugVisualizerCamera(self,*a,**k):pass
    def changeVisualShape(self,*a,**k):pass
    def getNumBodies(self):return len(self._bodies)
    def getBodyUniqueId(self,i):return i
    def _build(self):
        if not self._built:
            self._scene.build(n_envs=getattr(self,'_num_envs',1));self._built=True
            for b in self._bodies:self._allocate(b)
    def _allocate(self,b):
        e=b['entity'];b['force']=torch.zeros((getattr(self,'_num_envs',1),e.n_links,3),device=gs.device);b['torque']=torch.zeros_like(b['force'])
    def loadURDF(self,fileName,basePosition=(0,0,0),baseOrientation=(0,0,0,1),useFixedBase=False,flags=0,globalScaling=1.,**kwargs):
        name=Path(fileName).name
        if name!='plane.urdf':
            # Bullet tolerates the upstream rocket's duplicated closing tag; XML parsers do not.
            raw=Path(fileName).read_text();root=ET.fromstring(raw.split('</robot>')[0]+'</robot>')
            for mesh in root.findall('.//mesh'):
                mesh.set('filename',str((Path(fileName).parent/mesh.attrib['filename']).resolve()))
            runtime_path=Path(self._temp.name)/name;ET.ElementTree(root).write(runtime_path)
        if name=='landing_pad.urdf' and self._pending_pad is not None:
            e=self._pending_pad;self._pending_pad=None
        else:
            if self._built:raise NotImplementedError('Adding arbitrary bodies after scene build is unsupported')
            if name=='plane.urdf':
                # PyBullet's standard plane.urdf collision is a finite 200 x 200 x 10 box.
                e=self._scene.add_entity(gs.morphs.Box(pos=(0,0,-5*globalScaling),
                    size=(200*globalScaling,200*globalScaling,10*globalScaling),fixed=True),
                    material=gs.materials.Rigid(friction=1.))
            else:
                e=self._scene.add_entity(gs.morphs.URDF(file=str(runtime_path),
                    pos=tuple(basePosition),quat=(baseOrientation[3],*baseOrientation[:3]),fixed=useFixedBase,
                    scale=globalScaling,merge_fixed_links=False,align=False,recompute_inertia=False),
                    material=gs.materials.Rigid(friction=.5))
                if name=='rocket.urdf':
                    pad=Path(__file__).parent.parent/'original_pyflyt/models/landing_pad.urdf'
                    self._pending_pad=self._scene.add_entity(gs.morphs.URDF(file=str(pad),pos=(0,0,.1),fixed=True,align=False),material=gs.materials.Rigid(friction=.5))
        links=[]
        if name!='plane.urdf':
            links=[j.find('child').attrib['link'] for j in root.findall('joint')]
        frames={}
        if name!='plane.urdf':
            for link in root.findall('link'):
                origin=link.find('inertial/origin')
                rpy=[0,0,0] if origin is None else [float(x) for x in origin.attrib.get('rpy','0 0 0').split()]
                frames[link.attrib['name']]=Rotation.from_euler('xyz',rpy).as_matrix()
        b={'entity':e,'names':links,'file':str(fileName),'plane':name=='plane.urdf','fixed':useFixedBase,'inertial_rotations':frames}
        self._bodies.append(b)
        if self._built:self._allocate(b)
        return len(self._bodies)-1
    def _index(self,b,idx):
        if idx==-1:return 0
        name=b['names'][idx]
        return next(l.idx_local for l in b['entity'].links if l.name==name)
    def getNumJoints(self,body):return len(self._bodies[body]['names'])
    def getJointInfo(self,body,idx):
        a=[0]*17;a[12]=self._bodies[body]['names'][idx].encode();return tuple(a)
    def resetBasePositionAndOrientation(self,body,pos,quat):
        self._build();e=self._bodies[body]['entity'];e.set_pos(torch.as_tensor(np.asarray(pos)[None],dtype=gs.tc_float,device=gs.device),envs_idx=[self._env_idx],zero_velocity=True)
        e.set_quat(torch.tensor([[quat[3],*quat[:3]]],dtype=gs.tc_float,device=gs.device),envs_idx=[self._env_idx],zero_velocity=True)
    def resetBaseVelocity(self,body,linearVelocity=(0,0,0),angularVelocity=(0,0,0)):
        self._build();e=self._bodies[body]['entity'];q=e.get_quat()[self._env_idx:self._env_idx+1];from .dynamics import inverse_rotate
        w=inverse_rotate(q,torch.tensor([angularVelocity],dtype=gs.tc_float,device=gs.device))
        e.set_dofs_velocity(torch.cat((torch.tensor([linearVelocity],dtype=gs.tc_float,device=gs.device),w),-1),envs_idx=[self._env_idx])
    def getBasePositionAndOrientation(self,body):
        self._build();e=self._bodies[body]['entity'];p=e.get_pos()[self._env_idx].cpu().numpy();q=e.get_quat()[self._env_idx].cpu().numpy();return p,np.array([*q[1:],q[0]])
    def getBaseVelocity(self,body):
        self._build();e=self._bodies[body]['entity'];return e.get_vel()[self._env_idx].cpu().numpy(),e.get_ang()[self._env_idx].cpu().numpy()
    def getLinkStates(self,body,indices,computeLinkVelocity=False,computeForwardKinematics=False):
        self._build();b=self._bodies[body];e=b['entity'];ids=[self._index(b,i) for i in indices]
        ps=e.get_links_pos(links_idx_local=ids)[self._env_idx].cpu().numpy();qs=e.get_links_quat(links_idx_local=ids)[self._env_idx].cpu().numpy()
        vs=e.get_links_vel(links_idx_local=ids,ref=gs.link_ref_frame.link_COM)[self._env_idx].cpu().numpy();ws=e.get_links_ang(links_idx_local=ids)[self._env_idx].cpu().numpy()
        com=e.get_links_COM(links_idx_local=ids)[self._env_idx].cpu().numpy()
        rows=[]
        for idx,p,q,v,w,c in zip(ids,ps,qs,vs,ws,com):
            rot=Rotation.from_quat([*q[1:],q[0]]).as_matrix();pc=p+rot@c
            ir=b['inertial_rotations'][e.links[idx].name]
            rows.append((pc,Rotation.from_matrix(rot@ir).as_quat(),c,Rotation.from_matrix(ir).as_quat(),p,np.array([*q[1:],q[0]]),v,w))
        return rows
    def getLinkState(self,body,link,**kw):return self.getLinkStates(body,[link],**kw)[0]
    def changeDynamics(self,body,link,mass=None,localInertiaDiagonal=None,linearDamping=None,angularDamping=None,**kw):
        if kw:raise NotImplementedError(f'Unsupported dynamics settings: {list(kw)}')
        if mass is None and localInertiaDiagonal is None:
            if linearDamping not in (None,0.) or angularDamping not in (None,0.):raise NotImplementedError('Nonzero Bullet damping unsupported')
            return
        self._build();b=self._bodies[body];idx=self._index(b,link);e=b['entity']
        if mass is not None:e.set_links_mass(torch.tensor([[max(float(mass),1e-8)]],device=gs.device,dtype=gs.tc_float),links_idx_local=[idx],envs_idx=[self._env_idx])
        if localInertiaDiagonal is not None:
            basis=torch.tensor(inertial_basis(e.links[idx]),device=gs.device,dtype=gs.tc_float)
            inertia=basis.T@torch.diag(torch.tensor(localInertiaDiagonal,device=gs.device,dtype=gs.tc_float).clamp_min(1e-8))@basis
            e.set_links_inertia(inertia[None,None],links_idx_local=[idx],envs_idx=[self._env_idx])
    def applyExternalForce(self,body,link,forceObj,posObj,flags):
        self._build();b=self._bodies[body];idx=self._index(b,link)
        if flags!=self.LINK_FRAME:raise NotImplementedError('Only source component LINK_FRAME force calls supported')
        rot=self._inertial_world_rotation(b,idx)
        f=torch.tensor(rot@np.asarray(forceObj),device=gs.device,dtype=gs.tc_float);pos=torch.tensor(rot@np.asarray(posObj),device=gs.device,dtype=gs.tc_float)
        b['force'][self._env_idx,idx]+=f;b['torque'][self._env_idx,idx]+=torch.linalg.cross(pos,f)
    def _inertial_world_rotation(self,b,idx):
        e=b['entity'];q=e.get_links_quat(links_idx_local=[idx])[self._env_idx,0].cpu().numpy()
        return Rotation.from_quat([*q[1:],q[0]]).as_matrix()@b['inertial_rotations'][e.links[idx].name]
    def applyExternalTorque(self,body,link,torqueObj,flags):
        self._build();b=self._bodies[body];idx=self._index(b,link)
        if flags!=self.LINK_FRAME:raise NotImplementedError('Only LINK_FRAME torque supported')
        b['torque'][self._env_idx,idx]+=torch.tensor(self._inertial_world_rotation(b,idx)@np.asarray(torqueObj),device=gs.device,dtype=gs.tc_float)
    def stepSimulation(self):
        self._build()
        for b in self._bodies:
            if not b['fixed']:b['entity'].apply_links_external_wrench(force=b['force'],torque=b['torque'],ref=gs.link_ref_frame.link_COM,local=False)
        self._scene.step()
        for b in self._bodies:b['force'].zero_();b['torque'].zero_()
        self._contacts=[]
        for i,b in enumerate(self._bodies):
            if b['fixed']:continue
            for j,c in enumerate(self._bodies):
                if i==j:continue
                contacts=b['entity'].get_contacts(with_entity=c['entity'])
                if contacts['valid_mask'].any().item():self._contacts.append((0,i,j))
    def getContactPoints(self,*a,**k):return self._contacts

from .original_batch import ConfiguredBulletClient, engine_configuration

bullet_utils=SimpleNamespace(BulletClient=ConfiguredBulletClient)


def make_original_task(vehicle,backend='cpu',**kwargs):
    import functools
    with engine_configuration(backend=backend):
        task=_make_original_task(vehicle,**kwargs)
    original_reset=task.reset
    @functools.wraps(original_reset)
    def reset(*args,**reset_kwargs):
        with engine_configuration(backend=backend):
            return original_reset(*args,**reset_kwargs)
    task.reset=reset
    return task

def _make_original_task(vehicle,**kwargs):
    """Original default constructors; kwargs are passed through without replacement."""
    if kwargs.get('render_mode') is not None:raise NotImplementedError('Original-task Genesis backend is headless')
    if vehicle=='quadx':
        from ..original_pyflyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv
        return QuadXHoverEnv(**kwargs)
    if vehicle=='fixedwing':
        from ..original_pyflyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv
        return FixedwingWaypointsEnv(**kwargs)
    if vehicle=='rocket':
        from ..original_pyflyt.gym_envs.rocket_envs.rocket_landing_env import RocketLandingEnv
        return RocketLandingEnv(**kwargs)
    raise ValueError(vehicle)
