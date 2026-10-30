"""Frozen reference assets and an explicit, serializable controlled-plant contract."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from functools import lru_cache

ASSETS=Path(__file__).parent/'assets'


def inertial_basis(link):
    """Rotation from Genesis' inertial principal axes to its authored link axes."""
    w,x,y,z=np.asarray(link.desc.inertial_quat,dtype=float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


@lru_cache(maxsize=1)
def verify_installed_reference():
    import importlib.metadata
    import PyFlyt
    if importlib.metadata.version('PyFlyt')!='0.29.0':raise RuntimeError('Baseline must be frozen PyFlyt 0.29.0')
    base=Path(PyFlyt.__file__).parent
    manifest=json.loads((ASSETS/'reference_manifest.json').read_text())
    for name,expected in manifest['source_files_sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Baseline source changed: {name}')
    for name,expected in manifest['assets'].items():
        vehicle=name.split('.')[0]
        if hashlib.sha256((base/'models/vehicles'/vehicle/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Baseline asset changed: {name}')
    return 'PyFlyt 0.29.0 source and physical assets verified by SHA-256'


def verify_missile_assets():
    """SHA-256 check for the project-authored missile assets.

    The missile URDF/YAML are NOT PyFlyt 0.29.0 upstream files, so they must
    NOT be added to source_manifest.json (which certifies upstream provenance).
    They carry their own manifest next to them; TensorPlant calls this before
    loading the vehicle.
    """
    base=Path(__file__).parent/'original_pyflyt/models/vehicles/missile'
    manifest=json.loads((base/'asset_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'missile asset changed: {name}')
    return 'project missile assets verified by SHA-256'


def asset_inertials(vehicle):
    name='cf2x' if vehicle=='quadx' else vehicle
    text=(ASSETS/f'reference_{name}.urdf').read_text()
    # The original rocket XML has a repeated closing tag.
    root=ET.fromstring(text[:text.index('</robot>')+8])
    records={}
    for link in root.findall('link'):
        i=link.find('inertial');mass=float(i.find('mass').get('value'))
        inertia=i.find('inertia')
        diag=np.diag([float(inertia.get(k)) for k in ['ixx','iyy','izz']])
        for a,b,k in [(0,1,'ixy'),(0,2,'ixz'),(1,2,'iyz')]:diag[a,b]=diag[b,a]=float(inertia.get(k))
        origin=i.find('origin')
        com=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        records[link.get('name')]={'mass':mass,'inertia':diag,'position':np.zeros(3),'local_com':com}
    # All massive reference links have identity joint rotations; zero-mass legs do not affect aggregate inertia.
    for joint in root.findall('joint'):
        child=joint.find('child').get('link');o=joint.find('origin')
        if o is not None:records[child]['position']=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
    return records


def contract(vehicle,dt,decimation,mapping='default'):
    name='cf2x' if vehicle=='quadx' else vehicle
    manifest=json.loads((ASSETS/'reference_manifest.json').read_text())
    hashes={}
    for ext in ['urdf','yaml']:
        path=ASSETS/f'reference_{name}.{ext}'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=manifest['assets'][f'{name}.{ext}']:raise RuntimeError('Reference asset changed')
        hashes[ext]=actual
    plant={'reference':'PyFlyt 0.29.0','vehicle':vehicle,'reference_assets_sha256':hashes,
           'mapping':mapping,'frame':'ENU, body/world getters explicit, quaternion wxyz',
           'noise':'disabled','wind':'specified world velocity, per environment'}
    plant['plant_id']=hashlib.sha256(json.dumps(plant,sort_keys=True).encode()).hexdigest()[:20]
    return {'plant':plant,'simulation':{'dt':dt,'control_decimation':decimation,
            'genesis_integrator':'Euler','gravity_m_s2':[0,0,-9.81],
            'ground_contact':'not part of free-flight equivalence certificate'},
            'initial_conditions':'stored separately; fuel, velocities, pose and actuator states must match'}


def verify_rocket_v2_assets():
    """Keep the upstream asset intact; verify the separately versioned contact model."""
    base=Path(__file__).parent/'original_pyflyt/models/vehicles/rocket'
    manifest=json.loads((base/'benchmark_v2_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'rocket benchmark asset changed: {name}')
    return manifest
