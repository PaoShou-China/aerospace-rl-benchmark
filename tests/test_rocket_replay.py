"""Data/observer tests run with NumPy only; they do not simulate Genesis."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from visualization.rocket_data import Episode, SCHEMA, asset_geometry, playback_times, slerp
from visualization.rocket_recorder import RocketRecorder
from visualization.rocket_render import episode_digest, export_html, overlay


def fixture():
    geometry = asset_geometry()
    metadata = dict(geometry, position_frame='ENU', quaternion_order='wxyz', fuel_capacity_kg=410.9,
                    initial_fuel_mass_kg=20., terminal_height_m=20.)
    def sample(t, z, mass, contact=False, complete=False):
        return dict(t_s=t, position_m=[0.,0.,z], quaternion_wxyz=[1.,0.,0.,0.],
            linear_velocity_mps=[0.,0.,-1. if not contact else 0.], angular_velocity_radps=[0.,0.,0.],
            fin_actual=[0.]*4, gimbal_rad=[0.,0.], throttle_actual=.5, ignited=True,
            fuel_mass_kg=mass, fuel_fraction=mass/410.9, pad_contact=contact, ground_contact=False,
            complete=complete, terminated=False, truncated=False)
    return dict(schema=SCHEMA, demo=True, metadata=metadata,
        samples=[sample(0.,4.575,20.), sample(1.,3.575,19.), sample(2.,2.575,18.,True), sample(3.,2.575,18.,True,True)],
        events=[dict(t_s=1.,type='terminal_start'),dict(t_s=2.,type='pad_contact'),dict(t_s=3.,type='settled')])


def fake_env():
    state = [np.array([[0.,0.,10.]]),np.array([[1.,0.,0.,0.]]),np.array([[0.,0.,-1.]]),np.zeros((1,3))]
    plant = SimpleNamespace(state=lambda:tuple(a.copy() for a in state),
        drone=SimpleNamespace(get_pos=lambda:state[0]),_geometry=SimpleNamespace(_dt=1/240),
        params={'booster_params':dict(total_fuel=410.9,gimbal_range_degrees=5.)},
        surfaces=SimpleNamespace(actuation=np.zeros((1,4))),gimbal=np.zeros((1,2)),
        throttle=np.zeros((1,1)),ignited=np.array([False]),fuel=np.array([.05]),
        pad_contact=np.array([False]),ground_contact=np.array([False]))
    env=SimpleNamespace(num_envs=1,plant=plant,device='fake-test',cfg={},agent_hz=60,inner_steps=2,max_steps=1800,
        obs_buf=np.zeros((1,30)),complete=np.array([False]),terminated=np.array([False]),
        truncated=np.array([False]),reset_buf=np.array([False]),reward=np.array([0.]),reward_components={})
    return env,state


class DataTests(unittest.TestCase):
    def test_asset_calibration(self):
        g=asset_geometry()
        self.assertEqual(len(g['foot_points_body_m']),24)
        self.assertAlmostEqual(g['target_position_m'][2],2.575)

    def test_interpolation_and_no_future_flags(self):
        e=Episode(fixture())
        self.assertAlmostEqual(e.at(.5)['position_m'][2],4.075)
        self.assertFalse(e.at(2.99)['complete'])
        self.assertTrue(e.at(3.)['complete'])
        self.assertAlmostEqual(np.linalg.norm(slerp([1,0,0,0],[-1,0,0,0],.5)),1.)

    def test_event_clocks_and_fuel(self):
        e=Episode(fixture())
        m=e.metrics(1.5)
        self.assertIsNone(m['contact_time_s'])
        self.assertAlmostEqual(m['terminal_elapsed_s'],.5)
        end=e.metrics(3.)
        self.assertEqual(end['landing_elapsed_s'],2.)
        self.assertEqual(end['settled_time_s'],3.)
        self.assertAlmostEqual(end['fuel_used_kg'],2.)
        self.assertAlmostEqual(end['fuel_used_percent'],10.)
        self.assertAlmostEqual(end['foot_height_m'],0.)

    def test_real_time_and_slow_motion(self):
        self.assertEqual(len(playback_times(30,30,1)),900)
        self.assertEqual(len(playback_times(2,60,.25)),480)
        self.assertTrue(np.allclose(np.diff(playback_times(2,60,.25)),1/240))
        for fps,speed in [(0,1),(30,0)]:
            with self.assertRaises(ValueError):playback_times(2,fps,speed)

    def test_validation_and_roundtrip(self):
        d=fixture();d['samples'][2]['t_s']=1.
        with self.assertRaises(ValueError):Episode(d)
        d=fixture();d['samples'][0]['quaternion_wxyz']=[0.]*4
        with self.assertRaises(ValueError):Episode(d)
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'episode';e=Episode(fixture());e.save(path)
            restored=Episode.load(path)
            self.assertEqual(restored.metrics(3.)['contact_time_s'],2.)
            with np.load(path/'states.npz',allow_pickle=False) as data:
                self.assertEqual(data['position_m'].shape,(4,3))
            with self.assertRaises(FileExistsError):e.save(path)

    def test_reject_bad_frames_geometry_and_fuel_reference(self):
        for key,value in [('velocity_frame','BODY'),('initial_fuel_mass_kg',800.),('foot_points_body_m',[[0.,0.]])]:
            d=fixture();d['metadata'][key]=value
            with self.assertRaises(ValueError):Episode(d)

    def test_offline_html_requires_matching_video(self):
        e=Episode(fixture())
        with tempfile.TemporaryDirectory() as p:
            p=Path(p);v=p/'video.mp4';v.write_bytes(b'UNIT TEST PLACEHOLDER, NOT A VIDEO')
            sidecar=dict(episode_sha256=episode_digest(e),fps=30,speed=1.,start_s=0.,end_s=3.)
            v.with_suffix('.timeline.json').write_text(json.dumps(sidecar))
            export_html(e,v,p/'share.html')
            page=(p/'share.html').read_text()
            self.assertIn('data:video/mp4;base64,',page)
            self.assertNotIn('src="https://',page)
            self.assertIn('示意数据',page)
            changed=copy.deepcopy(e.data);changed['metadata']['initial_fuel_mass_kg']=21.
            with self.assertRaises(ValueError):export_html(Episode(changed),v,p/'wrong.html')

    def test_overlay_dimensions(self):
        frame=np.zeros((720,980,3),dtype=np.uint8);e=Episode(fixture())
        result=overlay(frame,frame,e.metrics(1.),e,1280,720)
        self.assertEqual(result.shape,(720,1280,3))
        self.assertGreater(int(result.max()),0)


class ObserverTests(unittest.TestCase):
    def test_pre_reset_snapshot_and_contact_latch(self):
        env,state=fake_env();r=RocketRecorder(env,seed=7,source='UNIT_TEST_NOT_SIMULATION',config={})
        r.begin_policy(env,np.zeros((1,7)),np.zeros((1,7)))
        env.plant.pad_contact[0]=True;state[0][0,2]=2.575
        for j in range(2):r.after_physics(env,np.array([True]),tuple(a.copy() for a in state),0,j)
        env.terminated[0]=True;env.reset_buf[0]=True
        r.after_control(env,np.array([True]),0,np.array([2.]),np.array([0.]),np.array([0.]),{'hard_pad_speed':np.array([True])})
        # An inactive control period must not append a fake continued trajectory.
        r.after_physics(env,np.array([False]),tuple(state),1,0)
        env.plant.pad_contact[0]=False
        r.after_policy(env)
        state[0][0,2]=450.;env.terminated[0]=False
        e=r.episode()
        self.assertEqual(len(e.times),3)
        self.assertAlmostEqual(e.duration,1/120)
        self.assertEqual(e.data['samples'][-1]['position_m'][2],2.575)
        self.assertTrue(e.data['samples'][-1]['terminated'])
        self.assertTrue(e.data['samples'][-1]['pad_contact'])
        self.assertEqual(len([x for x in e.data['events'] if x['type']=='pad_contact']),1)
        self.assertEqual(e.data['result']['failure_reasons'],['hard_pad_speed'])
        self.assertFalse(e.data['result']['collection_limited'])

    def test_limit_is_not_environment_timeout(self):
        env,state=fake_env();r=RocketRecorder(env,seed=7,source='UNIT_TEST_NOT_SIMULATION',config={})
        r.begin_policy(env,np.zeros((1,7)),np.zeros((1,7)))
        r.after_physics(env,np.array([True]),tuple(state),0,0)
        e=r.episode()
        self.assertTrue(e.data['result']['collection_limited'])
        self.assertFalse(e.data['result']['truncated'])


if __name__ == '__main__':unittest.main()
