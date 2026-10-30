import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch

from environments.common.rocket_benchmark import LandingMetrics, RocketVariation
from scripts.first_episode import FirstEpisodes

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_two_seconds_is_span_not_sample_count(self):
        for hz in (40,60):
            m = LandingMetrics(1,'cpu',hz)
            good = torch.ones(1,dtype=torch.bool)
            bad = ~good
            zero = torch.zeros(1,3)
            for _ in range(2*hz):m.update(good,zero,zero,torch.zeros(1),bad)
            self.assertFalse(m.ever_stable.item())
            m.update(good,zero,zero,torch.zeros(1),bad)
            self.assertTrue(m.ever_stable.item())
            m.update(good,zero,zero,torch.zeros(1),good)
            v = m.values(good,good)
            self.assertTrue(v['stable_then_failure'].item())
            self.assertFalse(v['terminal_stable_2s'].item())
            m.reset(good)
            self.assertFalse(m.ever_stable.item())

    def test_gap_and_threshold_boundary_break_streak(self):
        m = LandingMetrics(2,'cpu',40)
        good = torch.ones(2,dtype=torch.bool);zero = torch.zeros(2,3)
        for _ in range(80):m.update(good,zero,zero,torch.zeros(2),~good)
        speed = zero.clone();speed[0,0] = .02
        contact = good.clone();contact[1] = False
        m.update(contact,speed,zero,torch.zeros(2),~good)
        self.assertEqual(m.streak.tolist(),[0,0])
        self.assertFalse(m.ever_stable.any())

    def test_first_episode_excludes_fast_replica_repeats(self):
        env = SimpleNamespace(num_envs=2,device='cpu',completed_episodes=torch.zeros(2),
            completed_successes=torch.zeros(2),completed_returns=torch.zeros(2),
            completed_metrics={'ever_stable_2s':torch.zeros(2)})
        c = FirstEpisodes(env)
        env.completed_episodes[0] = 1;env.completed_returns[0] = -10
        self.assertFalse(c.update(env))
        # Fast replica succeeds in its SECOND episode, slow one still flying.
        env.completed_episodes[0] = 2;env.completed_successes[0] = 1
        env.completed_returns[0] = 90;env.completed_metrics['ever_stable_2s'][0] = 1
        self.assertFalse(c.update(env))
        env.completed_episodes[1] = 1;env.completed_successes[1] = 1
        env.completed_returns[1] = 50;env.completed_metrics['ever_stable_2s'][1] = 1
        self.assertTrue(c.update(env))
        r = c.report()
        self.assertEqual(r['successes'],1)
        self.assertEqual(r['mean_episode_return'],20)
        self.assertEqual(r['metric_counts']['ever_stable_2s'],1)

    def test_partial_cohort_never_marked_complete(self):
        env = SimpleNamespace(num_envs=2,device='cpu',completed_episodes=torch.tensor([0,0]),
            completed_successes=torch.tensor([0,0]),completed_returns=torch.zeros(2))
        c = FirstEpisodes(env);env.completed_episodes[0] = 1;c.update(env)
        self.assertFalse(c.report()['complete'])
        self.assertEqual(c.report()['episodes_finished'],1)

    def test_only_leg_collision_geometry_changed(self):
        base = ROOT/'environments/common/original_pyflyt/models/vehicles/rocket'
        manifest = json.loads((base/'benchmark_v2_manifest.json').read_text())
        trees = []
        for filename in ('rocket.urdf','rocket_v2.urdf'):
            data = (base/filename).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),manifest['files'][filename])
            trees.append(ET.fromstring(data.split(b'</robot>')[0]+b'</robot>'))
        changed = 0
        for a,b in zip(trees[0].findall('link'),trees[1].findall('link')):
            if ET.tostring(a) == ET.tostring(b):continue
            changed += 1
            self.assertEqual(len(a.findall('collision')),1)
            self.assertEqual(len(b.findall('collision')),2)
            self.assertEqual(a.find('collision/geometry/box').get('size'),'0.05 0.5 0.05')
            self.assertEqual([x.get('radius') for x in b.findall('collision/geometry/sphere')],['0.025']*2)
            for link in (a,b):
                for col in link.findall('collision'):link.remove(col)
            # Ignore whitespace formatting after removed children.
            self.assertEqual(ET.tostring(a).strip(),ET.tostring(b).strip())
        self.assertEqual(changed,3)
        self.assertEqual([ET.tostring(x) for x in trees[0].findall('joint')],
                         [ET.tostring(x) for x in trees[1].findall('joint')])

    def test_v5_corridor_and_single_fatal_penalty(self):
        from environments.rocket.env import REWARD_KEYS, rocket_shaping
        cfg = dict(hover=0.,hold=0.,lateral=0.,descent=0.,decel=0.,yaw=0.,tilt=0.,
                   contact_base=0.,contact_vel=0.,success=5000.,corridor=.5,
                   corridor_accel=2.,corridor_v0=.5,fatal=500.)
        p = torch.tensor([[0.,0.,400.],[0.,0.,400.],[0.,0.,1.],[0.,0.,0.]],
                         dtype=torch.float64)
        vel_z = torch.tensor([-100.,-40.,-2.5,-.5],dtype=torch.float64)
        newly_fatal = torch.tensor([True,False,True,False])
        _,comps = rocket_shaping(cfg,p,p,vel_z,vel_z,torch.zeros(4,3,dtype=torch.float64),
            torch.zeros(4,3,dtype=torch.float64),torch.zeros(4,dtype=torch.bool),
            torch.zeros(4,dtype=torch.float64),torch.zeros(4,dtype=torch.bool),newly_fatal)
        self.assertEqual(set(comps),set(REWARD_KEYS))
        torch.testing.assert_close(comps['corridor'],torch.tensor([-29.75,0.,0.,0.],dtype=torch.float64))
        torch.testing.assert_close(comps['fatal'],torch.tensor([-500.,0.,-500.,0.],dtype=torch.float64))
        # A later step with no new terminal transition pays no second fatal cost.
        _,next_comps = rocket_shaping(cfg,p,p,vel_z,vel_z,torch.zeros(4,3,dtype=torch.float64),
            torch.zeros(4,3,dtype=torch.float64),torch.zeros(4,dtype=torch.bool),
            torch.zeros(4,dtype=torch.float64),torch.zeros(4,dtype=torch.bool),
            torch.zeros(4,dtype=torch.bool))
        self.assertEqual(next_comps['fatal'].abs().sum().item(),0.)

    def test_variation_uses_independent_masked_stream(self):
        from environments.common.plant import TensorRandom
        rng = TensorRandom(3,'cpu',7)
        v = RocketVariation(3,'cpu',TensorRandom(3,'cpu',7+104729),'shifted')
        control = RocketVariation(3,'cpu',TensorRandom(3,'cpu',7+104729),'shifted')
        before = v.thrust.clone()
        v.sample(torch.tensor([True,False,False]))
        self.assertTrue(torch.equal(v.thrust[1:],before[1:]))
        control.sample(torch.tensor([True,False,False]))
        self.assertTrue(torch.equal(v.wind,control.wind))
        self.assertEqual(rng.counter.tolist(),[0,0,0])
        self.assertTrue(((v.thrust>=.92)&(v.thrust<=1.08)).all())
        self.assertTrue((v.wind.norm(dim=-1)<=5.).all())


if __name__=='__main__':unittest.main()
