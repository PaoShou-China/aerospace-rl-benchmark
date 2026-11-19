"""Regression tests for the v1 terminal result and historical eval reports."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.first_episode import FirstEpisodes
from scripts.landing_score import evaluation_rate, reusable_evaluation


class LandingReportTests(unittest.TestCase):
    def test_old_v1_score_uses_terminal_counts_and_requires_replay_for_rows(self):
        report = dict(benchmark={"version": "v1", "primary_metric": "ever_stable_2s"},
                      complete=True, episodes_expected=2, episodes_finished=2, successes=2,
                      success_rate=1.0, metric_counts={"terminal_stable_2s": 1},
                      episodes=[{"complete": True, "success": True} for _ in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "eval.json"
            p.write_text(json.dumps(report))
            self.assertEqual(evaluation_rate("rocket", report), 0.5)
            self.assertFalse(reusable_evaluation("rocket", p))
            report["benchmark"]["primary_metric"] = "terminal_stable_2s"
            report["episodes"][1]["success"] = False
            report.update(successes=1, success_rate=0.5)
            p.write_text(json.dumps(report))
            self.assertTrue(reusable_evaluation("rocket", p))
        self.assertEqual(evaluation_rate("rocket", {**report, "metric_counts": {"terminal_stable_2s": 0}}), 0.0)
        for invalid in ({"complete": False}, {"episodes_finished": 1},
                        {"metric_counts": {}}, {"metric_counts": {"terminal_stable_2s": 3}}):
            self.assertIsNone(evaluation_rate("rocket", {**report, **invalid}))
        report["benchmark"] = {"version": "v2", "primary_metric": "ever_stable_2s"}
        self.assertEqual(evaluation_rate("rocket", {**report, "success_rate": 1.0}), 1.0)

    @unittest.skipUnless(os.environ.get("ROCKET_INTEGRATION") == "1", "opt-in Genesis simulation")
    def test_first_episode_and_training_log_require_settling_at_the_end(self):
        from environments.rocket import RocketLandingEnv

        env = RocketLandingEnv(12, "cpu", 1011, benchmark_version="v1", aero="realistic")
        try:
            self.assertEqual(env.cfg["benchmark"]["primary_metric"], "terminal_stable_2s")
            p, q, v, w = [torch.zeros_like(x) for x in env.plant.state()]
            q[:, 0] = 1
            # Calculate upright support height from the collision asset (works with
            # both the upstream legs and the corrected leg-end collision model).
            import xml.etree.ElementTree as ET
            assets = Path(__file__).resolve().parents[1] / "environments/common/original_pyflyt/models"
            rocket = ET.parse(assets / "vehicles/rocket/rocket_falcon9_stage1.urdf")
            collision = rocket.find("link[@name='leg_1']/collision")
            leg_z = float(rocket.find("joint[@name='leg_1_joint']/origin").get("xyz").split()[2])
            ball_z = float(collision.find("origin").get("xyz").split()[2])
            radius = float(collision.find("geometry/sphere").get("radius"))
            p[:, 2] = 0.155 - leg_z - ball_z + radius
            env.plant.set_state(p, q, v, w)
            mask = torch.ones(env.num_envs, dtype=torch.bool)
            env._observe(mask)
            env._sensor.reset(mask)
            env._update_observation()
            env.max_steps = 130
            cohort = FirstEpisodes(env)
            actions = torch.zeros(env.num_envs, 7)
            actions[:, 3:5] = -1
            for tick in range(130):
                if tick == 100:
                    self.assertTrue(env.landing_metrics.ever_stable.all())
                    p, q, v, w = [x.clone() for x in env.plant.state()]
                    p[6:, 0] = 60
                    v[6:, 2] = -8
                    env.plant.set_state(p, q, v, w)
                _, _, term, trunc, extras = env.step(actions)
                cohort.update(env)
                if term.any():
                    self.assertEqual(float(extras["episode"]["success_rate"]), 0.0)
                    partial = cohort.report()
                    self.assertFalse(partial["complete"])
                    self.assertIsNone(partial["success_rate"])
                    self.assertTrue(all(row["success"] is None for row in partial["episodes"] if not row["complete"]))
                if trunc.any():
                    self.assertEqual(float(extras["episode"]["success_rate"]), 1.0)
            r = cohort.report()
            self.assertTrue(r["complete"])
            self.assertEqual(r["metric_counts"]["ever_stable_2s"], 12)
            self.assertEqual(r["metric_counts"]["stable_then_failure"], 6)
            self.assertEqual(r["metric_counts"]["terminal_stable_2s"], 6)
            self.assertEqual(r["success_rate"], 0.5)
            self.assertEqual(r["successes"], 6)
            self.assertEqual([row["success"] for row in r["episodes"]], [True]*6 + [False]*6)
        finally:
            env.close()
