"""Weather replay checks against the frozen observations, independent of landing policy."""

import json
import os
import unittest

import numpy as np
import torch

from environments.rocket.measured_wind import GROUND_Z_M, WEATHER, measured_field


class MeasuredWindTests(unittest.TestCase):
    def test_sounding_year_splits_and_masked_assignment(self):
        records = [json.loads(row) for row in (WEATHER / "kxmr.jsonl").read_text().splitlines()]
        expected = {"train": (361, 2018, 2021), "dev": (156, 2022, 2023), "test": (54, 2024, 2024)}
        for split, (count, start, end) in expected.items():
            wind = measured_field("kxmr-" + split, 2, "cpu", 0, 40, 0, "moderate")
            subset = [r for r in records if start <= int(r["utc"][:4]) <= end]
            self.assertEqual(len(wind.stamps), count)
            self.assertEqual(wind.stamps, [r["utc"] for r in subset])
            wind.reset(torch.tensor([True, True]))
            # Interpolate directly between reported sounding levels at two exact grid heights.
            heights = np.array([125.0, 1500.0])
            samples = wind.sample(torch.tensor(heights + GROUND_Z_M)).numpy()
            for lane, i in enumerate(wind.ids.tolist()):
                levels = np.array(subset[i]["levels"])
                uv = [np.interp(heights[lane], levels[:, 0], v) for v in levels[:, 1:].T]
                np.testing.assert_allclose(samples[lane], [*uv, 0], atol=2e-5)
            before = wind.ids.clone()
            wind.reset(torch.tensor([True, False]))
            self.assertNotEqual(before[0], wind.ids[0])
            self.assertEqual(before[1], wind.ids[1])

    def test_tower_clock_and_height_against_recorded_samples(self):
        with np.load(WEATHER / "cper.npz", allow_pickle=False) as data:
            for i, stamp in enumerate(data["utc"]):
                date = stamp[:10]
                wind = measured_field("cper-" + date, 3, "cpu", 60, 60, 0, "moderate")
                h = torch.tensor([data["mast"][i], 0, data["anchor"][i, 0]]) + GROUND_Z_M
                for tick in range(5):
                    flow = wind.step(altitude=h, airspeed=None).numpy()
                    raw = data["series"][i, tick // 3]
                    np.testing.assert_allclose(flow[0], raw, atol=3e-6)
                    np.testing.assert_allclose(flow[1], [*raw[:2], 0], atol=3e-6)
                    np.testing.assert_allclose(flow[2], [*data["anchor"][i, 1:], 0], atol=2e-5)
                wind.reset(torch.tensor([True, False, False]))
                out = wind.sample(h)
                np.testing.assert_allclose(out[0], data["series"][i, 0], atol=3e-6)
                np.testing.assert_allclose(out[1, :2], data["series"][i, 1, :2], atol=3e-6)

    def test_near_surface_correction_and_invalid_weather(self):
        name = "2025-04-21"
        base = measured_field("cper-" + name, 2, "cpu", 0, 40, 0, "moderate")
        scaled = measured_field("cper-surface-" + name, 2, "cpu", 0, 40, 0, "moderate")
        h = torch.tensor([1.5, base.mast]) + GROUND_Z_M
        value, corrected = base.sample(h), scaled.sample(h)
        torch.testing.assert_close(corrected[0, :2], value[0, :2] * 0.6256773560234744)
        torch.testing.assert_close(corrected[:, 2], value[:, 2])
        torch.testing.assert_close(corrected[1], value[1])
        with self.assertRaisesRegex(ValueError, "time-resolved"):
            measured_field("cper-" + name, 1, "cpu", 0, 40, 0.3, "moderate")
        with self.assertRaisesRegex(ValueError, "Unknown CPER date"):
            measured_field("cper-1999-01-01", 1, "cpu", 0, 40, 0, "moderate")


@unittest.skipUnless(os.environ.get("ROCKET_INTEGRATION") == "1", "opt-in Genesis simulation")
class MeasuredWindIntegrationTests(unittest.TestCase):
    def test_full_scale_step_reads_plant_height_and_masked_reset_preserves_neighbor(self):
        from environments.rocket import RocketLandingEnv

        for name in ("kxmr-test", "cper-2025-04-21"):
            env = RocketLandingEnv(
                2,
                "cpu",
                seed=7,
                benchmark_version="v1",
                measured_wind=name,
                aero="realistic",
                domain_randomize=False,
            )
            try:
                pos, *_ = env.plant.state()
                expected = env._wind.sample(pos[:, 2])
                obs, *_ = env.step(torch.zeros(2, 7, device=env.device))
                self.assertEqual(tuple(obs["policy"].shape), (2, 30))
                self.assertTrue(torch.isfinite(obs["policy"]).all())
                torch.testing.assert_close(env.plant.wind, expected.to(env.plant.wind))
                self.assertGreater(env.plant.wind.norm().item(), 0)
                before = env.plant.wind[1].clone()
                env.reset_idx(torch.tensor([True, False], device=env.device))
                torch.testing.assert_close(before, env.plant.wind[1])
                self.assertEqual(env.benchmark_env_kwargs["measured_wind"], name)
            finally:
                env.close()
