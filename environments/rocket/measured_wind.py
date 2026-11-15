"""Replay archived winds in world coordinates (east, north, up; metres per second)."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from environments.common.disturbances import WindDisturbance

WEATHER = Path(__file__).with_name("weather")
HEIGHTS_M = np.arange(0, 1521, 5)
SPLITS = {"train": range(2018, 2022), "dev": range(2022, 2024), "test": range(2024, 2025)}
# Ground reference used for the height lookup in the archived full-scale runs.
GROUND_Z_M = 0.15
# Five-minute scalar speed at the 1.5 m USCRN station relative to the 8.32 m CPER tower.
SURFACE_RATIO = 0.6256773560234744


def _grid_indices(height):
    scaled = (height / 5).clamp(0, len(HEIGHTS_M) - 1)
    lower = scaled.long()
    upper = (lower + 1).clamp_max(len(HEIGHTS_M) - 1)
    return lower, upper, (scaled - lower).unsqueeze(-1)


class SoundingWind:
    """One KXMR sounding per replica and episode, with optional synthetic gusts."""

    def __init__(self, split, n, device, seed, hz, dryden, severity):
        if split not in SPLITS:
            raise ValueError(f"Unknown KXMR split {split!r}; choose train, dev or test")
        grids = []
        self.stamps = []
        with (WEATHER / "kxmr.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                if int(record["utc"][:4]) in SPLITS[split]:
                    levels = np.asarray(record["levels"])
                    grids.append(
                        np.column_stack(
                            [
                                np.interp(HEIGHTS_M, levels[:, 0], component)
                                for component in levels[:, 1:].T
                            ]
                        )
                    )
                    self.stamps.append(record["utc"])
        self.grids = torch.as_tensor(np.stack(grids), device=device, dtype=torch.float32)
        # Preserve the archived lane assignment; a masked reset advances only that lane.
        count = len(grids)
        stride = next(i for i in range(17, 18 + count) if math.gcd(i, count) == 1)
        self.next_ids = (seed + torch.arange(n, device=device) * stride) % count
        self.ids = self.next_ids.clone()
        self.gusts = (
            WindDisturbance(n, device, seed + 42311, dryden=dryden, severity=severity, dt=1 / hz)
            if dryden
            else None
        )

    def reset(self, mask):
        self.ids.copy_(torch.where(mask, self.next_ids, self.ids))
        self.next_ids.copy_((self.next_ids + mask.long() * 31) % len(self.stamps))
        if self.gusts is not None:
            self.gusts.reset(mask)

    def sample(self, altitude):
        lo, hi, weight = _grid_indices(altitude.float() - GROUND_Z_M)
        horizontal = torch.lerp(self.grids[self.ids, lo], self.grids[self.ids, hi], weight)
        return torch.cat((horizontal, torch.zeros_like(horizontal[:, :1])), dim=-1)

    def step(self, *, altitude, airspeed):
        flow = self.sample(altitude)
        return flow + self.gusts.step() if self.gusts is not None else flow


class TowerWind:
    """Replay one 20 Hz CPER sonic minute beneath a regional upper-air column."""

    def __init__(self, date, n, device, hz, surface_correction):
        with np.load(WEATHER / "cper.npz", allow_pickle=False) as archive:
            dates = [stamp[:10] for stamp in archive["utc"].tolist()]
            if date not in dates:
                raise ValueError(f"Unknown CPER date {date!r}; available dates: {', '.join(dates)}")
            index = dates.index(date)
            self.series = torch.as_tensor(
                archive["series"][index], device=device, dtype=torch.float32
            )
            self.grid = torch.as_tensor(archive["grids"][index], device=device)
            self.anchor = torch.as_tensor(archive["anchor"][index], device=device)
            self.mast = float(archive["mast"][index])
        self.surface_correction = surface_correction
        self.hz = hz
        self.tick = torch.zeros(n, device=device, dtype=torch.long)

    def reset(self, mask):
        self.tick.masked_fill_(mask, 0)

    def sample(self, altitude):
        h = (altitude.float() - GROUND_Z_M).clamp_min(0)
        lo, hi, weight = _grid_indices(h)
        over = torch.lerp(self.grid[lo], self.grid[hi], weight)
        first, anchor_uv = self.anchor[0], self.anchor[1:]
        # A radar level can sit between grid points. Interpolate from the actual anchor.
        edge_weight = ((h - first) / (hi * 5 - first).clamp_min(1e-5)).clamp(0, 1)
        edge = torch.lerp(anchor_uv, self.grid[hi], edge_weight[:, None])
        over = torch.where((lo * 5 < first)[:, None], edge, over)
        frame = (self.tick * 20 // self.hz).clamp_max(len(self.series) - 1)
        sonic = self.series[frame]
        blend = ((h - self.mast) / (first - self.mast)).clamp(0, 1)
        below = torch.lerp(sonic[:, :2], anchor_uv, blend[:, None])
        uv = torch.where((h < first)[:, None], below, over)
        if self.surface_correction:
            fraction = ((h - 1.5) / (self.mast - 1.5)).clamp(0, 1)
            uv = uv * (SURFACE_RATIO + (1 - SURFACE_RATIO) * fraction[:, None])
        vertical = sonic[:, 2] * torch.minimum(h / self.mast, 1 - blend).clamp(0, 1)
        return torch.cat((uv, vertical[:, None]), dim=-1)

    def step(self, *, altitude, airspeed):
        flow = self.sample(altitude)
        self.tick.add_(1)
        return flow


def measured_field(name, n, device, seed, hz, dryden, severity):
    if name.startswith("kxmr-"):
        return SoundingWind(name.removeprefix("kxmr-"), n, device, seed, hz, dryden, severity)
    if name.startswith("cper-"):
        if dryden:
            raise ValueError("CPER already supplies time-resolved gusts; use dryden=0")
        suffix = name.removeprefix("cper-")
        return TowerWind(
            suffix.removeprefix("surface-"), n, device, hz, suffix.startswith("surface-")
        )
    raise ValueError(f"Unknown measured_wind {name!r}; use kxmr-<split> or cper-<date>")
