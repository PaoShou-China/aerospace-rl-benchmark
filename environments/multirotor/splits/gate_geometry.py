"""Analytic swept gate tests; local +X is the forward passage direction.

The source Y/Z dimensions are treated as a nominal aperture for this model.
Four rectangular frame bars extend outward by frame_width. A spherical vehicle
is conservatively approximated by expanding the bars on all local axes.
These configurable dimensions are project choices, not verified paper geometry.
"""
import math
import torch

from environments.multirotor.splits.split_s import gate_geometry


class GateGeometry:
    def __init__(self, device, vehicle_margin=.05, frame_width=.10):
        if not math.isfinite(vehicle_margin) or vehicle_margin < 0:
            raise ValueError('vehicle_margin must be finite and nonnegative')
        if not math.isfinite(frame_width) or frame_width <= 0:
            raise ValueError('frame_width must be finite and positive')
        data = gate_geometry(device)
        self.centres, self.rotations = data['centres'], data['rotations']
        self.sizes = data['sizes_xyz']
        if vehicle_margin >= float(self.sizes[:, 1:].min()) / 2:
            raise ValueError('vehicle_margin leaves no gate opening')
        self.margin, self.frame_width = vehicle_margin, frame_width

    def corners(self):
        """World-space corners, preserving the data file's fixed vertex order."""
        local=torch.tensor(((0.,-.5,-.5),(0.,.5,-.5),(0.,.5,.5),(0.,-.5,.5)),
                           device=self.centres.device,dtype=self.centres.dtype)[None]*self.sizes[:,None]
        return self.centres[:,None]+torch.einsum('nij,nkj->nki',self.rotations,local)

    def sweep(self, before, after):
        """Return forward passages (N,7), plane fractions (N,7), frame hits (N,).

        Full swept-box collision includes bar thickness and stationary overlap.
        Plane fractions allow the caller to enforce gate order within a substep.
        """
        a = torch.einsum('gji,ngj->ngi', self.rotations, before[:, None]-self.centres)
        b = torch.einsum('gji,ngj->ngi', self.rotations, after[:, None]-self.centres)
        delta = b-a
        moving = delta[:, :, 0].abs() > 1e-12
        fraction = -a[:, :, 0] / torch.where(moving, delta[:, :, 0], torch.ones_like(delta[:, :, 0]))
        plane = a + fraction[:, :, None]*delta
        half = self.sizes[:, 1:]/2-self.margin
        inside = (plane[:, :, 1:].abs() <= half).all(-1)
        passed = (a[:, :, 0] < 0) & (b[:, :, 0] >= 0) & moving & inside

        # Four solid frame bars in gate-local coordinates, then Minkowski-expand.
        x, y, z = self.sizes.unbind(-1)
        y, z, x = y/2, z/2, x/2
        width = self.frame_width
        lower = torch.stack((torch.stack((-x,-y-width,-z-width),-1),
                             torch.stack((-x,y,-z-width),-1),
                             torch.stack((-x,-y,-z-width),-1),
                             torch.stack((-x,-y,z),-1)),1)-self.margin
        upper = torch.stack((torch.stack((x,-y,z+width),-1),
                             torch.stack((x,y+width,z+width),-1),
                             torch.stack((x,y,-z),-1),
                             torch.stack((x,y,z+width),-1)),1)+self.margin
        start, direction = a[:, :, None], delta[:, :, None]
        parallel = direction.abs() <= 1e-12
        denominator = torch.where(parallel, torch.ones_like(direction), direction)
        t1, t2 = (lower-start)/denominator, (upper-start)/denominator
        enter = torch.where(parallel, -torch.inf, torch.minimum(t1,t2)).amax(-1)
        leave = torch.where(parallel, torch.inf, torch.maximum(t1,t2)).amin(-1)
        valid_axes = (~parallel | ((start >= lower) & (start <= upper))).all(-1)
        hit = valid_axes & (enter <= leave) & (leave >= 0) & (enter <= 1)
        return passed, fraction, hit.any(dim=(1,2))
