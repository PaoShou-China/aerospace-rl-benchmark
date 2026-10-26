"""Sample consecutive Split-S gate centres in original world coordinates (metres).

Source: https://github.com/uzh-rpg/marl-racing/blob/master/track.yaml
Gate1..Gate7 positions and loop=yes are preserved; no scaling or translation.
Geometry helpers are preparation for racing; the current env still uses only
sample_waypoints(). Gate-local +X is the proposed forward direction; Y/Z span
the gate plane. Source size is retained verbatim, not claimed to specify a
collision-accurate free aperture. Frame thickness/clearance remain to be defined.
"""
import torch


GATE_CENTRES = (
    (-0.6, -0.86, 3.68),
    (9.0, 6.45, 1.05),
    (8.85, -3.8, 1.05),
    (-4.3, -5.6, 3.4),
    (-4.3, -5.6, 1.42),
    (4.5, -0.45, 1.05),
    (-1.95, 6.81, 1.05),
)
SOURCE_URL = 'https://github.com/uzh-rpg/marl-racing/blob/master/track.yaml'
GATE_QUATERNIONS_WXYZ = (
    (.9848078, 0., 0., -.1736482),
    (1., 0., 0., 0.),
    (.4226183, 0., 0., -.9063078),
    (0., 0., 0., -1.),
    (1., 0., 0., 0.),
    (.7660444, 0., 0., .6427876),
    (.258819, 0., 0., -.9659258),
)
GATE_SIZES_XYZ = ((.4, 1.5, 1.5),) * len(GATE_CENTRES)
NUM_GATES = len(GATE_CENTRES)
DEFAULT_TRAIN_GATE_COUNT = 4
PAPER_LOOKAHEAD_GATES = 2  # Planned observation layout; current env is unchanged.
TRACK_IS_LOOP = True
BOUNDS_RADIUS = 15.0  # Original gates reach ~11.12 m from the origin; allow turn room.


def gate_geometry(device='cpu', dtype=torch.float64):
    """Return original-scale gate tensors; rotation maps gate-local to world.

    corners: (7, 4, 3), ordered (-Y,-Z), (+Y,-Z), (+Y,+Z), (-Y,+Z).
    They describe the nominal Y/Z size rectangle at local X=0, not a gate mesh.
    normals: (7, 3), world-space local +X, from entry side toward exit side.
    Rounded source quaternions are normalized only for geometry computation.
    """
    centres = torch.tensor(GATE_CENTRES, device=device, dtype=dtype)
    quaternions = torch.tensor(GATE_QUATERNIONS_WXYZ, device=device, dtype=dtype)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    sizes = torch.tensor(GATE_SIZES_XYZ, device=device, dtype=dtype)
    w, x, y, z = quaternions.unbind(-1)
    rotations = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                             2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                             2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(NUM_GATES, 3, 3)
    local = torch.tensor(((0.,-.5,-.5), (0.,.5,-.5), (0.,.5,.5), (0.,-.5,.5)),
                         device=device, dtype=dtype)[None] * sizes[:, None]
    corners = centres[:, None] + torch.einsum('nij,nkj->nki', rotations, local)
    return dict(centres=centres, quaternions_wxyz=quaternions, sizes_xyz=sizes,
                rotations=rotations, corners=corners, normals=rotations[:, :, 0])


def sequence_indices(start, count):
    """Expand a 1-D int64 start tensor into ordered, wrapping gate indices.

    count may exceed NUM_GATES for future multi-lap episodes; this is separate
    from the policy lookahead and the number of gates sampled for training.
    """
    if start.ndim != 1 or start.dtype != torch.long:
        raise ValueError('start must be a 1-D torch.long tensor')
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('count must be a positive integer')
    return (start[:, None] + torch.arange(count, device=start.device)) % NUM_GATES


def full_track_indices(device='cpu', laps=1):
    """Deterministic Gate1..Gate7 order per lap; consumes no random numbers."""
    if not isinstance(laps, int) or isinstance(laps, bool) or laps < 1:
        raise ValueError('laps must be a positive integer')
    return torch.arange(NUM_GATES, device=device).repeat(laps)


def sample_gate_indices(rng, mask, count=DEFAULT_TRAIN_GATE_COUNT):
    """Return (num_envs, count); preserve the original one-draw RNG protocol.

    Each of the seven starts is equally likely; indices wrap at Gate7.
    The caller applies the reset mask when updating its target buffer.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= NUM_GATES:
        raise ValueError('count must be between 1 and 7')
    start = (rng.uniform(1, mask)[:, 0] * NUM_GATES).to(torch.long)
    return sequence_indices(start, count)


def sample_waypoints(rng, mask, count):
    """Backward-compatible (num_envs, count, 3) gate-centre sampler for env.py."""
    indices = sample_gate_indices(rng, mask, count)
    gates = torch.tensor(GATE_CENTRES, device=mask.device, dtype=torch.float64)
    return gates[indices]
