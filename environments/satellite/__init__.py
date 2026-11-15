"""Spacecraft tasks: attitude control, orbit transfer, rendezvous & docking.

Subpackages
    attitude_control   SatelliteAttitudeEnv (CMG cluster, RK4 2.5 ms)
    orbit_transfer     LowThrustTransferEnv (two-body + 10 mN EP, RK4 10 s)
    rendezvous_docking SpaceDockingEnv / SpaceRendezvousEnv (CWH, symplectic Euler 1 s)
"""

from .attitude_control import SatelliteAttitudeEnv
from .orbit_transfer import LowThrustTransferEnv
from .rendezvous_docking import SpaceDockingEnv, SpaceRendezvousEnv

__all__ = ["SatelliteAttitudeEnv", "LowThrustTransferEnv", "SpaceDockingEnv", "SpaceRendezvousEnv"]
