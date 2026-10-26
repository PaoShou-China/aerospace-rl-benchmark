"""Independent multirotor scenarios; default package import uses Split-S."""
from .splits.env import QuadXSplitsEnv

# Compatibility for shared scripts that still import the old generic name.
QuadXWaypointsEnv = QuadXSplitsEnv
