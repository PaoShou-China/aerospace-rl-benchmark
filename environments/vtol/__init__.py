"""VTOL hybrid-UAV tasks.

Current task: QuadplaneWaypointsEnv — PyFlyt quadplane (4-quad motors +
fixedwing surfaces + forward pusher), waypoint navigation at 40 Hz agent
rate. No explicit mode switch in the plant or env; the physics of constant-
authority prop thrust vs v^2-proportional lifting-surface forces lets the
agent discover the transition regime naturally.
"""
