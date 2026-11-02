"""Shared utilities and batched physics primitives for the pyflyt-genesis RL environments.

This package owns the device-resident plant (``plant.TensorPlant`` and its
rate-PID mixer), the batched analytical dynamics and lifting-surface models
(``dynamics``, ``aerodynamics``), the lightweight action/observation spaces
(``spaces.Box``), and the rocket contact/variation helpers
(``rocket_benchmark``).  All components are written to work with batched
``(n, ...)`` tensors on a single device and expose consistent mask-based reset
contracts so inactive replicas can be frozen without consuming random numbers
or corrupting their state.
"""
