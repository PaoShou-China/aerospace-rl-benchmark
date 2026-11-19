"""Bind checked commands to the exact belief and short execution window."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ActionPacket:
    snapshot_id: str
    version: int
    modes: tuple[str, ...]
    expires_at: float
    command: tuple[float, ...]
    status: str


def consume(packet, snapshot, actual_command, now):
    if not snapshot.factual_valid:
        raise ValueError('invalid factual assumptions')
    if packet.version != snapshot.version or packet.snapshot_id != snapshot.fingerprint:
        raise ValueError("stale belief version/content")
    if packet.modes != snapshot.modes:
        raise ValueError("physical modes changed")
    if not math.isfinite(now) or not snapshot.clock <= now < packet.expires_at:
        raise ValueError("expired action packet")
    if packet.status not in ("empirical_support", "conditional_pass"):
        raise ValueError("assessment does not permit execution")
    if actual_command.shape != (7,) or not actual_command.is_floating_point() or not torch.isfinite(actual_command).all() or actual_command.abs().max() > 1:
        raise ValueError("invalid actuator command")
    # Compare values at their recorded precision. Casting the checked command
    # into the caller's dtype could truncate it and accept a different action.
    if tuple(actual_command.detach().cpu().tolist()) != packet.command:
        raise ValueError("actual command differs from checked command")
    return actual_command.detach().clone()
