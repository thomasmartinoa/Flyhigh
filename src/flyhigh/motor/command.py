from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCommand:
    """Normalised, body-frame command. The body layer (sim quadrotor, flybody, real drone)
    turns it into thrust and angles; `escape` may override everything for ~100 ms."""

    forward: float  # -1..1
    yaw: float  # -1..1, + = turn right
    lift: float  # -1..1
    escape: bool

    @classmethod
    def idle(cls, forward_bias: float = 0.2) -> MotorCommand:
        return cls(forward=forward_bias, yaw=0.0, lift=0.0, escape=False)
