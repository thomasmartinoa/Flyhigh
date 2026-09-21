"""Steering parameters for flybody's flight policy (the carrot itself lives in `multi.py`).

flybody's flight policy follows the next six reference root poses, given in the fly's own frame
(`ref_displacement`, `ref_root_quat`). Its "controller reuse" mode feeds those from a learned
high-level network; here they come from the brain's `MotorCommand` through a *reference point*
that moves at the commanded velocity and a reference heading that turns at the commanded rate.
The policy was trained to close the gap to a reference (a velocity-only command, with the gap
always reported as zero, lets it sag to the floor), so the point is kept within a body length of
the fly and the heading within 30°: a carrot on a short stick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BODY_PITCH_DEG = 47.5  # flybody's hovering body pitch (nose up), Muijres et al. 2014


@dataclass(frozen=True)
class SteerParams:
    """Command → flybody steering. Speeds are within the recorded flight dataset's range."""
    v_max: float = 30.0  # cm/s at forward = 1
    vz_max: float = 15.0  # cm/s at lift = 1
    w_max_deg: float = 180.0  # yaw rate at yaw = 1, + = right (720, a saccade rate, turned the
    # readout's ±0.02 noise into a ±15°/s wobble that swamped the optomotor signal)
    escape_up: float = 10.0  # a 1 cm hop in 100 ms; 20 cm/s sometimes tumbles the policy (roll > 100°)
    escape_back: float = 5.0
    escape_ms: float = 100.0
    start: tuple[float, float, float] = (-5.0, 0.0, 7.0)  # cm, mid-height of the drum
    start_yaw_deg: float = 0.0
    lead_max: float = 0.3  # cm the reference point may lead the fly (~1 body length)
    lead_max_deg: float = 30.0  # degrees the reference heading may lead


def yaw_quat(yaw_rad):
    return np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)])
