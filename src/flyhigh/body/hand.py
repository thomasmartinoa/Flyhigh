"""The hand: a mocap sphere that idles, approaches a point along a straight line, and retreats."""

from __future__ import annotations

import mujoco
import numpy as np


class Hand:
    def __init__(self, model, data, name: str = "hand", speed: float = 1.0, stop_short: float = 0.2):
        """Approaches stop when the hand's *surface* is `stop_short` from the target."""
        self.m, self.d = model, data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.mocap_id = int(model.body_mocapid[body])
        self.home = data.mocap_pos[self.mocap_id].copy()
        self.radius = float(model.geom_size[model.body_geomadr[body], 0])
        self.speed, self.stop_dist = speed, stop_short + self.radius
        self.target: np.ndarray | None = None
        self.state = "idle"

    @property
    def pos(self) -> np.ndarray:
        return self.d.mocap_pos[self.mocap_id].copy()

    def approach(self, target) -> None:
        """`target`: a point, or a callable returning the current point (a hand that follows the fly)."""
        self.target = target if callable(target) else np.asarray(target, dtype=float)
        if self.pos[2] < 0:  # parked: come back to the corner first
            self.d.mocap_pos[self.mocap_id] = self.home
        self.state = "approach"

    def idle(self) -> None:
        self.state = "idle"

    def park(self) -> None:
        """Take the hand out of the room (below the floor) until the next approach()."""
        self.state = "idle"
        self.d.mocap_pos[self.mocap_id] = self.home + np.array([0.0, 0.0, -10.0])

    def step(self, dt_s: float) -> None:
        if self.state == "idle":
            return
        if self.state == "approach":
            goal = np.asarray(self.target() if callable(self.target) else self.target, dtype=float)
        else:
            goal = self.home
        d = goal - self.pos
        dist = np.linalg.norm(d)
        stop = self.stop_dist if self.state == "approach" else 0.0
        if dist <= stop + 1e-9:
            self.state = "retreat" if self.state == "approach" else "idle"
            return
        move = min(self.speed * dt_s, dist - stop)
        self.d.mocap_pos[self.mocap_id] = self.pos + d / dist * move
