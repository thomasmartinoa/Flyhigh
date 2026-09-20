"""The hand: a mocap sphere that idles, approaches a point along a straight line, and retreats."""

from __future__ import annotations

import mujoco
import numpy as np


class Hand:
    def __init__(self, model, data, name: str = "hand", speed: float = 1.0, stop_dist: float = 0.2):
        self.m, self.d = model, data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.mocap_id = int(model.body_mocapid[body])
        self.home = data.mocap_pos[self.mocap_id].copy()
        self.speed, self.stop_dist = speed, stop_dist
        self.target: np.ndarray | None = None
        self.state = "idle"

    @property
    def pos(self) -> np.ndarray:
        return self.d.mocap_pos[self.mocap_id].copy()

    def approach(self, target) -> None:
        self.target = np.asarray(target, dtype=float)
        self.state = "approach"

    def idle(self) -> None:
        self.state = "idle"

    def step(self, dt_s: float) -> None:
        if self.state == "idle":
            return
        goal = self.target if self.state == "approach" else self.home
        d = goal - self.pos
        dist = np.linalg.norm(d)
        stop = self.stop_dist if self.state == "approach" else 0.0
        if dist <= stop + 1e-9:
            self.state = "retreat" if self.state == "approach" else "idle"
            return
        move = min(self.speed * dt_s, dist - stop)
        self.d.mocap_pos[self.mocap_id] = self.pos + d / dist * move
