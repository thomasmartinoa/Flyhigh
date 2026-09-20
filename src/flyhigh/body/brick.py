"""A flying brick: a free-floating MuJoCo body steered by body-frame forces and torques.

`MotorCommand` names *velocities* (forward speed, yaw rate, climb) and one reflex (escape); the
brick turns them into forces with a velocity controller, holds altitude when `lift` is zero and
keeps itself level, so the brain never has to fly the aircraft -- a real drone's flight
controller (M5) does exactly this job. Rotor-level dynamics can replace this class behind the
same `apply()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from flyhigh.motor.command import MotorCommand


@dataclass(frozen=True)
class BodyParams:
    v_max: float = 2.5  # m/s at forward = 1: the readout's 0.2 bias cruises at 0.5 m/s, so two flies
    # closing head-on loom at ~70°/s at half a metre (M2 validated 110°/s)
    w_max_deg: float = 180.0  # yaw rate at yaw = 1
    vz_max: float = 0.5  # m/s climb at lift = 1
    escape_up: float = 2.0  # m/s
    escape_back: float = 1.0  # m/s
    escape_ms: float = 100.0
    k_v: float = 6.0  # 1/s, velocity loop
    k_z: float = 20.0  # 1/s², altitude hold
    k_att: float = 200.0  # 1/s², level hold
    k_w: float = 15.0  # 1/s, rate loops
    g: float = 9.81


class Body:
    def __init__(self, model, data, name: str, params: BodyParams | None = None):
        self.m, self.d, self.p = model, data, params or BodyParams()
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.mass = float(model.body_subtreemass[self.id])
        self.inertia = model.body_inertia[self.id].copy()
        self.z_hold = float(data.xpos[self.id][2])
        self.escape_left_ms = 0.0
        self._vel = np.zeros(6)

    # ------------------------------------------------------------------ state
    @property
    def pos(self) -> np.ndarray:
        return self.d.xpos[self.id].copy()

    @property
    def rot(self) -> np.ndarray:
        return self.d.xmat[self.id].reshape(3, 3).copy()  # body → world

    @property
    def yaw_deg(self) -> float:
        R = self.rot
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    @property
    def roll_pitch_deg(self) -> tuple[float, float]:
        R = self.rot
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        return float(np.degrees(roll)), float(np.degrees(pitch))

    def local_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear, angular) velocity in the body frame."""
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.id, self._vel, 1)
        return self._vel[3:].copy(), self._vel[:3].copy()

    # ---------------------------------------------------------------- control
    def apply(self, cmd: MotorCommand, dt_ms: float) -> None:
        """Write this step's force and torque into data.xfrc_applied. Call before mj_step."""
        p = self.p
        if cmd.escape and self.escape_left_ms <= 0:
            self.escape_left_ms = p.escape_ms
        escaping = self.escape_left_ms > 0
        self.escape_left_ms = max(0.0, self.escape_left_ms - dt_ms)

        v, w = self.local_velocity()
        R = self.rot
        z = self.pos[2]
        if escaping:
            v_t = np.array([-p.escape_back, 0.0, p.escape_up])
            w_t = 0.0
        else:
            v_t = np.array([p.v_max * cmd.forward, 0.0, p.vz_max * cmd.lift])
            w_t = -np.radians(p.w_max_deg) * cmd.yaw  # + yaw = turn right = clockwise from above = -z
        # translational: velocity loop in the body frame, plus altitude hold and gravity
        a_local = p.k_v * (v_t - v)
        if escaping or cmd.lift != 0.0:
            self.z_hold = z
        else:
            a_local[2] += p.k_z * (self.z_hold - z)
        f_world = self.mass * (R @ a_local + np.array([0.0, 0.0, p.g]))
        # rotational: level roll/pitch, track the yaw rate
        roll, pitch = np.radians(self.roll_pitch_deg)
        alpha = np.array([
            -p.k_att * roll - p.k_w * w[0],
            -p.k_att * pitch - p.k_w * w[1],
            p.k_w * (w_t - w[2]),
        ])
        t_world = R @ (self.inertia * alpha)
        self.d.xfrc_applied[self.id, :3] = f_world
        self.d.xfrc_applied[self.id, 3:] = t_world
