"""SteeredFlight: flybody's flight-imitation task with the reference replaced by a command.

flybody's flight policy follows the next six reference root poses, given in the fly's own frame
(`ref_displacement`, `ref_root_quat`). Its "controller reuse" mode feeds those from a learned
high-level network; here they come from the brain's `MotorCommand` through a *reference point*
that moves at the commanded velocity and a reference heading that turns at the commanded rate.
The policy was trained to close the gap to a reference (a velocity-only command, with the gap
always reported as zero, lets it sag to the floor), so the point is kept within a body length of
the fly and the heading within 30°: a carrot on a short stick. No ghost body, no dataset, no
time limit; the wing-beat generator and action mapping are the parent's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from flybody.fruitfly import fruitfly
from flybody.quaternions import get_dquat_local, mult_quat
from flybody.tasks.constants import _FLY_CONTROL_TIMESTEP
from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import com2root
from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader

from flyhigh.motor.command import MotorCommand
from flyhigh.world.scene import FACE_FOVY, FACES

BODY_PITCH_DEG = 47.5  # flybody's hovering body pitch (nose up), Muijres et al. 2014


@dataclass(frozen=True)
class SteerParams:
    """Command → flybody steering. Speeds are within the recorded flight dataset's range."""
    v_max: float = 30.0  # cm/s at forward = 1
    vz_max: float = 15.0  # cm/s at lift = 1
    w_max_deg: float = 180.0  # yaw rate at yaw = 1, + = right (720, a saccade rate, turned the
    # readout's ±0.02 noise into a ±15°/s wobble that swamped the optomotor signal)
    escape_up: float = 20.0  # a 2 cm hop in 100 ms; 40 cm/s tumbles the policy (roll 173°)
    escape_back: float = 10.0
    escape_ms: float = 100.0
    start: tuple[float, float, float] = (-5.0, 0.0, 7.0)  # cm, mid-height of the drum
    start_yaw_deg: float = 0.0
    lead_max: float = 0.3  # cm the reference point may lead the fly (~1 body length)
    lead_max_deg: float = 30.0  # degrees the reference heading may lead


def yaw_quat(yaw_rad):
    return np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)])


class SteeredFlight(FlightImitationWBPG):
    def __init__(self, wbpg, arena, params: SteerParams | None = None, **kwargs):
        self.p = params or SteerParams()
        super().__init__(walker=fruitfly.FruitFly, arena=arena, wbpg=wbpg,
                         traj_generator=InferenceFlightTrajectoryLoader(), terminal_com_dist=float("inf"),
                         trajectory_sites=False, initialize_qvel=False, time_limit=1e5,
                         ghost_offset=np.array([0.0, 0.0, -1000.0]),  # the parent's ghost: out of sight
                         future_steps=5, joint_filter=0.0, disable_legs=True, **kwargs)
        self.dt = _FLY_CONTROL_TIMESTEP
        self.v_local = np.zeros(3)  # commanded body-frame velocity (horizontal frame), cm/s
        self.omega = 0.0  # commanded yaw rate, rad/s, + = counter-clockwise (left)
        self.escape_left_ms = 0.0
        self.ref_pos = np.zeros(3)  # the reference point (world, root frame) and heading
        self.ref_yaw = 0.0
        # the canonical flight attitude: nose up by the hovering body pitch
        q0, _ = constant_speed_trajectory(n_steps=2, speed=0.0, init_pos=(0, 0, 0), init_heading=0.0,
                                          body_rot_angle_y=-BODY_PITCH_DEG, control_timestep=self.dt)
        self.q_flight = q0[0, 3:7]
        # flybody renders for film: a 4K offscreen buffer and 8192² shadow maps. The eyes are 96 px.
        vis = self._walker.mjcf_model.visual
        getattr(vis, "global").offwidth, getattr(vis, "global").offheight = 640, 480
        vis.quality.shadowsize = 1024
        # cubemap eyes on the head, hidden from the fly's own geoms (groups 1, 3, 4, 5)
        head = self._walker.mjcf_model.find("body", "head")
        for face, axes in FACES.items():
            head.add("camera", name=f"cube_{face}", pos=(0, 0, 0), xyaxes=[float(v) for v in axes.split()],
                     fovy=FACE_FOVY)

    # ------------------------------------------------------------ the command
    def set_command(self, cmd: MotorCommand, dt_ms: float) -> None:
        p = self.p
        if cmd.escape and self.escape_left_ms <= 0:
            self.escape_left_ms = p.escape_ms
        escaping = self.escape_left_ms > 0
        self.escape_left_ms = max(0.0, self.escape_left_ms - dt_ms)
        if escaping:
            self.v_local = np.array([-p.escape_back, 0.0, p.escape_up])
            self.omega = 0.0
        else:
            self.v_local = np.array([p.v_max * cmd.forward, 0.0, p.vz_max * cmd.lift])
            self.omega = -np.radians(p.w_max_deg) * cmd.yaw  # + yaw command = turn right = clockwise = -z

    def heading(self, physics) -> float:
        """Yaw of the fly (rad), from its body x-axis projected on the ground."""
        w, x, y, z = np.array(self._walker.get_pose(physics)[1])  # a plain copy, not a physics view
        xaxis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z))
        return float(np.arctan2(xaxis[1], xaxis[0]))

    def v_world(self):
        """Commanded velocity in the world: forward along the reference heading, climb along z."""
        return np.array([np.cos(self.ref_yaw) * self.v_local[0], np.sin(self.ref_yaw) * self.v_local[0],
                         self.v_local[2]])

    def advance_reference(self, physics) -> None:
        """One control step of the carrot: move it, then keep it within reach of the fly."""
        p = self.p
        fly_pos = np.array(self._walker.get_pose(physics)[0])
        self.ref_pos = self.ref_pos + self.v_world() * self.dt
        lead = self.ref_pos - fly_pos
        dist = np.linalg.norm(lead)
        if dist > p.lead_max:
            self.ref_pos = fly_pos + lead * (p.lead_max / dist)
        self.ref_yaw += self.omega * self.dt
        gap = (self.ref_yaw - self.heading(physics) + np.pi) % (2 * np.pi) - np.pi
        limit = np.radians(p.lead_max_deg)
        if abs(gap) > limit:
            self.ref_yaw = self.heading(physics) + np.sign(gap) * limit

    def steering(self, physics):
        """(ref_displacement (6, 3), ref_root_quat (6, 4)): the carrot's next six poses, egocentric."""
        k = np.arange(self._future_steps + 1)[:, None] * self.dt
        fly_pos = np.array(self._walker.get_pose(physics)[0])
        disp = self._walker.transform_vec_to_egocentric_frame(physics, self.ref_pos + k * self.v_world() - fly_pos)
        q_fly = np.array(self._walker.get_pose(physics)[1])
        quats = np.stack([get_dquat_local(q_fly, mult_quat(yaw_quat(self.ref_yaw + self.omega * kk), self.q_flight))
                          for kk in k[:, 0]])
        return disp.astype(np.float32), quats.astype(np.float32)

    @composer.observable
    def ref_displacement(self):
        return observable.Generic(lambda physics: self.steering(physics)[0])

    @composer.observable
    def ref_root_quat(self):
        return observable.Generic(lambda physics: self.steering(physics)[1])

    # ----------------------------------------------------------- the episode
    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        # start hovering at `start`, facing `start_yaw_deg`, in the canonical attitude
        com = np.array(self.p.start, dtype=float)
        q = mult_quat(yaw_quat(np.radians(self.p.start_yaw_deg)), self.q_flight)
        root = com2root(com[None], q[None])[0]
        self._ref_qpos = np.repeat(np.concatenate([root, q])[None], self._future_steps + 2, axis=0)
        self._ref_qvel = np.zeros((self._future_steps + 2, 6))
        self.escape_left_ms = 0.0
        self.ref_pos = root.copy()
        self.ref_yaw = np.radians(self.p.start_yaw_deg)

    def before_step(self, physics, action, random_state):
        """The parent's wing-beat pattern step, with the carrot instead of its ghost."""
        self.advance_reference(physics)
        base_freq, rel_range = self._wbpg.base_beat_freq, self._wbpg.rel_freq_range
        ctrl_freq = base_freq * (1 + rel_range * action[self._user_idx_action])
        ctrl = self._wbpg.step(ctrl_freq=ctrl_freq)
        action[self._wing_inds_action] += ctrl - physics.bind(self._wing_joints).qpos
        super(FlightImitationWBPG, self).before_step(physics, action, random_state)

    def check_termination(self, physics) -> bool:
        return False  # the simulation decides when to stop; a crash shows in the log

    def get_reward_factors(self, physics):
        return (1.0,)
