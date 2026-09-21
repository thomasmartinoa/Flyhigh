"""The M3 loop with flybody as the body: render (head) → brain → command → policy → wings → physics."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import polars as pl
from dm_control import composer
from flybody.tasks.pattern_generators import WingBeatPatternGenerator

from flyhigh.body.hand import Hand
from flyhigh.flybody.arena import DrumParams, FlyDrum
from flyhigh.flybody.policy import DEFAULT_PATH, FlightPolicy
from flyhigh.flybody.task import SteeredFlight, SteerParams
from flyhigh.senses.frame import TICK_MS, PanoramicFrame
from flyhigh.world.eyes import CubemapPanorama
from flyhigh.world.scene import FACES

WPG_PATH = Path("data/flybody/datasets_flight-imitation/wing_pattern_fmech.npy")
OWN_GROUPS = (1, 3, 4, 5)  # flybody's mesh, site, collision and fluid geoms


class FlySimulation:
    """One flybody fly. Same API as `flyhigh.sim.Simulation` for n_agents = 1."""

    n_agents = 1

    def __init__(self, agent, drum: DrumParams | None = None, steer: SteerParams | None = None,
                 face_px: int = 96, policy_path=DEFAULT_PATH, wpg_path=WPG_PATH, settle_s: float = 0.5):
        self.agent = agent
        self.drum = drum or DrumParams()
        wbpg = WingBeatPatternGenerator(base_pattern_path=str(wpg_path))
        self.task = SteeredFlight(wbpg, FlyDrum(self.drum), steer)
        self.env = composer.Environment(task=self.task, time_limit=1e5, strip_singleton_obs_buffer_dim=True)
        self.ts = self.env.reset()
        self.physics = self.env.physics
        self.policy = FlightPolicy(policy_path)
        self.renderer = mujoco.Renderer(self.physics.model.ptr, face_px, face_px)
        head_frame = self._orient_eyes()
        mujoco.mj_forward(self.physics.model.ptr, self.physics.data.ptr)  # camera matrices from the new quats
        self.eye = CubemapPanorama(self.physics.model.ptr, face_px=face_px, eye_frame=head_frame,
                                  cam_names={f: f"walker/cube_{f}" for f in FACES}, hide_groups=OWN_GROUPS)
        self.hand = Hand(self.physics.model.ptr, self.physics.data.ptr, name="hand", speed=50.0, stop_short=2.0)
        self.policy_per_tick = round(TICK_MS / 1000 / self.env.control_timestep())
        self.tick = 0
        self.disturb_yaw: dict[int, float] = {}
        self.rows: list[dict] = []
        self.last_frames: list[PanoramicFrame] = []
        self.last_commands = []
        self.actions = np.zeros(self.env.action_spec().shape)
        ro = getattr(agent, "readout", None)
        self._ids = {} if ro is None else {"gf": ro.gf, "hs_l": ro.hs_l, "hs_r": ro.hs_r}
        self.settle(settle_s)

    def _orient_eyes(self) -> np.ndarray:
        """Point the head cameras as the M3 faces are defined -- image axes in the fly's *level
        heading* frame (x forward, y left, z up) -- whatever frame flybody's head mesh uses
        (its x points right, its y forward-up). Measured at the start pose; the cameras then
        move with the head, as a fly's eyes do. Returns the level->head rotation for the lookup."""
        m = self.physics.model.ptr
        head = np.array(self.physics.bind(self.task.walker.mjcf_model.find("body", "head")).xmat).reshape(3, 3)
        psi = self.task.heading(self.physics)
        level = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        M = head.T @ level  # level-frame vector -> head-frame vector
        for face, axes in FACES.items():
            v = np.array([float(a) for a in axes.split()])
            x, y = M @ v[:3], M @ v[3:]
            z = np.cross(x, y)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).ravel())
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"walker/cube_{face}")
            m.cam_quat[cid] = quat
        return M

    # ----------------------------------------------------------------- state
    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.task.walker.get_pose(self.physics)[0])

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(self.task.heading(self.physics)))

    def roll_pitch_deg(self) -> tuple[float, float]:
        """Attitude relative to the canonical flight attitude (0, 0 = flying level)."""
        from flybody.quaternions import get_dquat_local, mult_quat

        from flyhigh.flybody.task import yaw_quat
        q = np.array(self.task.walker.get_pose(self.physics)[1])
        dq = get_dquat_local(mult_quat(yaw_quat(self.task.heading(self.physics)), self.task.q_flight), q)
        w, x, y, z = dq
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        return float(roll), float(pitch)

    # -------------------------------------------------------------- stepping
    def settle(self, seconds: float) -> None:
        frames = self.frames()
        for _ in range(round(seconds * 1000 / TICK_MS)):
            self.agent.tick(frames)

    def frames(self) -> list[PanoramicFrame]:
        return [self.eye.render(self.renderer, self.physics.data.ptr)]

    def step(self):
        self.last_frames = self.frames()
        cmds = self.agent.tick(self.last_frames)
        self.task.set_command(cmds[0], TICK_MS)
        dt = self.env.control_timestep()
        for _ in range(self.policy_per_tick):
            self.actions = self.policy(self.ts.observation)
            for rate in self.disturb_yaw.values():
                self._impose_yaw(rate)
            self.hand.step(dt)
            self.ts = self.env.step(self.actions)
        self.tick += 1
        self.last_commands = cmds
        self._log(cmds)
        return cmds

    def run(self, seconds: float, on_tick=None) -> pl.DataFrame:
        for _ in range(round(seconds * 1000 / TICK_MS)):
            self.step()
            if on_tick is not None:
                on_tick(self)
        return self.log()

    def _log(self, cmds):
        pos = self.pos
        roll, pitch = self.roll_pitch_deg()
        cmd = cmds[0]
        rates = getattr(self.agent, "dn_rates", None)
        row = {"t_ms": self.tick * TICK_MS, "agent": 0, "x": pos[0], "y": pos[1], "z": pos[2], "yaw_deg": self.yaw_deg,
               "roll_deg": roll, "pitch_deg": pitch, "forward": float(cmd.forward), "yaw": float(cmd.yaw),
               "lift": float(cmd.lift), "escape": bool(cmd.escape), "escaping": bool(self.task.escape_left_ms > 0),
               "hand_dist": float(np.linalg.norm(self.hand.pos - pos)),
               "wingbeat_hz": float(self.task._wbpg.base_beat_freq * (1 + self.task._wbpg.rel_freq_range * self.actions[self.task._user_idx_action]))}
        if rates is not None and self._ids:
            for k, idx in self._ids.items():
                row[f"{k}_hz"] = float(rates[0, idx].mean()) if len(idx) else 0.0
        self.rows.append(row)

    def log(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows)

    # --------------------------------------------------------------- controls
    def _impose_yaw(self, rate_deg_s: float) -> None:
        """Set the root's angular velocity to a yaw about the world z (+ = right)."""
        joint = self.task.walker.mjcf_model.find("joint", "free")
        bound = self.physics.bind(joint)
        R = np.asarray(self.physics.bind(self.task.walker.root_body).xmat).reshape(3, 3)
        w_local = R.T @ np.array([0.0, 0.0, -np.radians(rate_deg_s)])
        qvel = np.array(bound.qvel)
        qvel[3:6] = w_local
        bound.qvel = qvel

    def speed(self, seconds: float = 1.0) -> float:
        n = round(seconds * 1000 / TICK_MS)
        t = time.perf_counter()
        for _ in range(n):
            self.step()
        return n / (time.perf_counter() - t)

    def close(self) -> None:
        self.renderer.close()
