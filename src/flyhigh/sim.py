"""The closed loop with a body: render → brain → command → physics, logged per tick.

One tick is one camera frame (10 ms) = 5 physics steps of 2 ms. The brain is `FlyAgent`
(or anything with `.tick(frames) -> commands` and, optionally, `.dn_rates`).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import polars as pl

from flyhigh.body.brick import Body, BodyParams
from flyhigh.body.hand import Hand
from flyhigh.senses.frame import TICK_MS, PanoramicFrame
from flyhigh.world.eyes import CubemapPanorama
from flyhigh.world.scene import RoomParams, build_mjcf


class Simulation:
    def __init__(self, agent, n_agents: int = 2, room: RoomParams | None = None, body: BodyParams | None = None,
                 face_px: int = 96, start=None):
        """`agent`: the brain for all n_agents (a FlyAgent built with the same n_agents)."""
        self.room = room or RoomParams()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(n_agents, self.room, start))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.n_agents = n_agents
        self.agent = agent
        self.bodies = [Body(self.model, self.data, f"agent{i}", body) for i in range(n_agents)]
        self.hand = Hand(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, face_px, face_px)
        self.eyes = [CubemapPanorama(self.model, i, face_px) for i in range(n_agents)]
        self.physics_per_tick = round(TICK_MS / (self.model.opt.timestep * 1000))
        self.dt_ms = TICK_MS / self.physics_per_tick
        self.tick = 0
        self.rows: list[dict] = []
        self.last_frames: list[PanoramicFrame] = []
        self.last_commands = []
        self._watch = getattr(getattr(agent, "readout", None), "watch", None)
        self._ids = self._readout_ids()

    def _readout_ids(self):
        ro = getattr(self.agent, "readout", None)
        if ro is None:
            return {}
        return {"gf": ro.gf, "hs_l": ro.hs_l, "hs_r": ro.hs_r}

    # ---------------------------------------------------------------- stepping
    def frames(self) -> list[PanoramicFrame]:
        return [eye.render(self.renderer, self.data) for eye in self.eyes]

    def step(self):
        self.last_frames = self.frames()
        cmds = self.agent.tick(self.last_frames)
        for _ in range(self.physics_per_tick):
            for body, cmd in zip(self.bodies, cmds):
                body.apply(cmd, self.dt_ms)
            self.hand.step(self.dt_ms / 1000.0)
            mujoco.mj_step(self.model, self.data)
        self.tick += 1
        self.last_commands = cmds
        self._log(cmds)
        return cmds

    def run(self, seconds: float, on_tick=None) -> pl.DataFrame:
        """Step for `seconds`; `on_tick(sim)` is called after every tick. Returns the log so far."""
        for _ in range(round(seconds * 1000 / TICK_MS)):
            self.step()
            if on_tick is not None:
                on_tick(self)
        return self.log()

    def _log(self, cmds):
        t = self.tick * TICK_MS
        rates = getattr(self.agent, "dn_rates", None)
        hand = self.hand.pos
        for i, (body, cmd) in enumerate(zip(self.bodies, cmds)):
            pos = body.pos
            roll, pitch = body.roll_pitch_deg
            row = {"t_ms": t, "agent": i, "x": pos[0], "y": pos[1], "z": pos[2], "yaw_deg": body.yaw_deg,
                   "roll_deg": roll, "pitch_deg": pitch, "forward": float(cmd.forward), "yaw": float(cmd.yaw),
                   "lift": float(cmd.lift), "escape": bool(cmd.escape), "escaping": bool(body.escape_left_ms > 0),
                   "hand_dist": float(np.linalg.norm(hand - pos))}
            if rates is not None and self._ids:
                for k, idx in self._ids.items():
                    row[f"{k}_hz"] = float(rates[i, idx].mean()) if len(idx) else 0.0
            self.rows.append(row)

    def log(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows)

    # --------------------------------------------------------------- controls
    def hide_agent(self, i: int) -> None:
        """Make agent i invisible to every eye (its geoms get alpha 0); physics unchanged."""
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}")
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == body:
                self.model.geom_rgba[g, 3] = 0.0

    def kick_yaw(self, i: int, rate_deg_s: float) -> None:
        """Impose a yaw rate on agent i (an external disturbance, + = right)."""
        adr = self.model.jnt_dofadr[self.model.body_jntadr[self.bodies[i].id]]
        self.data.qvel[adr + 5] = -np.radians(rate_deg_s)  # free-joint angular velocity, body frame
        mujoco.mj_forward(self.model, self.data)

    def speed(self, seconds: float = 1.0) -> float:
        """ticks per second, rendering included."""
        n = round(seconds * 1000 / TICK_MS)
        t = time.perf_counter()
        for _ in range(n):
            self.step()
        return n / (time.perf_counter() - t)

    def close(self) -> None:
        self.renderer.close()


class VideoWriter:
    """An mp4 of the overview camera with agent 0's panorama underneath, every `every` ticks."""

    def __init__(self, sim: Simulation, path, fps: int = 25, every: int = 4, width: int = 640):
        import imageio

        self.sim, self.every, self.width = sim, every, width
        self.overview = mujoco.Renderer(sim.model, width * 9 // 16, width)
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)

    def __call__(self, sim: Simulation) -> None:
        if sim.tick % self.every:
            return
        self.overview.update_scene(sim.data, camera="overview")
        top = self.overview.render()
        pano = sim.last_frames[0].lum if sim.last_frames else np.full((180, 360), 0.5, np.float32)
        cols = np.linspace(0, pano.shape[1] - 1, self.width).astype(int)
        rows = np.linspace(0, pano.shape[0] - 1, self.width // 2).astype(int)
        bottom = np.repeat((pano[rows][:, cols] * 255).astype(np.uint8)[..., None], 3, axis=-1)
        self.writer.append_data(np.concatenate([top, bottom], axis=0))

    def close(self) -> None:
        self.writer.close()
        self.overview.close()
