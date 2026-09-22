"""The M3 loop with flybody flies as bodies: render (heads) → brains → commands → policy → wings → physics."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import polars as pl
from dm_control import composer
from flybody.quaternions import get_dquat_local, mult_quat

from flyhigh.body.hand import Hand
from flyhigh.flybody.arena import DrumParams, FlyDrum
from flyhigh.flybody.multi import MultiFlyTask
from flyhigh.flybody.policy import DEFAULT_PATH, FlightPolicy
from flyhigh.flybody.task import SteerParams, yaw_quat
from flyhigh.motor.command import MotorCommand
from flyhigh.senses.frame import TICK_MS, PanoramicFrame
from flyhigh.world.eyes import CubemapPanorama
from flyhigh.world.scene import FACES

WPG_PATH = Path("data/flybody/datasets_flight-imitation/wing_pattern_fmech.npy")
HIDDEN_GROUPS = (3, 4, 5)  # flybody's sites, collision capsules and fluid ellipsoids: never shown to an eye


class FlySimulation:
    """Up to two flybody flies. Same API as `flyhigh.sim.Simulation`."""

    def __init__(self, agent, n_agents: int = 1, drum: DrumParams | None = None, steer: SteerParams | None = None,
                 face_px: int = 96, policy_path=DEFAULT_PATH, wpg_path=WPG_PATH, settle_s: float = 0.5,
                 exposures: int = 5, starts=None, warmup_s: float = 0.3, seed: int | None = 0):
        """`exposures`: sub-frames averaged into each tick's panorama. The head bobs ~2° with
        every wing beat (218 Hz); a single snapshot per 10 ms tick aliases that into motion the
        brain escapes from. Photoreceptors integrate over about a tick; so does this.
        `warmup_s`: the policy flies the fly alone for this long before the brain is connected.
        Each episode starts at a random wing-beat phase and the first ~0.2 s are a transient
        (roll up to 40°); a fly you start watching is already flying.
        `seed`: the episode's random state (the wing phase); None for a fresh one each run."""
        assert 1 <= n_agents <= 2, "eyes tell flies apart by geom group; groups 1 and 2 are available"
        self.agent, self.n_agents = agent, n_agents
        self.drum = drum or DrumParams()
        self.steer = steer or SteerParams()
        self.task = MultiFlyTask(FlyDrum(self.drum), wpg_path, n_agents, self.steer, starts)
        self.env = composer.Environment(task=self.task, time_limit=1e5, strip_singleton_obs_buffer_dim=True,
                                        random_state=np.random.RandomState(seed))
        self.ts = self.env.reset()
        self.physics = self.env.physics
        self.policy = FlightPolicy(policy_path)
        self.face_px = face_px
        self.renderer = None
        self._bind_model()
        self.policy_per_tick = round(TICK_MS / 1000 / self.env.control_timestep())
        self.exposures = exposures
        self._accum = None
        self.tick = 0
        self.spin_drum = 0.0  # °/s, + = clockwise seen from above (the scene moves right-to-left ahead)
        self._drum_angle = 0.0
        self.rows: list[dict] = []
        self.last_frames: list[PanoramicFrame] = []
        self.last_commands = []
        self.actions = np.zeros(self.env.action_spec().shape)
        ro = getattr(agent, "readout", None)
        self._ids = {} if ro is None else {"gf": ro.gf, "hs_l": ro.hs_l, "hs_r": ro.hs_r}
        self.warmup(warmup_s)
        self.settle(settle_s)

    def _bind_model(self) -> None:
        """(Re-)attach renderer, eyes, hand and drum to the current compiled model. `env.reset()`
        recompiles, so everything holding a model pointer or an id must be rebuilt."""
        m = self.physics.model.ptr
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(m, self.face_px, self.face_px)
        # each fly's visible (mesh) geoms get their own group so its eyes can skip its own body
        for i, f in enumerate(self.task.flies):
            for g in range(m.ngeom):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if name.startswith(f.name + "/") and m.geom_group[g] == 1:
                    m.geom_group[g] = 1 + i
        self.eyes = [CubemapPanorama(m, face_px=self.face_px, eye_frame=self._orient_eyes(f),
                                     cam_names={face: f"{f.name}/cube_{face}" for face in FACES},
                                     hide_groups=(1 + i, *HIDDEN_GROUPS))
                     for i, f in enumerate(self.task.flies)]
        mujoco.mj_forward(m, self.physics.data.ptr)
        self.hand = Hand(m, self.physics.data.ptr, name="hand", speed=50.0, stop_short=2.0)
        self._drum_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drum")])

    def _orient_eyes(self, fly) -> np.ndarray:
        """Point a fly's head cameras as the M3 faces are defined -- image axes in the fly's *level
        heading* frame (x forward, y left, z up) -- whatever frame flybody's head mesh uses (its x
        points right, its y forward-up). Measured at the start pose; the cameras then move with the
        head, as a fly's eyes do. Returns the level->head rotation for the panorama lookup."""
        m = self.physics.model.ptr
        head = np.array(self.physics.bind(fly.walker.mjcf_model.find("body", "head")).xmat).reshape(3, 3)
        psi = fly.carrot.heading(self.physics, fly.walker)
        level = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        M = head.T @ level
        for face, axes in FACES.items():
            v = np.array([float(a) for a in axes.split()])
            x, y = M @ v[:3], M @ v[3:]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.stack([x, y, np.cross(x, y)], axis=1).ravel())
            m.cam_quat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"{fly.name}/cube_{face}")] = quat
        return M

    # ----------------------------------------------------------------- state
    def pos_of(self, i: int) -> np.ndarray:
        f = self.task.flies[i]
        return np.asarray(f.walker.get_pose(self.physics)[0])

    def yaw_of(self, i: int) -> float:
        f = self.task.flies[i]
        return float(np.degrees(f.carrot.heading(self.physics, f.walker)))

    def roll_pitch_of(self, i: int) -> tuple[float, float]:
        """Attitude relative to the canonical flight attitude (0, 0 = flying level)."""
        f = self.task.flies[i]
        q = np.array(f.walker.get_pose(self.physics)[1])
        dq = get_dquat_local(mult_quat(yaw_quat(f.carrot.heading(self.physics, f.walker)), f.carrot.q_flight), q)
        w, x, y, z = dq
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        return float(roll), float(pitch)

    @property
    def pos(self) -> np.ndarray:  # fly 0, for the single-fly scripts
        return self.pos_of(0)

    @property
    def yaw_deg(self) -> float:
        return self.yaw_of(0)

    def roll_pitch_deg(self) -> tuple[float, float]:
        return self.roll_pitch_of(0)

    # -------------------------------------------------------------- stepping
    def warmup(self, seconds: float) -> None:
        """Let the policy fly the flies alone (hover command) until the start transient is over."""
        for f in self.task.flies:
            f.carrot.set_command(MotorCommand.idle(0.0), TICK_MS)
        for _ in range(round(seconds * 1000 / self.env.control_timestep() / 1000)):
            self.actions = self.policy.batch(self._observations()).ravel()
            self.ts = self.env.step(self.actions)

    def settle(self, seconds: float) -> None:
        frames = self.frames()
        for _ in range(round(seconds * 1000 / TICK_MS)):
            self.agent.tick(frames)

    def frames(self) -> list[PanoramicFrame]:
        """The last tick's time-averaged panoramas (single snapshots before the first tick)."""
        if self._accum is None:
            return [eye.render(self.renderer, self.physics.data.ptr) for eye in self.eyes]
        return [PanoramicFrame(a / self.exposures) for a in self._accum]

    def _observations(self) -> list[dict]:
        """Each fly's observation dict, keyed as the policy expects (`walker/...`)."""
        out = []
        for f in self.task.flies:
            pre = f.name + "/"
            out.append({"walker/" + k[len(pre):]: v for k, v in self.ts.observation.items() if k.startswith(pre)})
        return out

    def step(self):
        self.last_frames = self.frames()
        cmds = self.agent.tick(self.last_frames)
        for f, cmd in zip(self.task.flies, cmds):
            f.carrot.set_command(cmd, TICK_MS)
        dt = self.env.control_timestep()
        every = max(1, self.policy_per_tick // self.exposures)
        accum = [np.zeros((eye.h, eye.w), dtype=np.float32) for eye in self.eyes]
        n_exp = 0
        for k in range(self.policy_per_tick):
            self.actions = self.policy.batch(self._observations()).ravel()
            if self.spin_drum:
                self._drum_angle -= np.radians(self.spin_drum) * dt
                self.physics.data.ptr.mocap_quat[self._drum_mocap] = [np.cos(self._drum_angle / 2), 0, 0,
                                                                     np.sin(self._drum_angle / 2)]
            self.hand.step(dt)
            self.ts = self.env.step(self.actions)
            if (k + 1) % every == 0 and n_exp < self.exposures:
                for a, eye in zip(accum, self.eyes):
                    a += eye.render(self.renderer, self.physics.data.ptr).lum
                n_exp += 1
        self._accum = [a * (self.exposures / n_exp) for a in accum]
        self.tick += 1
        self.last_commands = cmds
        self._log(cmds)
        return cmds

    def reset(self, warmup_s: float = 0.3, settle_s: float = 0.5) -> None:
        """A fresh episode (new wing-beat phase, flies back at their starts) on the same model,
        brain and policy -- a new trial without paying for a second brain."""
        self.ts = self.env.reset()
        self.physics = self.env.physics
        self._bind_model()
        self.spin_drum = 0.0
        self._drum_angle = 0.0
        self._accum = None
        self.tick = 0
        self.rows.clear()
        self.warmup(warmup_s)
        self.settle(settle_s)

    def run(self, seconds: float, on_tick=None) -> pl.DataFrame:
        for _ in range(round(seconds * 1000 / TICK_MS)):
            self.step()
            if on_tick is not None:
                on_tick(self)
        return self.log()

    def _log(self, cmds):
        rates = getattr(self.agent, "dn_rates", None)
        hand = self.hand.pos
        n_act = self.actions.size // self.n_agents
        for i, (f, cmd) in enumerate(zip(self.task.flies, cmds)):
            pos = self.pos_of(i)
            roll, pitch = self.roll_pitch_of(i)
            user = self.actions[i * n_act + f.user_idx]
            row = {"t_ms": self.tick * TICK_MS, "agent": i, "x": pos[0], "y": pos[1], "z": pos[2], "yaw_deg": self.yaw_of(i),
                   "roll_deg": roll, "pitch_deg": pitch, "forward": float(cmd.forward), "yaw": float(cmd.yaw),
                   "lift": float(cmd.lift), "escape": bool(cmd.escape), "escaping": bool(f.carrot.escape_left_ms > 0),
                   "hand_dist": float(np.linalg.norm(hand - pos)),
                   "wingbeat_hz": float(f.wbpg.base_beat_freq * (1 + f.wbpg.rel_freq_range * user))}
            if rates is not None and self._ids:
                for k, idx in self._ids.items():
                    row[f"{k}_hz"] = float(rates[i, idx].mean()) if len(idx) else 0.0
            self.rows.append(row)

    def log(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows)

    # --------------------------------------------------------------- controls
    def hide_fly(self, i: int) -> None:
        """Make fly i a ghost: invisible to every eye and without contacts."""
        m = self.physics.model.ptr
        pre = self.task.flies[i].name + "/"
        for g in range(m.ngeom):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(pre):
                m.geom_rgba[g, 3] = 0.0
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0

    def speed(self, seconds: float = 1.0) -> float:
        n = round(seconds * 1000 / TICK_MS)
        t = time.perf_counter()
        for _ in range(n):
            self.step()
        return n / (time.perf_counter() - t)

    def close(self) -> None:
        self.renderer.close()
